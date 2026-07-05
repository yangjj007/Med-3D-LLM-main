"""
Trainer for Sparse SDF VQVAE model.
Multi-region weighted L2 loss design following Direct3D-S2 §3.2:
  L_total = λ_inside * L_inside + λ_sharp * L_sharp + λ_extra * L_extra
          + λ_vq * L_vq + λ_commitment * L_commitment

Data flow:
  - npz files are preprocessed with a wide SDF band (preprocessing_extra_band_factor,
    e.g. 4.0 voxels from surface).  The full band is loaded by the dataset and used
    to build a GT lookup table covering all possible decoder outputs.
  - The ENCODER receives only the tight surface band (input_band_factor, e.g. 0.5 voxels).
    In normalised SDF space (where 1.0 = udf_max = extra_band/resolution):
      encoder input threshold = input_band_factor / preprocessing_extra_band_factor
      default: 0.5 / 4.0 = 0.125 in normalised SDF units

Regions (all in decoder output space):
  - inside region : decoder output voxels matching encoder INPUT coords (tight band)
  - sharp region  : inside voxels flagged as sharp/edge (edge_mask=True)
  - extra region  : decoder output voxels NOT in encoder input; these lie in the
                    wider band (0.5–4.0 voxels) and are supervised using GT SDF
                    values from the wide-band GT lookup table.
  - OOR           : extra voxels with no GT coverage (beyond the full 4.0-band);
                    their samples are skipped from the loss.

VQ commitment/codebook losses replace the KL term from the VAE version.
"""

from typing import *
import copy
import json
import os
import re
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
from easydict import EasyDict as edict

from ..basic import BasicTrainer


def _vqvae_disable_gradient_checkpoint_subtree(vqvae: torch.nn.Module) -> int:
    """
    将 vqvae 子树中所有 ``use_checkpoint`` 置为 False（含 encoder/decoder 内各 Block）。

    当 ``batch_split==1`` 时无 micro-batch 累积，整批一次前向即可完成一步，无需 checkpoint 换显存；
    同时可避免 torchsparse ImplicitGEMM 等与 gradient checkpoint 组合时的不稳定问题。
    """
    n = 0
    for m in vqvae.modules():
        if hasattr(m, "use_checkpoint") and getattr(m, "use_checkpoint", False):
            m.use_checkpoint = False
            n += 1
    return n
from ...modules import sparse as sp


# ---------------------------------------------------------------------------
# GT-lookup helpers
# ---------------------------------------------------------------------------

def build_gt_lookup(
    sparse_sdf: torch.Tensor,       # [total_N, 1]
    sparse_index: torch.Tensor,     # [total_N, 3]  int
    batch_idx: torch.Tensor,        # [total_N]     int
    resolution: int,
) -> Tuple[str, Any]:
    """
    Build a structure to query GT SDF values at arbitrary (batch, x, y, z) coords.

    ``resolution`` must be the **voxel grid size** of the sparse SDF data (same as
    dataset / npz ``…_r{R}.npz``), NOT the VQVAE model's internal ``resolution``
    hyperparameter (often 64 while data are 512³).

    Strategy:
      - If resolution <= 128  → dense scatter into [B, R, R, R] float tensor.
        Unoccupied voxels are filled with NaN (sentinel for OOR).
      - Otherwise             → sorted-key hash table using int64 keys.
        Key = batch * R^3 + x * R^2 + y * R + z.

    Returns:
        (mode, payload) where mode is 'dense' or 'hash'.
    """
    device = sparse_sdf.device
    B = int(batch_idx.max().item()) + 1
    R = resolution
    R3 = R * R * R

    sdf_flat = sparse_sdf.squeeze(-1)  # [total_N]
    ix = sparse_index[:, 0].long()
    iy = sparse_index[:, 1].long()
    iz = sparse_index[:, 2].long()
    bi = batch_idx.long()

    # Host-side bounds check (avoids CUDA indexKernel asserts on scatter)
    if ix.numel() > 0:
        max_x = int(ix.max().item())
        max_y = int(iy.max().item())
        max_z = int(iz.max().item())
        min_x = int(ix.min().item())
        min_y = int(iy.min().item())
        min_z = int(iz.min().item())
        if min_x < 0 or min_y < 0 or min_z < 0:
            raise ValueError(
                f"[build_gt_lookup] sparse_index must be non-negative; "
                f"got min ({min_x},{min_y},{min_z})"
            )
        if max_x >= R or max_y >= R or max_z >= R:
            raise ValueError(
                f"[build_gt_lookup] sparse_index out of grid [0,{R-1}]³: "
                f"max ({max_x},{max_y},{max_z}) for R={R}.  "
                "Pass grid_resolution from the dataset (npz voxel grid size). "
                "Do not use VQVAE.model.resolution when data were voxelized at a different R."
            )
        if bi.numel() > 0 and (int(bi.min().item()) < 0 or int(bi.max().item()) >= B):
            raise ValueError(
                f"[build_gt_lookup] batch_idx out of range [0,{B-1}]: "
                f"min={int(bi.min().item())} max={int(bi.max().item())}"
            )

    if R <= 128:
        # Dense mode: [B, R, R, R] with NaN sentinels
        vol = torch.full((B, R, R, R), float('nan'), device=device, dtype=torch.float32)
        vol[bi, ix, iy, iz] = sdf_flat.float()
        return ('dense', vol)
    else:
        # Hash mode: sorted int64 keys + corresponding SDF values
        keys = bi * R3 + ix * R * R + iy * R + iz  # [total_N] int64
        sort_idx = torch.argsort(keys)
        sorted_keys = keys[sort_idx]
        sorted_vals = sdf_flat.float()[sort_idx]
        return ('hash', (sorted_keys, sorted_vals, R3, R))


def _query_gt_lookup(
    lookup: Tuple[str, Any],
    out_batch_ids: torch.Tensor,  # [M] int  (batch index per output voxel)
    out_coords: torch.Tensor,     # [M, 3] int (x,y,z of output voxels)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Query GT SDF values for M output voxel positions.

    Returns:
        gt_vals : [M] float  – GT SDF for each output voxel (NaN / -inf for OOR)
        in_range: [M] bool   – True if the voxel has a valid GT value
    """
    mode, payload = lookup
    ix = out_coords[:, 0].long()
    iy = out_coords[:, 1].long()
    iz = out_coords[:, 2].long()
    bi = out_batch_ids.long()

    if mode == 'dense':
        vol = payload  # [B, R, R, R]
        B, R, _, _ = vol.shape
        # Clamp coords to [0, R-1] for safe indexing, then check OOR separately
        cx = ix.clamp(0, R - 1)
        cy = iy.clamp(0, R - 1)
        cz = iz.clamp(0, R - 1)
        cb = bi.clamp(0, B - 1)
        gt_vals = vol[cb, cx, cy, cz]  # NaN where unoccupied
        # OOR: coords out of grid bounds, or batch id out of range, or NaN sentinel
        oob = (ix < 0) | (ix >= R) | (iy < 0) | (iy >= R) | (iz < 0) | (iz >= R) | (bi >= B)
        in_range = ~oob & ~torch.isnan(gt_vals)
    else:
        sorted_keys, sorted_vals, R3, R = payload
        B_max = sorted_keys.max().item() // R3 + 1 if sorted_keys.numel() > 0 else 0
        R_t = torch.tensor(R, device=ix.device, dtype=torch.long)
        keys = bi * R3 + ix * R * R + iy * R + iz
        oob = (ix < 0) | (ix >= R_t) | (iy < 0) | (iy >= R_t) | (iz < 0) | (iz >= R_t)
        # searchsorted to find position
        pos = torch.searchsorted(sorted_keys, keys)
        pos_clamped = pos.clamp(0, sorted_keys.shape[0] - 1)
        found = (sorted_keys[pos_clamped] == keys) & ~oob
        gt_vals = torch.where(found, sorted_vals[pos_clamped], torch.tensor(float('nan'), device=ix.device))
        in_range = found

    return gt_vals, in_range


def classify_output_voxels(
    out_coords: torch.Tensor,   # [M, 4]  (batch, x, y, z) from recon.coords
    gt_lookup: Tuple[str, Any],
    instance_ids: Optional[List[str]] = None,
    resolution: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Classify output voxels into inside vs extra, query GT for extra, detect OOR.

    Returns:
        inside_mask   : [M] bool – output voxels that have a GT match (in_range)
        extra_mask    : [M] bool – output voxels NOT in GT (not inside)
        gt_for_extra  : [M] float – GT SDF values for extra voxels (valid only where extra & ~oor)
        oor_mask      : [M] bool – True for extra voxels with OOR coord (no GT)
        oor_flags     : [B] bool – per-sample flag: True if ANY of its extras are OOR
    """
    device = out_coords.device
    out_batch_ids = out_coords[:, 0]
    out_xyz = out_coords[:, 1:4]

    gt_vals, in_range = _query_gt_lookup(gt_lookup, out_batch_ids, out_xyz)

    inside_mask = in_range
    extra_mask  = ~in_range

    # For extra voxels: gt_vals is NaN when OOR
    oor_mask = extra_mask & torch.isnan(gt_vals)

    # per-sample OOR flag
    B_max = int(out_batch_ids.max().item()) + 1 if out_coords.shape[0] > 0 else 1
    oor_flags = torch.zeros(B_max, dtype=torch.bool, device=device)
    if oor_mask.any():
        oor_sample_ids = out_batch_ids[oor_mask].long()
        oor_flags.scatter_(0, oor_sample_ids, torch.ones_like(oor_sample_ids, dtype=torch.bool))

    return inside_mask, extra_mask, gt_vals, oor_mask, oor_flags


def _print_diagnose_zero_encoder_band(
    *,
    sparse_sdf: torch.Tensor,
    sparse_index: torch.Tensor,
    batch_idx: torch.Tensor,
    input_sdf_thresh: float,
    preprocessing_extra_band_factor: float,
    input_band_factor: float,
    R_gt: int,
    N_total: int,
    instance_ids: Optional[List[str]],
    dataset_indices: Optional[torch.Tensor],
    step: int,
) -> None:
    """
    N_enc==0 时打印诊断：帮助区分「阈值过严」「npz 尺度异常」「DDP 某 rank 抽到坏样本」等。
    """
    try:
        import torch.distributed as dist
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else -1
        world = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    except Exception:
        rank, world = -1, 1

    absv = sparse_sdf.detach().abs().float().reshape(-1)
    sdf1 = sparse_sdf.detach().float().reshape(-1)
    t = float(input_sdf_thresh)
    n_le = int((absv <= t).sum().item())
    n_le_05 = int((absv <= 0.05).sum().item())
    n_le_20 = int((absv <= 0.20).sum().item())
    n_pos = int((sdf1 > 0).sum().item())
    n_neg = int((sdf1 < 0).sum().item())
    n_zero = int((sdf1 == 0).sum().item())

    qs = torch.tensor([0.0, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0], device=absv.device)
    qv = torch.quantile(absv, qs).detach().cpu().tolist()

    sep = '=' * 88
    print(f"\n{sep}\n[ZERO-ENC-BAND DIAG] step={step} rank={rank}/{world} N_total={N_total} R_gt={R_gt}")
    if instance_ids:
        print(f"  instance_ids (sha256): {instance_ids}")
    if dataset_indices is not None:
        print(f"  dataset_indices: {dataset_indices.detach().cpu().tolist()}")
    print(
        f"  Trainer: input_band_factor={input_band_factor}  "
        f"preprocessing_extra_band_factor={preprocessing_extra_band_factor}  "
        f"=> input_sdf_thresh={t:.6f} (normalized |SDF| must be <= this for encoder)"
    )
    print(
        f"  Counts: |sdf|<={t:.4f}: {n_le}  |sdf|<=0.05: {n_le_05}  |sdf|<=0.20: {n_le_20}  "
        f"sign(+/-/0): {n_pos}/{n_neg}/{n_zero}"
    )
    print(f"  |sdf| quantiles (p -> value): {list(zip(qs.tolist(), qv))}")
    if sparse_index.numel() > 0:
        ix = sparse_index[:, 0].long()
        iy = sparse_index[:, 1].long()
        iz = sparse_index[:, 2].long()
        print(
            f"  sparse_index bbox: x=[{ix.min().item()},{ix.max().item()}] "
            f"y=[{iy.min().item()},{iy.max().item()}] z=[{iz.min().item()},{iz.max().item()}]"
        )
    cap_hint = ""
    if N_total in (500_000, 1_000_000) or (N_total >= 400_000 and N_total % 1000 == 0):
        cap_hint = (
            f"\n  Hint: N_total={N_total} 常见于 dataset max_points 上限子采样。"
            "若全量 npz 中近表面点很少，随机子采样可能抽不到 |sdf|<=阈值的体素（概率极低但理论存在）；"
            "更常见是本条 npz 的 SDF 未按 mesh_utils 归一化或壳层不含近表面。"
        )
    print(
        f"  常见原因:\n"
        f"    1) DDP: 各 GPU 样本不同；仅某一 rank 报错 = 该 rank 当前 batch 的 npz 异常或阈值与预处理不一致。\n"
        f"    2) 预处理 voxelize 使用的 extra_band_factor 与训练 preprocessing_extra_band_factor 不一致，"
        f"归一化 udf_max 错导致 |sdf| 整体偏大。\n"
        f"    3) input_band_factor 过严（相对预处理带宽）。\n"
        f"    4) 损坏/非标准 npz（例如缺少 |sdf|≈0 附近的壳层点）。{cap_hint}"
    )
    print(f"{sep}\n", flush=True)


class SparseSDF_VQVAETrainer(BasicTrainer):
    """
    Trainer for Sparse SDF VQVAE.
    
    Args:
        models (dict[str, nn.Module]): Models to train. Should contain 'vqvae' key.
        dataset (torch.utils.data.Dataset): Dataset returning sparse SDF data.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU.
        batch_split (int): Split batch with gradient accumulation.
            若为 ``1``，训练器会在初始化时关闭 ``vqvae`` 全树的 ``gradient checkpoint``
            （与整批一次前向一致，无需再省激活显存）。
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
        fp16_scale_growth (float): Scale growth for FP16.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print / i_log / i_sample / i_save / i_ddpcheck: Logging intervals.

        lambda_vq (float): VQ codebook loss weight (gradient mode only).
        lambda_commitment (float): VQ commitment loss weight.
        lambda_inside (float): Weight for the inside-region L2 reconstruction loss.
        lambda_sharp (float): Weight for the sharp/edge-region L2 loss (subset of inside).
        lambda_extra (float): Weight for the extra-region L2 loss (decoder-only voxels).
        loss_type (str): Per-voxel loss function: 'l2' (MSE), 'l1', or 'l1_l2'.
        extra_oor_skip_policy (str): When ``extra_oor_ratio_threshold`` is set and
            ``n_oor / M_out`` (M_out = decoder output voxel count) exceeds it:
            ``'sample'`` / ``'step'`` / ``'skip_step'`` – skip backward and optimizer
            for this step on **all** DDP ranks if any rank triggers (frees graph, avoids
            OOM on catastrophic OOR ratios); ``'inside_sharp_only'`` – still backward
            but set ``λ_extra * L_extra`` to zero (only inside + sharp + VQ/commit).
            ``'none'`` – ignore threshold.
        extra_oor_ratio_threshold (float | None): Ratio ``n_oor / M_out``; ``None``
            disables the gate (default).
        debug_loss_verbose (bool): Print per-step detailed debug output.
        debug_loss_histogram_every_n_steps (int): Print SDF residual histogram every N steps.
        input_band_factor (float): Encoder-input tight band (voxel units, e.g. 0.5).
        preprocessing_extra_band_factor (float): Preprocessing wide SDF shell half-width (e.g. 4.0).
        decoder_band_prune (bool): If True, decoder drops voxels outside the wide-band AABB
            (GPU vectorized) during forward for speed and alignment with GT coverage.
        min_sharp_voxels_for_backward (int): If ``n_sharp`` (sharp inside voxels in decoder
            output) is **strictly less** than this value on any rank, **all** DDP ranks skip
            backward/optimizer for this step (same mechanism as OOR skip_step): releases the
            graph and avoids unstable / exploding ``L_sharp`` when only 0–1 sharp voxels exist.
            Default ``2`` skips exactly ``n_sharp in {0, 1}``. Set ``0`` to disable.
        pretrained_vae_path (str): Path to pretrained VAE checkpoint.
        training_stage (int): 1=codebook only; 2=joint; 3=decoder only.
    """
    
    def __init__(
        self,
        models,
        dataset,
        *,
        lambda_vq: float = 1.0,
        lambda_commitment: float = 0.25,
        lambda_inside: float = 1.0,
        lambda_sharp: float = 1.0,
        lambda_extra: float = 0.1,
        # Legacy param kept for checkpoint compatibility – prints deprecation warning
        lambda_unmatched_output: float = None,
        loss_type: str = 'l2',
        extra_oor_skip_policy: str = 'sample',
        extra_oor_ratio_threshold: Optional[float] = None,
        min_sharp_voxels_for_backward: int = 2,
        debug_loss_verbose: bool = True,
        debug_loss_histogram_every_n_steps: int = 50,
        # ── Input-band filtering ─────────────────────────────────────────────
        # The npz files are preprocessed with a wide SDF band
        # (preprocessing_extra_band_factor, e.g. 4.0 voxels) so the GT lookup
        # covers decoder-predicted extra voxels.  Only a tighter subset of
        # voxels (input_band_factor, e.g. 0.5 voxels) is fed to the encoder;
        # points beyond that are the "extra" region the decoder must reconstruct.
        #
        # In normalized SDF space (SDF in npz is divided by udf_max =
        # preprocessing_extra_band_factor / resolution):
        #   encoder input threshold = input_band_factor / preprocessing_extra_band_factor
        #   e.g. 0.5 / 4.0 = 0.125
        input_band_factor: float = 0.5,
        preprocessing_extra_band_factor: float = 4.0,
        decoder_band_prune: bool = True,
        # ── Inference-time pruning (no GT leak) ──────────────────────────────
        # inference_band_factor: 推理 Decode 时的紧带宽度（R_final voxel 单位）。
        #   None → 回退到 input_band_factor（默认 0.5）。
        # inference_occ_resolution: 推理 geometry occ 的分辨率上限。
        #   None → 256（R_final=512 上 2 voxel 紧度；想要 1 voxel 用 512）。
        inference_band_factor: Optional[float] = None,
        inference_occ_resolution: Optional[int] = None,
        # ────────────────────────────────────────────────────────────────────
        pretrained_vae_path: str = None,
        training_stage: int = 1,
        load_dir=None,
        step=None,
        **kwargs
    ):
        # Legacy compat
        if lambda_unmatched_output is not None:
            import warnings
            warnings.warn(
                "lambda_unmatched_output is deprecated and ignored.  "
                "Use lambda_extra instead (extra-region is now supervised with GT SDF).",
                DeprecationWarning,
                stacklevel=2,
            )

        # 保存配置，稍后使用
        self.lambda_vq = lambda_vq
        self.lambda_commitment = lambda_commitment
        self.lambda_inside = lambda_inside
        self.lambda_sharp = lambda_sharp
        self.lambda_extra = lambda_extra
        self.loss_type = loss_type
        self.extra_oor_skip_policy = extra_oor_skip_policy
        self.extra_oor_ratio_threshold = extra_oor_ratio_threshold
        self.min_sharp_voxels_for_backward = int(min_sharp_voxels_for_backward)
        self.debug_loss_verbose = debug_loss_verbose
        self.debug_loss_histogram_every_n_steps = debug_loss_histogram_every_n_steps
        self.input_band_factor = input_band_factor
        self.preprocessing_extra_band_factor = preprocessing_extra_band_factor
        self.decoder_band_prune = bool(decoder_band_prune)
        # Normalized SDF threshold for the encoder input (0.125 for default 0.5/4.0)
        self.input_sdf_thresh = input_band_factor / preprocessing_extra_band_factor
        # 推理时裁剪参数（无 GT 泄露路径）
        self.inference_band_factor    = inference_band_factor    # None → 用 input_band_factor
        self.inference_occ_resolution = inference_occ_resolution # None → 256
        self.training_stage = training_stage
        self.pretrained_vae_path = pretrained_vae_path
        self._oor_skip_streak = 0  # consecutive steps with >50% samples skipped
        
        # 检查是否有实际的checkpoint要加载
        # load_dir 可能被设置为 output_dir，但如果没有checkpoint文件，就不算"恢复训练"
        has_checkpoint_to_load = False
        if load_dir is not None and step is not None:
            # 显式提供了step，肯定要加载checkpoint
            has_checkpoint_to_load = True
        elif load_dir is not None:
            # 只提供了load_dir，检查是否真的有checkpoint文件
            import os
            import glob
            ckpt_files = glob.glob(os.path.join(load_dir, 'ckpts', 'misc_*.pt'))
            has_checkpoint_to_load = len(ckpt_files) > 0
        
        # 只有当：1) 提供了pretrained_vae_path，且 2) 没有checkpoint要恢复时，才加载预训练权重
        self._should_load_pretrained = (pretrained_vae_path is not None and 
                                       not has_checkpoint_to_load)
        
        print(f"\n{'='*80}")
        print(f"🔍 [DEBUG] SparseSDF_VQVAETrainer.__init__ 参数检查")
        print(f"{'='*80}")
        print(f"  pretrained_vae_path: {pretrained_vae_path}")
        print(f"  load_dir: {load_dir}")
        print(f"  step: {step}")
        print(f"  has_checkpoint_to_load: {has_checkpoint_to_load}")
        print(f"  _should_load_pretrained: {self._should_load_pretrained}")
        print(f"{'='*80}\n")

        # batch_split==1：整批一次前向，关闭 vqvae 全树 gradient checkpoint
        self._n_gc_disabled_for_batch_split_one = 0
        _bs = int(kwargs.get("batch_split", 1) or 1)
        if _bs == 1 and isinstance(models, dict) and "vqvae" in models:
            self._n_gc_disabled_for_batch_split_one = _vqvae_disable_gradient_checkpoint_subtree(
                models["vqvae"]
            )

        # 调用父类初始化
        super().__init__(models, dataset, load_dir=load_dir, step=step, **kwargs)

        if self._n_gc_disabled_for_batch_split_one > 0 and self.is_master:
            print(
                f"[SparseSDF_VQVAETrainer] batch_split=1：已在 vqvae 子模块关闭 gradient checkpoint "
                f"（共 {self._n_gc_disabled_for_batch_split_one} 处 use_checkpoint→False）",
                flush=True,
            )
    
    def init_models_and_more(self, **kwargs):
        """
        重写父类方法，在初始化DDP和收集参数之前先配置训练阶段（冻结参数）
        
        关键时序：
        1. 先加载预训练权重（如果需要且不是从checkpoint恢复）
        2. 配置训练阶段（冻结不需要训练的参数）
        3. 调用父类方法（收集参数、初始化DDP、optimizer等）
        
        这样可以确保DDP和optimizer只包含真正需要训练的参数，避免
        "parameters that were not used in producing loss" 错误
        """
        # ===== 阶段1：加载预训练权重（仅当不从checkpoint恢复时）=====
        if self._should_load_pretrained:
            if self.is_master:
                print(f"\n{'='*80}")
                print(f"[INFO] Loading pretrained VAE weights...")
                print(f"{'='*80}")
            self._load_pretrained_vae(self.pretrained_vae_path)
        
        # ===== 阶段2：配置训练阶段（冻结参数）=====
        # ⚠️ 关键：必须在父类的init_models_and_more之前调用
        # 这样父类收集model_params时就不会包含冻结的参数
        self._configure_training_stage()
        
        # ===== 阶段3：调用父类方法 =====
        # 此时会：
        # - 收集model_params（只包含requires_grad=True的参数）✅
        # - 初始化DDP（只包含可训练参数，避免unused parameter错误）✅
        # - 初始化optimizer（只优化可训练参数）✅
        if self.is_master:
            print(f"\n{'='*80}")
            print(f"[INFO] Initializing trainer components with correct parameter set...")
            print(f"{'='*80}")
        super().init_models_and_more(**kwargs)
        
        if self.is_master:
            print(f"\n{'='*80}")
            print(f"[SUCCESS] Trainer initialization complete!")
            print(f"  - Model parameters: {len(self.model_params)}")
            print(f"  - Optimizer parameters: {sum(len(g['params']) for g in self.optimizer.param_groups)}")
            print(f"{'='*80}\n")
    
    def _load_pretrained_vae(self, pretrained_vae_path: str):
        """
        Load pretrained VAE weights.
        
        Args:
            pretrained_vae_path: Path to pretrained VAE checkpoint (.pth file)
        """
        if self.is_master:
            print(f'\n{"="*80}')
            print(f'🔧 [DEBUG] _load_pretrained_vae 被调用')
            print(f'{"="*80}')
            print(f'📁 预训练权重路径: {pretrained_vae_path}')
            print(f'📋 self._should_load_pretrained: {self._should_load_pretrained}')
        
        # Load checkpoint
        if self.is_master:
            print(f'\n📦 正在加载 checkpoint...')
        checkpoint = torch.load(pretrained_vae_path, map_location='cpu', weights_only=True)
        if self.is_master:
            print(f'✅ Checkpoint 加载成功')
            print(f'   顶层键: {list(checkpoint.keys())}')
        
        # Extract VAE state dict
        if 'vae' in checkpoint:
            vae_state_dict = checkpoint['vae']
            if self.is_master:
                print(f'   使用键: "vae"')
        elif 'state_dict' in checkpoint:
            vae_state_dict = checkpoint['state_dict']
            if self.is_master:
                print(f'   使用键: "state_dict"')
        else:
            vae_state_dict = checkpoint
            if self.is_master:
                print(f'   直接使用整个 checkpoint')
        
        # Get encoder and decoder state dicts
        encoder_state_dict = {
            k.replace('encoder.', ''): v
            for k, v in vae_state_dict.items()
            if k.startswith('encoder.')
        }
        decoder_state_dict = {
            k.replace('decoder.', ''): v
            for k, v in vae_state_dict.items()
            if k.startswith('decoder.')
        }
        
        # Get VQ state dict (if exists)
        vq_state_dict = {
            k.replace('vq.', ''): v
            for k, v in vae_state_dict.items()
            if k.startswith('vq.')
        }
        
        if self.is_master:
            print(f'\n📊 提取的参数统计:')
            print(f'   Encoder: {len(encoder_state_dict)} 个参数')
            print(f'   Decoder: {len(decoder_state_dict)} 个参数')
            print(f'   VQ: {len(vq_state_dict)} 个参数')
            
            if vq_state_dict:
                print(f'\n   VQ 参数详情:')
                for key, value in vq_state_dict.items():
                    if isinstance(value, torch.Tensor):
                        print(f'     - {key}: shape={value.shape}, dtype={value.dtype}')
                        if key == 'embeddings.weight':
                            print(f'       统计: min={value.min().item():.6f}, max={value.max().item():.6f}, '
                                  f'mean={value.mean().item():.6f}, std={value.std().item():.6f}')
                            print(f'       前3个code的前5维:')
                            for i in range(min(3, value.shape[0])):
                                print(f'         Code {i}: {value[i, :5].tolist()}')
            else:
                print(f'   ⚠️  VQ state dict 是空的！')
        
        # Load into VQVAE model
        vqvae = self.models['vqvae']
        if hasattr(vqvae, 'module'):
            vqvae = vqvae.module
            if self.is_master:
                print(f'\n🔧 模型被 DDP 包装，使用 module 属性')
        
        if self.is_master:
            print(f'\n🔍 检查模型是否有 load_pretrained_vae 方法...')
            print(f'   hasattr(vqvae, "load_pretrained_vae"): {hasattr(vqvae, "load_pretrained_vae")}')
        
        if hasattr(vqvae, 'load_pretrained_vae'):
            if self.is_master:
                print(f'\n📥 调用 vqvae.load_pretrained_vae()...')
                # 在加载前记录当前codebook
                print(f'\n📊 加载前的 Codebook 统计:')
                current_embeddings = vqvae.vq.embeddings.weight.data
                print(f'   Shape: {current_embeddings.shape}')
                print(f'   Min: {current_embeddings.min().item():.6f}, Max: {current_embeddings.max().item():.6f}')
                print(f'   Mean: {current_embeddings.mean().item():.6f}, Std: {current_embeddings.std().item():.6f}')
                print(f'   前3个code的前5维:')
                for i in range(min(3, current_embeddings.shape[0])):
                    print(f'     Code {i}: {current_embeddings[i, :5].tolist()}')
            
            vqvae.load_pretrained_vae(encoder_state_dict, decoder_state_dict, vq_state_dict)
            
            if self.is_master:
                print(f'\n📊 加载后的 Codebook 统计:')
                new_embeddings = vqvae.vq.embeddings.weight.data
                print(f'   Shape: {new_embeddings.shape}')
                print(f'   Min: {new_embeddings.min().item():.6f}, Max: {new_embeddings.max().item():.6f}')
                print(f'   Mean: {new_embeddings.mean().item():.6f}, Std: {new_embeddings.std().item():.6f}')
                print(f'   前3个code的前5维:')
                for i in range(min(3, new_embeddings.shape[0])):
                    print(f'     Code {i}: {new_embeddings[i, :5].tolist()}')
                
                # 检查是否真的改变了（若码本条数不同，仅在重叠前缀上比较）
                if vq_state_dict and 'embeddings.weight' in vq_state_dict:
                    original_embeddings = vq_state_dict['embeddings.weight']
                    new_embeddings_cpu = new_embeddings.cpu()
                    if new_embeddings_cpu.shape[1] != original_embeddings.shape[1]:
                        print(
                            f'\n   ⚠️  预训练与当前 embedding_dim 不一致 '
                            f'({original_embeddings.shape[1]} vs {new_embeddings_cpu.shape[1]})，跳过差异统计'
                        )
                    else:
                        n_overlap = min(new_embeddings_cpu.shape[0], original_embeddings.shape[0])
                        diff = (
                            new_embeddings_cpu[:n_overlap] - original_embeddings[:n_overlap]
                        ).abs().max().item()
                        print(f'\n   ✅ 与前 {n_overlap} 个预训练 code 的最大差异: {diff:.6e}')
                        if original_embeddings.shape[0] != new_embeddings_cpu.shape[0]:
                            print(
                                f'   ℹ️  码本大小: checkpoint={original_embeddings.shape[0]}, '
                                f'当前={new_embeddings_cpu.shape[0]}；'
                                f'差异仅在重叠前缀 [{n_overlap}] 上计算'
                            )
                        if diff < 1e-6:
                            print(f'   ✅ 重叠前缀与预训练权重一致')
                        else:
                            print(f'   ⚠️  重叠前缀与预训练权重仍有数值差异（可能含 dtype/设备转换）')
                
                print(f'\n✅ 预训练 VAE 权重加载完成')
                print(f'{"="*80}\n')
        else:
            if self.is_master:
                print(f'❌ 警告: VQVAE 模型没有 load_pretrained_vae 方法')
                print(f'{"="*80}\n')
    
    def _configure_training_stage(self):
        """
        Configure model parameters based on training stage.
        
        Stage 1: Freeze encoder and decoder, train only codebook
        Stage 2: Unfreeze all parameters for joint training
        Stage 3: Freeze encoder and codebook, train decoder only
        """
        vqvae = self.models['vqvae']
        if hasattr(vqvae, 'module'):
            vqvae = vqvae.module

        if hasattr(vqvae, 'freeze_codebook_updates'):
            vqvae.freeze_codebook_updates = False
        
        if self.training_stage == 1:
            # Stage 1: Freeze encoder and decoder
            if hasattr(vqvae, 'encoder'):
                for param in vqvae.encoder.parameters():
                    param.requires_grad = False
            
            if hasattr(vqvae, 'decoder'):
                for param in vqvae.decoder.parameters():
                    param.requires_grad = False
            
            # Ensure VQ parameters are trainable (except for EMA mode where embeddings have no grad)
            if hasattr(vqvae, 'vq'):
                for param in vqvae.vq.parameters():
                    param.requires_grad = True
                # 但如果是EMA模式，码本权重不需要梯度
                if hasattr(vqvae, 'use_ema_update') and vqvae.use_ema_update:
                    vqvae.vq.embeddings.weight.requires_grad = False
            
            if self.is_master:
                print("\n" + "=" * 80)
                print("[Stage 1] Encoder and Decoder frozen, training Codebook only")
                
                # 打印码本更新模式
                if hasattr(vqvae, 'use_ema_update'):
                    if vqvae.use_ema_update:
                        print(f"[Codebook Update Mode] EMA (decay={vqvae.vq.decay}, epsilon={vqvae.vq.epsilon})")
                    else:
                        print(f"[Codebook Update Mode] Gradient (lambda_vq={self.lambda_vq})")
                
                print("=" * 80)
                
                # Count trainable parameters
                total_params = sum(p.numel() for p in vqvae.parameters())
                trainable_params = sum(p.numel() for p in vqvae.parameters() if p.requires_grad)
                print(f"Total parameters: {total_params:,}")
                print(f"Trainable parameters: {trainable_params:,}")
                print(f"Frozen parameters: {total_params - trainable_params:,}")
                print("=" * 80 + "\n")
        
        elif self.training_stage == 2:
            # Stage 2: Unfreeze all parameters
            for param in vqvae.parameters():
                param.requires_grad = True
            
            # 但如果是EMA模式，码本权重不需要梯度
            if hasattr(vqvae, 'use_ema_update') and vqvae.use_ema_update:
                vqvae.vq.embeddings.weight.requires_grad = False
            
            if self.is_master:
                print("\n" + "=" * 80)
                print("[Stage 2] Joint training: Encoder + Decoder + Codebook")
                
                # 打印码本更新模式
                if hasattr(vqvae, 'use_ema_update'):
                    if vqvae.use_ema_update:
                        print(f"[Codebook Update Mode] EMA (decay={vqvae.vq.decay}, epsilon={vqvae.vq.epsilon})")
                    else:
                        print(f"[Codebook Update Mode] Gradient (lambda_vq={self.lambda_vq})")
                
                print("=" * 80)
                
                # Count trainable parameters
                total_params = sum(p.numel() for p in vqvae.parameters())
                trainable_params = sum(p.numel() for p in vqvae.parameters() if p.requires_grad)
                print(f"Total parameters: {total_params:,}")
                print(f"Trainable parameters: {trainable_params:,}")
                print("=" * 80 + "\n")
        
        elif self.training_stage == 3:
            # Stage 3: Freeze encoder and codebook; train only decoder.
            if hasattr(vqvae, 'encoder'):
                for param in vqvae.encoder.parameters():
                    param.requires_grad = False

            if hasattr(vqvae, 'vq'):
                for param in vqvae.vq.parameters():
                    param.requires_grad = False

            if hasattr(vqvae, 'decoder'):
                for param in vqvae.decoder.parameters():
                    param.requires_grad = True

            if hasattr(vqvae, 'freeze_codebook_updates'):
                vqvae.freeze_codebook_updates = True
            if hasattr(vqvae, 'train_full_block_decode'):
                vqvae.train_full_block_decode = True

            if self.is_master:
                print("\n" + "=" * 80)
                print("[Stage 3] Decoder-only training: Encoder + Codebook frozen")
                print("[Stage 3] Full-block decode is enabled to match inference Decode()")
                if hasattr(vqvae, 'use_ema_update') and vqvae.use_ema_update:
                    print("[Codebook Update Mode] Frozen (EMA/K-means updates disabled)")
                print("=" * 80)

                total_params = sum(p.numel() for p in vqvae.parameters())
                trainable_params = sum(p.numel() for p in vqvae.parameters() if p.requires_grad)
                print(f"Total parameters: {total_params:,}")
                print(f"Trainable parameters: {trainable_params:,}")
                print(f"Frozen parameters: {total_params - trainable_params:,}")
                print("=" * 80 + "\n")
        
        else:
            raise ValueError(f"Invalid training_stage: {self.training_stage}. Must be 1, 2, or 3.")
    
    def _per_voxel_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute per-voxel scalar loss according to self.loss_type."""
        if self.loss_type in ('l2', 'mse'):
            return F.mse_loss(pred, target, reduction='mean')
        elif self.loss_type == 'l1':
            return F.l1_loss(pred, target, reduction='mean')
        elif self.loss_type == 'l1_l2':
            return (0.5 * F.l1_loss(pred, target, reduction='mean') +
                    0.5 * F.mse_loss(pred, target, reduction='mean'))
        else:
            raise ValueError(f'Invalid loss_type: {self.loss_type!r}')

    def training_losses(
        self,
        sparse_sdf: torch.Tensor,
        sparse_index: torch.Tensor,
        batch_idx: torch.Tensor,
        edge_mask: Optional[torch.Tensor] = None,
        grid_resolution: Optional[Union[torch.Tensor, int]] = None,
        instance_ids: Optional[List[str]] = None,
        dataset_indices: Optional[torch.Tensor] = None,
        data_roots: Optional[List[str]] = None,
        **kwargs
    ) -> Tuple[Dict, Dict]:
        """
        Compute multi-region weighted L2 training losses.

        Regions:
          inside  – decoder output voxels matching GT input coordinates
          sharp   – inside voxels with edge_mask=True (subset of inside)
          extra   – decoder output voxels NOT in GT; supervised by wider GT band

        OOR safety: if any decoder extra voxel has no GT (coord outside the stored
        SDF band), the entire sample is skipped from that loss region (with a log).

        Returns:
            (terms_dict, status_dict)
        """
        step = self.step
        verbose = self.debug_loss_verbose
        SEP = '=' * 100

        print(f"\n{SEP}")
        print(f"[LOSS] ===== Step {step} - multi-region VQVAE loss =====")
        print(f"{SEP}")

        if instance_ids is not None:
            di = dataset_indices.detach().cpu().tolist() if dataset_indices is not None else None
            print(
                f"[BATCH_IDS] rank={self.rank} step={step} | "
                f"sha256={instance_ids} | dataset_index={di}"
            )

        # ── VQVAE model resolution vs data voxel grid ───────────────────────
        # GT lookup must use the same R as the npz voxel grid (e.g. 512), not
        # the model's internal resolution (often 64) — mixing them causes CUDA
        # index out of bounds in dense scatter.
        vqvae = self.training_models['vqvae']
        if hasattr(vqvae, 'module'):
            vqvae_module = vqvae.module
        else:
            vqvae_module = vqvae
        model_resolution = int(getattr(vqvae_module, 'resolution', 64))

        if grid_resolution is not None:
            if isinstance(grid_resolution, torch.Tensor):
                R_gt = int(grid_resolution.reshape(-1)[0].item())
            else:
                R_gt = int(grid_resolution)
        else:
            R_gt = int(sparse_index.max().item()) + 1 if sparse_index.numel() > 0 else model_resolution
            print(
                f"[training_losses] WARNING: grid_resolution not in batch; "
                f"inferred R_gt={R_gt} from sparse_index max. "
                f"Collate should pass grid_resolution from SparseSDF for correctness.",
                flush=True,
            )

        # ── Default edge_mask (all False) when not provided ─────────────────
        if edge_mask is None:
            edge_mask = torch.zeros(sparse_sdf.shape[0], dtype=torch.bool, device=sparse_sdf.device)

        N_total = sparse_sdf.shape[0]
        B = int(batch_idx.max().item()) + 1

        # ════════════════════════════════════════════════════════════════════
        # PHASE A – Input distribution debug (full wide-band dataset stats)
        # ════════════════════════════════════════════════════════════════════
        print(f"\n[PHASE-A] Dataset stats (full wide-band, N={N_total}, B={B}, "
              f"grid_resolution={R_gt}, model_resolution={model_resolution})")
        print(f"  preprocessing_extra_band_factor={self.preprocessing_extra_band_factor}  "
              f"input_band_factor={self.input_band_factor}  "
              f"input_sdf_thresh={self.input_sdf_thresh:.4f} (normalized)")
        print(f"  sparse_sdf  shape={sparse_sdf.shape} dtype={sparse_sdf.dtype}")
        sdf_min_v = sparse_sdf.min().item()
        sdf_max_v = sparse_sdf.max().item()
        sdf_mean_v = sparse_sdf.mean().item()
        near_surf = (sparse_sdf.abs() < 0.05).float().mean().item()
        print(f"  sparse_sdf  min={sdf_min_v:.4f} max={sdf_max_v:.4f} "
              f"mean={sdf_mean_v:.4f} |sdf|<0.05={near_surf:.3f}")
        n_sharp_total = int(edge_mask.sum().item())
        print(f"  edge_mask   sharp={n_sharp_total}/{N_total} "
              f"({100.*n_sharp_total/max(N_total,1):.2f}%)")

        # ── Build GT lookup from ALL wide-band data (4.0-band) ───────────────
        # This is used to look up GT SDF for decoder-predicted "extra" voxels
        # that lie beyond the encoder input band (0.5-band).
        gt_lookup_full = build_gt_lookup(sparse_sdf, sparse_index, batch_idx, R_gt)
        print(f"  GT lookup (full band) mode: {gt_lookup_full[0]}")

        # ── Filter encoder input to tight surface band (input_band_factor) ───
        # The encoder sees only the core surface voxels (|sdf| <= input_sdf_thresh).
        # Voxels in the wider band (0.5 to 4.0 voxels) will appear as "extra"
        # in the decoder output and are supervised via gt_lookup_full.
        enc_mask = sparse_sdf.abs().squeeze(-1) <= self.input_sdf_thresh
        enc_sdf       = sparse_sdf[enc_mask]       # [N_enc, 1]
        enc_index     = sparse_index[enc_mask]      # [N_enc, 3]
        enc_edge_mask = edge_mask[enc_mask]         # [N_enc]  bool
        enc_batch_idx = batch_idx[enc_mask]         # [N_enc]

        N_enc = enc_sdf.shape[0]
        print(f"\n[PHASE-A] Encoder input (tight band |sdf|<={self.input_sdf_thresh:.4f}): "
              f"N_enc={N_enc}/{N_total} ({100.*N_enc/max(N_total,1):.1f}%)")
        n_sharp_enc = int(enc_edge_mask.sum().item())
        print(f"  edge_mask (encoder): sharp={n_sharp_enc}/{N_enc} "
              f"({100.*n_sharp_enc/max(N_enc,1):.2f}%)")
        for bid in range(B):
            m = enc_batch_idx == bid
            em = enc_edge_mask[m]
            if m.any():
                print(f"    sample[{bid}]: N_enc={int(m.sum())} sharp_enc={int(em.sum())} "
                      f"sdf=[{enc_sdf[m].min().item():.3f},{enc_sdf[m].max().item():.3f}]")
            else:
                print(f"    sample[{bid}]: N_enc=0 (all wide-band, no tight-band points!)")

        if N_enc == 0:
            _print_diagnose_zero_encoder_band(
                sparse_sdf=sparse_sdf,
                sparse_index=sparse_index,
                batch_idx=batch_idx,
                input_sdf_thresh=self.input_sdf_thresh,
                preprocessing_extra_band_factor=self.preprocessing_extra_band_factor,
                input_band_factor=self.input_band_factor,
                R_gt=R_gt,
                N_total=N_total,
                instance_ids=instance_ids,
                dataset_indices=dataset_indices,
                step=int(self.step),
            )
            raise RuntimeError(
                f"[training_losses] N_enc=0: 全部 {N_total} 个体素的归一化 |SDF| > "
                f"{self.input_sdf_thresh:.4f}（encoder 紧带为空）。"
                f"已打印 [ZERO-ENC-BAND DIAG] 块。处理：检查该 sha256 的 npz、"
                f"voxelize 的 extra_band 与训练配置是否一致，或放宽 input_band_factor；"
                f"DDP 下仅某一 rank 报错属正常（该卡分到的样本更差）。"
            )

        # ── Build encoder-input lookup (for inside/extra classification) ─────
        # "inside" = decoder output matches an encoder INPUT coord (tight band)
        # "extra"  = decoder output is beyond the encoder input → supervised via full GT
        enc_lookup = build_gt_lookup(enc_sdf, enc_index, enc_batch_idx, R_gt)
        print(f"  Encoder lookup mode: {enc_lookup[0]}")

        # ── Model param stats ────────────────────────────────────────────────
        if verbose:
            enc_grad = any(p.requires_grad for p in vqvae_module.encoder.parameters())
            dec_grad = any(p.requires_grad for p in vqvae_module.decoder.parameters())
            vq_grad  = any(p.requires_grad for p in vqvae_module.vq.parameters())
            print(f"  grad flags: encoder={enc_grad} decoder={dec_grad} vq={vq_grad}")
            emb = vqvae_module.vq.embeddings.weight
            print(f"  codebook: shape={emb.shape} "
                  f"min={emb.min().item():.4f} max={emb.max().item():.4f} "
                  f"mean={emb.mean().item():.4f} std={emb.std().item():.4f} "
                  f"requires_grad={emb.requires_grad}")

        # ── Forward pass (encoder input = tight-band voxels only) ────────────
        enc_coords = torch.cat([enc_batch_idx.unsqueeze(-1), enc_index], dim=-1).int()
        x = sp.SparseTensor(enc_sdf, enc_coords)
        print(f"\n[PHASE-A] Calling VQVAE forward (N_enc={N_enc})...")
        band_prune = None
        if self.decoder_band_prune:
            band_prune = {
                "output_resolution": R_gt,
                "extra_band_factor": float(self.preprocessing_extra_band_factor),
                "wide_batch_idx": batch_idx,
                "wide_xyz": sparse_index,
            }

        # ── GT 键集合精确裁剪（在每个上采样步后使用 GT 占用格裁剪，
        #    最终输出层前使用 sorted_keys 精确匹配）──────────────────────────
        # 目的：把 decoder 在各中间分辨率（128/256/512）的活跃体素数精确限制在
        # GT 宽频带（preprocessing_extra_band_factor）覆盖的范围内，避免
        # 上千万体素在 torchsparse sparse conv backward 中申请 GiB 级临时缓冲区
        # 导致 CUDA OOM。sparse_index 是全宽频带 GT 坐标（R_gt 分辨率），与
        # preprocessing_extra_band_factor 对应，用作裁剪锚点。
        gt_prune = {
            'mode':         'keys',
            'gt_xyz':       sparse_index,   # [N, 3] int，全宽频带，at R_gt
            'gt_batch_idx': batch_idx,      # [N] per-voxel batch index
            'B':            B,
            'resolution':   R_gt,
        }
        outputs = vqvae(x, current_step=step, band_prune=band_prune, gt_prune=gt_prune)
        recon = outputs['reconst_x']
        vq_loss = outputs.get('vq_loss')
        commitment_loss = outputs['commitment_loss']
        codebook_stats = outputs.get('codebook_stats', {})

        recon_min = recon.feats.min().item()
        recon_max = recon.feats.max().item()
        recon_mean = recon.feats.mean().item()
        print(f"  recon: N_out={recon.feats.shape[0]} "
              f"min={recon_min:.4f} max={recon_max:.4f} mean={recon_mean:.4f} "
              f"requires_grad={recon.feats.requires_grad}")
        if vq_loss is not None:
            print(f"  vq_loss={vq_loss.item():.6f} requires_grad={vq_loss.requires_grad}")
        else:
            print(f"  vq_loss=None (EMA mode)")
        print(f"  commitment_loss={commitment_loss.item():.6f} "
              f"requires_grad={commitment_loss.requires_grad}")

        # ════════════════════════════════════════════════════════════════════
        # PHASE B – Classify output voxels into inside / extra / OOR
        #
        # inside = decoder output matches encoder INPUT coord (tight 0.5-band)
        # extra  = decoder output is beyond the encoder input (0.5 to 4.0 band)
        # OOR    = extra voxels not covered even by the full 4.0-band GT data
        # ════════════════════════════════════════════════════════════════════
        out_coords = recon.coords          # [M, 4]  (batch, x, y, z)
        M_out = out_coords.shape[0]

        # Step 1: inside/extra classification against encoder INPUT lookup
        enc_gt_vals, inside_in_enc = _query_gt_lookup(
            enc_lookup, out_coords[:, 0], out_coords[:, 1:4]
        )
        inside_mask = inside_in_enc
        extra_mask  = ~inside_in_enc

        # Step 2: get GT SDF values for ALL decoder outputs from full wide-band lookup
        gt_vals_full, full_in_range = _query_gt_lookup(
            gt_lookup_full, out_coords[:, 0], out_coords[:, 1:4]
        )
        # Use gt_vals_full as the authoritative GT for all decoder outputs
        gt_vals = gt_vals_full

        # Step 3: OOR = extra voxels not in the full GT lookup
        oor_mask = extra_mask & ~full_in_range

        # Per-sample OOR flags
        B_max = int(out_coords[:, 0].max().item()) + 1 if M_out > 0 else B
        oor_flags = torch.zeros(B_max, dtype=torch.bool, device=sparse_sdf.device)
        if oor_mask.any():
            oor_sample_ids = out_coords[:, 0][oor_mask].long()
            oor_flags.scatter_(0, oor_sample_ids,
                               torch.ones_like(oor_sample_ids, dtype=torch.bool))

        n_inside = int(inside_mask.sum().item())
        n_extra  = int(extra_mask.sum().item())
        n_oor    = int(oor_mask.sum().item())
        n_extra_valid = int((extra_mask & ~oor_mask).sum().item())

        print(f"\n[PHASE-B] Output voxel classification (M_out={M_out}, N_enc={N_enc})")
        print(f"  inside (matched encoder input, tight band): {n_inside}/{M_out}")
        print(f"  extra  (beyond encoder input → need full-band GT): {n_extra}/{M_out}  "
              f"valid={n_extra_valid}  OOR={n_oor}")
        if n_oor > 0:
            oor_sample_ids = out_coords[:, 0][oor_mask].unique().tolist()
            oor_coords_sample = out_coords[oor_mask][:5].tolist()
            print(f"  [OOR WARNING] OOR samples: {oor_sample_ids}")
            print(f"  [OOR WARNING] sample OOR coords (first 5): {oor_coords_sample}")
            print(f"  [OOR WARNING] full GT coord range: "
                  f"x=[{sparse_index[:,0].min().item()},{sparse_index[:,0].max().item()}] "
                  f"y=[{sparse_index[:,1].min().item()},{sparse_index[:,1].max().item()}] "
                  f"z=[{sparse_index[:,2].min().item()},{sparse_index[:,2].max().item()}]")
            if instance_ids:
                for sid in oor_sample_ids:
                    if sid < len(instance_ids):
                        print(f"    OOR sample sha256: {instance_ids[sid]}")

        # ── OOR handling: exclude OOR voxels from extra loss only ───────────
        # OOR voxels are already excluded in PHASE C via:
        #   extra_valid_mask = extra_mask & ~oor_mask
        # Skipping the entire sample is overly aggressive because it also removes
        # valid inside-region voxels. We simply log OOR counts and let PHASE C
        # handle the exclusion correctly.
        n_skip = int(oor_flags.sum().item())
        if n_skip > 0:
            print(f"\n[PHASE-B][OOR] {n_oor} OOR extra voxels detected across {n_skip}/{B} samples. "
                  f"OOR voxels will be excluded from extra loss in PHASE C (inside loss unaffected).")
        self._oor_skip_streak = 0

        # ── OOR ratio gate: ``n_oor / M_out`` (M_out = decoder output voxel count) ──
        M_out_i = max(int(out_coords.shape[0]), 1)
        oor_ratio = float(n_oor) / float(M_out_i)
        pol = (self.extra_oor_skip_policy or "none").strip().lower()
        thr = self.extra_oor_ratio_threshold
        force_zero_extra_loss = False
        trigger_local = False
        if thr is not None and float(thr) >= 0.0 and pol not in ("none", "", "off"):
            trigger_local = oor_ratio > float(thr)

        trigger = trigger_local
        if self.world_size > 1:
            try:
                if dist.is_available() and dist.is_initialized():
                    t = torch.tensor(
                        [1 if trigger_local else 0],
                        device=sparse_sdf.device,
                        dtype=torch.int32,
                    )
                    dist.all_reduce(t, op=dist.ReduceOp.MAX)
                    trigger = bool(t.item() != 0)
            except Exception:
                trigger = trigger_local

        skip_backward = bool(trigger and pol in ("sample", "step", "skip_step"))
        if trigger and pol in ("inside_sharp_only", "inside_sharp"):
            force_zero_extra_loss = True

        if skip_backward:
            thr_show = float(thr) if thr is not None else -1.0
            print(
                f"\n[PHASE-B][OOR_SKIP_STEP] oor_ratio={oor_ratio:.6f} > thr={thr_show:.6f} "
                f"(n_oor={n_oor}, M_out={M_out_i}) policy={pol!r} — 本步所有 DDP rank 跳过 backward/optimizer，"
                f"并释放解码计算图以避免 OOM。",
                flush=True,
            )
            codebook_stats_skip = dict(outputs.get("codebook_stats", {}))
            del outputs, recon, out_coords, inside_mask, extra_mask, oor_mask, oor_flags, gt_vals
            del enc_lookup, gt_lookup_full, enc_coords, x
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            dev = sparse_sdf.device
            zt = torch.zeros((), dtype=torch.float32, device=dev)
            terms = edict(
                loss=zt,
                loss_inside=zt,
                loss_sharp=zt,
                loss_extra=zt,
                commitment=zt,
            )
            status = edict(
                n_inside=0,
                n_sharp=0,
                n_extra=0,
                n_oor=n_oor,
                n_skip_samples=n_skip,
                loss_inside=0.0,
                loss_sharp=0.0,
                loss_extra=0.0,
                skip_backward=True,
                oor_ratio=oor_ratio,
                oor_skip_step=1.0,
                sharp_skip_step=0.0,
                codebook_perplexity=codebook_stats_skip.get("perplexity", 0.0),
                codebook_entropy=codebook_stats_skip.get("entropy", 0.0),
                codebook_utilization_ratio=codebook_stats_skip.get("utilization_ratio", 0.0),
                codebook_unique_count=codebook_stats_skip.get("unique_count", 0),
                codebook_batch_unique_count=codebook_stats_skip.get("batch_unique_count", 0),
            )
            print(
                f"\n[LOSS-SUMMARY] step={step} | SKIPPED backward (oor_ratio gate) "
                f"oor_ratio={oor_ratio:.4f}\n{SEP}\n",
                flush=True,
            )
            return terms, status

        if force_zero_extra_loss:
            thr_show = float(thr) if thr is not None else -1.0
            print(
                f"\n[PHASE-B][OOR_POLICY] inside_sharp_only: λ_extra→0 "
                f"(oor_ratio={oor_ratio:.6f} > thr={thr_show:.6f})",
                flush=True,
            )

        # ════════════════════════════════════════════════════════════════════
        # PHASE C – Compute multi-region losses
        # ════════════════════════════════════════════════════════════════════
        extra_w = 0.0 if force_zero_extra_loss else self.lambda_extra

        # ── Prepare edge_mask aligned to output inside voxels ────────────────
        # We use the ENCODER INPUT edge_mask (enc_edge_mask) since "inside" voxels
        # are exactly those matching the encoder input (tight 0.5-band).
        # Build a lookup keyed by encoder input coords → edge_mask value.
        _em_lookup_mode = enc_lookup[0]
        if _em_lookup_mode == 'dense':
            enc_vol = enc_lookup[1]
            B_v, R_v = enc_vol.shape[0], enc_vol.shape[1]
            em_vol = torch.zeros(B_v, R_v, R_v, R_v, dtype=torch.bool, device=enc_edge_mask.device)
            bi_in = enc_batch_idx.long()
            ix_in = enc_index[:, 0].long()
            iy_in = enc_index[:, 1].long()
            iz_in = enc_index[:, 2].long()
            em_vol[bi_in, ix_in, iy_in, iz_in] = enc_edge_mask
            # Look up for inside output voxels
            inside_out_coords = out_coords[inside_mask]  # [n_inside, 4]
            if inside_out_coords.shape[0] > 0:
                bi_o = inside_out_coords[:, 0].long().clamp(0, B_v - 1)
                xo   = inside_out_coords[:, 1].long().clamp(0, R_v - 1)
                yo   = inside_out_coords[:, 2].long().clamp(0, R_v - 1)
                zo   = inside_out_coords[:, 3].long().clamp(0, R_v - 1)
                edge_mask_inside = em_vol[bi_o, xo, yo, zo]  # [n_inside] bool
            else:
                edge_mask_inside = torch.zeros(0, dtype=torch.bool, device=enc_edge_mask.device)
        else:
            # Hash mode: build key→edge_mask table from encoder input
            sorted_keys_enc, sorted_vals_enc, R3, R_v = enc_lookup[1]
            bi_in = enc_batch_idx.long()
            ix_in = enc_index[:, 0].long()
            iy_in = enc_index[:, 1].long()
            iz_in = enc_index[:, 2].long()
            em_keys = bi_in * R3 + ix_in * R_v * R_v + iy_in * R_v + iz_in
            em_sort_idx = torch.argsort(em_keys)
            em_sorted_keys = em_keys[em_sort_idx]
            em_sorted_vals = enc_edge_mask.float()[em_sort_idx]

            inside_out_coords = out_coords[inside_mask]
            if inside_out_coords.shape[0] > 0:
                bi_o = inside_out_coords[:, 0].long()
                xo   = inside_out_coords[:, 1].long()
                yo   = inside_out_coords[:, 2].long()
                zo   = inside_out_coords[:, 3].long()
                q_keys = bi_o * R3 + xo * R_v * R_v + yo * R_v + zo
                pos = torch.searchsorted(em_sorted_keys, q_keys).clamp(0, em_sorted_keys.shape[0] - 1)
                found = em_sorted_keys[pos] == q_keys
                edge_mask_inside = (em_sorted_vals[pos].bool()) & found
            else:
                edge_mask_inside = torch.zeros(0, dtype=torch.bool, device=enc_edge_mask.device)

        # Build full-M bool tensor mapping inside output positions to edge_mask values
        full_edge_inside = torch.zeros(M_out, dtype=torch.bool, device=inside_mask.device)
        if inside_out_coords.shape[0] > 0 and edge_mask_inside.shape[0] > 0:
            full_edge_inside[inside_mask] = edge_mask_inside

        # ── Inside region loss (non-sharp inside voxels) ─────────────────────
        non_sharp_inside = inside_mask & ~full_edge_inside

        n_inside_non_sharp = int(non_sharp_inside.sum().item())
        n_sharp = int((inside_mask & full_edge_inside).sum().item())

        # ── Degenerate sharp count: skip backward (DDP-safe, same pattern as OOR skip) ──
        # n_sharp==0: L_sharp is identically zero (no signal); n_sharp==1: MSE over one voxel
        # is extremely high-variance and tends to destabilize AMP + sparse conv backward.
        min_sv = self.min_sharp_voxels_for_backward
        sharp_skip_local = min_sv > 0 and n_sharp < min_sv
        sharp_skip = sharp_skip_local
        if self.world_size > 1:
            try:
                if dist.is_available() and dist.is_initialized():
                    t = torch.tensor(
                        [1 if sharp_skip_local else 0],
                        device=sparse_sdf.device,
                        dtype=torch.int32,
                    )
                    dist.all_reduce(t, op=dist.ReduceOp.MAX)
                    sharp_skip = bool(t.item() != 0)
            except Exception:
                sharp_skip = sharp_skip_local

        if sharp_skip:
            print(
                f"\n[PHASE-C][SHARP_SKIP_STEP] n_sharp={n_sharp} < min_sharp_voxels_for_backward={min_sv} "
                f"(rank={self.rank}) — 任一 DDP rank 触发则本步所有 rank 跳过 backward/optimizer，"
                f"释放解码计算图（与 OOR skip_step 一致）。",
                flush=True,
            )
            codebook_stats_skip = dict(outputs.get("codebook_stats", {}))
            del outputs, recon, out_coords, inside_mask, extra_mask, oor_mask, oor_flags, gt_vals
            del enc_lookup, gt_lookup_full, enc_coords, x
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            dev = sparse_sdf.device
            zt = torch.zeros((), dtype=torch.float32, device=dev)
            terms = edict(
                loss=zt,
                loss_inside=zt,
                loss_sharp=zt,
                loss_extra=zt,
                commitment=zt,
            )
            status = edict(
                n_inside=n_inside_non_sharp,
                n_sharp=n_sharp,
                n_extra=n_extra_valid,
                n_oor=n_oor,
                n_skip_samples=n_skip,
                loss_inside=0.0,
                loss_sharp=0.0,
                loss_extra=0.0,
                skip_backward=True,
                oor_ratio=oor_ratio,
                oor_skip_step=0.0,
                sharp_skip_step=1.0,
                force_zero_extra_loss=float(force_zero_extra_loss),
                codebook_perplexity=codebook_stats_skip.get("perplexity", 0.0),
                codebook_entropy=codebook_stats_skip.get("entropy", 0.0),
                codebook_utilization_ratio=codebook_stats_skip.get("utilization_ratio", 0.0),
                codebook_unique_count=codebook_stats_skip.get("unique_count", 0),
                codebook_batch_unique_count=codebook_stats_skip.get("batch_unique_count", 0),
            )
            print(
                f"\n[LOSS-SUMMARY] step={step} | SKIPPED backward (degenerate sharp) "
                f"n_sharp={n_sharp} min_sharp={min_sv} rank={self.rank} oor_ratio={oor_ratio:.4f}\n{SEP}\n",
                flush=True,
            )
            return terms, status

        print(f"\n[PHASE-C] Loss computation (loss_type={self.loss_type!r})")

        if n_inside_non_sharp > 0:
            recon_inside_ns = recon.feats[non_sharp_inside]
            gt_inside_ns    = gt_vals[non_sharp_inside].unsqueeze(-1)
            loss_inside = self._per_voxel_loss(recon_inside_ns, gt_inside_ns)
        else:
            loss_inside = torch.tensor(0.0, device=recon.feats.device, requires_grad=False)

        # ── Sharp region loss ─────────────────────────────────────────────────
        sharp_mask_full = inside_mask & full_edge_inside
        if sharp_mask_full.sum() > 0:
            recon_sharp = recon.feats[sharp_mask_full]
            gt_sharp    = gt_vals[sharp_mask_full].unsqueeze(-1)
            loss_sharp = self._per_voxel_loss(recon_sharp, gt_sharp)
        else:
            loss_sharp = torch.tensor(0.0, device=recon.feats.device, requires_grad=False)

        # ── Extra region loss ─────────────────────────────────────────────────
        extra_valid_mask = extra_mask & ~oor_mask
        n_extra_valid = int(extra_valid_mask.sum().item())
        if n_extra_valid > 0:
            recon_extra = recon.feats[extra_valid_mask]
            gt_extra    = gt_vals[extra_valid_mask].unsqueeze(-1)
            loss_extra = self._per_voxel_loss(recon_extra, gt_extra)
        else:
            loss_extra = torch.tensor(0.0, device=recon.feats.device, requires_grad=False)

        # ── Debug: per-region stats ───────────────────────────────────────────
        print(f"  [inside non-sharp] N={n_inside_non_sharp}  "
              f"loss_raw={loss_inside.item():.6f}  "
              f"weighted={self.lambda_inside * loss_inside.item():.6f}  "
              f"requires_grad={loss_inside.requires_grad}")
        print(f"  [sharp  ]          N={n_sharp}  "
              f"loss_raw={loss_sharp.item():.6f}  "
              f"weighted={self.lambda_sharp * loss_sharp.item():.6f}  "
              f"requires_grad={loss_sharp.requires_grad}")
        print(f"  [extra  ]          N={n_extra_valid}  "
              f"loss_raw={loss_extra.item():.6f}  "
              f"weighted={extra_w * loss_extra.item():.6f}  "
              f"requires_grad={loss_extra.requires_grad}")
        if verbose:
            # loss_raw 为各区域内 MSE 的 mean（对体素数归一），不是对解码全体素求和
            def _rmse(t: torch.Tensor) -> float:
                return float(torch.sqrt(torch.clamp(t.detach(), min=0.0)).item())

            print(
                "  [loss scale] 各区域 loss_raw = F.mse_loss(..., reduction='mean') 于该区域内体素；"
                "总损失 = λ_in*L_in + λ_sh*L_sh + λ_ex*L_ex + (VQ/)commitment。"
                f" RMSE≈√MSE: in={_rmse(loss_inside):.4f} sh={_rmse(loss_sharp):.4f} ex={_rmse(loss_extra):.4f}（归一化 SDF）"
            )

        if n_inside_non_sharp > 0 and verbose:
            res_inside = (recon_inside_ns - gt_inside_ns).squeeze(-1)
            print(f"  [inside residual] min={res_inside.min().item():.4f} "
                  f"max={res_inside.max().item():.4f} mean={res_inside.mean().item():.4f} "
                  f"std={res_inside.std().item():.4f}")
        if n_extra_valid > 0 and verbose:
            res_extra = (recon_extra - gt_extra).squeeze(-1)
            print(f"  [extra  residual] min={res_extra.min().item():.4f} "
                  f"max={res_extra.max().item():.4f} mean={res_extra.mean().item():.4f} "
                  f"std={res_extra.std().item():.4f}")
            print(f"  [extra  GT range] min={gt_extra.min().item():.4f} "
                  f"max={gt_extra.max().item():.4f} mean={gt_extra.mean().item():.4f}")

        # Histogram of residuals (inside) every N steps
        if (verbose and n_inside_non_sharp > 0 and
                self.debug_loss_histogram_every_n_steps > 0 and
                step % self.debug_loss_histogram_every_n_steps == 0):
            with torch.no_grad():
                r = (recon_inside_ns - gt_inside_ns).squeeze(-1).abs()
                bins = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, float('inf')]
                counts = [(r < b).sum().item() for b in bins[1:]]
                hist_parts = []
                prev_c = 0
                for b, c in zip(bins[1:], counts):
                    hist_parts.append(f"|err|<{b:.2f}: {c - prev_c}")
                    prev_c = c
                print(f"  [inside residual histogram] {' | '.join(hist_parts)}")

        print(f"  commitment_loss={commitment_loss.item():.6f}  "
              f"weighted={self.lambda_commitment * commitment_loss.item():.6f}")
        if vq_loss is not None:
            print(f"  vq_loss={vq_loss.item():.6f}  "
                  f"weighted={self.lambda_vq * vq_loss.item():.6f}")

        # ── Assemble total loss ───────────────────────────────────────────────
        recon_component = (self.lambda_inside * loss_inside +
                           self.lambda_sharp * loss_sharp +
                           extra_w * loss_extra)

        if self.training_stage == 3:
            total_loss = recon_component
            vq_str = "stage3 (no VQ)"
        elif vq_loss is not None:
            total_loss = (recon_component +
                          self.lambda_vq * vq_loss +
                          self.lambda_commitment * commitment_loss)
            vq_str = f"λ_vq*{vq_loss.item():.4f}+λ_cmt*{commitment_loss.item():.4f}"
        else:
            total_loss = recon_component + self.lambda_commitment * commitment_loss
            vq_str = f"EMA: λ_cmt*{commitment_loss.item():.4f}"

        print(f"\n[PHASE-C] Total loss: {total_loss.item():.6f}")
        print(f"  = λ_in({self.lambda_inside})*{loss_inside.item():.4f}"
              f" + λ_sh({self.lambda_sharp})*{loss_sharp.item():.4f}"
              f" + λ_ex({extra_w})*{loss_extra.item():.4f}"
              f" + {vq_str}")
        print(f"  total requires_grad={total_loss.requires_grad}")

        # ════════════════════════════════════════════════════════════════════
        # PHASE D – Gradient health (logged every i_log steps)
        # ════════════════════════════════════════════════════════════════════
        if verbose and hasattr(self, 'i_log') and step % self.i_log == 0:
            print(f"\n[PHASE-D] Gradient health check (step={step})")
            print(f"  total_loss requires_grad={total_loss.requires_grad}")
            print(f"  recon.feats requires_grad={recon.feats.requires_grad}")
            print(f"  commitment_loss requires_grad={commitment_loss.requires_grad}")
            if vq_loss is not None:
                print(f"  vq_loss requires_grad={vq_loss.requires_grad}")

        # ════════════════════════════════════════════════════════════════════
        # PHASE E – Per-step summary
        # ════════════════════════════════════════════════════════════════════
        vq_val = vq_loss.item() if vq_loss is not None else 0.
        print(
            f"\n[LOSS-SUMMARY] step={step} | "
            f"n_in={n_inside_non_sharp} | n_sharp={n_sharp} | "
            f"n_extra={n_extra_valid} | n_oor={n_oor} | n_skip={n_skip} | "
            f"L_in={loss_inside.item():.4f} L_sharp={loss_sharp.item():.4f} "
            f"L_extra={loss_extra.item():.4f} L_vq={vq_val:.4f} "
            f"L_cmt={commitment_loss.item():.4f} L_total={total_loss.item():.4f}"
        )
        print(f"{SEP}\n")

        # ── Assemble return dicts ─────────────────────────────────────────────
        terms = edict(
            loss=total_loss,
            loss_inside=loss_inside,
            loss_sharp=loss_sharp,
            loss_extra=loss_extra,
            commitment=commitment_loss,
        )
        if vq_loss is not None:
            terms['vq'] = vq_loss

        status = edict(
            n_inside=n_inside_non_sharp,
            n_sharp=n_sharp,
            n_extra=n_extra_valid,
            n_oor=n_oor,
            n_skip_samples=n_skip,
            loss_inside=loss_inside.item(),
            loss_sharp=loss_sharp.item(),
            loss_extra=loss_extra.item(),
            skip_backward=False,
            oor_ratio=oor_ratio,
            oor_skip_step=0.0,
            sharp_skip_step=0.0,
            force_zero_extra_loss=float(force_zero_extra_loss),
            codebook_perplexity=codebook_stats.get('perplexity', 0.),
            codebook_entropy=codebook_stats.get('entropy', 0.),
            codebook_utilization_ratio=codebook_stats.get('utilization_ratio', 0.),
            codebook_unique_count=codebook_stats.get('unique_count', 0),
            codebook_batch_unique_count=codebook_stats.get('batch_unique_count', 0),
        )

        return terms, status
    
    @torch.no_grad()
    def snapshot(self, suffix=None, num_samples=16, batch_size=1, verbose=False):  # 默认batch_size=1避免内存问题
        """Take a snapshot of the model's performance."""
        # Use training batch_size if not specified, default to 2 for safety
        if batch_size is None:
            batch_size = getattr(self, 'batch_size_per_gpu', 2)
        super().snapshot(suffix=suffix, num_samples=num_samples, batch_size=batch_size, verbose=verbose)
    
    @torch.no_grad()
    def run_snapshot(
        self,
        num_samples: int,
        batch_size: int,
        verbose: bool = False,
        snapshot_export_suffix: Optional[str] = None,
        **kwargs,
    ) -> Dict:
        """
        Run snapshot inference.
        
        Args:
            num_samples: Number of samples to generate
            batch_size: Batch size for inference
            verbose: Whether to print verbose output
        
        Returns:
            Dictionary of samples for visualization
        """
        try:
            # Create a dataset copy with potentially reduced max_points for stability
            dataset_copy = copy.deepcopy(self.dataset)

            dataloader = DataLoader(
                dataset_copy,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                collate_fn=dataset_copy.collate_fn if hasattr(dataset_copy, 'collate_fn') else None,
            )
            # 持久迭代器：避免每轮 ``next(iter(dataloader))`` 重置 shuffle 后反复取到同一批
            data_iter = iter(dataloader)

            # Get VQVAE model
            vqvae = self.models['vqvae']
            if hasattr(vqvae, 'module'):
                vqvae = vqvae.module

            device = next(vqvae.parameters()).device
            
            # Get model dtype for fp16 compatibility
            model_dtype = vqvae.encoder.dtype if hasattr(vqvae.encoder, 'dtype') else torch.float32
            
            # 检查encoder的out_layer权重（仅在首次快照时检查）
            if self.step == 0 or self.step % (self.i_sample * 10) == 0:
                if hasattr(vqvae.encoder, 'out_layer'):
                    out_layer = vqvae.encoder.out_layer
                    weight_sum = out_layer.weight.abs().sum().item()
                    if weight_sum < 1e-6:
                        print(f"\n⚠️  警告: Encoder out_layer权重几乎为零！")
                        print(f"  这可能表示模型未被训练或checkpoint未正确加载")
                        print(f"  权重绝对值总和: {weight_sum:.6e}\n")
            
            # Inference：每轮子 batch 的 ``batch_idx`` 均为 0..B-1，合并前必须加上全局偏移，
            # 否则 ``sparse_sample_dict_to_trimeshes`` 会把多个物体混进同一 512³ 体素网格。
            gts = []
            recons = []
            grid_resolution_tensor: Optional[torch.Tensor] = None
            running_global = 0
            last_mesh_export_root: Optional[str] = None

            while running_global < num_samples:
                want = min(batch_size, num_samples - running_global)
                try:
                    data = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    data = next(data_iter)

                sparse_sdf = data['sparse_sdf'].to(device)
                sparse_index = data['sparse_index'].to(device)
                batch_idx = data['batch_idx'].to(device)

                if 'grid_resolution' in data and isinstance(data['grid_resolution'], torch.Tensor):
                    gr = data['grid_resolution'].detach().view(-1).to(device=device, dtype=torch.long)
                    if grid_resolution_tensor is None:
                        grid_resolution_tensor = gr.clone()
                    elif int(grid_resolution_tensor.reshape(-1)[0].item()) != int(
                        gr.reshape(-1)[0].item()
                    ):
                        raise ValueError(
                            "[run_snapshot] snapshot batch mixes grid_resolution values; "
                            f"had R={int(grid_resolution_tensor.reshape(-1)[0].item())}, "
                            f"new R={int(gr.reshape(-1)[0].item())}"
                        )

                B_loaded = int(batch_idx.max().item()) + 1 if batch_idx.numel() > 0 else 0
                take = min(want, B_loaded)
                if take <= 0:
                    raise RuntimeError(
                        "[run_snapshot] empty or invalid batch from DataLoader "
                        f"(B_loaded={B_loaded}, want={want}). Check dataset is non-empty."
                    )

                mask = batch_idx < take
                sparse_sdf = sparse_sdf[mask]
                sparse_index = sparse_index[mask]
                batch_idx_local = batch_idx[mask]
                # 全局 batch 下标，供 MC / 可视化按样本分组
                batch_idx_global = batch_idx_local + int(running_global)

                # Convert to model dtype (for fp16 compatibility)
                sparse_sdf = sparse_sdf.to(dtype=model_dtype)

                # Check for any NaN or Inf
                if torch.isnan(sparse_sdf).any() or torch.isinf(sparse_sdf).any():
                    print(f"⚠️  警告: 检测到NaN或Inf值在sparse_sdf中！")

                # ── 与训练保持一致：encoder 只看紧带体素（|SDF| <= input_sdf_thresh）──
                # 训练时 training_losses 对 enc_mask 做了同样的过滤；
                # 若此处不过滤，encoder 会收到训练时从未见过的宽频 SDF 分布，导致
                # 量化索引错误，重建可视化严重失真。
                enc_mask = sparse_sdf.abs().squeeze(-1) <= self.input_sdf_thresh
                enc_sdf       = sparse_sdf[enc_mask]
                enc_index     = sparse_index[enc_mask]
                enc_batch_idx = batch_idx_local[enc_mask]

                if enc_sdf.shape[0] == 0:
                    print(f"⚠️  [run_snapshot] 紧带为空（input_sdf_thresh={self.input_sdf_thresh:.4f}），跳过本 batch")
                    running_global += take
                    continue

                print(
                    f"[run_snapshot] tight-band filter: "
                    f"{enc_sdf.shape[0]}/{sparse_sdf.shape[0]} voxels "
                    f"(|SDF|<={self.input_sdf_thresh:.4f})"
                )

                # ── 确定输出分辨率，用于 gt_prune ──────────────────────────────
                if grid_resolution_tensor is not None:
                    R_out = int(grid_resolution_tensor.reshape(-1)[0].item())
                else:
                    R_out = 512  # 与配置 dataset.resolution 保持一致的默认值

                # Encode and decode
                try:
                    # 使用紧带过滤后的数据构建 batch_dict，与 training_losses 的 encoder 输入一致
                    batch_dict = {
                        'sparse_sdf': enc_sdf,
                        'sparse_index': enc_index,
                        'batch_idx': enc_batch_idx,
                    }

                    encoding_indices = vqvae.Encode(batch_dict)
                except Exception as e:
                    print(f"❌ 编码错误: {type(e).__name__}: {e}")
                    raise

                # ── 推理 Decode：仅用 encoding_indices 的 coords/feats，无 GT 泄露 ──
                # band_prune mode='seed'：以 block 级 latent 坐标为 seed，按 inference_band_factor
                # 扩张 AABB 粗剪（不依赖任何 GT 坐标）。
                # gt_prune mode='geometry'：每个 subdivide 步按 R_target 分辨率动态构造 dilated occ，
                # 带宽也用 inference_band_factor（而非 preprocessing_extra_band_factor），紧度可调。
                inf_band  = (self.inference_band_factor
                             if self.inference_band_factor is not None
                             else float(self.input_band_factor))
                inf_occ_R = (self.inference_occ_resolution
                             if self.inference_occ_resolution is not None
                             else 256)
                R_latent_block = int(vqvae.resolution // vqvae.vq_block_side)

                band_prune_infer = {
                    "mode":              "seed",
                    "seed_coords":       encoding_indices.coords,
                    "seed_resolution":   R_latent_block,
                    "output_resolution": R_out,
                    "extra_band_factor": inf_band,
                } if self.decoder_band_prune else None

                gt_prune_infer = {
                    "mode":              "geometry",
                    "extra_band_factor": inf_band,
                    "resolution":        R_out,
                    "occ_resolution":    inf_occ_R,
                }

                print(
                    f"[run_snapshot] inference prune: band={inf_band} R_final-voxel "
                    f"(input_band_factor={self.input_band_factor}, "
                    f"NOT preprocessing_extra_band_factor={self.preprocessing_extra_band_factor}), "
                    f"occ_resolution={inf_occ_R}, "
                    f"max R_final-voxel tightness={R_out // min(inf_occ_R, R_out)}"
                )

                recon = vqvae.Decode(
                    encoding_indices,
                    band_prune=band_prune_infer,
                    gt_prune=gt_prune_infer,
                )

                recon_coords = recon.coords.clone()
                recon_coords[:, 0] = recon_coords[:, 0].long() + int(running_global)

                # ── 验证路径 2：GT / recon 各自导出 mesh 到该 snapshot 下按样本分子目录 ─
                from ...utils.sparse_sdf_marching_cubes import sparse_sample_dict_to_trimeshes

                snap_suffix = snapshot_export_suffix or f"step{self.step:07d}"
                mesh_root = os.path.join(self.output_dir, "samples", snap_suffix, "meshes")
                if getattr(self, "world_size", 1) > 1:
                    mesh_root = os.path.join(mesh_root, f"rank{getattr(self, 'rank', 0)}")
                mc_th = float(getattr(self.dataset, "snapshot_mc_threshold", 0.0))
                instance_ids = data.get("instance_ids") or []
                ds_idx_t = data.get("dataset_indices")

                mesh_stats: Dict[str, Dict[str, Any]] = {}

                def _record_and_export_mesh(
                    name: str,
                    mesh,
                    sparse_values: torch.Tensor,
                    sparse_count: int,
                    sample_dir: str,
                ) -> None:
                    vals = sparse_values.detach().float().cpu().reshape(-1)
                    stats = {
                        "num_sparse_voxels": int(sparse_count),
                        "sdf_min": float(vals.min().item()) if vals.numel() else None,
                        "sdf_max": float(vals.max().item()) if vals.numel() else None,
                        "sdf_mean": float(vals.mean().item()) if vals.numel() else None,
                        "sdf_neg": int((vals < 0).sum().item()) if vals.numel() else 0,
                        "sdf_pos": int((vals > 0).sum().item()) if vals.numel() else 0,
                        "sdf_zero": int((vals == 0).sum().item()) if vals.numel() else 0,
                        "num_vertices": 0,
                        "num_faces": 0,
                        "exported": False,
                    }
                    if mesh is not None:
                        stats["num_vertices"] = int(len(mesh.vertices))
                        stats["num_faces"] = int(len(mesh.faces))
                        if stats["num_vertices"] > 0 and stats["num_faces"] > 0:
                            out_path = os.path.join(sample_dir, f"{name}_mesh.obj")
                            try:
                                mesh.export(
                                    out_path,
                                    file_type="obj",
                                    include_attributes=False,
                                )
                            except TypeError:
                                mesh.export(out_path, file_type="obj")
                            stats["exported"] = True
                            stats["path"] = out_path
                    mesh_stats[name] = stats

                for b in range(take):
                    sid = instance_ids[b] if b < len(instance_ids) else ""
                    if not (isinstance(sid, str) and sid.strip()):
                        if ds_idx_t is not None and b < int(ds_idx_t.shape[0]):
                            sid = f"dataset_index_{int(ds_idx_t[b].item())}"
                        else:
                            sid = f"batch_{b}"
                    safe = re.sub(r"[^\w\-\.]", "_", str(sid))[:240]
                    sample_dir = os.path.join(mesh_root, safe)
                    os.makedirs(sample_dir, exist_ok=True)
                    mesh_stats = {}

                    m_gt = batch_idx_local == b
                    if bool(m_gt.any().item()):
                        gt_export = {
                            "sparse_sdf": sparse_sdf[m_gt].detach().cpu(),
                            "sparse_index": sparse_index[m_gt].detach().cpu(),
                            "batch_idx": torch.zeros(
                                int(m_gt.sum().item()), dtype=torch.long
                            ),
                        }
                        gt_meshes = sparse_sample_dict_to_trimeshes(
                            gt_export, R_out, mc_th
                        )
                        _record_and_export_mesh(
                            "gt",
                            gt_meshes[0] if gt_meshes else None,
                            sparse_sdf[m_gt],
                            int(m_gt.sum().item()),
                            sample_dir,
                        )

                    m_rb = recon.coords[:, 0].long() == b
                    if bool(m_rb.any().item()):
                        recon_export = {
                            "sparse_sdf": recon.feats[m_rb].detach().cpu(),
                            "sparse_index": recon.coords[m_rb, 1:4].detach().cpu(),
                            "batch_idx": torch.zeros(
                                int(m_rb.sum().item()), dtype=torch.long
                            ),
                        }
                        r_meshes = sparse_sample_dict_to_trimeshes(
                            recon_export, R_out, mc_th
                        )
                        _record_and_export_mesh(
                            "recon",
                            r_meshes[0] if r_meshes else None,
                            recon.feats[m_rb],
                            int(m_rb.sum().item()),
                            sample_dir,
                        )
                    with open(
                        os.path.join(sample_dir, "mesh_stats.json"),
                        "w",
                        encoding="utf-8",
                    ) as f:
                        json.dump(mesh_stats, f, ensure_ascii=False, indent=2)

                last_mesh_export_root = mesh_root

                # 清理CUDA缓存，避免内存累积
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Store results（使用全局 batch_idx，与 recon.coords[:,0] 一致）
                gts.append({
                    'sparse_sdf': sparse_sdf,
                    'sparse_index': sparse_index,
                    'batch_idx': batch_idx_global,
                })
                recons.append({
                    'sparse_sdf': recon.feats,
                    'sparse_index': recon_coords[:, 1:],
                    'batch_idx': recon_coords[:, 0],
                })

                running_global += take
            
            if last_mesh_export_root is not None:
                print(
                    f"[run_snapshot] GT/recon mesh 已写入（mc_threshold="
                    f"{float(getattr(self.dataset, 'snapshot_mc_threshold', 0.0))}）: "
                    f"{last_mesh_export_root}",
                    flush=True,
                )

            # Combine results
            gt_combined = {
                'sparse_sdf': torch.cat([g['sparse_sdf'] for g in gts], dim=0),
                'sparse_index': torch.cat([g['sparse_index'] for g in gts], dim=0),
                'batch_idx': torch.cat([g['batch_idx'] for g in gts], dim=0),
            }
            if grid_resolution_tensor is not None:
                gt_combined['grid_resolution'] = grid_resolution_tensor

            recon_combined = {
                'sparse_sdf': torch.cat([r['sparse_sdf'] for r in recons], dim=0),
                'sparse_index': torch.cat([r['sparse_index'] for r in recons], dim=0),
                'batch_idx': torch.cat([r['batch_idx'] for r in recons], dim=0),
            }
            if grid_resolution_tensor is not None:
                recon_combined['grid_resolution'] = grid_resolution_tensor
            
            sample_dict = {
                'gt': {'value': gt_combined, 'type': 'sample'},
                'recon': {'value': recon_combined, 'type': 'sample'},
            }
            
            return sample_dict
            
        except Exception as e:
            print(f"\n❌ run_snapshot错误:")
            print(f"  异常类型: {type(e).__name__}")
            print(f"  异常信息: {str(e)}")
            import traceback
            print(f"\n完整堆栈跟踪:")
            traceback.print_exc()
            raise

