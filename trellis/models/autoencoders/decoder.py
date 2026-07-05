from typing import *
import math
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...modules.utils import zero_module, convert_module_to_f16, convert_module_to_f32
from ...modules import sparse as sp
from .base import SparseTransformerBase

# torchsparse / ImplicitGEMM 诊断：DECODER_TORCHSPARSE_DEBUG 或 DECODER_SUBDIVIDE_DEBUG=1；
# 卷积细日志另可 DECODER_SPARSE_CONV_DEBUG；DECODER_TSDBG_COORD_SAMPLE 控制坐标重复估计行数（默认 5e5）。


def _env_flag_true(name: str) -> bool:
    return os.environ.get(name, "").strip() in ("1", "true", "True", "yes", "YES")


def _decoder_subdivide_step_debug() -> bool:
    """SparseSubdivideBlock3d 内各阶段细日志。DECODER_TORCHSPARSE_DEBUG 或 DECODER_SUBDIVIDE_DEBUG=1。"""
    return _env_flag_true("DECODER_TORCHSPARSE_DEBUG") or _env_flag_true("DECODER_SUBDIVIDE_DEBUG")


def _debug_dump_sparse_subdivide(phase: str, x: sp.SparseTensor, block_tag: str) -> None:
    """
    诊断 ImplicitGEMM backward 崩溃：坐标范围、体素数、dtype、torchsparse 元数据、坐标重复（子采样）。
    需设置环境变量 DECODER_TORCHSPARSE_DEBUG 或 DECODER_SUBDIVIDE_DEBUG。
    """
    if not _decoder_subdivide_step_debug():
        return
    c = x.coords
    f = x.feats
    n = int(c.shape[0])
    dev = c.device
    lines = [
        f"[TSDBG subdivide] {block_tag} | {phase}",
        f"  N={n}  feats={tuple(f.shape)} dtype feats={f.dtype} coords={c.dtype} device={dev}",
        f"  _scale={getattr(x, '_scale', None)}  layout_batches={x.shape[0] if hasattr(x, 'shape') else '?'}",
    ]
    if n > 0 and c.ndim == 2 and c.shape[1] >= 4:
        xyz = c[:, 1:4].long()
        b = c[:, 0].long()
        lines.append(
            f"  batch_id: min={b.min().item()} max={b.max().item()} "
            f"xyz min=[{xyz[:,0].min().item()},{xyz[:,1].min().item()},{xyz[:,2].min().item()}] "
            f"max=[{xyz[:,0].max().item()},{xyz[:,1].max().item()},{xyz[:,2].max().item()}]"
        )
        sample_n = int(os.environ.get("DECODER_TSDBG_COORD_SAMPLE", "500000"))
        sample_n = max(1000, min(sample_n, n))
        sub = c[:sample_n]
        uniq = torch.unique(sub, dim=0).shape[0]
        dup = sample_n - int(uniq)
        lines.append(
            f"  coord_dup_estimate: first_{sample_n}_rows unique={int(uniq)} dup≈{dup} "
            f"(dup_ratio≈{dup / sample_n:.6f}; 全量 unique 未算以省显存)"
        )
    if f.numel() > 0:
        fn = f.detach().float()
        lines.append(
            f"  feats stats: min={fn.min().item():.6f} max={fn.max().item():.6f} "
            f"mean={fn.mean().item():.6f} std={fn.std().item():.6f} finite={torch.isfinite(fn).all().item()}"
        )
    ts = getattr(x, "data", None)
    if ts is not None:
        extra = []
        for attr in ("spatial_range", "stride"):
            if hasattr(ts, attr):
                try:
                    extra.append(f"{attr}={getattr(ts, attr)}")
                except Exception as e:  # noqa: BLE001
                    extra.append(f"{attr}=<err {e}>")
        if extra:
            lines.append("  torchsparse.SparseTensor: " + " ".join(extra))
    print("\n".join(lines), flush=True)


def build_band_prune_aabb(
    band_prune: dict,
    device: torch.device,
    dtype: torch.dtype = torch.int64,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    由宽频带体素索引或编码器种子坐标，计算解码器输出网格上的按-batch AABB（用于 GPU 矢量化裁剪）。

    Args:
        band_prune: 字典，字段如下之一：
          - ``enabled``：若为 ``False``，不进行裁剪（推理无坐标信息时由调用方传入）。
          - **训练 / 有 GT**：``mode='wide'``（可省略，若提供 ``wide_xyz`` 则自动为 wide）
            ``wide_batch_idx`` [N]、``wide_xyz`` [N,3]（与预处理 npz 相同分辨率，通常为 512）、
            ``output_resolution``、``extra_band_factor``（= preprocessing_extra_band_factor）。
            AABB 取宽频带坐标 min/max，并向外扩 ``max(1, ceil(extra*0.25))`` 个体素作为安全余量。
          - **仅解码推理**：``mode='seed'``（可省略，若提供 ``seed_coords`` 且无 ``wide_xyz``）
            ``seed_coords`` [N,4] (batch, x, y, z)、``seed_resolution``（种子格分辨率，如 64）、
            ``output_resolution``（解码输出分辨率，如 512）、``extra_band_factor``。
            将每个种子体素映射到输出分辨率下的子块 ``[s*scale, s*scale+scale-1]`` 再取并集，
            并向外扩 ``ceil(extra_band_factor)`` 个体素（与预处理壳层宽度同量级）。

    Returns:
        (bbox_min, bbox_max, grid_upper)，均为 ``[B, 3]``，坐标闭区间 ``[bbox_min, bbox_max]``，
        ``grid_upper`` 为合法坐标上界（开区间），即有效坐标满足 ``0 <= c < grid_upper``。
    """
    out_r = int(band_prune["output_resolution"])
    if "extra_band_factor" in band_prune:
        extra = float(band_prune["extra_band_factor"])
    elif "preprocessing_extra_band_factor" in band_prune:
        extra = float(band_prune["preprocessing_extra_band_factor"])
    else:
        raise KeyError("band_prune needs extra_band_factor or preprocessing_extra_band_factor.")
    wide_xyz = band_prune.get("wide_xyz")
    wide_batch_idx = band_prune.get("wide_batch_idx")
    seed_coords = band_prune.get("seed_coords")
    seed_resolution = band_prune.get("seed_resolution")
    mode = band_prune.get("mode")
    if mode is None:
        if wide_xyz is not None and wide_batch_idx is not None:
            mode = "wide"
        elif seed_coords is not None and seed_resolution is not None:
            mode = "seed"
        else:
            raise ValueError(
                "band_prune: need either (wide_batch_idx, wide_xyz) or (seed_coords, seed_resolution)."
            )

    if mode == "wide":
        if wide_xyz is None or wide_batch_idx is None:
            raise ValueError("band_prune mode=wide requires wide_xyz and wide_batch_idx.")
        wb = wide_batch_idx.to(device=device, dtype=torch.long)
        xyz = wide_xyz.to(device=device, dtype=torch.long)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError(f"wide_xyz must be [N,3], got {tuple(xyz.shape)}")
        B = int(wb.max().item()) + 1 if wb.numel() > 0 else 1
        pad = max(1, int(math.ceil(extra * 0.25)))
        bbox_min = torch.zeros(B, 3, device=device, dtype=dtype)
        bbox_max = torch.zeros(B, 3, device=device, dtype=dtype)
        for bid in range(B):
            m = wb == bid
            if m.any():
                lo = xyz[m].min(dim=0).values.to(dtype) - pad
                hi = xyz[m].max(dim=0).values.to(dtype) + pad
            else:
                lo = torch.zeros(3, device=device, dtype=dtype)
                hi = torch.full((3,), out_r - 1, device=device, dtype=dtype)
            bbox_min[bid] = lo.clamp(0, out_r - 1)
            bbox_max[bid] = hi.clamp(0, out_r - 1)
            for d in range(3):
                if bbox_min[bid, d] > bbox_max[bid, d]:
                    bbox_max[bid, d] = bbox_min[bid, d]
        return bbox_min, bbox_max, out_r

    if mode == "seed":
        if seed_coords is None:
            raise ValueError("band_prune mode=seed requires seed_coords.")
        sr = int(seed_resolution)
        if out_r % sr != 0:
            raise ValueError(f"output_resolution {out_r} must be divisible by seed_resolution {sr}.")
        scale = out_r // sr
        pad = max(1, int(math.ceil(extra)))
        sc = seed_coords.to(device=device, dtype=torch.long)
        if sc.ndim != 2 or sc.shape[1] != 4:
            raise ValueError(f"seed_coords must be [N,4] (b,x,y,z), got {tuple(sc.shape)}")
        B = int(sc[:, 0].max().item()) + 1 if sc.numel() > 0 else 1
        bbox_min = torch.zeros(B, 3, device=device, dtype=dtype)
        bbox_max = torch.zeros(B, 3, device=device, dtype=dtype)
        for bid in range(B):
            m = sc[:, 0] == bid
            if m.any():
                s = sc[m, 1:4].to(dtype)
                lo = s.min(dim=0).values * scale - pad
                hi = s.max(dim=0).values * scale + (scale - 1) + pad
            else:
                lo = torch.zeros(3, device=device, dtype=dtype)
                hi = torch.full((3,), out_r - 1, device=device, dtype=dtype)
            bbox_min[bid] = lo.clamp(0, out_r - 1)
            bbox_max[bid] = hi.clamp(0, out_r - 1)
            for d in range(3):
                if bbox_min[bid, d] > bbox_max[bid, d]:
                    bbox_max[bid, d] = bbox_min[bid, d]
        return bbox_min, bbox_max, out_r

    raise ValueError(f"band_prune mode must be 'wide' or 'seed', got {mode!r}.")


def scale_band_prune_boxes_to_grid(
    bbox_min: torch.Tensor,
    bbox_max: torch.Tensor,
    R_out: int,
    R_cur: int,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    将定义在 ``R_out`` 网格上的 AABB（闭区间）映射到当前解码子步的 ``R_cur`` 网格。

    解码器在 128³ / 256³ 阶段的坐标仍是粗网格索引；若直接用 512³ 的包围盒裁剪，
    会把几乎所有体素判为越界（例如要求 x≥167 而粗格 x≤127），导致 n_inside=0、
    仅剩错误子集或触发 DDP 静态图参数未使用错误。

    映射：输出体素索引 ``xf`` 属于粗格 ``xc`` 当且仅当 ``xc = xf // S``，
    ``S = R_out // R_cur``。AABB ``[Lf, Hf]`` 在粗格上保留的 ``xc`` 范围为
    ``Lf // S`` 到 ``Hf // S``（对非负坐标与标准 subdivide 一致）。
    """
    if R_cur <= 0 or R_out <= 0:
        raise ValueError(f"Invalid R_cur={R_cur}, R_out={R_out}")
    if R_out % R_cur != 0:
        raise ValueError(f"R_out={R_out} must be divisible by R_cur={R_cur} for band prune scaling.")
    S = R_out // R_cur
    lo = (bbox_min // S).clamp(0, R_cur - 1)
    hi = (bbox_max // S).clamp(0, R_cur - 1)
    bad = lo > hi
    if bad.any():
        hi = torch.where(bad, lo, hi)
    return lo, hi, R_cur


def _prune_sparse_by_batch_aabb(
    x: sp.SparseTensor,
    bbox_min: torch.Tensor,
    bbox_max: torch.Tensor,
    grid_upper: int,
) -> sp.SparseTensor:
    """矢量化按 batch 的 AABB 裁剪，丢弃网格外或盒外的体素。"""
    coords = x.coords
    b = coords[:, 0].long().clamp(0, bbox_min.shape[0] - 1)
    xyz = coords[:, 1:4].long()
    lo = bbox_min[b]
    hi = bbox_max[b]
    in_grid = (xyz >= 0).all(dim=1) & (xyz < grid_upper).all(dim=1)
    in_box = (xyz >= lo).all(dim=1) & (xyz <= hi).all(dim=1)
    keep = in_grid & in_box
    if keep.all():
        return x
    if not keep.any():
        if os.environ.get("DECODER_PRUNE_DEBUG", ""):
            print(
                "[DECODER_PRUNE_DEBUG] _prune_sparse_by_batch_aabb: keep mask empty — "
                "skip prune (check R_cur vs bbox scale)."
            )
        return x
    new_coords = coords[keep].contiguous()
    new_feats = x.feats[keep].contiguous()
    out = sp.SparseTensor(new_feats, new_coords)
    out._scale = getattr(x, "_scale", None)
    # 勿继承父张量的 spatial_cache：其中序列化/窗口索引对应旧 N，裁剪后会导致底层 kernel 越界。
    out._spatial_cache = {}
    return out


def _clamp_sparse_coords(x: sp.SparseTensor, max_resolution: int) -> sp.SparseTensor:
    """
    过滤掉空间坐标超出 [0, max_resolution) 的稀疏点。

    背景：在反向传播的 sparse convolution 中，如果上游算子（例如 SparseSubdivide
    将坐标 ×2 之后，再叠加卷积感受野的扩张）产生了非法/超大坐标，会触发底层
    cuda kernel 显存越界访问导致崩溃。该函数会直接丢弃越界点，保证坐标始终
    位于合法立方体范围内。

    Args:
        x: 待裁剪的 SparseTensor，coords 形状 [N, 4]，第 0 列为 batch idx。
        max_resolution: 当前层期望的空间分辨率（坐标合法上界，开区间）。
    """
    coords = x.coords
    spatial = coords[:, 1:]
    valid_mask = (spatial >= 0).all(dim=1) & (spatial < max_resolution).all(dim=1)

    if valid_mask.all():
        return x

    num_total = coords.shape[0]
    num_dropped = int((~valid_mask).sum().item())
    if num_dropped > 0:
        spatial_min = spatial.min(0).values.tolist()
        spatial_max = spatial.max(0).values.tolist()
        print(
            f"[WARN _clamp_sparse_coords] Dropped {num_dropped}/{num_total} OOB points "
            f"(max_resolution={max_resolution}, range={spatial_min}..{spatial_max})."
        )

    new_coords = coords[valid_mask].contiguous()
    new_feats = x.feats[valid_mask].contiguous()
    out = sp.SparseTensor(new_feats, new_coords)
    out._scale = x._scale
    out._spatial_cache = {}
    return out


# ─────────────────────────────────────────────────────────────────────────────
# GT 键集合 / 几何带状精确裁剪辅助函数
#
# 设计目标：在 SparseSDFDecoder 的每个上采样步之后，把解码中间表示裁剪到
# 与 GT 稀疏数据对齐的范围内，而不是仅依赖宽松的 AABB。
#   - 训练：每步用粗化 GT 占用格裁剪，最终输出层前用 sorted_keys 精确匹配。
#   - 推理：以 decoder 输入坐标为锚点，按 extra_band_factor 扩张生成占用格裁剪。
# ─────────────────────────────────────────────────────────────────────────────

def _build_gt_coarse_occ(
    gt_xyz:   torch.Tensor,       # [N, 3] int – GT 体素坐标，at gt_R 分辨率
    gt_bidx:  torch.Tensor,       # [N] int – per-voxel batch index
    B:        int,
    gt_R:     int,                # GT 数据的分辨率（如 512）
    target_R: int,                # 目标粗化分辨率（如 128、256）
    device:   torch.device,
) -> torch.Tensor:                # [B, target_R, target_R, target_R] bool
    """
    把 gt_R 分辨率下的稀疏 GT 坐标映射（取整除）到 target_R 粗化格，
    生成 bool 占用格：若某粗化格内有任意 GT 体素则标记为 True。

    内存开销：B × target_R³ bytes（如 1×128³ ≈ 2 MB）。
    """
    assert gt_R % target_R == 0, (
        f"gt_R={gt_R} 必须能被 target_R={target_R} 整除"
    )
    S = gt_R // target_R
    R = target_R
    tx = (gt_xyz[:, 0].long() // S).clamp(0, R - 1)
    ty = (gt_xyz[:, 1].long() // S).clamp(0, R - 1)
    tz = (gt_xyz[:, 2].long() // S).clamp(0, R - 1)
    bi = gt_bidx.long().clamp(0, B - 1)
    occ = torch.zeros(B, R, R, R, dtype=torch.bool, device=device)
    occ[bi, tx, ty, tz] = True
    return occ


def _build_geometry_occ_at_R_model(
    input_coords:    torch.Tensor,  # [M, 4] (batch,x,y,z) – decoder 输入坐标，at R_model
    B:               int,
    R_model:         int,           # decoder 输入分辨率（如 64）
    extra_band_factor: float,       # 带宽，单位为 R_final 体素（如 4.0）
    R_final:         int,           # decoder 最终输出分辨率（如 512）
    device:          torch.device,
) -> torch.Tensor:                  # [B, R_model, R_model, R_model] bool – 扩张后占用格
    """
    推理时几何剪枝用的占用格，构建于 R_model 分辨率。

    策略：在 R_model 格标记 decoder 输入体素，然后做 max_pool3d 扩张。
    扩张半径 d 保证：当中间分辨率 R_cur 的体素查其 R_model 父格时，
    所有在 R_final 分辨率下距任意输入体素 ≤ extra_band_factor 的体素
    均被保留（保守覆盖）。

    对于 extra_band_factor=4, R_final=512, R_model=64：
        d = ceil(4 / (512/64)) + 1 = ceil(0.5) + 1 = 2
    """
    scale_to_final = R_final // R_model   # e.g. 8
    d = max(1, int(math.ceil(extra_band_factor / scale_to_final)) + 1)

    bi = input_coords[:, 0].long().clamp(0, B - 1)
    xi = input_coords[:, 1].long().clamp(0, R_model - 1)
    yi = input_coords[:, 2].long().clamp(0, R_model - 1)
    zi = input_coords[:, 3].long().clamp(0, R_model - 1)

    occ = torch.zeros(B, R_model, R_model, R_model, dtype=torch.float32, device=device)
    occ[bi, xi, yi, zi] = 1.0
    ks  = 2 * d + 1
    occ = F.max_pool3d(occ.unsqueeze(1), kernel_size=ks, stride=1, padding=d).squeeze(1)
    return occ > 0.5


def _build_geometry_occ_at_R_target(
    seed_coords:         torch.Tensor,  # [M, 4] (b,x,y,z) at R_seed (= R_model, e.g. 64)
    B:                   int,
    R_seed:              int,           # decoder latent 分辨率，如 64
    R_target:            int,           # 本步占用格分辨率（动态选取，≥ R_seed 且整除）
    band_factor_R_final: float,         # 紧带宽度，单位为 R_final 体素（如 0.5）
    R_final:             int,           # 最终输出分辨率，如 512
    device:              torch.device,
) -> torch.Tensor:                      # [B, R_target, R_target, R_target] bool
    """
    基于 latent seed 坐标（R_seed）在任意目标分辨率 R_target 上构造 dilated 占用格。

    算法：
      1. 把每个 seed 体素（R_seed）splat 到 R_target 上的 scale³ 个子格
         （scale = R_target // R_seed），得到初始 occ。
      2. 用 max_pool3d 扩张：半径 d = max(0, ceil(band_factor_R_final * R_target / R_final))。
         当 band_factor_R_final=0.5, R_target=256, R_final=512 时：
           d = ceil(0.5 * 256 / 512) = ceil(0.25) = 1
         当 band_factor_R_final=0.5, R_target=64, R_final=512 时：
           d = ceil(0.5 * 64 / 512) = ceil(0.0625) = 1，但 d=0 时跳过 pool 节省开销。

    注意：不加 +1 安全余量（与旧 _build_geometry_occ_at_R_model 的 +1 不同），
    以实现"尽量紧"的 band_factor 量级覆盖。
    R_target 必须是 R_seed 的整数倍。
    """
    assert R_target % R_seed == 0, (
        f"R_target={R_target} 必须是 R_seed={R_seed} 的整数倍"
    )
    scale = R_target // R_seed  # 每个 seed 体素在 R_target 上对应的边长

    bi = seed_coords[:, 0].long().clamp(0, B - 1)
    xi = seed_coords[:, 1].long().clamp(0, R_seed - 1)
    yi = seed_coords[:, 2].long().clamp(0, R_seed - 1)
    zi = seed_coords[:, 3].long().clamp(0, R_seed - 1)

    occ = torch.zeros(B, R_target, R_target, R_target, dtype=torch.float32, device=device)

    if scale == 1:
        # R_target == R_seed：直接标记
        occ[bi, xi, yi, zi] = 1.0
    else:
        # 把每个 seed 体素 splat 到 scale³ 个子格（向量化）
        # 生成 scale³ 个块内偏移
        g = torch.arange(scale, device=device)
        offsets = torch.stack(
            torch.meshgrid(g, g, g, indexing="ij"), dim=-1
        ).reshape(-1, 3)  # [scale³, 3]

        # seed 在 R_target 上的起始坐标
        sx = (xi * scale).unsqueeze(1) + offsets[:, 0].unsqueeze(0)  # [M, scale³]
        sy = (yi * scale).unsqueeze(1) + offsets[:, 1].unsqueeze(0)
        sz = (zi * scale).unsqueeze(1) + offsets[:, 2].unsqueeze(0)
        sbi = bi.unsqueeze(1).expand_as(sx)

        sx = sx.clamp(0, R_target - 1).reshape(-1)
        sy = sy.clamp(0, R_target - 1).reshape(-1)
        sz = sz.clamp(0, R_target - 1).reshape(-1)
        sbi = sbi.reshape(-1)

        occ[sbi, sx, sy, sz] = 1.0

    # 扩张：半径 d = ceil(band_factor_R_final * R_target / R_final)
    d = int(math.ceil(band_factor_R_final * R_target / R_final))
    if d > 0:
        ks  = 2 * d + 1
        occ = F.max_pool3d(occ.unsqueeze(1), kernel_size=ks, stride=1, padding=d).squeeze(1)

    return occ > 0.5


def _prune_sparse_by_occ_parent(
    x:    sp.SparseTensor,
    occ:  torch.Tensor,   # [B, R_occ, R_occ, R_occ] bool
    R_x:  int,            # x 坐标所在分辨率（R_occ ≤ R_x，且 R_x % R_occ == 0）
) -> sp.SparseTensor:
    """
    按占用格裁剪稀疏张量：每个体素的坐标除以 (R_x // R_occ) 找到其父格，
    保留父格为 True 的体素，丢弃其余（含越界点）。

    当 R_occ == R_x 时为直接查询；R_occ < R_x 时为父格查询（用于推理几何模式）。
    安全保护：若 keep 全为 False 则原样返回，防止误清空。
    """
    coords  = x.coords
    B_occ   = occ.shape[0]
    R_occ   = occ.shape[1]

    bi = coords[:, 0].long().clamp(0, B_occ - 1)
    xi = coords[:, 1].long()
    yi = coords[:, 2].long()
    zi = coords[:, 3].long()

    oob = (xi < 0) | (xi >= R_x) | (yi < 0) | (yi >= R_x) | (zi < 0) | (zi >= R_x)

    if R_occ == R_x:
        xi_q = xi.clamp(0, R_occ - 1)
        yi_q = yi.clamp(0, R_occ - 1)
        zi_q = zi.clamp(0, R_occ - 1)
    else:
        S    = R_x // R_occ
        xi_q = (xi // S).clamp(0, R_occ - 1)
        yi_q = (yi // S).clamp(0, R_occ - 1)
        zi_q = (zi // S).clamp(0, R_occ - 1)

    keep = occ[bi, xi_q, yi_q, zi_q] & ~oob

    if keep.all():
        return x
    if not keep.any():
        return x   # 安全保护：不全清空

    new_coords = coords[keep].contiguous()
    new_feats  = x.feats[keep].contiguous()
    out = sp.SparseTensor(new_feats, new_coords)
    out._scale         = getattr(x, "_scale", None)
    out._spatial_cache = {}
    return out


def _prune_sparse_by_sorted_keys(
    x:           sp.SparseTensor,
    sorted_keys: torch.Tensor,   # int64，升序排列
    R:           int,            # 体素格分辨率
) -> sp.SparseTensor:
    """
    精确 key 匹配裁剪：保留 (batch, x, y, z) 对应 key = batch*R³+x*R²+y*R+z
    存在于 sorted_keys 中的体素，其余丢弃。

    安全保护：若匹配集为空则原样返回。
    """
    coords = x.coords
    R3     = R * R * R

    bi = coords[:, 0].long()
    xi = coords[:, 1].long()
    yi = coords[:, 2].long()
    zi = coords[:, 3].long()

    oob  = (xi < 0) | (xi >= R) | (yi < 0) | (yi >= R) | (zi < 0) | (zi >= R)
    keys = bi * R3 + xi * (R * R) + yi * R + zi

    pos   = torch.searchsorted(sorted_keys, keys)
    pos_c = pos.clamp(0, sorted_keys.shape[0] - 1)
    found = (sorted_keys[pos_c] == keys) & ~oob

    if found.all():
        return x
    if not found.any():
        return x   # 安全保护：不全清空

    new_coords = coords[found].contiguous()
    new_feats  = x.feats[found].contiguous()
    out = sp.SparseTensor(new_feats, new_coords)
    out._scale         = getattr(x, "_scale", None)
    out._spatial_cache = {}
    return out


class SparseSubdivideBlock3d(nn.Module):

    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        use_checkpoint: bool = False,
        max_resolution: Optional[int] = None,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_checkpoint = use_checkpoint
        self.max_resolution = max_resolution

        self.act_layers = nn.Sequential(
            sp.SparseConv3d(channels, self.out_channels, 3, padding=1),
            sp.SparseSiLU()
        )
        
        self.sub = sp.SparseSubdivide()
        
        self.out_layers = nn.Sequential(
            sp.SparseConv3d(self.out_channels, self.out_channels, 3, padding=1),
            sp.SparseSiLU(),
        )
        self._tsdbg_block_tag = (
            f"SparseSubdivideBlock3d(in={channels},out={self.out_channels},"
            f"max_res={max_resolution},use_ckpt={use_checkpoint})"
        )

    def _forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        tag = self._tsdbg_block_tag
        _debug_dump_sparse_subdivide("00_input", x, tag)
        h = self.act_layers(x)
        _debug_dump_sparse_subdivide("10_after_act_layers", h, tag)
        h = self.sub(h)
        _debug_dump_sparse_subdivide("20_after_subdivide", h, tag)
        if self.max_resolution is not None:
            h = _clamp_sparse_coords(h, self.max_resolution)
            _debug_dump_sparse_subdivide("30_after_clamp_pre_out", h, tag)
        if _decoder_subdivide_step_debug():
            print(
                f"[TSDBG subdivide] {tag} | 40_calling_out_layers(Sequential: Conv3d+SiLU) "
                f"← 常见 ImplicitGEMM backward 崩溃点",
                flush=True,
            )
        h = self.out_layers(h)
        _debug_dump_sparse_subdivide("50_after_out_layers", h, tag)
        if self.max_resolution is not None:
            h = _clamp_sparse_coords(h, self.max_resolution)
            _debug_dump_sparse_subdivide("60_after_clamp_final", h, tag)
        return h

    def forward(self, x: torch.Tensor):
        if _decoder_subdivide_step_debug():
            print(
                f"[TSDBG subdivide] {self._tsdbg_block_tag} | forward entry "
                f"use_checkpoint={self.use_checkpoint} training={self.training}",
                flush=True,
            )
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)


class SparseSDFDecoder(SparseTransformerBase):
    def __init__(
        self,
        resolution: int,
        model_channels: int,
        latent_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        attn_mode: Literal["full", "shift_window", "shift_sequence", "shift_order", "swin"] = "swin",
        window_size: int = 8,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        qk_rms_norm: bool = False,
        representation_config: dict = None,
        out_channels: int = 1,
        chunk_size: int = 1,
        extra_down_up_levels: int = 0,
    ):
        super().__init__(
            in_channels=latent_channels,
            model_channels=model_channels,
            num_blocks=num_blocks,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            mlp_ratio=mlp_ratio,
            attn_mode=attn_mode,
            window_size=window_size,
            pe_mode=pe_mode,
            use_fp16=use_fp16,
            use_checkpoint=use_checkpoint,
            qk_rms_norm=qk_rms_norm,
        )
        self.resolution = resolution
        self.rep_config = representation_config
        self.out_channels = out_channels
        self.chunk_size = chunk_size
        self.extra_down_up_levels = int(extra_down_up_levels)
        if self.extra_down_up_levels < 0:
            raise ValueError("extra_down_up_levels must be >= 0")
        den = 2 ** self.extra_down_up_levels
        if resolution % den != 0:
            raise ValueError(
                f"resolution={resolution} must be divisible by 2**extra_down_up_levels={den}"
            )
        # 与 encoder 多出来的下采样级对称：decoder 输入（latent）坐标系相对 self.resolution 再粗 2^L 倍
        self.decoder_input_resolution = resolution // den

        # 每经过一次 SparseSubdivide 分辨率翻倍，记录合法坐标上界用于裁剪越界点
        # （防止反向传播时 sparse conv 因坐标越界产生显存非法访问而崩溃）。
        # 基线 4 个上采样块（512→256→128→64→32 通道语义）+ L 个前置同宽 subdivide，共 4+L 步 ×2 空间上采样
        n_up = 4 + self.extra_down_up_levels
        L = self.extra_down_up_levels
        res_clamps = [int(resolution * (2 ** (i + 1 - L))) for i in range(n_up)]
        # torchsparse ImplicitGEMM 在 backward 上与 torch.utils.checkpoint 组合时，
        # 大体素数（数百万级）易触发 illegal memory access（见 torchsparse#361 等）。
        # 上采样块显存主要来自 subdivide 后的稀疏体素，此处默认关闭 ckpt；需省显存可设环境变量开启。
        subdiv_ckpt = use_checkpoint and os.environ.get(
            "DECODER_SUBDIVIDE_USE_CHECKPOINT", ""
        ).strip() in ("1", "true", "True", "yes", "YES")
        upsample_blocks = []
        for j in range(L):
            upsample_blocks.append(
                SparseSubdivideBlock3d(
                    channels=model_channels,
                    out_channels=model_channels,
                    use_checkpoint=subdiv_ckpt,
                    max_resolution=res_clamps[j],
                )
            )
        C = model_channels
        upsample_blocks.extend(
            [
                SparseSubdivideBlock3d(
                    channels=C,
                    out_channels=C // 2,
                    use_checkpoint=subdiv_ckpt,
                    max_resolution=res_clamps[L + 0],
                ),
                SparseSubdivideBlock3d(
                    channels=C // 2,
                    out_channels=C // 4,
                    use_checkpoint=subdiv_ckpt,
                    max_resolution=res_clamps[L + 1],
                ),
                SparseSubdivideBlock3d(
                    channels=C // 4,
                    out_channels=C // 8,
                    use_checkpoint=subdiv_ckpt,
                    max_resolution=res_clamps[L + 2],
                ),
                SparseSubdivideBlock3d(
                    channels=C // 8,
                    out_channels=C // 16,
                    use_checkpoint=subdiv_ckpt,
                    max_resolution=res_clamps[L + 3],
                ),
            ]
        )
        self.upsample = nn.ModuleList(upsample_blocks)

        self.out_layer = sp.SparseLinear(model_channels // 16, self.out_channels)
        self.out_active = sp.SparseTanh()

        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()

    def initialize_weights(self) -> None:
        super().initialize_weights()
        # # Zero-out output layers:
        # nn.init.constant_(self.out_layer.weight, 0)
        # nn.init.constant_(self.out_layer.bias, 0)

    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        self.use_fp16 = True
        self.dtype = torch.float16
        super().convert_to_fp16()
        self.upsample.apply(convert_module_to_f16)
        self.out_layer.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        self.use_fp16 = False
        self.dtype = torch.float32
        super().convert_to_fp32()
        self.upsample.apply(convert_module_to_f32)
        self.out_layer.apply(convert_module_to_f32)  
    
    @torch.no_grad()
    def split_for_meshing(self, x: sp.SparseTensor, chunk_size=4, padding=4):
        
        sub_resolution = self.resolution // chunk_size
        upsample_ratio = 2 ** len(self.upsample)
        assert sub_resolution % padding == 0
        out = []
        
        for i in range(chunk_size):
            for j in range(chunk_size):
                for k in range(chunk_size):
                    # Calculate padded boundaries
                    start_x = max(0, i * sub_resolution - padding)
                    end_x = min((i + 1) * sub_resolution + padding, self.resolution)
                    start_y = max(0, j * sub_resolution - padding)
                    end_y = min((j + 1) * sub_resolution + padding, self.resolution)
                    start_z = max(0, k * sub_resolution - padding)
                    end_z = min((k + 1) * sub_resolution + padding, self.resolution)
                    
                    # Store original (unpadded) boundaries for later cropping
                    orig_start_x = i * sub_resolution
                    orig_end_x = (i + 1) * sub_resolution
                    orig_start_y = j * sub_resolution
                    orig_end_y = (j + 1) * sub_resolution
                    orig_start_z = k * sub_resolution
                    orig_end_z = (k + 1) * sub_resolution

                    mask = torch.logical_and(
                        torch.logical_and(
                            torch.logical_and(x.coords[:, 1] >= start_x, x.coords[:, 1] < end_x),
                            torch.logical_and(x.coords[:, 2] >= start_y, x.coords[:, 2] < end_y)
                        ),
                        torch.logical_and(x.coords[:, 3] >= start_z, x.coords[:, 3] < end_z)
                    )

                    if mask.sum() > 0:
                        # Get the coordinates and shift them to local space
                        coords = x.coords[mask].clone()
                        # Shift to local coordinates
                        coords[:, 1:] = coords[:, 1:] - torch.tensor([start_x, start_y, start_z], 
                                                                    device=coords.device).view(1, 3)

                        chunk_tensor = sp.SparseTensor(x.feats[mask], coords)
                        # Store the boundaries and offsets as metadata for later reconstruction
                        chunk_tensor.bounds = {
                            'original': (orig_start_x * upsample_ratio, orig_end_x * upsample_ratio + (upsample_ratio - 1), orig_start_y * upsample_ratio, orig_end_y * upsample_ratio + (upsample_ratio - 1), orig_start_z * upsample_ratio, orig_end_z * upsample_ratio + (upsample_ratio - 1)),
                            'offsets': (start_x * upsample_ratio, start_y * upsample_ratio, start_z * upsample_ratio)  # Store offsets for reconstruction
                        }
                        out.append(chunk_tensor)

                    del mask
                    torch.cuda.empty_cache()
        return out
    
    @torch.no_grad()
    def split_single_chunk(self, x: sp.SparseTensor, chunk_size=4, padding=4):
        sub_resolution = self.resolution // chunk_size
        upsample_ratio = 2 ** len(self.upsample)
        assert sub_resolution % padding == 0

        mask_sum = -1
        while mask_sum < 1:
            orig_start_x = random.randint(0, self.resolution - sub_resolution)
            orig_end_x = orig_start_x + sub_resolution
            orig_start_y = random.randint(0, self.resolution - sub_resolution)
            orig_end_y = orig_start_y + sub_resolution
            orig_start_z = random.randint(0, self.resolution - sub_resolution)
            orig_end_z = orig_start_z + sub_resolution
            start_x = max(0, orig_start_x - padding)
            end_x = min(orig_end_x + padding, self.resolution)
            start_y = max(0, orig_start_y - padding)
            end_y = min(orig_end_y + padding, self.resolution)
            start_z = max(0, orig_start_z - padding)
            end_z = min(orig_end_z + padding, self.resolution)

            mask_ori = torch.logical_and(
                torch.logical_and(
                    torch.logical_and(x.coords[:, 1] >= orig_start_x, x.coords[:, 1] < orig_end_x),
                    torch.logical_and(x.coords[:, 2] >= orig_start_y, x.coords[:, 2] < orig_end_y)
                ),
                torch.logical_and(x.coords[:, 3] >= orig_start_z, x.coords[:, 3] < orig_end_z)
            )
            mask_sum = mask_ori.sum()

        # Store the boundaries and offsets as metadata for later reconstruction
        bounds = {
            'original': (orig_start_x * upsample_ratio, orig_end_x * upsample_ratio + (upsample_ratio - 1), orig_start_y * upsample_ratio, orig_end_y * upsample_ratio + (upsample_ratio - 1), orig_start_z * upsample_ratio, orig_end_z * upsample_ratio + (upsample_ratio - 1)),
            'start': (start_x, end_x, start_y, end_y, start_z, end_z),
            'offsets': (start_x * upsample_ratio, start_y * upsample_ratio, start_z * upsample_ratio)  # Store offsets for reconstruction
        }
        return bounds
    
    def forward_single_chunk(self, x: sp.SparseTensor, padding=4):
        
        bounds = self.split_single_chunk(x, self.chunk_size, padding=padding)

        start_x, end_x, start_y, end_y, start_z, end_z = bounds['start']
        mask = torch.logical_and(
            torch.logical_and(
                torch.logical_and(x.coords[:, 1] >= start_x, x.coords[:, 1] < end_x),
                torch.logical_and(x.coords[:, 2] >= start_y, x.coords[:, 2] < end_y)
            ),
            torch.logical_and(x.coords[:, 3] >= start_z, x.coords[:, 3] < end_z)
        )

        # Shift to local coordinates
        coords = x.coords.clone()
        coords[:, 1:] = coords[:, 1:] - torch.tensor([start_x, start_y, start_z], 
                                                    device=coords.device).view(1, 3)

        chunk = sp.SparseTensor(x.feats[mask], coords[mask])
        # upsamples 内坐标为 chunk 局部系，不能与全局 output_resolution 的 AABB 混用
        chunk_result = self.upsamples(chunk, band_prune_boxes=None)

        coords = chunk_result.coords.clone()

        # Restore global coordinates
        offsets = torch.tensor(bounds['offsets'], 
                                device=coords.device).view(1, 3)
        coords[:, 1:] = coords[:, 1:] + offsets

        # Filter points within original bounds
        original = bounds['original']
        within_bounds = torch.logical_and(
            torch.logical_and(
                torch.logical_and(
                    coords[:, 1] >= original[0],
                    coords[:, 1] < original[1]
                ),
                torch.logical_and(
                    coords[:, 2] >= original[2],
                    coords[:, 2] < original[3]
                )
            ),
            torch.logical_and(
                coords[:, 3] >= original[4],
                coords[:, 3] < original[5]
            )
        )

        final_coords = coords[within_bounds]
        final_feats = chunk_result.feats[within_bounds]

        return sp.SparseTensor(final_feats, final_coords)

    def upsamples(self, x, return_feat: bool = False, band_prune_boxes=None):
        input_dtype = x.feats.dtype  # 保存原始 dtype，最后转回
        for i, block in enumerate(self.upsample):
            x = block(x)
            if band_prune_boxes is not None:
                bm0, bx0, R_out = band_prune_boxes
                R_cur = int(self.resolution * (2 ** (i + 1 - self.extra_down_up_levels)))
                bm, bx, gu = scale_band_prune_boxes_to_grid(bm0, bx0, R_out, R_cur)
                x = _prune_sparse_by_batch_aabb(x, bm, bx, gu)
        # out_layer 是 fp16，在转回 input_dtype 之前调用
        output = self.out_active(self.out_layer(x))
        # 最后才转回原始 dtype
        output = output.type(input_dtype)

        if return_feat:
            return output, x.type(input_dtype)
        else:
            return output
    
    def forward(
        self,
        x: sp.SparseTensor,
        factor: float = None,
        return_feat: bool = False,
        band_prune: Optional[dict] = None,
        gt_prune: Optional[dict] = None,
    ):
        """
        Args:
            x: 量化后的 latent SparseTensor（与 encoder 输出同坐标系）。
            factor: 位置编码缩放因子。
            return_feat: 是否同时返回 out_layer 前的特征。
            band_prune: 可选，按预处理宽频壳层范围裁剪解码中间体素，见 ``build_band_prune_aabb``。
                传入 ``{"enabled": False}`` 可关闭裁剪（推理无 ``wide_xyz`` / ``seed_coords`` 时）。
                仅 ``chunk_size<=1`` 时在全局坐标系下生效；``chunk_size>1`` 的 chunk 路径因局部坐标
                与全局 AABB 不对齐，内部不做 band 裁剪。
            gt_prune: 可选，在每个上采样步后做 GT 键集合对齐裁剪（比 AABB 更精确），
                最终输出层（out_layer）执行前做精确 key 匹配，保证只有 GT 带宽内的体素
                参与 out_layer 的前向 / 反向传播，从根本上避免 OOM。

                **训练模式** (``mode='keys'``)::

                    {
                        'mode':         'keys',
                        'gt_xyz':       Tensor[N, 3] int,   # GT 稀疏坐标（resolution 分辨率）
                        'gt_batch_idx': Tensor[N] int,      # per-voxel batch index
                        'B':            int,                # batch size
                        'resolution':   int,                # GT 分辨率，如 512
                    }

                每个上采样步后按粗化 GT 占用格（取整除 scale）裁剪；最终输出层前
                按精确 sorted_keys 匹配，体素数严格限制于 GT 宽频带内。

                **推理模式** (``mode='geometry'``)::

                    {
                        'mode':               'geometry',
                        'extra_band_factor':  float,   # 带宽（resolution 体素单位，如 4.0）
                        'resolution':         int,     # 输出分辨率，如 512
                    }

                以 decoder 输入坐标 ``x.coords``（``self.resolution`` 分辨率）为锚点，
                按 ``extra_band_factor`` 扩张生成占用格，在每个上采样步后裁剪。
                无需 GT 数据，适用于 LLM token 解码的推理场景。

                两种模式均不改变保留体素的计算逻辑（数学上与原始 decoder 一致），
                仅对每步输出做 mask-index，超出带宽的体素不进入后续 sparse conv，
                其梯度图自然断开，不参与反向传播。
        """
        h = super().forward(x, factor)
        if _decoder_subdivide_step_debug():
            print(
                f"[TSDBG decoder] after transformer torso: N={h.feats.shape[0]} "
                f"training={self.training} "
                f"band_prune={'on' if (band_prune is not None and band_prune.get('enabled', True) is not False) else 'off'} "
                f"gt_prune={'on' if gt_prune is not None else 'off'}",
                flush=True,
            )
        band_prune_boxes = None
        if band_prune is not None and band_prune.get("enabled", True) is not False:
            band_prune_boxes = build_band_prune_aabb(band_prune, h.coords.device)

        # ── 预计算 gt_prune 所需占用格 / sorted_keys ─────────────────────────
        _gtp_mode         = None
        _gtp_gt_occ       = {}   # R_cur (int) -> [B, R_cur^3] bool 占用格（训练模式中间步）
        _gtp_sorted_keys  = None  # int64 sorted Tensor（训练模式最终精确匹配）
        _gtp_R_final      = None
        # geometry 模式：循环内按步动态构造，以下变量在循环内使用
        _gtp_geom_seed    = None  # [M, 4] seed coords at R_model（推理模式）
        _gtp_geom_B       = None  # batch size
        _gtp_geom_extra   = None  # 紧带宽度（R_final voxel 单位）
        _gtp_geom_occ_max = None  # occ 分辨率上限

        if gt_prune is not None:
            _gtp_mode    = gt_prune.get('mode', 'keys')
            _gtp_R_final = int(gt_prune['resolution'])
            device       = h.coords.device
            R_model      = self.resolution  # 64

            if _gtp_mode == 'keys':
                gt_xyz  = gt_prune['gt_xyz']        # [N, 3] at R_final
                gt_bidx = gt_prune['gt_batch_idx']  # [N]
                B_gt    = int(gt_prune['B'])
                R_final = _gtp_R_final

                # 每个上采样步对应的中间分辨率
                R_cur = self.decoder_input_resolution
                for i in range(len(self.upsample)):
                    R_cur = R_cur * 2
                    if R_cur < R_final:
                        # 中间分辨率：用粗化 GT 占用格
                        _gtp_gt_occ[R_cur] = _build_gt_coarse_occ(
                            gt_xyz, gt_bidx, B_gt, R_final, R_cur, device
                        )
                    # R_cur == R_final：由下方 sorted_keys 处理

                # 构建最终分辨率的精确 sorted_keys
                R3    = R_final * R_final * R_final
                bi_k  = gt_bidx.long()
                gx    = gt_xyz[:, 0].long()
                gy    = gt_xyz[:, 1].long()
                gz    = gt_xyz[:, 2].long()
                keys  = bi_k * R3 + gx * (R_final * R_final) + gy * R_final + gz
                _gtp_sorted_keys = keys.sort().values

            elif _gtp_mode == 'geometry':
                # 推理模式：不预先构造占用格，每个上采样步根据 R_cur 动态构造。
                # 只保存 seed 坐标和参数，节省峰值显存。
                _gtp_geom_extra   = float(gt_prune['extra_band_factor'])
                _gtp_geom_B       = (int(x.coords[:, 0].max().item()) + 1
                                     if x.coords.shape[0] > 0 else 1)
                _gtp_geom_seed    = x.coords   # [M, 4] at R_model
                # occ_resolution：允许调用方通过 gt_prune['occ_resolution'] 指定，
                # 默认 R_model（保持旧行为；调用方传 256/512 可获得更细粒度裁剪）。
                _gtp_geom_occ_max = int(gt_prune.get('occ_resolution', R_model))

        if self.chunk_size <= 1:
            input_dtype = x.feats.dtype  # 保存原始 dtype，最后转回
            R_cur = self.decoder_input_resolution
            for i, block in enumerate(self.upsample):
                h     = block(h)
                R_cur = R_cur * 2

                # ── 原 AABB 裁剪（始终保留，作为快速粗剪） ────────────────
                if band_prune_boxes is not None:
                    bm0, bx0, R_out = band_prune_boxes
                    bm, bx, gu = scale_band_prune_boxes_to_grid(bm0, bx0, R_out, R_cur)
                    n0 = h.feats.shape[0]
                    h  = _prune_sparse_by_batch_aabb(h, bm, bx, gu)
                    if os.environ.get("DECODER_PRUNE_DEBUG", ""):
                        print(
                            f"[DECODER_PRUNE_DEBUG][AABB] block={i} R_cur={R_cur} "
                            f"N {n0}->{h.feats.shape[0]}  "
                            f"lo0={bm[0].tolist()} hi0={bx[0].tolist()}"
                        )

                # ── GT / 几何精确裁剪（比 AABB 更细，与 GT 键对齐） ──────
                if _gtp_mode == 'keys' and R_cur in _gtp_gt_occ:
                    n0 = h.feats.shape[0]
                    h  = _prune_sparse_by_occ_parent(h, _gtp_gt_occ[R_cur], R_cur)
                    if os.environ.get("DECODER_PRUNE_DEBUG", ""):
                        print(
                            f"[DECODER_PRUNE_DEBUG][GT-OCC] block={i} R_cur={R_cur} "
                            f"N {n0}->{h.feats.shape[0]}"
                        )
                elif _gtp_mode == 'geometry' and _gtp_geom_seed is not None:
                    # 每步独立构造 occ，在 R_target = min(R_cur, occ_max) 上做 splat+dilate。
                    # 扩张半径以 R_final voxel 为单位，紧度由 band_factor 决定（而非旧的 +1 保守量）。
                    R_target = min(R_cur, _gtp_geom_occ_max)
                    R_seed = self.decoder_input_resolution
                    # R_target 必须是 latent 坐标分辨率 R_seed 的整数倍
                    if R_target % R_seed != 0:
                        R_target = R_seed  # fallback
                    occ_cur = _build_geometry_occ_at_R_target(
                        _gtp_geom_seed, _gtp_geom_B, R_seed,
                        R_target, _gtp_geom_extra, _gtp_R_final, device,
                    )
                    n0 = h.feats.shape[0]
                    h  = _prune_sparse_by_occ_parent(h, occ_cur, R_cur)
                    del occ_cur  # 立即释放，避免多步累积显存
                    if os.environ.get("DECODER_PRUNE_DEBUG", ""):
                        print(
                            f"[DECODER_PRUNE_DEBUG][GEOM] block={i} R_cur={R_cur} "
                            f"R_target={R_target} extra={_gtp_geom_extra} "
                            f"N {n0}->{h.feats.shape[0]}"
                        )

            # ── 最终输出层前：精确 sorted_keys 匹配（训练模式） ───────────
            # 作用：把体素数从「上采样后全部」精确缩减到「GT 宽频带内体素数」，
            # 使 out_layer（SparseLinear）及其反向传播只处理必要的体素，
            # 从根本上消除因 OOM 导致的训练崩溃。
            if _gtp_mode == 'keys' and _gtp_sorted_keys is not None:
                n0 = h.feats.shape[0]
                h  = _prune_sparse_by_sorted_keys(h, _gtp_sorted_keys, _gtp_R_final)
                if os.environ.get("DECODER_PRUNE_DEBUG", ""):
                    print(
                        f"[DECODER_PRUNE_DEBUG][GT-KEYS] R={_gtp_R_final} "
                        f"N {n0}->{h.feats.shape[0]}"
                    )

            # out_layer 是 fp16，必须在转回 input_dtype 之前调用
            if return_feat:
                result = self.out_active(self.out_layer(h))
                return result.type(input_dtype), h.type(input_dtype)

            h = self.out_layer(h)
            h = self.out_active(h)
            # 最后才转回原始 dtype
            h = h.type(input_dtype)
            return h
        else:
            if self.training:
                return self.forward_single_chunk(h)
            else:
                batch_size = x.shape[0]                
                chunks = self.split_for_meshing(h, chunk_size=self.chunk_size)
                all_coords, all_feats = [], []
                for chunk_idx, chunk in enumerate(chunks):
                    chunk_result = self.upsamples(chunk, band_prune_boxes=None)

                    for b in range(batch_size):
                        mask = torch.nonzero(chunk_result.coords[:, 0] == b).squeeze(-1)
                        if mask.numel() > 0:
                            coords = chunk_result.coords[mask].clone()

                            # Restore global coordinates
                            offsets = torch.tensor(chunk.bounds['offsets'], 
                                                    device=coords.device).view(1, 3)
                            coords[:, 1:] = coords[:, 1:] + offsets

                            # Filter points within original bounds
                            bounds = chunk.bounds['original']
                            within_bounds = torch.logical_and(
                                torch.logical_and(
                                    torch.logical_and(
                                        coords[:, 1] >= bounds[0],
                                        coords[:, 1] < bounds[1]
                                    ),
                                    torch.logical_and(
                                        coords[:, 2] >= bounds[2],
                                        coords[:, 2] < bounds[3]
                                    )
                                ),
                                torch.logical_and(
                                    coords[:, 3] >= bounds[4],
                                    coords[:, 3] < bounds[5]
                                )
                            )
                            
                            if within_bounds.any():
                                all_coords.append(coords[within_bounds])
                                all_feats.append(chunk_result.feats[mask][within_bounds])
                    
                    if not self.training:
                        torch.cuda.empty_cache()

                final_coords = torch.cat(all_coords)
                final_feats = torch.cat(all_feats)
                
                return sp.SparseTensor(final_feats, final_coords)
            