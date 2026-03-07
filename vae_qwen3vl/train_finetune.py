"""
Fine-tune the 3D projector (and optional LoRA) on Qwen2-VL / Qwen3-VL checkpoint
using 3D-text paired data. VAE is frozen.
Supports multi-GPU training via HuggingFace Accelerate.
"""

import argparse
import os
import time
import math
from datetime import timedelta

# 确保 3D-VL 训练使用 torchsparse（新终端直接 python 运行时不会走 shell 脚本的 export）
os.environ.setdefault("SPARSE_BACKEND", "torchsparse")

import sys
import json
from datetime import datetime

import contextlib
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

# Add project root for trellis
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from accelerate import Accelerator
    from accelerate.utils import DistributedType
    from accelerate.utils import set_seed
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False


def _gpu_memory_stats(device=None) -> str:
    """返回当前 GPU 显存统计字符串，用于 OOM 调试。"""
    if not torch.cuda.is_available():
        return "CUDA not available"
    try:
        if device is None:
            device = torch.device("cuda")
        if isinstance(device, str):
            device = torch.device(device)
        idx = device.index if device.index is not None else 0
        allocated = torch.cuda.memory_allocated(idx) / (1024**3)
        reserved = torch.cuda.memory_reserved(idx) / (1024**3)
        total = torch.cuda.get_device_properties(idx).total_memory / (1024**3)
        free = total - reserved
        peak = torch.cuda.max_memory_allocated(idx) / (1024**3)
        return (
            f"GPU{idx}: alloc={allocated:.2f}GiB reserved={reserved:.2f}GiB "
            f"free≈{free:.2f}GiB total={total:.2f}GiB peak={peak:.2f}GiB"
        )
    except Exception as e:
        return f"mem_stats_err: {e}"


def _rank_prefix() -> str:
    """分布式下返回 rank 前缀，便于定位哪个进程到达哪个 checkpoint。"""
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            return f"[rank{dist.get_rank()}]"
    except Exception:
        pass
    return ""


def _dist_context_str() -> str:
    """返回统一的分布式上下文字符串，便于跨文件检索日志。"""
    rank = os.environ.get("RANK", "?")
    local_rank = os.environ.get("LOCAL_RANK", "?")
    world_size = os.environ.get("WORLD_SIZE", "?")
    return f"rank={rank} local_rank={local_rank} world={world_size}"


def _debug_parallel_enabled() -> bool:
    return os.environ.get("PARALLEL_DEBUG", "0") == "1"


def _debug_parallel_verbose_enabled() -> bool:
    return os.environ.get("PARALLEL_DEBUG_VERBOSE", "0") == "1"


def _dlog(msg: str, *, all_ranks: bool = True, force: bool = False) -> None:
    """
    并行训练统一 debug 日志：
    - force=True: 总是打印
    - 否则仅在 PARALLEL_DEBUG=1 时打印
    - all_ranks=False: 仅 rank0 打印
    """
    if not (force or _debug_parallel_enabled()):
        return
    if not all_ranks:
        try:
            if dist.is_initialized() and dist.get_rank() != 0:
                return
        except Exception:
            pass
    ts = time.strftime("%H:%M:%S")
    print(f"[DDBG {ts}] {_rank_prefix()}[{_dist_context_str()}] {msg}", flush=True)


def _gpu_memory_stats_all(include_peak: bool = False, compact: bool = True) -> str:
    """返回所有可见 GPU 的显存统计。compact=True 时输出简短格式。"""
    if not torch.cuda.is_available():
        return "no_cuda"
    parts = []
    for i in range(torch.cuda.device_count()):
        try:
            a = torch.cuda.memory_allocated(i) / (1024**3)
            p = torch.cuda.max_memory_allocated(i) / (1024**3) if include_peak else a
            if compact:
                parts.append(f"G{i}:{a:.1f}/{p:.1f}")
            else:
                r = torch.cuda.memory_reserved(i) / (1024**3)
                t = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                parts.append(f"GPU{i}: alloc={a:.2f} res={r:.2f} total={t:.2f} peak={p:.2f} GiB")
        except Exception:
            parts.append(f"G{i}:err")
    return " ".join(parts)


def load_vae_from_config(vae_config_path: str, vae_ckpt_path: str, device: str = "cuda"):
    """Load SparseSDFVQVAE from project config and checkpoint."""
    from trellis.models import SparseSDFVQVAE
    with open(vae_config_path, "r") as f:
        config = json.load(f)
    vae_args = config["models"]["vqvae"]["args"]
    model = SparseSDFVQVAE(**vae_args)
    ckpt = torch.load(vae_ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    model.load_state_dict(ckpt, strict=False)
    model = model.to(device)
    model.eval()
    return model


class Dummy3DTextDataset(Dataset):
    """
    Placeholder dataset: yields (feats_3d, coords_3d, input_ids, attention_mask, labels).
    latent_dim must match the VAE codebook dimension (e.g. 16). Replace with your
    3D-text pairs for real alignment training.
    """

    def __init__(
        self,
        num_samples: int = 100,
        max_3d_tokens: int = 512,
        seq_len: int = 64,
        vocab_size: int = 152064,
        latent_dim: int = 16,
    ):
        self.num_samples = num_samples
        self.max_3d_tokens = max_3d_tokens
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        torch.manual_seed(idx)
        N = torch.randint(100, self.max_3d_tokens + 1, (1,)).item()
        feats_3d = torch.randn(N, self.latent_dim)
        coords_3d = torch.cat([
            torch.zeros(N, 1, dtype=torch.long),
            torch.randint(0, 64, (N, 3)),
        ], dim=1)
        input_ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        attention_mask = torch.ones(self.seq_len)
        labels = input_ids.clone()
        labels[:-1] = -100
        return {
            "feats_3d": feats_3d,
            "coords_3d": coords_3d,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def _training_log_stream(
    log_path: str,
    orig_stdout,
    echo_to_console: bool = True,
    echo_debug_memory: bool = False,
    echo_all: bool = False,
):
    """
    Returns a file-like object that:
    - Writes all output to log_path
    - If echo_to_console: echoes to orig_stdout
    - echo_all=True: 回显全部输出（含 [DEBUG]），便于控制台实时查看
    - echo_all=False: 仅回显训练相关；echo_debug_memory=True 时额外回显 [MEM]/[DEBUG_MEMORY]
    - 多卡时仅 rank0 echo_to_console=True，避免多进程输出交错
    """

    class TrainingLogTee:
        def __init__(self, path, original, echo, echo_mem, echo_all_flag):
            self._log = open(path, "w", encoding="utf-8")
            self._orig = original
            self._echo = echo
            self._echo_mem = echo_mem
            self._echo_all = echo_all_flag

        def write(self, data):
            self._log.write(data)
            self._log.flush()
            if self._echo and data.strip():
                if self._echo_all:
                    self._orig.write(data)
                    self._orig.flush()
                elif self._echo_mem and ("[DEBUG_MEMORY]" in data or "[MEM]" in data):
                    self._orig.write(data)
                    self._orig.flush()
                elif "[DEBUG" not in data:
                    self._orig.write(data)
                    self._orig.flush()

        def flush(self):
            self._log.flush()
            if self._echo:
                self._orig.flush()

        def close(self):
            self._log.close()

    return TrainingLogTee(log_path, orig_stdout, echo_to_console, echo_debug_memory, echo_all)


def _to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, dict):
        return {k: _to_device(v, device) for k, v in x.items()}
    return x


def collate_3d_text(batch, latent_dim: int = 16):
    """Collate batch: pad 3D to max N in batch, stack text. latent_dim must match projector."""
    feats = [b["feats_3d"] for b in batch]
    coords = [b["coords_3d"] for b in batch]
    max_n = max(f.size(0) for f in feats)
    pad_feats = []
    pad_coords = []
    for i, (f, c) in enumerate(zip(feats, coords)):
        n = f.size(0)
        if n < max_n:
            f = torch.cat([f, torch.zeros(max_n - n, latent_dim)], dim=0)
            c = torch.cat([c, torch.zeros(max_n - n, 4, dtype=c.dtype)], dim=0)
        pad_feats.append(f)
        pad_coords.append(c)
    return {
        "feats_3d": torch.stack(pad_feats),
        "coords_3d": torch.stack(pad_coords),
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


def _make_output_subdir(args) -> str:
    """
    根据关键训练参数生成输出子目录名，便于多次训练不混淆。
    格式: ep{epochs}_lr{lr}_bs{batch_size}_n{samples}_lora{lora_r}_{timestamp}
    """
    lr_str = f"{args.lr:.0e}".replace("-0", "-").replace("+0", "+")  # 1e-4, 5e-5
    n_str = "all" if args.max_samples == 0 else str(args.max_samples)
    if args.dummy_data:
        n_str = f"d{args.dummy_samples}"
    lora_str = f"lora{args.lora_r}" if args.use_lora else "nolora"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"ep{args.epochs}_lr{lr_str}_bs{args.batch_size}_n{n_str}_{lora_str}_{ts}"


def _load_config(config_path: str, project_root: str) -> dict:
    """Load YAML config. Returns flattened dict (excluding nested 'accelerate')."""
    try:
        import yaml
    except ImportError:
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not cfg:
        return {}
    out = {}
    path_keys = ("vl_model", "vae_config", "vae_ckpt", "data_dir", "output_dir")
    for k, v in cfg.items():
        if k == "accelerate":
            out["_accelerate_config"] = v
            continue
        if isinstance(v, dict):
            for sk, sv in v.items():
                out[sk] = sv
        else:
            out[k] = v
    for k in path_keys:
        if k in out and isinstance(out[k], str) and out[k] and not os.path.isabs(out[k]):
            out[k] = os.path.normpath(os.path.join(project_root, out[k]))
    return out


def main():
    parser = argparse.ArgumentParser(description="Fine-tune 3D projector on Qwen2-VL / Qwen3-VL")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config (training + paths)")
    parser.add_argument("--vl_model", type=str, default=None, help="HuggingFace model name or path")
    parser.add_argument("--vae_config", type=str, default=None, help="Path to VAE config JSON")
    parser.add_argument("--vae_ckpt", type=str, default=None, help="Path to VAE checkpoint")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--max_3d_tokens", type=int, default=None, help="Max 3D token length")
    parser.add_argument("--truncate_mode", type=str, default="head", choices=["head", "random_sample"],
                        help="When 3D tokens exceed max_3d_tokens: head=take first L; random_sample=randomly sample L (better coverage, avoids OOM)")
    parser.add_argument("--use_3d_pos", action="store_true", help="Use 3D positional encoding")
    parser.add_argument("--projector_layers", type=int, default=1, help="Projector MLP layers (1 = Linear)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--weight_decay", type=float, default=0.0, help="AdamW weight decay")
    parser.add_argument("--grad_clip", type=float, default=0.0, help="Gradient clipping max norm (0=disabled)")
    parser.add_argument("--use_lr_scheduler", action="store_true", help="Use cosine LR schedule with warmup")
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="Warmup ratio of total steps")
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA on LLM")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=None, help="LoRA alpha (default 2*r)")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--dummy_data", action="store_true", help="Use dummy dataset")
    parser.add_argument("--dummy_samples", type=int, default=50)
    parser.add_argument("--data_dir", type=str, default=None, help="Path to SDF dataset")
    parser.add_argument("--data_format", type=str, default="sdf_caption", choices=["sdf_caption"])
    parser.add_argument("--max_samples", type=int, default=2000, help="Max samples (0 = all)")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader CPU worker 数（0=主进程加载；>0 多进程预取，可按 CPU 核数调整）")
    parser.add_argument("--prompt", type=str, default="Describe this 3D shape in one sentence:", help="Caption prompt")
    parser.add_argument("--use_discrete_3d_tokens", action="store_true", help="Use discrete mesh tokens (8x8x8 pool) and no Projector")
    parser.add_argument("--use_variable_length_3d_tokens", action="store_true", help="Variable-length 3D: keep all VAE points, Morton sort, dynamic pad (use with use_discrete_3d_tokens)")
    parser.add_argument("--max_safe_3d_length", type=int, default=15000, help="Soft cap: FPS downsample only when N > this (variable-length mode)")
    parser.add_argument("--coord_max_3d", type=int, default=64, help="Max coord for Morton (current VAE: 64^3 latent=64; if VAE outputs 512^3 use 512)")
    parser.add_argument("--max_length_variable", type=int, default=32768, help="Max token length when variable-length 3D (avoid truncating 8k~12k mesh)")
    parser.add_argument("--reconstruction_ratio", type=float, default=0.0, help="Fraction of samples as 3D reconstruction task (0..1), only when use_discrete_3d_tokens")
    parser.add_argument("--training_stage", type=str, default="sft", choices=["warmup", "sft"], help="warmup = only embed+lm_head; sft = LoRA + embed")
    parser.add_argument("--warmup_output_dir", type=str, default=None,
                        help="SFT 时从 warmup 输出目录加载 tokenizer 与 embed/lm_head；留空则从 vl_model 重新建词表")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation (recommend 4~8 for variable-length with batch_size=1)")
    parser.add_argument("--use_flash_attn_2", action="store_true", help="Use Flash Attention 2 for long sequences (variable-length)")
    parser.add_argument("--use_gradient_checkpointing", action="store_true", help="Gradient checkpointing to save VRAM (variable-length)")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Tensor parallel degree (1=disabled; 2=TP+FSDP2 2D parallel). "
                             "For Qwen3-VL 2B with GQA (2 KV heads) max is 2. "
                             "With 4 GPUs: tp=2 gives TP=2 + DP=2.")
    parser.add_argument("--sequence_parallel_size", type=int, default=1,
                        help="Sequence parallel degree (1=disabled). TP+SP hybrid: "
                             "e.g. 4 GPUs with tp=2, sp=2 -> DP=1, SP=2, TP=2.")
    parser.add_argument("--debug_vl_forward", action="store_true",
                        help="打印 VL forward 输入的详细 debug 信息（combined_embeds 形状等）")
    parser.add_argument("--debug_qwen3vl_patch", action="store_true",
                        help="开启 qwen3vl_debug_patch 的详细日志（如 transformers 版本）")
    parser.add_argument("--debug_memory", action="store_true",
                        help="在每个训练步骤的关键阶段打印 GPU 显存统计，用于定位 OOM 步骤")
    parser.add_argument("--echo_all", action="store_true",
                        help="控制台回显全部输出（含 [DEBUG] 等），便于实时查看；否则仅回显部分训练信息")
    parser.add_argument("--debug_parallel", action="store_true",
                        help="并行调试总开关：输出 TP/SP/DP 组、collective 与同步点详细日志")
    parser.add_argument("--debug_parallel_verbose", action="store_true",
                        help="并行调试详细模式：输出每层 TP/SP 打补丁与梯度同步细节（日志量大）")
    parser.add_argument("--debug_sync_points", action="store_true",
                        help="在关键阶段插入同步点日志（barrier 前后耗时），用于定位卡住位置")
    parser.add_argument("--debug_loss", action="store_true",
                        help="每步打印 valid_label_tokens 与 lr，用于排查 loss=NaN/Inf（默认仅前 5 步打印）")
    parser.add_argument("--dist_timeout_seconds", type=int, default=600,
                        help="torch.distributed init_process_group 超时时间（秒）")
    # torchrun/accelerate 可能注入 --local-rank，用 parse_known_args 避免未知参数导致失败
    parser.add_argument("--local_rank", "--local-rank", type=int, default=-1,
                        help="Local rank (injected by torchrun, prefer env LOCAL_RANK)")
    args, remaining = parser.parse_known_args()
    known_keys = {a.dest for a in parser._actions if a.dest != "help"}

    # 将 config 路径解析为绝对路径，避免 torchrun 子进程 cwd 不同导致相对路径找不到文件
    config_path = None
    if args.config:
        config_path = args.config if os.path.isabs(args.config) else os.path.normpath(os.path.join(PROJECT_ROOT, args.config))
    cfg = {}
    if config_path and os.path.isfile(config_path):
        cfg = _load_config(config_path, PROJECT_ROOT)
        overrides = {k: v for k, v in cfg.items() if not k.startswith("_") and k in known_keys}
        parser.set_defaults(**overrides)
        args, _ = parser.parse_known_args(remaining)
    else:
        args, _ = parser.parse_known_args(remaining if remaining else [])

    # 兜底：config 已加载但 overrides 可能未覆盖（如 known_keys 差异），显式补齐关键参数
    for key in ("vl_model", "vae_config", "vae_ckpt", "data_dir", "output_dir", "warmup_output_dir"):
        if key in cfg and getattr(args, key, None) is None and cfg[key]:
            setattr(args, key, cfg[key])

    if not getattr(args, "vl_model", None):
        raise ValueError(
            "vl_model 未设置。请确保 config YAML 中有 vl_model 字段，"
            "或通过 --vl_model 传参。当前 config_path=%r" % (config_path,)
        )

    # 并行调试开关写入环境变量，供其它模块（TP/SP/ring attention）复用。
    if getattr(args, "debug_parallel", False):
        os.environ["PARALLEL_DEBUG"] = "1"
    if getattr(args, "debug_parallel_verbose", False):
        os.environ["PARALLEL_DEBUG_VERBOSE"] = "1"
    if getattr(args, "debug_parallel", False) and "RING_ATTN_DEBUG" not in os.environ:
        # 统一打开 ring attention 关键路径日志（send/recv 与 step）。
        os.environ["RING_ATTN_DEBUG"] = "1"
    if getattr(args, "debug_loss", False):
        # 供 model / sequence_parallel_utils 读取，输出 loss 相关详细日志。
        os.environ["DEBUG_LOSS"] = "1"

    # -----------------------------------------------------------------------
    # Distributed setup: TP+FSDP2 path vs. legacy Accelerate path
    # -----------------------------------------------------------------------
    tp_size = getattr(args, "tensor_parallel_size", 1)
    sp_size = getattr(args, "sequence_parallel_size", 1)
    use_tp = tp_size > 1
    use_sp = sp_size > 1
    tp_mesh = None
    dp_mesh = None
    mesh_2d = None
    mesh_3d = None
    sp_group = None

    if use_tp:
        # Native torch.distributed init (torchrun sets LOCAL_RANK / RANK / WORLD_SIZE)
        if not dist.is_initialized():
            _dlog(
                f"init_process_group start backend=nccl timeout={args.dist_timeout_seconds}s "
                f"MASTER_ADDR={os.environ.get('MASTER_ADDR', '?')} "
                f"MASTER_PORT={os.environ.get('MASTER_PORT', '?')}",
                force=True,
            )
            _init_t0 = time.time()
            dist.init_process_group(
                backend="nccl",
                timeout=timedelta(seconds=max(1, int(args.dist_timeout_seconds))),
            )
            _dlog(
                f"init_process_group done in {time.time() - _init_t0:.3f}s",
                force=True,
            )
        # ---- 多机分布式诊断 ----
        _diag_rank = os.environ.get("RANK", "?")
        _diag_local_rank = os.environ.get("LOCAL_RANK", "?")
        _diag_world = os.environ.get("WORLD_SIZE", "?")
        _diag_local_world = os.environ.get("LOCAL_WORLD_SIZE", "?")
        _diag_group_rank = os.environ.get("GROUP_RANK", "?")
        _diag_role = os.environ.get("ROLE", "?")
        _diag_master_addr = os.environ.get("MASTER_ADDR", "?")
        _diag_master_port = os.environ.get("MASTER_PORT", "?")
        _diag_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "(not set)")
        _diag_gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        _diag_gpu_names = []
        for _gi in range(_diag_gpu_count):
            try:
                _diag_gpu_names.append(f"  GPU{_gi}: {torch.cuda.get_device_name(_gi)}")
            except Exception as _ge:
                _diag_gpu_names.append(f"  GPU{_gi}: error={_ge}")
        import socket as _socket
        _diag_hostname = _socket.gethostname()
        _diag_lines = [
            "=" * 60,
            f"[Multi-Node Diagnostic] hostname={_diag_hostname}",
            f"  RANK={_diag_rank}  LOCAL_RANK={_diag_local_rank}  WORLD_SIZE={_diag_world}",
            f"  LOCAL_WORLD_SIZE={_diag_local_world}  GROUP_RANK={_diag_group_rank}  ROLE={_diag_role}",
            f"  MASTER_ADDR={_diag_master_addr}  MASTER_PORT={_diag_master_port}",
            f"  CUDA_VISIBLE_DEVICES={_diag_cuda_visible}",
            f"  torch.cuda.device_count()={_diag_gpu_count}",
        ] + _diag_gpu_names + ["=" * 60]
        _diag_msg = "\n".join(_diag_lines)
        print(_diag_msg, flush=True)
        _diag_log_path = os.path.join(
            PROJECT_ROOT, "configs", f"train_diag_rank{_diag_rank}.txt"
        )
        os.makedirs(os.path.dirname(_diag_log_path), exist_ok=True)
        with open(_diag_log_path, "w", encoding="utf-8") as _df:
            _df.write(_diag_msg + "\n")
        # ---- 诊断结束 ----

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if local_rank >= _diag_gpu_count:
            _fix_msg = (
                f"[WARN] LOCAL_RANK={local_rank} >= device_count={_diag_gpu_count} on {_diag_hostname}! "
                f"This typically means the cluster framework is passing the GLOBAL rank as LOCAL_RANK. "
                f"Attempting fix: local_rank = RANK % device_count = {int(_diag_rank) % _diag_gpu_count if _diag_rank != '?' and _diag_gpu_count > 0 else '??'}"
            )
            print(_fix_msg, flush=True)
            with open(_diag_log_path, "a", encoding="utf-8") as _df:
                _df.write(_fix_msg + "\n")
            if _diag_gpu_count > 0 and _diag_rank != "?":
                local_rank = int(_diag_rank) % _diag_gpu_count
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        world_size = dist.get_world_size()
        global_rank = dist.get_rank()
        is_main_process = (global_rank == 0)
        _dlog(
            f"dist ready backend=nccl rank={global_rank}/{world_size} local_rank={local_rank} "
            f"cuda_device={torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'}",
            force=True,
        )

        if use_sp:
            from vae_qwen3vl.tensor_parallel_utils import get_3d_mesh
            from vae_qwen3vl.sequence_parallel_utils import get_sp_group_from_mesh
            mesh_3d, dp_mesh, sp_mesh, tp_mesh = get_3d_mesh(tp_size, sp_size, world_size)
            sp_group = get_sp_group_from_mesh(mesh_3d)
            accelerator = None
            torch.manual_seed(42)
            dp_sz = dp_mesh.size()
            _dlog(
                f"mesh_3d ready dims=(dp={dp_sz},sp={sp_size},tp={tp_size}) "
                f"global_rank={global_rank} local_rank={local_rank}",
                all_ranks=True,
                force=True,
            )
        else:
            from torch.distributed.device_mesh import init_device_mesh
            dp_size = world_size // tp_size
            if world_size % tp_size != 0:
                raise ValueError(
                    f"world_size={world_size} must be divisible by "
                    f"tensor_parallel_size={tp_size}"
                )
            mesh_2d = init_device_mesh(
                "cuda", (dp_size, tp_size), mesh_dim_names=("dp", "tp")
            )
            dp_mesh = mesh_2d["dp"]
            tp_mesh = mesh_2d["tp"]
            accelerator = None
            torch.manual_seed(42)
            _dlog(
                f"mesh_2d ready dims=(dp={dp_size},tp={tp_size}) "
                f"global_rank={global_rank} local_rank={local_rank}",
                all_ranks=True,
                force=True,
            )
    else:
        accelerator = Accelerator(
            gradient_accumulation_steps=getattr(args, "gradient_accumulation_steps", 1),
            mixed_precision="bf16" if torch.cuda.is_available() else "no",
        ) if ACCELERATE_AVAILABLE else None
        device = accelerator.device if accelerator is not None else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        is_main_process = (accelerator is None or accelerator.is_main_process)
        if accelerator is not None:
            set_seed(42)
            accelerator.print(f"Accelerate: {accelerator.num_processes} processes, device={device}")
        _dlog(
            f"legacy accelerate path enabled distributed={accelerator is not None}",
            force=getattr(args, "debug_parallel", False),
        )

    output_base = args.output_dir or os.path.join(PROJECT_ROOT, "outputs_3d_align")
    args.output_dir = os.path.join(output_base, _make_output_subdir(args))
    if is_main_process:
        print(f"Output dir: {args.output_dir}", flush=True)
    os.makedirs(args.output_dir, exist_ok=True)

    rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))
    log_dir = os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}_rank{rank}.log")
    _orig_stdout = sys.stdout
    _log_tee = _training_log_stream(
        log_path, _orig_stdout,
        echo_to_console=(rank == 0),
        echo_debug_memory=getattr(args, "debug_memory", False),
        echo_all=getattr(args, "echo_all", False),
    )
    sys.stdout = _log_tee
    if is_main_process:
        print(f"[Train] Full log -> {log_path}", flush=True)
        _dlog(
            "train args snapshot: "
            f"use_tp={use_tp} tp={tp_size} sp={sp_size} "
            f"stage={training_stage if 'training_stage' in locals() else getattr(args, 'training_stage', 'sft')} "
            f"batch_size={args.batch_size} grad_accum={getattr(args, 'gradient_accumulation_steps', 1)} "
            f"debug_memory={getattr(args, 'debug_memory', False)} "
            f"debug_parallel={getattr(args, 'debug_parallel', False)} "
            f"debug_parallel_verbose={getattr(args, 'debug_parallel_verbose', False)}",
            all_ranks=False,
            force=True,
        )

    metrics_file = None
    use_discrete = getattr(args, "use_discrete_3d_tokens", False)
    training_stage = getattr(args, "training_stage", "sft")
    try:
        vae_model = None
        if args.vae_config and args.vae_ckpt:
            vae_model = load_vae_from_config(args.vae_config, args.vae_ckpt, str(device))

        from transformers import AutoTokenizer
        from vae_qwen3vl import Qwen3VLWith3DBranch, add_mesh_tokens_to_tokenizer, resize_token_embeddings_and_init_mesh

        # SFT 时从 warmup 输出加载已扩展的 tokenizer；否则从 vl_model 建词表
        warmup_dir = getattr(args, "warmup_output_dir", None) or os.environ.get("WARMUP_OUTPUT_DIR")
        if warmup_dir and not os.path.isabs(warmup_dir):
            warmup_dir = os.path.normpath(os.path.join(PROJECT_ROOT, warmup_dir))
        if use_discrete and warmup_dir and os.path.isdir(warmup_dir):
            for tok_name in ("tokenizer_final", "tokenizer_epoch0"):
                tok_path = os.path.join(warmup_dir, tok_name)
                if os.path.isdir(tok_path):
                    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
                    if is_main_process:
                        _print(f"Loaded tokenizer from warmup: {tok_path}")
                    break
            else:
                tokenizer = AutoTokenizer.from_pretrained(args.vl_model, trust_remote_code=True)
                add_mesh_tokens_to_tokenizer(tokenizer)
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.vl_model, trust_remote_code=True)
            if use_discrete:
                add_mesh_tokens_to_tokenizer(tokenizer)

        vl_kwargs = {}
        if getattr(args, "use_flash_attn_2", False):
            vl_kwargs["attn_implementation"] = "flash_attention_2"
        if getattr(args, "debug_qwen3vl_patch", False):
            from vae_qwen3vl.qwen3vl_debug_patch import enable_debug_logging, print_transformers_info
            enable_debug_logging(True)
            print_transformers_info()
        model = Qwen3VLWith3DBranch(
            model_name_or_path=args.vl_model,
            vae_model=vae_model,
            latent_dim=16,
            max_3d_tokens=args.max_3d_tokens or 512,
            use_3d_pos=args.use_3d_pos,
            projector_num_layers=args.projector_layers,
            torch_dtype=torch.bfloat16 if str(device).startswith("cuda") else torch.float32,
            truncate_mode=args.truncate_mode,
            use_discrete_3d_tokens=use_discrete,
            **vl_kwargs,
        )
        if getattr(args, "debug_vl_forward", False):
            _model = getattr(model, "module", model)
            _model._debug_vl_forward = True
            if is_main_process:
                print("[Debug] 已开启 VL forward 详细日志 (combined_embeds 形状等)", flush=True)
        model = model.to(device)
        use_gc = bool(getattr(args, "use_gradient_checkpointing", False))
        if not use_tp:
            # Legacy Accelerate/DeepSpeed path: disable GC when warmup+DeepSpeed to
            # avoid NoneType.next_functions backward hook error in ZeRO-2.
            is_deepspeed = bool(
                accelerator is not None
                and getattr(accelerator.state, "distributed_type", None) == DistributedType.DEEPSPEED
            )
            if use_gc and training_stage == "warmup" and is_deepspeed:
                print(
                    "[Warn] Disable gradient checkpointing for warmup+DeepSpeed "
                    "to avoid backward hook error.", flush=True
                )
                use_gc = False
        # TP mode: no DeepSpeed, GC is always safe — apply unconditionally.
        # GC must be enabled BEFORE FSDP2 wrapping so FSDP2 tracks recompute boundaries.
        if use_gc and hasattr(model.vl_model, "gradient_checkpointing_enable"):
            model.vl_model.gradient_checkpointing_enable()
            # Required for HF models under GC so trainable params backprop correctly.
            if hasattr(model.vl_model, "enable_input_require_grads"):
                model.vl_model.enable_input_require_grads()
        # Training path should always disable KV cache; this avoids incompatible
        # attention return signatures under GC/SP patching.
        if hasattr(model.vl_model, "config"):
            model.vl_model.config.use_cache = False
        if use_discrete:
            resize_token_embeddings_and_init_mesh(model, tokenizer)

        # SFT 时从 warmup 输出加载已训练的 embed_tokens / lm_head
        if use_discrete and training_stage == "sft" and warmup_dir and os.path.isdir(warmup_dir):
            embed_ckpt = os.path.join(warmup_dir, "warmup_embed_lmhead_final.pt")
            if not os.path.isfile(embed_ckpt):
                import glob
                candidates = glob.glob(os.path.join(warmup_dir, "warmup_embed_lmhead_epoch*.pt"))
                if candidates:
                    def _epoch_num(p):
                        b = os.path.basename(p)
                        import re
                        m = re.search(r"warmup_embed_lmhead_epoch(\d+)\.pt", b)
                        return int(m.group(1)) if m else -1
                    embed_ckpt = max(candidates, key=_epoch_num)
            if os.path.isfile(embed_ckpt):
                try:
                    ckpt = torch.load(embed_ckpt, map_location="cpu", weights_only=True)
                except TypeError:
                    ckpt = torch.load(embed_ckpt, map_location="cpu")
                if ckpt:
                    # 可能被 FSDP/DP 包装，仅 main 加载并 broadcast 较复杂；这里各 rank 都加载同一文件
                    missing, unexpected = model.load_state_dict(ckpt, strict=False)
                    if is_main_process:
                        _print(f"Loaded warmup embed+lm_head from {embed_ckpt} (missing={len(missing)}, unexpected={len(unexpected)})")
            elif is_main_process:
                _print(f"[Warmup] No warmup_embed_lmhead*.pt in {warmup_dir}, SFT 使用当前初始化的 embed/lm_head")

        latent_dim = model.projector.latent_dim if model.projector is not None else 16

        # Stage1 warmup: only embedding + lm_head; Stage2 sft: projector (if any) or LoRA + embed
        for name, p in model.named_parameters():
            p.requires_grad = False
        if training_stage == "warmup":
            for name, p in model.named_parameters():
                if "embed_tokens" in name or "embedding" in name or "lm_head" in name:
                    p.requires_grad = True
        else:
            if model.projector is not None:
                for name, p in model.named_parameters():
                    if "projector" in name:
                        p.requires_grad = True
            for name, p in model.named_parameters():
                if "embed_tokens" in name or "embedding" in name or "lm_head" in name:
                    p.requires_grad = True
        use_lora_applied = False
        if args.use_lora and training_stage == "sft":
            try:
                from peft import get_peft_model, LoraConfig, TaskType
                lora_alpha = args.lora_alpha if args.lora_alpha is not None else (2 * args.lora_r)
                lora_config = LoraConfig(
                    r=args.lora_r,
                    lora_alpha=lora_alpha,
                    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                    lora_dropout=args.lora_dropout,
                    bias="none",
                    task_type=TaskType.CAUSAL_LM,
                )
                model.vl_model = get_peft_model(model.vl_model, lora_config)
                for name, p in model.vl_model.named_parameters():
                    if "lora" in name:
                        p.requires_grad = True
                use_lora_applied = True
                print("LoRA applied successfully.", flush=True)
            except ImportError:
                print("peft not installed; skipping LoRA", flush=True)

        # -------------------------------------------------------------------
        # Tensor Parallelism + Sequence Parallelism + FSDP2 (TP mode only)
        # -------------------------------------------------------------------
        if use_tp:
            from vae_qwen3vl.tensor_parallel_utils import (
                apply_tp_to_qwen3vl,
                apply_fsdp2_dp,
                register_dp_grad_hooks,
            )
            # FSDP2 要求所有参数具有统一的 dtype，否则会触发:
            # AssertionError: FSDP expects uniform original parameter dtype but got {torch.bfloat16, torch.float32}
            target_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            model = model.to(target_dtype)
            # In SFT+LoRA: LoRA wraps attention projections (q/k/v/o_proj).
            apply_attn_tp = not use_lora_applied
            apply_tp_to_qwen3vl(
                model.vl_model,
                tp_mesh,
                apply_to_attention=apply_attn_tp,
                apply_to_mlp=True,
            )
            _dlog(
                f"apply_tp_to_qwen3vl finished attn={apply_attn_tp} mlp=True",
                all_ranks=True,
                force=getattr(args, "debug_parallel", False),
            )
            if use_sp and sp_group is not None:
                from vae_qwen3vl.sequence_parallel_utils import apply_sp_attention_patch
                apply_sp_attention_patch(model, sp_group)
                model.sp_group = sp_group
                _dlog(
                    "apply_sp_attention_patch finished; model.sp_group attached",
                    all_ranks=True,
                    force=getattr(args, "debug_parallel", False),
                )
            if use_sp:
                # 3D mesh (dp, sp, tp): PyTorch 要求 tp 为最内层维度，导致 ("dp","tp")
                # 不相邻 → FSDP2+TP 组合失败。改用梯度钩子做 DP 梯度同步。
                # TP 已通过 DTensor 自动处理 TP 内部梯度规约；钩子负责跨 dp ranks 平均。
                register_dp_grad_hooks(model, dp_mesh)
                _dlog(
                    "register_dp_grad_hooks finished (SP path, no FSDP2)",
                    all_ranks=True,
                    force=getattr(args, "debug_parallel", False),
                )
            else:
                # 2D mesh (dp, tp): tp 在最内层且 ("dp","tp") 相邻 → FSDP2+TP 正常工作
                apply_fsdp2_dp(model, dp_mesh)
                _dlog(
                    "apply_fsdp2_dp finished (TP+DP path)",
                    all_ranks=True,
                    force=getattr(args, "debug_parallel", False),
                )

        # DTensor + AdamW foreach kernels are not fully supported yet
        # (e.g., aten._foreach_mul_.Scalar cross-mesh), so disable foreach
        # explicitly in TP+FSDP mode.
        adamw_kwargs = {}
        if use_tp:
            adamw_kwargs["foreach"] = False
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr,
            weight_decay=args.weight_decay,
            **adamw_kwargs,
        )

        if args.dummy_data:
            dataset = Dummy3DTextDataset(
                num_samples=args.dummy_samples,
                max_3d_tokens=min(256, args.max_3d_tokens),
                latent_dim=latent_dim,
            )
            if use_tp:
                _sampler = DistributedSampler(
                    dataset,
                    num_replicas=dp_mesh.size(),
                    rank=dp_mesh.get_local_rank(),
                    shuffle=True,
                    drop_last=True,
                )
                dataloader = DataLoader(
                    dataset,
                    batch_size=args.batch_size,
                    sampler=_sampler,
                    collate_fn=lambda b: collate_3d_text(b, latent_dim=latent_dim),
                    num_workers=getattr(args, "num_workers", 0),
                )
            else:
                dataloader = DataLoader(
                    dataset,
                    batch_size=args.batch_size,
                    shuffle=True,
                    collate_fn=lambda b: collate_3d_text(b, latent_dim=latent_dim),
                    num_workers=getattr(args, "num_workers", 0),
                )
        elif args.data_dir:
            from vae_qwen3vl.dataset_sdf_caption import (
                SDF3DCaptionDataset,
                collate_sdf_caption,
                collate_sdf_caption_discrete,
            )
            if not use_discrete:
                tokenizer = AutoTokenizer.from_pretrained(args.vl_model, trust_remote_code=True)
            dataset = SDF3DCaptionDataset(
                sdf_dir=args.data_dir,
                resolution=512,
                min_points=100,
                max_points=500000,
                max_samples=args.max_samples if args.max_samples > 0 else None,
            )
            if use_discrete:
                if vae_model is None:
                    raise ValueError("Discrete 3D token mode requires --vae_config and --vae_ckpt for collate encoding.")
                reconstruction_ratio = getattr(args, "reconstruction_ratio", 0.0)
                use_variable_length_3d = getattr(args, "use_variable_length_3d_tokens", False)
                # TP+FSDP: vae_model is wrapped (params are DTensors); collate needs plain tensors.
                # Use a separate unwrapped VAE instance for collate when use_tp.
                vae_for_collate = vae_model
                if use_tp:
                    vae_for_collate = load_vae_from_config(args.vae_config, args.vae_ckpt, str(device))
                collate_fn = lambda b: collate_sdf_caption_discrete(
                    b,
                    tokenizer,
                    vae_for_collate,
                    device,
                    prompt=args.prompt,
                    max_length=2048,
                    reconstruction_ratio=reconstruction_ratio,
                    use_variable_length_3d=use_variable_length_3d,
                    max_safe_3d_length=getattr(args, "max_safe_3d_length", 15000),
                    coord_max_3d=getattr(args, "coord_max_3d", 64),
                    max_length_variable=getattr(args, "max_length_variable", 32768) if use_variable_length_3d else None,
                )
            else:
                collate_fn = lambda b: collate_sdf_caption(b, tokenizer, prompt=args.prompt, max_length=256)
            if use_tp:
                # TP mode: shard data across DP ranks only (TP-group peers see same data)
                dp_rank = dp_mesh.get_local_rank()
                dp_world = dp_mesh.size()
                _sampler = DistributedSampler(
                    dataset,
                    num_replicas=dp_world,
                    rank=dp_rank,
                    shuffle=True,
                    drop_last=True,
                )
                dataloader = DataLoader(
                    dataset,
                    batch_size=args.batch_size,
                    sampler=_sampler,
                    collate_fn=collate_fn,
                    num_workers=getattr(args, "num_workers", 0),
                )
            else:
                dataloader = DataLoader(
                    dataset,
                    batch_size=args.batch_size,
                    shuffle=True,
                    collate_fn=collate_fn,
                    num_workers=getattr(args, "num_workers", 0),
                )
        else:
            if use_discrete:
                raise ValueError("Discrete 3D token mode requires --data_dir (no dummy_data path).")
            raise ValueError("Provide --dummy_data or --data_dir. Use --dummy_data for a quick test.")

        _dlog(
            f"dataloader ready len={len(dataloader)} batch_size={args.batch_size} "
            f"dataset_len={len(dataset)} sampler={type(getattr(dataloader, 'sampler', None)).__name__}",
            all_ranks=True,
            force=getattr(args, "debug_parallel", False),
        )

        if not use_tp and accelerator is not None:
            model, opt, dataloader = accelerator.prepare(model, opt, dataloader)

        # LR scheduler (cosine with warmup)
        scheduler = None
        if getattr(args, "use_lr_scheduler", False):
            num_training_steps = len(dataloader) * args.epochs
            num_warmup_steps = int(getattr(args, "warmup_ratio", 0.05) * num_training_steps)
            try:
                from transformers import get_cosine_schedule_with_warmup
                scheduler = get_cosine_schedule_with_warmup(
                    opt, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
                )
                if not use_tp and accelerator is not None:
                    scheduler = accelerator.prepare(scheduler)
                if is_main_process:
                    print(f"LR scheduler: cosine, warmup {num_warmup_steps} / {num_training_steps} steps", flush=True)
            except ImportError:
                pass

        debug_memory = getattr(args, "debug_memory", False)
        _debug_all_ranks = os.environ.get("DEBUG_MEMORY_ALL_RANKS", "0") == "1" or (use_tp and debug_memory)
        if _debug_all_ranks and "DEBUG_MEMORY_ALL_RANKS" not in os.environ:
            os.environ["DEBUG_MEMORY_ALL_RANKS"] = "1"  # 让 model 内 forward hooks 也打印所有 rank

        def _print(*a, **kw):
            if is_main_process:
                kw.setdefault("flush", True)
                print(*a, **kw)

        def _print_mem(*a, **kw):
            """DEBUG_MEMORY 专用：默认仅 main；DEBUG_MEMORY_ALL_RANKS=1 时所有 rank 打印（用于死锁排查）。"""
            kw.setdefault("flush", True)
            if _debug_all_ranks or is_main_process:
                print(*a, **kw)

        def _barrier():
            t0 = time.time()
            if getattr(args, "debug_sync_points", False):
                _dlog("barrier enter", all_ranks=True, force=True)
            if use_tp:
                dist.barrier()
            elif accelerator is not None:
                accelerator.wait_for_everyone()
            if getattr(args, "debug_sync_points", False):
                _dlog(f"barrier leave waited={time.time() - t0:.3f}s", all_ranks=True, force=True)

        def _should_log_loss(step_idx: int, valid_tokens: int) -> bool:
            return bool(getattr(args, "debug_loss", False) or step_idx < 5 or valid_tokens == 0)

        def _log_detailed_loss_debug(
            *,
            epoch_idx: int,
            step_idx: int,
            batch_dict: dict,
            outputs_obj,
            loss_tensor: torch.Tensor,
            valid_tokens: int,
            total_tokens: int,
        ) -> None:
            if not _should_log_loss(step_idx, valid_tokens):
                return
            if loss_tensor is None:
                _print_mem(f"[LOSSDBG] {_rank_prefix()} ep={epoch_idx} step={step_idx} loss is None")
                return
            if "labels" not in batch_dict or not isinstance(batch_dict["labels"], torch.Tensor):
                _print_mem(f"[LOSSDBG] {_rank_prefix()} ep={epoch_idx} step={step_idx} labels missing")
                return

            with torch.no_grad():
                labels_now = batch_dict["labels"]
                attn_now = batch_dict.get("attention_mask")
                logits_now = getattr(outputs_obj, "logits", None)
                if logits_now is None and isinstance(outputs_obj, dict):
                    logits_now = outputs_obj.get("logits", None)
                lr_now = opt.param_groups[0]["lr"] if opt.param_groups else 0.0
                loss_val_now = float(loss_tensor.detach().float().item())
                loss_state = "finite"
                if math.isnan(loss_val_now):
                    loss_state = "nan"
                elif math.isinf(loss_val_now):
                    loss_state = "inf"
                labels_valid_mask = labels_now.ne(-100)
                per_sample_valid = labels_valid_mask.sum(dim=1).detach().cpu().tolist()
                attn_tokens = int(attn_now.sum().item()) if isinstance(attn_now, torch.Tensor) else -1
                _print_mem(
                    f"[LOSSDBG] {_rank_prefix()} ep={epoch_idx} step={step_idx} "
                    f"loss={loss_val_now:.8e} ({loss_state}) lr={lr_now:.3e} "
                    f"valid={valid_tokens}/{total_tokens} attn_tokens={attn_tokens} "
                    f"labels_shape={tuple(labels_now.shape)}"
                )
                _print_mem(
                    f"[LOSSDBG] {_rank_prefix()} per_sample_valid={per_sample_valid[:8]} "
                    f"(show<=8) grad_accum={grad_accum_steps}"
                )
                if labels_now.numel() > 0:
                    tail_len = min(12, labels_now.shape[1])
                    tail_labels = labels_now[0, -tail_len:].detach().cpu().tolist()
                    _print_mem(f"[LOSSDBG] {_rank_prefix()} sample0_tail_labels={tail_labels}")

                if not isinstance(logits_now, torch.Tensor):
                    _print_mem(f"[LOSSDBG] {_rank_prefix()} logits missing in outputs, skip CE recompute")
                    return

                logits_det = logits_now.detach().float()
                finite_ratio = float(torch.isfinite(logits_det).float().mean().item())
                logits_absmax = float(logits_det.abs().max().item())
                logits_mean = float(logits_det.mean().item())
                logits_std = float(logits_det.std(unbiased=False).item())
                _print_mem(
                    f"[LOSSDBG] {_rank_prefix()} logits_shape={tuple(logits_now.shape)} dtype={logits_now.dtype} "
                    f"finite_ratio={finite_ratio:.6f} mean={logits_mean:.6e} std={logits_std:.6e} absmax={logits_absmax:.6e}"
                )

                if logits_now.ndim != 3 or labels_now.ndim != 2 or logits_now.shape[:2] != labels_now.shape:
                    # SP 模式下 logits 已在 model.forward 内部被 split_for_sp 切成本 rank 的片段
                    # (形状 [B, S/sp_size, V])，而 batch["labels"] 仍是送入模型前的完整序列
                    # (形状 [B, S])，长度不等属于正常现象，CE 重算交由 SP-LOSSDBG 展示。
                    _sp_ratio = (
                        f"≈1/{round(labels_now.shape[1] / logits_now.shape[1])}"
                        if logits_now.ndim == 3 and labels_now.ndim == 2 and logits_now.shape[1] > 0
                        else "?"
                    )
                    _print_mem(
                        f"[LOSSDBG] {_rank_prefix()} SP 模式：logits 是本 rank 的序列切片 "
                        f"logits_seq={logits_now.shape[1] if logits_now.ndim==3 else '?'} "
                        f"labels_seq={labels_now.shape[1] if labels_now.ndim==2 else '?'} "
                        f"(ratio {_sp_ratio})，CE 重算结果见上方 SP-LOSSDBG 日志"
                    )
                    return

                shift_logits = logits_det[..., :-1, :].contiguous().view(-1, logits_det.size(-1))
                shift_labels = labels_now[..., 1:].contiguous().view(-1).to(shift_logits.device)
                shift_valid = shift_labels.ne(-100)
                local_valid = int(shift_valid.sum().item())
                if local_valid <= 0:
                    _print_mem(
                        f"[LOSSDBG WARN] {_rank_prefix()} shift 后 valid_label_tokens=0 "
                        f"(原始valid={valid_tokens}/{total_tokens})"
                    )
                    return

                ce_sum_local = torch.nn.functional.cross_entropy(
                    shift_logits,
                    shift_labels,
                    ignore_index=-100,
                    reduction="sum",
                )
                ce_mean_local = ce_sum_local / max(local_valid, 1)
                pred = shift_logits.argmax(dim=-1)
                valid_pred = pred[shift_valid]
                valid_labels = shift_labels[shift_valid]
                token_acc = float((valid_pred == valid_labels).float().mean().item())
                ppl = float(torch.exp(torch.clamp(ce_mean_local, max=20.0)).item())
                ce_sum_global = ce_sum_local.detach().clone()
                valid_global_t = torch.tensor(
                    float(local_valid),
                    device=ce_sum_local.device,
                    dtype=ce_sum_local.dtype,
                )
                if use_tp and sp_group is not None and dist.is_initialized():
                    try:
                        if dist.get_world_size(group=sp_group) > 1:
                            dist.all_reduce(ce_sum_global, op=dist.ReduceOp.SUM, group=sp_group)
                            dist.all_reduce(valid_global_t, op=dist.ReduceOp.SUM, group=sp_group)
                    except Exception as e:
                        _print_mem(f"[LOSSDBG WARN] {_rank_prefix()} all_reduce debug stats failed: {e}")
                global_valid = max(float(valid_global_t.item()), 1.0)
                ce_mean_global = float((ce_sum_global / global_valid).item())
                _print_mem(
                    f"[LOSSDBG] {_rank_prefix()} shift_valid_local={local_valid} "
                    f"ce_local={float(ce_mean_local.item()):.8e} ce_global={ce_mean_global:.8e} "
                    f"token_acc={token_acc:.4f} ppl={ppl:.4f}"
                )
                loss_abs_diff = abs(loss_val_now - ce_mean_global)
                _print_mem(
                    f"[LOSSDBG] {_rank_prefix()} reported_loss={loss_val_now:.8e} "
                    f"vs_recomputed_global_ce={ce_mean_global:.8e} diff={loss_abs_diff:.8e}"
                )
                if loss_val_now < 1e-8 and ce_mean_global > 1e-4:
                    _print_mem(
                        f"[LOSSDBG WARN] {_rank_prefix()} loss 接近 0 但重算 CE={ce_mean_global:.6e}，"
                        "请重点检查 SP loss 的 local/global 聚合。"
                    )

        grad_accum_steps = getattr(args, "gradient_accumulation_steps", 1)
        if debug_memory and is_main_process:
            _print("[DEBUG_MEMORY] 显存调试已开启。")
            _print("  - 每步: @batch_to_device | @before_forward | @after_forward | @before_backward | @after_backward | @after_opt_step")
            _print("  - discrete: @before_vl_model | vl_L0/N..vl_LN-1/N | @after_vl_model")
            if use_tp:
                _print("  - TP 模式下已自动开启 all_ranks，各 rank 会分别打印 checkpoint")
            _print(
                "  - 缓解: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True | 降低 max_length | 增大 grad_accum | gradient_checkpointing"
            )
        model.train()
        # FSDP2 does not add a .module wrapper; DDP (Accelerate) does.
        # Use getattr fallback so both paths work.
        _model = getattr(model, "module", model)
        if debug_memory:
            setattr(_model, "_debug_memory_fine", True)
        global_step = 0
        # 训练指标记录（仅 main process），用于可视化
        metrics_path = os.path.join(args.output_dir, "training_metrics.jsonl")
        metrics_file = open(metrics_path, "w", encoding="utf-8") if is_main_process else None
        grad_clip = getattr(args, "grad_clip", 0.0)
        # TP+FSDP: clip_grad_norm_ 会触发 "DTensor does not support cross-mesh operation"
        # (aten._foreach_norm)，已知限制。强制禁用 grad_clip 以避免崩溃。
        if use_tp and grad_clip > 0:
            if is_main_process:
                print(
                    "[Warn] TP+FSDP 下 clip_grad_norm_ 与 DTensor 不兼容，已禁用梯度裁剪 (grad_clip→0)。"
                    " 若需裁剪请使用非 TP 模式。",
                    flush=True,
                )
            grad_clip = 0.0
        _accum_loss = 0.0  # accumulates loss for logging averaged over accum window
        _accum_count = 0
        opt.zero_grad()  # zero once before the accumulation loop

        for epoch in range(args.epochs):
            _epoch_t0 = time.time()
            # Update DistributedSampler epoch so shuffle is different each epoch
            if use_tp and hasattr(dataloader, "sampler") and hasattr(dataloader.sampler, "set_epoch"):
                dataloader.sampler.set_epoch(epoch)

            for step, batch in enumerate(dataloader):
                _step_t0 = time.time()
                if getattr(args, "debug_parallel", False) and (step < 3 or _debug_parallel_verbose_enabled()):
                    _dlog(f"step_begin epoch={epoch} step={step}", all_ranks=True)
                # 每步开始重置 peak，便于定位本步内显存峰值
                if debug_memory and torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        torch.cuda.reset_peak_memory_stats(i)
                batch = {k: _to_device(v, device) for k, v in batch.items()}
                _t_data = time.time()
                use_inputs_3d = "inputs_3d" in batch and batch["inputs_3d"] is not None
                labels = batch.get("labels", None)
                valid_label_tokens = -1
                total_label_tokens = -1
                if isinstance(labels, torch.Tensor):
                    valid_label_tokens = int((labels != -100).sum().item())
                    total_label_tokens = int(labels.numel())
                    if _should_log_loss(step, valid_label_tokens):
                        lr_now = opt.param_groups[0]["lr"] if opt.param_groups else 0.0
                        _print_mem(
                            f"[LOSSCHK] {_rank_prefix()} step={step} valid_label_tokens="
                            f"{valid_label_tokens}/{total_label_tokens} lr={lr_now:.3e}"
                        )
                    if valid_label_tokens == 0:
                        # All labels masked out: CE denominator is zero and loss may become NaN.
                        # Skip this batch to keep optimizer states healthy and print actionable hints.
                        _print_mem(
                            f"[LOSSCHK WARN] {_rank_prefix()} step={step} all labels are -100; skip batch. "
                            "请检查 tokenizer 截断是否把 assistant 回复截没了（常见于变长 3D + max_length 不足）。"
                        )
                        continue

                # [DEBUG_MEMORY] 步骤开始：batch 已加载到 GPU
                if debug_memory and torch.cuda.is_available():
                    seq_len = batch["input_ids"].shape[1] if "input_ids" in batch else 0
                    batch_info = (
                        f"bs={batch['input_ids'].shape[0]} seq={seq_len} "
                        f"3d={'inputs_3d' if use_inputs_3d else 'discrete'}"
                    )
                    if use_inputs_3d and "inputs_3d" in batch and batch["inputs_3d"] is not None:
                        inp = batch["inputs_3d"]
                        if isinstance(inp, dict) and "sparse_sdf" in inp:
                            batch_info += f" pts={inp['sparse_sdf'].shape[0]}"
                    _print_mem(f"\n[MEM] {_rank_prefix()} step={step} @batch_to_device | {batch_info} | {_gpu_memory_stats_all(include_peak=True)}")

                # [DEBUG_MEMORY] 即将进入 forward
                if debug_memory and torch.cuda.is_available():
                    _print_mem(f"[MEM] {_rank_prefix()} step={step} @before_forward | mode={'discrete' if use_discrete else '3d'} | {_gpu_memory_stats_all(include_peak=True)}")

                # BF16 autocast:
                #   TP path: explicit torch.autocast required (no Accelerate wrapper)
                #   Accelerate path: Accelerate applies mixed_precision="bf16" internally,
                #                    but an extra autocast doesn't hurt correctness
                #   CPU path: no-op (nullcontext)
                if torch.cuda.is_available():
                    _ctx = torch.autocast("cuda", dtype=torch.bfloat16)
                else:
                    _ctx = contextlib.nullcontext()
                try:
                    with _ctx:
                        if use_discrete or not use_inputs_3d:
                            outputs = _model(
                                input_ids=batch["input_ids"],
                                attention_mask=batch.get("attention_mask"),
                                labels=batch["labels"],
                                use_cache=False,
                            )
                        elif use_inputs_3d:
                            outputs = _model.forward_with_3d(
                                input_ids=batch["input_ids"],
                                attention_mask=batch["attention_mask"],
                                labels=batch["labels"],
                                inputs_3d=batch["inputs_3d"],
                                use_cache=False,
                            )
                        else:
                            outputs = _model.forward_with_3d(
                                input_ids=batch["input_ids"],
                                attention_mask=batch["attention_mask"],
                                labels=batch["labels"],
                                feats_3d=batch["feats_3d"],
                                coords_3d=batch["coords_3d"],
                                use_cache=False,
                            )
                except Exception as e:
                    import traceback
                    _print_mem(f"\n[MEM ERROR] {_rank_prefix()} step={step} forward 异常: {type(e).__name__}: {e}")
                    _print_mem("[MEM ERROR] 最后成功的 checkpoint 即为异常位置，traceback:")
                    traceback.print_exc()
                    if "out of memory" in str(e).lower():
                        _print_mem(f"[MEM OOM] 显存: {_gpu_memory_stats_all(include_peak=True)}")
                    raise

                _t_fwd = time.time()
                # [DEBUG_MEMORY] Forward 完成
                loss = getattr(outputs, "loss", None) or outputs.get("loss")
                loss_val = loss.detach().float().item() if loss is not None else float("nan")
                if debug_memory and torch.cuda.is_available():
                    _print_mem(f"[MEM] {_rank_prefix()} step={step} @after_forward | loss={loss_val:.4f} | {_gpu_memory_stats_all(include_peak=True)}")
                if loss is not None:
                    _nan = torch.isnan(loss)
                    _inf = torch.isinf(loss)
                    if _nan.any().item() or _inf.any().item():
                        lr_now = opt.param_groups[0]["lr"] if opt.param_groups else 0.0
                        seq_now = int(batch["input_ids"].shape[1]) if "input_ids" in batch else -1
                        _print_mem(
                            f"[MEM WARN] {_rank_prefix()} step={step} loss=NaN/Inf ({loss_val}) "
                            f"| valid_label_tokens={valid_label_tokens}/{total_label_tokens} "
                            f"| seq={seq_now} | lr={lr_now:.3e}"
                        )
                        _print_mem(
                            "[MEM WARN] 建议优先检查：1) labels 是否全 -100；2) max_length_variable 是否导致 assistant 被截断；"
                            "3) 学习率是否过高（可先降到 2e-5 验证）。"
                        )
                        opt.zero_grad()
                        continue
                _log_detailed_loss_debug(
                    epoch_idx=epoch,
                    step_idx=step,
                    batch_dict=batch,
                    outputs_obj=outputs,
                    loss_tensor=loss,
                    valid_tokens=valid_label_tokens,
                    total_tokens=total_label_tokens,
                )
                # Scale loss for gradient accumulation
                scaled_loss = loss / grad_accum_steps

                # [DEBUG_MEMORY] 即将 backward（此处最易 OOM：需分配梯度与激活重计算）
                if debug_memory and torch.cuda.is_available():
                    _print_mem(f"[MEM] {_rank_prefix()} step={step} @before_backward | {_gpu_memory_stats_all(include_peak=True)}")
                if use_tp:
                    scaled_loss.backward()
                elif accelerator is not None:
                    accelerator.backward(scaled_loss)
                else:
                    scaled_loss.backward()
                if getattr(args, "debug_parallel", False) and (step < 3 or _debug_parallel_verbose_enabled()):
                    _dlog(
                        f"backward_done epoch={epoch} step={step} "
                        f"scaled_loss={scaled_loss.detach().float().item():.6f}",
                        all_ranks=True,
                    )

                _t_bwd = time.time()
                # [DEBUG_MEMORY] Backward 完成
                if debug_memory and torch.cuda.is_available():
                    _print_mem(f"[MEM] {_rank_prefix()} step={step} @after_backward | {_gpu_memory_stats_all(include_peak=True)}")
                _accum_loss += loss.detach().float().item()
                _accum_count += 1

                is_last_accum = (_accum_count % grad_accum_steps == 0) or (step == len(dataloader) - 1)
                if is_last_accum:
                    if grad_clip > 0:
                        if use_tp:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                        elif accelerator is not None:
                            accelerator.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                        else:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                    opt.step()
                    if getattr(args, "debug_parallel", False) and (step < 3 or _debug_parallel_verbose_enabled()):
                        _dlog(
                            f"optimizer_step_done epoch={epoch} step={step} global_step(next)={global_step + 1}",
                            all_ranks=True,
                        )
                    if scheduler is not None:
                        scheduler.step()
                    opt.zero_grad()
                    global_step += 1
                    _t_opt = time.time()
                    # [DEBUG_MEMORY] Optimizer step + zero_grad 完成（梯度已释放）
                    if debug_memory and torch.cuda.is_available():
                        _print_mem(f"[MEM] {_rank_prefix()} step={step} @after_opt_step gs={global_step} | {_gpu_memory_stats_all(include_peak=True)}")

                    avg_loss = _accum_loss / _accum_count
                    _accum_loss = 0.0
                    _accum_count = 0

                    if metrics_file is not None:
                        lr = opt.param_groups[0]["lr"] if opt.param_groups else 0.0
                        metrics_file.write(
                            json.dumps({
                                "step": global_step, "epoch": epoch,
                                "loss": round(avg_loss, 6), "lr": lr,
                            }) + "\n"
                        )
                        metrics_file.flush()
                    if global_step % 10 == 0:
                        _print(f"Epoch {epoch} step {step} global_step {global_step} loss {avg_loss:.4f}")
                    if step < 5 or global_step % 50 == 0:
                        _print(
                            f"[DEBUG step] ep={epoch} step={step} gs={global_step} "
                            f"data={_t_data-_step_t0:.2f}s fwd={_t_fwd-_t_data:.2f}s "
                            f"bwd={_t_bwd-_t_fwd:.2f}s opt={_t_opt-_t_bwd:.2f}s "
                            f"total={_t_opt-_step_t0:.2f}s"
                        )

            # --- End-of-epoch checkpoint ---
            _dlog(f"epoch_end enter epoch={epoch}", all_ranks=True, force=getattr(args, "debug_sync_points", False))
            _barrier()
            ckpt_path = os.path.join(args.output_dir, f"projector_epoch{epoch}.pt")
            if use_tp:
                from vae_qwen3vl.tensor_parallel_utils import save_projector_tp, save_lora_tp, save_warmup_embed_lmhead_tp
                if _model.projector is not None:
                    save_projector_tp(model, ckpt_path, is_main=is_main_process)
                if use_discrete and is_main_process:
                    tok_path = os.path.join(args.output_dir, f"tokenizer_epoch{epoch}")
                    tokenizer.save_pretrained(tok_path)
                    _print(f"Saved tokenizer to {tok_path}")
                if use_discrete and training_stage == "warmup":
                    warmup_embed_path = os.path.join(args.output_dir, f"warmup_embed_lmhead_epoch{epoch}.pt")
                    save_warmup_embed_lmhead_tp(model, warmup_embed_path, is_main=is_main_process)
                if use_lora_applied:
                    lora_dir = os.path.join(args.output_dir, f"lora_epoch{epoch}")
                    save_lora_tp(model, lora_dir, is_main=is_main_process)
                    _print(f"Saved LoRA to {lora_dir}")
            else:
                raw_model = accelerator.unwrap_model(model) if accelerator is not None else model
                if is_main_process:
                    if raw_model.projector is not None:
                        torch.save(raw_model.projector.state_dict(), ckpt_path)
                        _print(f"Saved {ckpt_path}")
                    if use_discrete:
                        tok_path = os.path.join(args.output_dir, f"tokenizer_epoch{epoch}")
                        tokenizer.save_pretrained(tok_path)
                        _print(f"Saved tokenizer to {tok_path}")
                    if use_discrete and training_stage == "warmup":
                        vl_sd = raw_model.vl_model.state_dict()
                        embed_lmhead_sd = {
                            "vl_model." + k: v.cpu().clone()
                            for k, v in vl_sd.items()
                            if "embed_tokens" in k or "embedding" in k or "lm_head" in k
                        }
                        if embed_lmhead_sd:
                            warmup_embed_path = os.path.join(args.output_dir, f"warmup_embed_lmhead_epoch{epoch}.pt")
                            torch.save(embed_lmhead_sd, warmup_embed_path)
                            _print(f"Saved warmup embed+lm_head ({len(embed_lmhead_sd)} keys) → {warmup_embed_path}")
                    if use_lora_applied:
                        lora_dir = os.path.join(args.output_dir, f"lora_epoch{epoch}")
                        raw_model.vl_model.save_pretrained(lora_dir)
                        _print(f"Saved LoRA to {lora_dir}")
            _barrier()
            _dlog(f"epoch_end leave epoch={epoch}", all_ranks=True, force=getattr(args, "debug_sync_points", False))

        # --- Final checkpoint ---
        _dlog("final_checkpoint enter", all_ranks=True, force=getattr(args, "debug_sync_points", False))
        _barrier()
        final_proj_path = os.path.join(args.output_dir, "projector_final.pt")
        if use_tp:
            from vae_qwen3vl.tensor_parallel_utils import save_projector_tp, save_lora_tp, save_warmup_embed_lmhead_tp
            if _model.projector is not None:
                save_projector_tp(model, final_proj_path, is_main=is_main_process)
            if use_discrete and is_main_process:
                tokenizer.save_pretrained(os.path.join(args.output_dir, "tokenizer_final"))
            if use_discrete and training_stage == "warmup":
                save_warmup_embed_lmhead_tp(model, os.path.join(args.output_dir, "warmup_embed_lmhead_final.pt"), is_main=is_main_process)
            if use_lora_applied:
                save_lora_tp(model, os.path.join(args.output_dir, "lora_final"), is_main=is_main_process)
                _print("Saved LoRA final.")
        else:
            raw_model = accelerator.unwrap_model(model) if accelerator is not None else model
            if is_main_process:
                if raw_model.projector is not None:
                    torch.save(raw_model.projector.state_dict(), final_proj_path)
                if use_discrete:
                    tokenizer.save_pretrained(os.path.join(args.output_dir, "tokenizer_final"))
                if use_discrete and training_stage == "warmup":
                    vl_sd = raw_model.vl_model.state_dict()
                    embed_lmhead_sd = {
                        "vl_model." + k: v.cpu().clone()
                        for k, v in vl_sd.items()
                        if "embed_tokens" in k or "embedding" in k or "lm_head" in k
                    }
                    if embed_lmhead_sd:
                        torch.save(embed_lmhead_sd, os.path.join(args.output_dir, "warmup_embed_lmhead_final.pt"))
                        _print("Saved warmup embed+lm_head → warmup_embed_lmhead_final.pt")
                if use_lora_applied:
                    lora_final_dir = os.path.join(args.output_dir, "lora_final")
                    raw_model.vl_model.save_pretrained(lora_final_dir)
                    _print(f"Saved LoRA final to {lora_final_dir}")
        _dlog("final_checkpoint leave", all_ranks=True, force=getattr(args, "debug_sync_points", False))
        if is_main_process:
            with open(os.path.join(args.output_dir, "train_args.json"), "w") as f:
                json.dump(vars(args), f, indent=2)
            _print("Done.")
    finally:
        if metrics_file is not None:
            try:
                metrics_file.close()
            except Exception:
                pass
        sys.stdout = _orig_stdout
        _log_tee.close()


if __name__ == "__main__":
    import traceback
    import sys
    import socket as _err_socket

    def _error_rank_label():
        """用 RANK（全局）命名文件，避免多机 LOCAL_RANK 冲突。"""
        global_rank = os.environ.get("RANK", None)
        local_rank = os.environ.get("LOCAL_RANK", "0")
        if global_rank is not None:
            return f"grank{global_rank}"
        return f"rank{local_rank}"

    def _save_error(exc_type, exc_val, exc_tb):
        if exc_type is not None:
            tb = "".join(traceback.format_exception(exc_type, exc_val, exc_tb))
            label = _error_rank_label()
            hostname = _err_socket.gethostname()
            header = (
                f"hostname={hostname}  RANK={os.environ.get('RANK', '?')}  "
                f"LOCAL_RANK={os.environ.get('LOCAL_RANK', '?')}  "
                f"WORLD_SIZE={os.environ.get('WORLD_SIZE', '?')}  "
                f"LOCAL_WORLD_SIZE={os.environ.get('LOCAL_WORLD_SIZE', '?')}  "
                f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(not set)')}\n\n"
            )
            err_path = os.path.join(PROJECT_ROOT, "configs", f"train_error_{label}.txt")
            os.makedirs(os.path.dirname(err_path), exist_ok=True)
            with open(err_path, "w", encoding="utf-8") as f:
                f.write(header + tb)
            # 同时追加到 Launcher 指定的 --log-file，确保 align_debug.log 中有完整 traceback
            align_log = os.environ.get("ALIGN_DEBUG_LOG")
            if align_log:
                try:
                    with open(align_log, "a", encoding="utf-8") as f:
                        f.write("\n" + "=" * 60 + "\n")
                        f.write(f"[train_finetune] 异常已写入 {err_path}\n")
                        f.write("=" * 60 + "\n" + header + tb)
                except Exception:
                    pass
    sys.excepthook = _save_error
    try:
        main()
    except BaseException as e:
        _save_error(type(e), e, e.__traceback__)
        label = _error_rank_label()
        err_path = os.path.join(PROJECT_ROOT, "configs", f"train_error_{label}.txt")
        try:
            print(f"[{label}] Error saved to {err_path}", file=sys.stderr)
        except Exception:
            pass
        raise
