#!/usr/bin/env python3
"""
3D-VL 对齐训练启动脚本。

支持两种并行模式：
  1. TP 模式（默认，tensor_parallel_size > 1）：
       torchrun --nproc_per_node=<N> vae_qwen3vl/train_finetune.py ...
     DTensor 张量并行 + FSDP2 数据并行，无需 Accelerate/DeepSpeed。
     多机时自动检测 GPU 数并设置 --nnodes / --node_rank / --master_addr。

  2. Legacy Accelerate 模式（tensor_parallel_size=1 时自动启用）：
       accelerate launch --config_file configs/accelerate.yaml ...
     兼容旧版 DeepSpeed ZeRO-2 配置。

用法：
  python scripts/run_3d_align_train.py
  python scripts/run_3d_align_train.py --config configs/3d_align_train_variable_length.yaml
  python scripts/run_3d_align_train.py --debug   # 单卡调试
"""

import os
import sys
import subprocess
import argparse
import socket

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

os.environ["SPARSE_BACKEND"] = "torchsparse"


def _detect_gpus_per_node() -> int:
    """通过 nvidia-smi 检测当前节点的 GPU 数量，失败则 fallback 到 torch。"""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0:
            count = len([l for l in r.stdout.strip().splitlines() if l.strip()])
            if count > 0:
                return count
    except Exception:
        pass
    try:
        r = subprocess.run(
            [sys.executable, "-c", "import torch; print(torch.cuda.device_count())"],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode == 0:
            return int(r.stdout.strip())
    except Exception:
        pass
    return 0


def _detect_node_rank() -> int:
    """
    从环境变量或 hostname 推断当前节点的 rank。
    优先级：NODE_RANK > GROUP_RANK > hostname 解析 > 0。
    PyTorchJob hostname 格式：{name}-master-0 → rank 0，{name}-worker-{i} → rank i+1。
    """
    for var in ("NODE_RANK", "GROUP_RANK"):
        val = os.environ.get(var)
        if val is not None:
            return int(val)
    hostname = socket.gethostname()
    if "-master-" in hostname:
        return 0
    if "-worker-" in hostname:
        try:
            return int(hostname.rsplit("-", 1)[-1]) + 1
        except ValueError:
            pass
    return 0


def main():
    parser = argparse.ArgumentParser(description="Launch 3D-VL alignment training")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(PROJECT_ROOT, "configs", "3d_align_train_variable_length.yaml"),
        help="Path to unified YAML config",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="单卡调试模式（nproc=1），便于查看完整报错",
    )
    args = parser.parse_args()
    config_path = os.path.normpath(
        os.path.join(PROJECT_ROOT, args.config) if not os.path.isabs(args.config) else args.config
    )
    if not os.path.isfile(config_path):
        print(f"Config not found: {config_path}")
        sys.exit(1)

    try:
        import yaml
    except ImportError:
        print("Please install PyYAML: pip install pyyaml")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not cfg:
        print("Config is empty")
        sys.exit(1)

    # Resolve paths
    def abspath(p):
        if not p or os.path.isabs(p):
            return p
        return os.path.normpath(os.path.join(PROJECT_ROOT, p))

    vae_ckpt = abspath(os.environ.get("VAE_CKPT") or cfg.get("vae_ckpt", ""))
    if not vae_ckpt or not os.path.isfile(vae_ckpt):
        print(f"VAE checkpoint not found: {vae_ckpt}")
        print("Set vae_ckpt in config or export VAE_CKPT")
        sys.exit(1)

    train_script = os.path.join(PROJECT_ROOT, "vae_qwen3vl", "train_finetune.py")
    env = os.environ.copy()
    error_file = os.path.join(PROJECT_ROOT, "configs", "elastic_error.txt")
    env.setdefault("TORCHELASTIC_ERROR_FILE", error_file)

    # Decide launch mode based on tensor_parallel_size
    tp_size = int(cfg.get("tensor_parallel_size", 1))
    acc_cfg = dict(cfg.get("accelerate", {}))
    total_processes = 1 if args.debug else int(acc_cfg.get("num_processes", 4))

    if tp_size > 1:
        # ---- TP mode: torchrun (no Accelerate/DeepSpeed) ----
        # 多机支持：自动检测每节点 GPU 数，推算 nnodes 和 node_rank
        gpus_per_node = int(cfg.get("gpus_per_node", 0))
        if gpus_per_node <= 0:
            gpus_per_node = _detect_gpus_per_node()
        if gpus_per_node <= 0:
            print("[Launcher] WARNING: 无法检测 GPU 数量，fallback 到 num_processes")
            gpus_per_node = total_processes

        if args.debug:
            gpus_per_node = 1

        nnodes = int(cfg.get("num_nodes", 0))
        if nnodes <= 0:
            nnodes = max(1, total_processes // gpus_per_node)

        nproc_per_node = gpus_per_node
        node_rank = _detect_node_rank()

        master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
        master_port = os.environ.get("MASTER_PORT", "29500")
        # PyTorchJob 有时用 master 的 service name 做 MASTER_ADDR，
        # 也可以在 config 里手动指定。
        if cfg.get("master_addr"):
            master_addr = cfg["master_addr"]
        if cfg.get("master_port"):
            master_port = str(cfg["master_port"])

        hostname = socket.gethostname()
        print(
            f"[Launcher] TP mode on {hostname}:\n"
            f"  tensor_parallel_size={tp_size}, total_processes={total_processes}\n"
            f"  gpus_per_node={gpus_per_node}, nnodes={nnodes}, node_rank={node_rank}\n"
            f"  master_addr={master_addr}, master_port={master_port}\n"
            f"  dp_size={total_processes // tp_size}"
        )

        cmd = [
            "torchrun",
            f"--nproc_per_node={nproc_per_node}",
            f"--nnodes={nnodes}",
            f"--node_rank={node_rank}",
            f"--master_addr={master_addr}",
            f"--master_port={master_port}",
            train_script,
            "--config", config_path,
        ]
        if os.environ.get("VAE_CKPT"):
            cmd.extend(["--vae_ckpt", vae_ckpt])
        print("Running:", " ".join(cmd))
        print("(若失败，报错会写入 configs/train_error_grank*.txt 和 configs/train_diag_rank*.txt)")
        subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)

    else:
        # Legacy Accelerate mode (tp_size=1): write accelerate.yaml and launch
        if args.debug:
            acc_cfg["num_processes"] = 1
            print("Debug 模式: 单卡运行 (num_processes=1)")
        if cfg.get("use_deepspeed", False):
            acc_cfg["distributed_type"] = "DEEPSPEED"
            acc_cfg.setdefault("fsdp_config", {})
            grad_accum = cfg.get("gradient_accumulation_steps", 1)
            grad_clip = cfg.get("grad_clip", 0.0) or 1.0
            acc_cfg["deepspeed_config"] = {
                "gradient_accumulation_steps": grad_accum,
                "gradient_clipping": grad_clip,
                "offload_optimizer_device": "none",
                "offload_param_device": "none",
                "zero_stage": 2,
            }
            print(
                f"[Launcher] Accelerate + DeepSpeed ZeRO-2 "
                f"(grad_accum={grad_accum}, grad_clip={grad_clip})"
            )
        else:
            print(f"[Launcher] Accelerate MULTI_GPU (num_processes={acc_cfg.get('num_processes', 4)})")

        acc_path = os.path.join(PROJECT_ROOT, "configs", "accelerate.yaml")
        os.makedirs(os.path.dirname(acc_path), exist_ok=True)
        with open(acc_path, "w", encoding="utf-8") as f:
            yaml.dump(acc_cfg, f, default_flow_style=False, allow_unicode=True)

        cmd = [
            "accelerate", "launch",
            "--config_file", acc_path,
            train_script,
            "--config", config_path,
        ]
        if os.environ.get("VAE_CKPT"):
            cmd.extend(["--vae_ckpt", vae_ckpt])
        print("Running:", " ".join(cmd))
        print("(若失败，报错会写入 configs/train_error_grank*.txt 和 configs/train_diag_rank*.txt)")
        subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
