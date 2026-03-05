#!/usr/bin/env python3
"""
3D-VL 对齐训练启动脚本。

支持两种并行模式：
  1. TP 模式（默认，tensor_parallel_size > 1）：
       torchrun --nproc_per_node=<N> vae_qwen3vl/train_finetune.py ...
     DTensor 张量并行 (TP=2) + FSDP2 数据并行 (DP=2)，无需 Accelerate/DeepSpeed。

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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def _ensure_conda_env(env_name: str = "trellis") -> None:
    """若当前 Python 不在指定 conda 环境，自动用该环境的 Python 原地重启脚本。

    原理：os.execv 直接替换当前进程（不产生子进程），argv 完整保留，
    重启后 CONDA_DEFAULT_ENV == env_name，下次调用时直接返回。
    """
    if os.environ.get("CONDA_DEFAULT_ENV") == env_name:
        return

    try:
        result = subprocess.run(
            ["conda", "run", "-n", env_name, "python", "-c",
             "import sys; print(sys.executable)"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip())
        trellis_python = result.stdout.strip()
        if not os.path.isfile(trellis_python):
            raise FileNotFoundError(f"找不到可执行文件: {trellis_python}")
        print(f"[Launcher] 当前环境={os.environ.get('CONDA_DEFAULT_ENV', '未知')}, "
              f"切换至 {env_name}: {trellis_python}")
        os.execv(trellis_python, [trellis_python] + sys.argv)
    except Exception as exc:
        print(f"[Launcher] 警告: 无法自动切换至 {env_name} 环境 ({exc})，继续使用当前 Python")


_ensure_conda_env("trellis")

os.environ["SPARSE_BACKEND"] = "torchsparse"


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
    error_file = os.path.join(PROJECT_ROOT, "configs", ".elastic_error.txt")
    env.setdefault("TORCHELASTIC_ERROR_FILE", error_file)

    # 使用当前 Python 解释器路径，确保子进程与 launcher 处于同一 conda 环境
    # 等效于 "python -m torch.distributed.run" / "python -m accelerate.commands.launch"
    PYTHON = sys.executable

    # Decide launch mode based on tensor_parallel_size
    tp_size = int(cfg.get("tensor_parallel_size", 1))
    acc_cfg = dict(cfg.get("accelerate", {}))
    num_gpus = 1 if args.debug else int(acc_cfg.get("num_processes", 4))

    if tp_size > 1:
        # TP mode: use torchrun (no Accelerate/DeepSpeed needed)
        print(
            f"[Launcher] TP mode: tensor_parallel_size={tp_size}, "
            f"nproc_per_node={num_gpus}, dp_size={num_gpus // tp_size}"
        )
        print(f"[Launcher] Python: {PYTHON}")
        cmd = [
            PYTHON, "-m", "torch.distributed.run",
            f"--nproc_per_node={num_gpus}",
            "--master_port=29500",
            train_script,
            "--config", config_path,
        ]
        if os.environ.get("VAE_CKPT"):
            cmd.extend(["--vae_ckpt", vae_ckpt])
        print("Running:", " ".join(cmd))
        print("(若失败，报错会写入 configs/.train_error_rank*.txt)")
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

        print(f"[Launcher] Python: {PYTHON}")
        cmd = [
            PYTHON, "-m", "accelerate.commands.launch",
            "--config_file", acc_path,
            train_script,
            "--config", config_path,
        ]
        if os.environ.get("VAE_CKPT"):
            cmd.extend(["--vae_ckpt", vae_ckpt])
        print("Running:", " ".join(cmd))
        print("(若失败，报错会写入 configs/.train_error_rank*.txt)")
        subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
