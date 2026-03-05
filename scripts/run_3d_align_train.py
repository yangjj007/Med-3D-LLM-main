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


def _find_env_python(env_name: str) -> str | None:
    """不依赖 conda activate/run，直接定位指定 conda 环境的 Python 路径。

    优先用 `conda info --base` 获取 conda 根目录；失败时遍历常见安装路径。
    """
    # 方法1：conda info --base（轻量，不激活环境，通常 <1s）
    try:
        r = subprocess.run(
            ["conda", "info", "--base"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0:
            candidate = os.path.join(r.stdout.strip(), "envs", env_name, "bin", "python")
            if os.path.isfile(candidate):
                return candidate
    except Exception:
        pass

    # 方法2：遍历常见 conda 安装根目录
    home = os.path.expanduser("~")
    for base in [
        f"{home}/anaconda3", f"{home}/miniconda3", f"{home}/miniforge3",
        "/opt/conda", "/usr/local/anaconda3", "/usr/local/miniconda3",
    ]:
        candidate = os.path.join(base, "envs", env_name, "bin", "python")
        if os.path.isfile(candidate):
            return candidate

    return None


def _setup_conda_env_vars(env_name: str, python_path: str) -> None:
    """设置 conda 环境变量（CONDA_DEFAULT_ENV / CONDA_PREFIX / PATH），
    使后续 subprocess 能找到该环境下的 torchrun、accelerate 等命令。"""
    env_prefix = os.path.dirname(os.path.dirname(python_path))  # .../envs/trellis
    os.environ["CONDA_DEFAULT_ENV"] = env_name
    os.environ["CONDA_PREFIX"] = env_prefix
    env_bin = os.path.join(env_prefix, "bin")
    path = os.environ.get("PATH", "")
    if env_bin not in path.split(os.pathsep):
        os.environ["PATH"] = env_bin + os.pathsep + path


def _ensure_conda_env(env_name: str = "trellis") -> None:
    """若当前 Python 不在指定 conda 环境，用 os.execv 原地切换后重启。"""
    if os.environ.get("CONDA_DEFAULT_ENV") == env_name:
        return

    python_path = _find_env_python(env_name)
    if python_path is None:
        print(f"[Launcher] 警告: 找不到 {env_name} 环境的 Python，继续使用当前 Python")
        return

    if os.path.realpath(sys.executable) == os.path.realpath(python_path):
        print(f"[Launcher] 已在 {env_name} 环境中 ({python_path})，设置环境变量")
        _setup_conda_env_vars(env_name, python_path)
        return

    print(f"[Launcher] 切换至 {env_name} 环境: {python_path}", flush=True)
    _setup_conda_env_vars(env_name, python_path)
    os.execv(python_path, [python_path] + sys.argv)


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
