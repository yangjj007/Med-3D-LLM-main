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

    # 激活 conda 环境：用 conda run 包裹后续命令，等效于 conda activate trellis
    CONDA_ENV = "trellis"
    def wrap_conda(cmd_list):
        return ["conda", "run", "--no-capture-output", "-n", CONDA_ENV] + cmd_list

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
        cmd = [
            "torchrun",
            f"--nproc_per_node={num_gpus}",
            "--master_port=29500",
            train_script,
            "--config", config_path,
        ]
        if os.environ.get("VAE_CKPT"):
            cmd.extend(["--vae_ckpt", vae_ckpt])
        final_cmd = wrap_conda(cmd)
        print("Running:", " ".join(final_cmd))
        print("(若失败，报错会写入 configs/.train_error_rank*.txt)")
        subprocess.run(final_cmd, cwd=PROJECT_ROOT, env=env, check=True)

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
        final_cmd = wrap_conda(cmd)
        print("Running:", " ".join(final_cmd))
        print("(若失败，报错会写入 configs/.train_error_rank*.txt)")
        subprocess.run(final_cmd, cwd=PROJECT_ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
