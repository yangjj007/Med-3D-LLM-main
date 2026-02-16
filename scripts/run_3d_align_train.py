#!/usr/bin/env python3
"""
3D-VL 对齐训练启动脚本：从统一配置加载，启动 accelerate 多卡训练。
用法：python scripts/run_3d_align_train.py
      或：python scripts/run_3d_align_train.py --config configs/3d_align_train.yaml
"""

import os
import sys
import subprocess
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def main():
    parser = argparse.ArgumentParser(description="Launch 3D-VL alignment training")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(PROJECT_ROOT, "configs", "3d_align_train.yaml"),
        help="Path to unified YAML config",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="单卡调试模式（num_processes=1），便于查看完整报错",
    )
    args = parser.parse_args()
    config_path = os.path.normpath(os.path.join(PROJECT_ROOT, args.config) if not os.path.isabs(args.config) else args.config)
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

    # Write accelerate config
    acc_cfg = cfg.get("accelerate", {})
    if args.debug:
        acc_cfg = {**acc_cfg, "num_processes": 1}
        print("Debug 模式: 单卡运行 (num_processes=1)")
    acc_path = os.path.join(PROJECT_ROOT, "configs", "accelerate.yaml")
    os.makedirs(os.path.dirname(acc_path), exist_ok=True)
    with open(acc_path, "w", encoding="utf-8") as f:
        yaml.dump(acc_cfg, f, default_flow_style=False, allow_unicode=True)

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
    cmd = [
        "accelerate", "launch",
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
