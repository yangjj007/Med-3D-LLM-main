#!/bin/bash
# 3D-VL 评估：使用训练配置与真实数据，加载 LoRA 进行推理测试
# 用法: bash scripts/run_3d_align_eval.sh [config路径]

set -e
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

CONFIG="${1:-${PROJECT_ROOT}/configs/3d_align_train.yaml}"
export SPARSE_BACKEND=spconv

echo "=== 3D-VL 评估（LoRA + 真实数据）==="
echo "Config: $CONFIG"
echo ""

python vae_qwen3vl/eval_3d_vl.py --config "$CONFIG"
