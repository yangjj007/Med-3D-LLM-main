#!/bin/bash
# 3D-VL 对齐训练：从统一配置 configs/3d_align_train.yaml 启动
# 修改 configs/3d_align_train.yaml 即可调整训练参数与分布式设置

set -e
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

CONFIG="${1:-${PROJECT_ROOT}/configs/3d_align_train.yaml}"
export SPARSE_BACKEND=spconv

echo "=== 3D-VL 对齐训练 ==="
echo "Config: $CONFIG"
echo ""

python scripts/run_3d_align_train.py --config "$CONFIG"
