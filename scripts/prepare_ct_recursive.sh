#!/bin/bash
# CT数据递归预处理脚本
# 自动扫描大文件夹中的所有数据集并处理

if [ $# -lt 2 ]; then
    echo "用法: bash scripts/prepare_ct_recursive.sh <root_dir> <output_dir> [organ_labels.json] [num_workers] [max_depth]"
    echo ""
    echo "参数:"
    echo "  root_dir       : 根目录（包含多个数据集）"
    echo "  output_dir     : 输出基础目录"
    echo "  organ_labels   : 器官标签映射JSON文件（可选，用于NIfTI格式）"
    echo "  num_workers    : 并行进程数（可选，默认4）"
    echo "  max_depth      : 最大递归深度（可选，默认5）"
    echo ""
    echo "示例:"
    echo "  # 处理包含多个数据集的大文件夹"
    echo "  bash scripts/prepare_ct_recursive.sh \\"
    echo "       /path/to/all_datasets \\"
    echo "       ./processed_all"
    echo ""
    echo "  # 指定器官映射和并行数"
    echo "  bash scripts/prepare_ct_recursive.sh \\"
    echo "       /path/to/all_datasets \\"
    echo "       ./processed_all \\"
    echo "       ./organ_mapping.json \\"
    echo "       8"
    echo ""
    echo "支持的数据格式:"
    echo "  1. NIfTI格式:"
    echo "     your_dataset/"
    echo "     ├── imagesTr/"
    echo "     │   ├── case_001_0000.nii.gz"
    echo "     │   └── ..."
    echo "     └── labelsTr/"
    echo "         ├── case_001.nii.gz"
    echo "         └── ..."
    echo ""
    echo "  2. M3D-Seg格式:"
    echo "     dataset_0000/"
    echo "     ├── 0000.json"
    echo "     ├── 1/"
    echo "     │   ├── image.npy"
    echo "     │   └── mask_*.npz"
    echo "     └── ..."
    exit 1
fi

# 参数解析
ROOT_DIR=$1
OUTPUT_DIR=$2
ORGAN_LABELS=${3:-""}
NUM_WORKERS=${4:-4}
MAX_DEPTH=${5:-5}

echo "=========================================="
echo "   CT数据递归预处理"
echo "=========================================="
echo "根目录: $ROOT_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "器官标签: ${ORGAN_LABELS:-未指定}"
echo "并行进程数: $NUM_WORKERS"
echo "最大递归深度: $MAX_DEPTH"
echo "=========================================="
echo ""

# 检查输入目录
if [ ! -d "$ROOT_DIR" ]; then
    echo "错误: 根目录不存在: $ROOT_DIR"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 构建Python命令
CMD="python dataset_toolkits/process_ct_recursive.py \
    --root_dir \"$ROOT_DIR\" \
    --output_dir \"$OUTPUT_DIR\" \
    --num_workers $NUM_WORKERS \
    --max_depth $MAX_DEPTH"

# 如果指定了器官标签，添加参数
if [ -n "$ORGAN_LABELS" ] && [ -f "$ORGAN_LABELS" ]; then
    CMD="$CMD --organ_labels \"$ORGAN_LABELS\""
fi

echo "执行命令:"
echo "$CMD"
echo ""

# 执行递归处理
eval $CMD

if [ $? -ne 0 ]; then
    echo ""
    echo "✗ 递归处理失败"
    exit 1
fi

echo ""
echo "=========================================="
echo "   递归处理完成！"
echo "=========================================="
echo ""
echo "查看处理结果:"
echo "  总结报告: $OUTPUT_DIR/processing_summary.json"
echo ""
echo "各数据集输出目录:"
echo "  $OUTPUT_DIR/"
echo "    ├── dataset_1/"
echo "    │   ├── metadata.csv"
echo "    │   └── processed/"
echo "    ├── dataset_2/"
echo "    │   ├── metadata.csv"
echo "    │   └── processed/"
echo "    └── ..."
echo ""

