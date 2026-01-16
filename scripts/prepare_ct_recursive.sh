#!/bin/bash
# CT数据递归预处理脚本
# 自动扫描大文件夹中的所有数据集并处理

if [ $# -lt 2 ]; then
    echo "用法: bash scripts/prepare_ct_recursive.sh <root_dir> <output_dir> [organ_labels.json] [num_workers] [max_depth] [--compute_sdf] [--replace_npy] [--use_mask]"
    echo ""
    echo "参数:"
    echo "  root_dir       : 根目录（包含多个数据集）"
    echo "  output_dir     : 输出基础目录"
    echo "  organ_labels   : 器官标签映射JSON文件（可选，仅用于NIfTI格式）"
    echo "                  注意：M3D-Seg格式会自动从数据集JSON中读取标签，无需此参数"
    echo "  num_workers    : 并行进程数（可选，默认4）"
    echo "  max_depth      : 最大递归深度（可选，默认5）"
    echo "  --compute_sdf  : 计算窗口数据的SDF表示（需要CUDA和TRELLIS）"
    echo "  --replace_npy  : 用NPZ文件替换原NPY文件"
    echo "  --use_mask     : 直接使用分割掩码生成二值化体素网格，跳过窗位窗宽处理"
    echo ""
    echo "示例:"
    echo "  # 处理M3D-Seg格式（自动读取数据集自带的标签信息）"
    echo "  bash scripts/prepare_ct_recursive.sh \\"
    echo "       /path/to/M3D_Seg \\"
    echo "       ./processed_m3d_seg \\"
    echo "       \"\" \\"
    echo "       8 \\"
    echo "       5 \\"
    echo "       --compute_sdf \\"
    echo "       --replace_npy"
    echo ""
    echo "  # 使用掩码模式（跳过窗位窗宽，直接从掩码提取器官）"
    echo "  bash scripts/prepare_ct_recursive.sh \\"
    echo "       /path/to/datasets \\"
    echo "       ./processed_masks \\"
    echo "       ./organ_labels.json \\"
    echo "       8 \\"
    echo "       5 \\"
    echo "       --use_mask \\"
    echo "       --compute_sdf \\"
    echo "       --replace_npy"
    echo ""
    echo "  # 处理NIfTI格式（需要提供器官映射文件）"
    echo "  bash scripts/prepare_ct_recursive.sh \\"
    echo "       /path/to/nifti_datasets \\"
    echo "       ./processed_nifti \\"
    echo "       ./organ_labels.json \\"
    echo "       8 \\"
    echo "       5"
    echo ""
    echo "支持的数据格式:"
    echo "  1. NIfTI格式（需要organ_labels.json）:"
    echo "     your_dataset/"
    echo "     ├── imagesTr/"
    echo "     │   ├── case_001_0000.nii.gz"
    echo "     │   └── ..."
    echo "     └── labelsTr/"
    echo "         ├── case_001.nii.gz"
    echo "         └── ..."
    echo ""
    echo "  2. M3D-Seg格式（自动从0000.json等文件读取标签）:"
    echo "     dataset_0000/"
    echo "     ├── 0000.json        # 包含labels字段"
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

# 解析额外标志参数
COMPUTE_SDF=""
REPLACE_NPY=""
USE_MASK=""
shift 5 2>/dev/null || true  # 跳过前5个位置参数
while [ $# -gt 0 ]; do
    case "$1" in
        --compute_sdf)
            COMPUTE_SDF="--compute_sdf"
            ;;
        --replace_npy)
            REPLACE_NPY="--replace_npy"
            ;;
        --use_mask)
            USE_MASK="--use_mask"
            ;;
    esac
    shift
done

echo "=========================================="
echo "   CT数据递归预处理"
echo "=========================================="
echo "根目录: $ROOT_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "器官标签: ${ORGAN_LABELS:-未指定（M3D-Seg格式会自动读取）}"
echo "并行进程数: $NUM_WORKERS"
echo "最大递归深度: $MAX_DEPTH"
echo "使用掩码模式: ${USE_MASK:-否}"
echo "计算SDF: ${COMPUTE_SDF:-否}"
echo "替换NPY: ${REPLACE_NPY:-否}"
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

# 如果指定了器官标签，添加参数（仅用于NIfTI格式，M3D-Seg格式会自动读取）
if [ -n "$ORGAN_LABELS" ] && [ -f "$ORGAN_LABELS" ]; then
    CMD="$CMD --organ_labels \"$ORGAN_LABELS\""
    echo "注意: organ_labels仅用于NIfTI格式数据集"
    echo "      M3D-Seg格式会自动从数据集JSON中读取标签信息"
    echo ""
fi

# 添加SDF相关参数
if [ -n "$COMPUTE_SDF" ]; then
    CMD="$CMD $COMPUTE_SDF"
fi

if [ -n "$REPLACE_NPY" ]; then
    CMD="$CMD $REPLACE_NPY"
fi

if [ -n "$USE_MASK" ]; then
    CMD="$CMD $USE_MASK"
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

