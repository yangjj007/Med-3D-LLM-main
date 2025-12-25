#!/bin/bash
# CT数据集预处理脚本
# 自动化执行完整的CT数据预处理流程

# 用法检查
if [ $# -lt 2 ]; then
    echo "用法: bash scripts/prepare_medical_ct_dataset.sh <nifti_root> <output_dir> [organ_labels.json] [num_workers]"
    echo ""
    echo "参数:"
    echo "  nifti_root      : NIfTI数据根目录（包含imagesTr和labelsTr）"
    echo "  output_dir      : 输出目录"
    echo "  organ_labels    : 器官标签映射JSON文件（可选）"
    echo "  num_workers     : 并行进程数（可选，默认4）"
    echo ""
    echo "示例:"
    echo "  bash scripts/prepare_medical_ct_dataset.sh \\"
    echo "       ./data/3Dircad \\"
    echo "       ./data/processed_ct \\"
    echo "       ./dataset_toolkits/ct_preprocessing/organ_mapping_example.json \\"
    echo "       8"
    exit 1
fi

# 参数解析
NIFTI_ROOT=$1
OUTPUT_DIR=$2
ORGAN_LABELS=${3:-""}
NUM_WORKERS=${4:-4}
DEFAULT_RESOLUTION=512

echo "=========================================="
echo "   CT数据集预处理流程"
echo "=========================================="
echo "NIfTI数据根目录: $NIFTI_ROOT"
echo "输出目录: $OUTPUT_DIR"
echo "器官标签映射: ${ORGAN_LABELS:-未指定}"
echo "并行进程数: $NUM_WORKERS"
echo "默认分辨率: ${DEFAULT_RESOLUTION}³"
echo "=========================================="
echo ""

# 检查输入目录
if [ ! -d "$NIFTI_ROOT" ]; then
    echo "错误: NIfTI数据根目录不存在: $NIFTI_ROOT"
    exit 1
fi

if [ ! -d "$NIFTI_ROOT/imagesTr" ]; then
    echo "错误: imagesTr目录不存在: $NIFTI_ROOT/imagesTr"
    echo "请确保目录结构正确:"
    echo "  $NIFTI_ROOT/"
    echo "    ├── imagesTr/"
    echo "    └── labelsTr/"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 步骤1: CT预处理（分辨率适配、窗口处理、器官提取）
echo "=========================================="
echo "步骤1: CT数据预处理"
echo "=========================================="
echo ""

# 构建命令
CMD="python dataset_toolkits/process_medical_ct.py \
    --data_root \"$NIFTI_ROOT\" \
    --output_dir \"$OUTPUT_DIR\" \
    --default_resolution $DEFAULT_RESOLUTION \
    --num_workers $NUM_WORKERS"

# 如果指定了器官标签映射，添加参数
if [ -n "$ORGAN_LABELS" ] && [ -f "$ORGAN_LABELS" ]; then
    CMD="$CMD --organ_labels \"$ORGAN_LABELS\""
    echo "使用器官标签映射: $ORGAN_LABELS"
fi

echo "执行命令:"
echo "$CMD"
echo ""

# 执行预处理
eval $CMD

if [ $? -ne 0 ]; then
    echo ""
    echo "✗ CT预处理失败"
    exit 1
fi

echo ""
echo "✓ CT预处理完成"
echo ""

# 步骤2: 构建元数据（用于后续处理）
echo "=========================================="
echo "步骤2: 构建元数据"
echo "=========================================="
echo ""

python dataset_toolkits/datasets/MedicalCT.py \
    --processed_dir "$OUTPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --data_type processed

if [ $? -ne 0 ]; then
    echo ""
    echo "✗ 元数据构建失败"
    exit 1
fi

echo ""
echo "✓ 元数据构建完成"
echo ""

# 步骤3（可选）: 生成Sparse SDF用于TRELLIS训练
echo "=========================================="
echo "步骤3: 生成Sparse SDF（可选）"
echo "=========================================="
echo ""
echo "提示: 如果需要训练TRELLIS模型，可以运行以下命令生成Sparse SDF:"
echo ""
echo "  python dataset_toolkits/compute_sparse_sdf.py \\"
echo "      --output_dir \"$OUTPUT_DIR\" \\"
echo "      --resolutions 512 \\"
echo "      --input_type voxel \\"
echo "      --max_workers 8"
echo ""
echo "跳过此步骤（需要CUDA支持）..."
echo ""

# 步骤4: 生成数据统计报告
echo "=========================================="
echo "步骤4: 数据统计报告"
echo "=========================================="
echo ""

python -c "
import pandas as pd
import json
import os

output_dir = '$OUTPUT_DIR'
metadata_path = os.path.join(output_dir, 'metadata.csv')
config_path = os.path.join(output_dir, 'dataset_config.json')

if os.path.exists(metadata_path):
    metadata = pd.read_csv(metadata_path)
    print(f'总病例数: {len(metadata)}')
    
    if 'has_segmentation' in metadata.columns:
        has_seg = metadata['has_segmentation'].sum()
        print(f'有分割标签: {has_seg} ({has_seg/len(metadata)*100:.1f}%)')
    
    if 'resolution' in metadata.columns:
        print(f'')
        print(f'分辨率分布:')
        for res, count in metadata['resolution'].value_counts().sort_index().items():
            print(f'  {res}³: {count} 病例')
    
    if 'file_size_mb' in metadata.columns:
        total_size = metadata['file_size_mb'].sum()
        avg_size = metadata['file_size_mb'].mean()
        print(f'')
        print(f'存储统计:')
        print(f'  总大小: {total_size:.2f} MB ({total_size/1024:.2f} GB)')
        print(f'  平均大小: {avg_size:.2f} MB/病例')
    
    if 'processing_time_sec' in metadata.columns:
        total_time = metadata['processing_time_sec'].sum()
        avg_time = metadata['processing_time_sec'].mean()
        print(f'')
        print(f'处理时间:')
        print(f'  总时间: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)')
        print(f'  平均时间: {avg_time:.2f} 秒/病例')

if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    print(f'')
    print(f'数据集配置:')
    print(f'  名称: {config.get(\"dataset_name\", \"Unknown\")}')
    print(f'  模态: {config.get(\"modality\", \"Unknown\")}')
    print(f'  处理日期: {config.get(\"processing_date\", \"Unknown\")}')
"

echo ""
echo "=========================================="
echo "   预处理流程完成！"
echo "=========================================="
echo ""
echo "输出目录结构:"
echo "  $OUTPUT_DIR/"
echo "    ├── metadata.csv                # 元数据"
echo "    ├── dataset_config.json         # 数据集配置"
echo "    └── processed/                  # 处理后的数据"
echo "        ├── case_001/"
echo "        │   ├── ct_original_512.npy # 原始CT（HU值，已适配分辨率）"
echo "        │   ├── windows/            # 窗口二值化结果（基于原始HU值）"
echo "        │   ├── organs/             # 器官特定窗口"
echo "        │   ├── masks/              # 分割掩码"
echo "        │   └── info.json"
echo "        └── ..."
echo ""
echo "下一步:"
echo "  1. 查看数据: 检查输出目录中的processed文件夹"
echo "  2. 训练模型: 使用处理后的数据训练TRELLIS"
echo "  3. 数据加载示例: 参考ct_preprocess_README.md"
echo ""

