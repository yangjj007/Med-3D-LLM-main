#!/bin/bash
# 调试预处理脚本 - 对比成功和失败的数据集
# 用于查找 0008 数据集处理失败的原因

set -e  # 遇到错误时停止（可选）

echo "=========================================="
echo "   数据预处理调试脚本"
echo "=========================================="
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 配置参数
ROOT_DIR="./M3D_Seg"
OUTPUT_DIR="./debug_processed"
NUM_WORKERS=1  # 使用单进程便于调试
LOG_DIR="./debug_logs"

# 创建输出和日志目录
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# 清理旧的输出（避免"跳过已处理"）
echo "清理旧的调试输出..."
rm -rf "$OUTPUT_DIR/0001"
rm -rf "$OUTPUT_DIR/0008"
echo "✓ 清理完成"
echo ""

# ============================================
# 函数：检查数据集基本信息
# ============================================
check_dataset_info() {
    local dataset_path=$1
    local dataset_name=$2
    
    echo "----------------------------------------"
    echo "检查数据集: $dataset_name"
    echo "----------------------------------------"
    echo "路径: $dataset_path"
    
    # 检查目录是否存在
    if [ ! -d "$dataset_path" ]; then
        echo "❌ 错误: 目录不存在"
        return 1
    fi
    
    # 检查 JSON 文件
    json_files=$(find "$dataset_path" -maxdepth 1 -name "*.json" 2>/dev/null)
    if [ -n "$json_files" ]; then
        echo "✓ JSON配置文件:"
        echo "$json_files" | while read -r file; do
            echo "  - $(basename "$file")"
            echo "    内容预览:"
            head -n 20 "$file" | sed 's/^/    /'
        done
    else
        echo "⚠ 未找到JSON配置文件"
    fi
    
    # 统计病例数
    case_count=$(find "$dataset_path" -mindepth 1 -maxdepth 1 -type d | wc -l)
    echo "✓ 子文件夹数量: $case_count"
    
    # 检查几个病例的文件
    echo "✓ 病例样本检查:"
    find "$dataset_path" -mindepth 1 -maxdepth 1 -type d | head -n 3 | while read -r case_dir; do
        case_name=$(basename "$case_dir")
        echo "  - $case_name:"
        
        # 检查 image.npy
        if [ -f "$case_dir/image.npy" ]; then
            image_size=$(ls -lh "$case_dir/image.npy" | awk '{print $5}')
            echo "    image.npy: $image_size"
        else
            echo "    ❌ image.npy 不存在"
        fi
        
        # 检查 mask 文件
        mask_files=$(ls "$case_dir"/mask_*.npz 2>/dev/null)
        if [ -n "$mask_files" ]; then
            echo "    mask文件:"
            echo "$mask_files" | while read -r mask; do
                mask_name=$(basename "$mask")
                mask_size=$(ls -lh "$mask" | awk '{print $5}')
                echo "      $mask_name: $mask_size"
            done
        else
            echo "    ❌ mask文件不存在"
        fi
    done
    
    echo ""
}

# ============================================
# 函数：处理单个数据集
# ============================================
process_dataset() {
    local dataset_name=$1
    local dataset_path=$2
    local output_path=$3
    local log_file=$4
    
    echo "=========================================="
    echo "处理数据集: $dataset_name"
    echo "=========================================="
    echo "输入路径: $dataset_path"
    echo "输出路径: $output_path"
    echo "日志文件: $log_file"
    echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # 记录开始时间
    start_time=$(date +%s)
    
    # 执行预处理（启用Python的详细输出）
    echo "执行预处理命令..."
    python -u dataset_toolkits/process_m3d_seg_format.py \
        --data_root "$dataset_path" \
        --output_dir "$output_path" \
        --num_workers $NUM_WORKERS \
        --use_mask \
        --compute_sdf \
        --replace_npy \
        --no_skip \
        > "$log_file" 2>&1
    
    exit_code=$?
    
    # 记录结束时间
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    echo ""
    echo "----------------------------------------"
    echo "处理结果: $dataset_name"
    echo "----------------------------------------"
    echo "退出码: $exit_code"
    echo "耗时: ${duration}秒"
    
    if [ $exit_code -eq 0 ]; then
        echo "状态: ✓ 成功"
    else
        echo "状态: ❌ 失败"
        echo ""
        echo "错误日志（最后50行）:"
        tail -n 50 "$log_file" | sed 's/^/  /'
    fi
    
    # 检查输出结果
    echo ""
    echo "输出检查:"
    if [ -d "$output_path/processed" ]; then
        processed_count=$(find "$output_path/processed" -mindepth 1 -maxdepth 1 -type d | wc -l)
        echo "  处理的病例数: $processed_count"
        
        # 检查第一个病例的输出
        first_case=$(find "$output_path/processed" -mindepth 1 -maxdepth 1 -type d | head -n 1)
        if [ -n "$first_case" ]; then
            case_name=$(basename "$first_case")
            echo "  样本病例: $case_name"
            
            if [ -d "$first_case/masks" ]; then
                mask_files=$(ls "$first_case/masks" 2>/dev/null | wc -l)
                echo "    masks/ 文件数: $mask_files"
                ls -lh "$first_case/masks/" 2>/dev/null | sed 's/^/      /'
            else
                echo "    ❌ masks/ 目录不存在"
            fi
            
            if [ -f "$first_case/info.json" ]; then
                echo "    ✓ info.json 存在"
            else
                echo "    ❌ info.json 不存在"
            fi
        fi
    else
        echo "  ❌ processed/ 目录不存在"
    fi
    
    echo ""
    echo "完整日志位置: $log_file"
    echo "=========================================="
    echo ""
    
    return $exit_code
}

# ============================================
# 主流程
# ============================================

# 1. 检查成功数据集的基本信息
check_dataset_info "$ROOT_DIR/0001/0001" "0001 (成功案例)"

# 2. 检查失败数据集的基本信息
check_dataset_info "$ROOT_DIR/0008/0008" "0008 (失败案例)"

# 3. 处理成功数据集 (0001)
echo ""
echo "================================================"
echo "第1步: 处理成功数据集 0001"
echo "================================================"
echo ""

success_result=0
process_dataset \
    "0001" \
    "$ROOT_DIR/0001/0001" \
    "$OUTPUT_DIR/0001" \
    "$LOG_DIR/0001_preprocess.log" || success_result=$?

# 4. 处理失败数据集 (0008)
echo ""
echo "================================================"
echo "第2步: 处理失败数据集 0008"
echo "================================================"
echo ""

failure_result=0
process_dataset \
    "0008" \
    "$ROOT_DIR/0008/0008" \
    "$OUTPUT_DIR/0008" \
    "$LOG_DIR/0008_preprocess.log" || failure_result=$?

# ============================================
# 生成对比报告
# ============================================
echo ""
echo "=========================================="
echo "   对比分析报告"
echo "=========================================="
echo ""

echo "数据集 0001 (成功案例):"
echo "  处理结果: $([ $success_result -eq 0 ] && echo '✓ 成功' || echo '❌ 失败')"
echo "  日志文件: $LOG_DIR/0001_preprocess.log"
echo ""

echo "数据集 0008 (失败案例):"
echo "  处理结果: $([ $failure_result -eq 0 ] && echo '✓ 成功' || echo '❌ 失败')"
echo "  日志文件: $LOG_DIR/0008_preprocess.log"
echo ""

# 如果0008失败，提取关键错误信息
if [ $failure_result -ne 0 ]; then
    echo "----------------------------------------"
    echo "0008 失败原因分析"
    echo "----------------------------------------"
    
    # 搜索常见错误模式
    echo ""
    echo "1. Python异常:"
    grep -i "error\|exception\|traceback" "$LOG_DIR/0008_preprocess.log" | tail -n 20 | sed 's/^/  /'
    
    echo ""
    echo "2. 文件相关问题:"
    grep -i "file\|not found\|permission" "$LOG_DIR/0008_preprocess.log" | tail -n 10 | sed 's/^/  /'
    
    echo ""
    echo "3. 数据相关问题:"
    grep -i "shape\|dimension\|array\|mask" "$LOG_DIR/0008_preprocess.log" | tail -n 10 | sed 's/^/  /'
    
    echo ""
    echo "完整错误日志请查看: $LOG_DIR/0008_preprocess.log"
fi

# 生成对比摘要文件
summary_file="$LOG_DIR/comparison_summary.txt"
{
    echo "数据预处理对比摘要"
    echo "生成时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    echo "=================================="
    echo ""
    echo "数据集 0001:"
    echo "  状态: $([ $success_result -eq 0 ] && echo '成功' || echo '失败')"
    echo "  路径: $ROOT_DIR/0001/0001"
    echo "  输出: $OUTPUT_DIR/0001"
    echo "  日志: $LOG_DIR/0001_preprocess.log"
    echo ""
    echo "数据集 0008:"
    echo "  状态: $([ $failure_result -eq 0 ] && echo '成功' || echo '失败')"
    echo "  路径: $ROOT_DIR/0008/0008"
    echo "  输出: $OUTPUT_DIR/0008"
    echo "  日志: $LOG_DIR/0008_preprocess.log"
    echo ""
    echo "=================================="
    echo ""
    echo "查看详细日志:"
    echo "  成功案例: cat $LOG_DIR/0001_preprocess.log"
    echo "  失败案例: cat $LOG_DIR/0008_preprocess.log"
    echo ""
    echo "查看错误信息:"
    echo "  grep -i error $LOG_DIR/0008_preprocess.log"
} > "$summary_file"

echo ""
echo "=========================================="
echo "对比摘要已保存到: $summary_file"
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
echo ""

# 提示如何查看日志
echo "下一步操作:"
echo "  1. 查看完整日志对比:"
echo "     diff $LOG_DIR/0001_preprocess.log $LOG_DIR/0008_preprocess.log"
echo ""
echo "  2. 查看0008的错误信息:"
echo "     grep -A 10 -i 'error\|exception' $LOG_DIR/0008_preprocess.log"
echo ""
echo "  3. 查看处理的详细步骤:"
echo "     cat $LOG_DIR/0008_preprocess.log | less"
echo ""

