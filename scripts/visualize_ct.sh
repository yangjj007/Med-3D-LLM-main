#!/bin/bash
# CT数据集可视化脚本
# 
# 使用方法:
#   bash scripts/visualize_ct.sh <数据集路径> [输出目录]
#
# 示例:
#   bash scripts/visualize_ct.sh /processed_dataset/processed/0000
#   bash scripts/visualize_ct.sh /processed_dataset/processed/0000 /custom/output

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 打印横幅
print_banner() {
    echo "================================================================================"
    echo "                      CT数据集3D可视化工具"
    echo "================================================================================"
}

# 检查参数
if [ $# -lt 1 ]; then
    print_error "缺少必要参数！"
    echo ""
    echo "使用方法:"
    echo "  bash scripts/visualize_ct.sh <数据集路径> [输出目录]"
    echo ""
    echo "示例:"
    echo "  bash scripts/visualize_ct.sh /processed_dataset/processed/0000"
    echo "  bash scripts/visualize_ct.sh /processed_dataset/processed/0000 /custom/output"
    echo ""
    exit 1
fi

DATASET_PATH=$1
OUTPUT_DIR=${2:-""}

# 打印横幅
print_banner

# 检查数据集路径
print_info "检查数据集路径..."
if [ ! -d "$DATASET_PATH" ]; then
    print_error "数据集路径不存在: $DATASET_PATH"
    exit 1
fi
print_success "数据集路径有效: $DATASET_PATH"

# 检查必要的数据文件
print_info "检查数据文件..."
CT_FILE_FOUND=false

if [ -f "$DATASET_PATH/ct_original_512.npy" ]; then
    print_success "找到: ct_original_512.npy"
    CT_FILE_FOUND=true
elif [ -f "$DATASET_PATH/ct_original_1024.npy" ]; then
    print_success "找到: ct_original_1024.npy"
    CT_FILE_FOUND=true
elif [ -f "$DATASET_PATH/ct_normalized_512.npy" ]; then
    print_success "找到: ct_normalized_512.npy (旧版本)"
    CT_FILE_FOUND=true
elif [ -f "$DATASET_PATH/ct_normalized_1024.npy" ]; then
    print_success "找到: ct_normalized_1024.npy (旧版本)"
    CT_FILE_FOUND=true
fi

if [ "$CT_FILE_FOUND" = false ]; then
    print_error "未找到CT数据文件 (ct_original_*.npy 或 ct_normalized_*.npy)"
    exit 1
fi

# 检查可选的目录
if [ -d "$DATASET_PATH/windows" ]; then
    WINDOW_COUNT=$(ls -1 "$DATASET_PATH/windows"/*.npy 2>/dev/null | wc -l)
    print_info "发现 $WINDOW_COUNT 个窗口文件"
fi

if [ -d "$DATASET_PATH/organs" ]; then
    ORGAN_COUNT=$(ls -d "$DATASET_PATH/organs"/*/ 2>/dev/null | wc -l)
    print_info "发现 $ORGAN_COUNT 个器官目录"
fi

if [ -d "$DATASET_PATH/masks" ]; then
    print_info "发现masks目录"
fi

# 检查Python依赖
print_info "检查Python依赖..."
python -c "import numpy, plotly, skimage" 2>/dev/null
if [ $? -ne 0 ]; then
    print_error "缺少必要的Python依赖包"
    echo ""
    echo "请安装以下依赖:"
    echo "  pip install numpy plotly scikit-image kaleido"
    echo ""
    exit 1
fi
print_success "Python依赖检查通过"

# 构建命令
CMD="python dataset_toolkits/visualize_ct_dataset.py --dataset_path \"$DATASET_PATH\""

if [ -n "$OUTPUT_DIR" ]; then
    CMD="$CMD --output_dir \"$OUTPUT_DIR\""
    print_info "输出目录: $OUTPUT_DIR"
else
    print_info "输出目录: $DATASET_PATH/visualization (默认)"
fi

echo "================================================================================"
print_info "开始可视化..."
echo "================================================================================"

# 执行可视化
eval $CMD

# 检查执行结果
if [ $? -eq 0 ]; then
    echo ""
    echo "================================================================================"
    print_success "可视化完成！"
    echo "================================================================================"
    
    # 确定实际的输出目录
    if [ -n "$OUTPUT_DIR" ]; then
        ACTUAL_OUTPUT="$OUTPUT_DIR"
    else
        ACTUAL_OUTPUT="$DATASET_PATH/visualization"
    fi
    
    # 打印可视化文件列表
    if [ -d "$ACTUAL_OUTPUT" ]; then
        echo ""
        print_info "生成的可视化文件:"
        ls -lh "$ACTUAL_OUTPUT"/*.html 2>/dev/null | awk '{print "  - " $9 " (" $5 ")"}'
        
        echo ""
        print_info "打开以下文件查看可视化:"
        INDEX_FILE="$ACTUAL_OUTPUT/index.html"
        
        if [ -f "$INDEX_FILE" ]; then
            # 转换为绝对路径
            ABS_INDEX_FILE=$(cd "$(dirname "$INDEX_FILE")" && pwd)/$(basename "$INDEX_FILE")
            echo "  file://$ABS_INDEX_FILE"
            
            # 尝试在浏览器中打开（可选）
            echo ""
            read -p "是否在浏览器中打开可视化? (y/n): " -n 1 -r
            echo ""
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                print_info "正在打开浏览器..."
                
                # 根据操作系统选择命令
                if [[ "$OSTYPE" == "linux-gnu"* ]]; then
                    xdg-open "$INDEX_FILE" 2>/dev/null || print_warning "无法自动打开浏览器"
                elif [[ "$OSTYPE" == "darwin"* ]]; then
                    open "$INDEX_FILE" 2>/dev/null || print_warning "无法自动打开浏览器"
                elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
                    start "$INDEX_FILE" 2>/dev/null || print_warning "无法自动打开浏览器"
                else
                    print_warning "未识别的操作系统，请手动打开文件"
                fi
            fi
        fi
    fi
    
    echo "================================================================================"
else
    echo ""
    echo "================================================================================"
    print_error "可视化过程中出现错误"
    echo "================================================================================"
    exit 1
fi

