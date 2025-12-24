#!/bin/bash
# CT数据集批量可视化脚本
# 
# 使用方法:
#   bash scripts/visualize_ct_batch.sh <processed目录> [最大数量]
#
# 示例:
#   bash scripts/visualize_ct_batch.sh /processed_dataset/processed
#   bash scripts/visualize_ct_batch.sh /processed_dataset/processed 5

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

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

print_banner() {
    echo "================================================================================"
    echo "                    CT数据集批量3D可视化工具"
    echo "================================================================================"
}

# 检查参数
if [ $# -lt 1 ]; then
    print_error "缺少必要参数！"
    echo ""
    echo "使用方法:"
    echo "  bash scripts/visualize_ct_batch.sh <processed目录> [最大数量]"
    echo ""
    echo "示例:"
    echo "  bash scripts/visualize_ct_batch.sh /processed_dataset/processed"
    echo "  bash scripts/visualize_ct_batch.sh /processed_dataset/processed 5"
    echo ""
    exit 1
fi

PROCESSED_DIR=$1
MAX_COUNT=${2:-999999}

print_banner

# 检查目录
print_info "检查processed目录..."
if [ ! -d "$PROCESSED_DIR" ]; then
    print_error "目录不存在: $PROCESSED_DIR"
    exit 1
fi
print_success "目录有效: $PROCESSED_DIR"

# 查找所有病例
print_info "扫描病例..."
CASES=()
for case_dir in "$PROCESSED_DIR"/*; do
    if [ -d "$case_dir" ]; then
        # 检查是否有CT数据文件
        if [ -f "$case_dir/ct_normalized_512.npy" ] || [ -f "$case_dir/ct_normalized_1024.npy" ]; then
            CASES+=("$case_dir")
        fi
    fi
done

TOTAL_CASES=${#CASES[@]}
print_success "发现 $TOTAL_CASES 个有效病例"

if [ $TOTAL_CASES -eq 0 ]; then
    print_error "未找到任何有效病例"
    exit 1
fi

# 限制处理数量
if [ $TOTAL_CASES -gt $MAX_COUNT ]; then
    print_info "限制处理数量: $MAX_COUNT (总共 $TOTAL_CASES)"
    TOTAL_CASES=$MAX_COUNT
fi

echo "================================================================================"
print_info "开始批量可视化..."
echo "================================================================================"

# 统计变量
SUCCESS_COUNT=0
FAILED_COUNT=0
FAILED_CASES=()

START_TIME=$(date +%s)

# 处理每个病例
for i in "${!CASES[@]}"; do
    if [ $i -ge $MAX_COUNT ]; then
        break
    fi
    
    case_dir="${CASES[$i]}"
    case_name=$(basename "$case_dir")
    current=$((i + 1))
    
    echo ""
    echo "--------------------------------------------------------------------------------"
    echo -e "${CYAN}[$current/$TOTAL_CASES]${NC} 处理病例: $case_name"
    echo "--------------------------------------------------------------------------------"
    
    # 执行可视化
    python dataset_toolkits/visualize_ct_dataset.py --dataset_path "$case_dir" 2>&1
    
    if [ $? -eq 0 ]; then
        print_success "病例 $case_name 可视化完成"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        print_error "病例 $case_name 可视化失败"
        FAILED_COUNT=$((FAILED_COUNT + 1))
        FAILED_CASES+=("$case_name")
    fi
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

# 打印总结
echo ""
echo "================================================================================"
echo "                           批量可视化完成总结"
echo "================================================================================"
echo ""
echo "总体统计:"
echo "  ✓ 成功: $SUCCESS_COUNT"
echo "  ✗ 失败: $FAILED_COUNT"
echo "  ⏱ 总耗时: ${ELAPSED}秒 ($((ELAPSED / 60))分钟)"
echo "  ⚡ 平均速度: $((ELAPSED / TOTAL_CASES))秒/病例"

if [ $FAILED_COUNT -gt 0 ]; then
    echo ""
    echo "失败的病例:"
    for case_name in "${FAILED_CASES[@]}"; do
        echo "  ✗ $case_name"
    done
fi

echo ""
echo "可视化文件位置:"
echo "  每个病例的可视化在: <病例目录>/visualization/"
echo "  例如: $PROCESSED_DIR/0000/visualization/index.html"

echo ""
echo "================================================================================"

# 生成批量总结索引
print_info "生成批量可视化总索引..."
SUMMARY_FILE="$PROCESSED_DIR/visualization_summary.html"

cat > "$SUMMARY_FILE" << EOF
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>批量可视化总览</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }
        .header h1 { font-size: 2.5em; margin-bottom: 10px; }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            padding: 30px;
            background: #f8f9fa;
        }
        .stat-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .stat-card h3 { color: #667eea; font-size: 2em; margin-bottom: 5px; }
        .stat-card p { color: #6c757d; }
        .content { padding: 30px; }
        .case-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .case-card {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            transition: all 0.3s;
            border: 2px solid #e9ecef;
        }
        .case-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
            border-color: #667eea;
        }
        .case-card h3 {
            color: #495057;
            margin-bottom: 15px;
            font-size: 1.3em;
        }
        .case-card a {
            display: inline-block;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 10px 20px;
            border-radius: 20px;
            text-decoration: none;
            font-weight: bold;
            transition: transform 0.2s;
        }
        .case-card a:hover { transform: scale(1.05); }
        .success { color: #28a745; }
        .failed { color: #dc3545; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏥 CT数据集批量可视化总览</h1>
            <p>处理时间: $(date '+%Y-%m-%d %H:%M:%S')</p>
        </div>
        
        <div class="stats">
            <div class="stat-card">
                <h3>$SUCCESS_COUNT</h3>
                <p class="success">✓ 成功</p>
            </div>
            <div class="stat-card">
                <h3>$FAILED_COUNT</h3>
                <p class="failed">✗ 失败</p>
            </div>
            <div class="stat-card">
                <h3>${ELAPSED}s</h3>
                <p>总耗时</p>
            </div>
            <div class="stat-card">
                <h3>$((ELAPSED / TOTAL_CASES))s</h3>
                <p>平均速度</p>
            </div>
        </div>
        
        <div class="content">
            <h2 style="color: #667eea; margin-bottom: 20px;">📁 病例列表</h2>
            <div class="case-grid">
EOF

# 添加每个病例的卡片
for case_dir in "${CASES[@]}"; do
    case_name=$(basename "$case_dir")
    vis_dir="$case_dir/visualization"
    
    if [ -f "$vis_dir/index.html" ]; then
        rel_path="$(basename "$PROCESSED_DIR")/$case_name/visualization/index.html"
        cat >> "$SUMMARY_FILE" << EOF
                <div class="case-card">
                    <h3>📊 $case_name</h3>
                    <p class="success">✓ 可视化成功</p>
                    <br>
                    <a href="$case_name/visualization/index.html" target="_blank">查看可视化 →</a>
                </div>
EOF
    else
        cat >> "$SUMMARY_FILE" << EOF
                <div class="case-card">
                    <h3>📊 $case_name</h3>
                    <p class="failed">✗ 可视化失败</p>
                </div>
EOF
    fi
done

cat >> "$SUMMARY_FILE" << EOF
            </div>
        </div>
    </div>
</body>
</html>
EOF

print_success "总索引已生成: $SUMMARY_FILE"
ABS_SUMMARY=$(cd "$(dirname "$SUMMARY_FILE")" && pwd)/$(basename "$SUMMARY_FILE")
echo "  file://$ABS_SUMMARY"
echo "================================================================================"

