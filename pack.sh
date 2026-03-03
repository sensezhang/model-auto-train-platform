#!/bin/bash

# ========================================
#   YOLO/RF-DETR 训练标注平台 - 打包脚本
# ========================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="yolo-train-test"
OUTPUT_FILE="${PROJECT_NAME}-deploy.tar.gz"

cd "$(dirname "$SCRIPT_DIR")"

echo "========================================"
echo "  YOLO/RF-DETR 训练标注平台 - 打包脚本"
echo "========================================"
echo ""
echo "正在打包项目（仅代码和配置）..."
echo ""

# 排除列表
EXCLUDES=(
    # 依赖和虚拟环境
    "node_modules"
    ".venv"
    "venv"
    "backend/.venv"
    "backend/venv"
    "__pycache__"
    "*.pyc"

    # IDE和版本控制
    ".git"
    ".idea"
    ".claude"
    ".vscode"

    # 缓存
    ".ultralytics"
    "backend/.ultralytics"
    ".cache"
    ".torch"

    # 数据目录
    "datasets"
    "backend/datasets"
    "models"
    "backend/models"
    "data"
    "backend/data"
    "output"
    "backend/output"
    "deploy/data"
    "datasets_coco"
    "backend/datasets_coco"

    # 数据文件
    "*.db"
    "*.pth"
    "*.pt"
    "*.onnx"
    "*.log"
    "*.tar.gz"

    # 其他
    "nul"
)

# 构建排除参数
EXCLUDE_ARGS=""
for item in "${EXCLUDES[@]}"; do
    EXCLUDE_ARGS="$EXCLUDE_ARGS --exclude=$item"
done

# 打包
tar -czvf "$OUTPUT_FILE" $EXCLUDE_ARGS "$PROJECT_NAME"

echo ""
echo "========================================"
echo "打包完成！"
echo ""
echo "输出文件: $(pwd)/$OUTPUT_FILE"
echo "文件大小: $(du -h "$OUTPUT_FILE" | cut -f1)"
echo ""
echo "部署步骤:"
echo "  1. 上传到服务器: scp $OUTPUT_FILE user@server:/opt/"
echo "  2. 解压: cd /opt && tar -xzvf $OUTPUT_FILE"
echo "  3. 启动: cd $PROJECT_NAME/deploy && ./deploy.sh start"
echo "========================================"
