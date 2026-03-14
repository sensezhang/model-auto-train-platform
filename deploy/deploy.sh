#!/bin/bash

# ============================================
# YOLO/RF-DETR 训练标注平台 - 一键部署脚本
# ============================================
# 使用方法:
#   ./deploy.sh start       # 启动服务（GPU 模式）
#   ./deploy.sh stop        # 停止服务
#   ./deploy.sh restart     # 重启服务
#   ./deploy.sh logs        # 查看日志
#   ./deploy.sh update      # 更新并重启
#   ./deploy.sh upgrade     # 从压缩包更新代码（保留数据）
#   ./deploy.sh status      # 查看状态
#   ./deploy.sh clean       # 清理（保留数据）
#   ./deploy.sh clean-all   # 完全清理（删除数据）
# ============================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="/mnt/data"

cd "$SCRIPT_DIR"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# 检查 Docker
check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker 未安装"
        echo ""
        echo "请先安装 Docker："
        echo "  Ubuntu: curl -fsSL https://get.docker.com | sh"
        echo "  或参考: https://docs.docker.com/engine/install/"
        exit 1
    fi

    if ! docker info &> /dev/null; then
        log_error "Docker 服务未运行或当前用户无权限"
        echo ""
        echo "请尝试："
        echo "  1. 启动 Docker: sudo systemctl start docker"
        echo "  2. 添加用户到 docker 组: sudo usermod -aG docker \$USER"
        echo "  3. 重新登录后再试"
        exit 1
    fi

    log_info "Docker 检查通过 ✓"
}

# 检查 NVIDIA GPU
check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi &> /dev/null; then
            GPU_COUNT=$(nvidia-smi -L | wc -l)
            log_info "检测到 $GPU_COUNT 个 NVIDIA GPU ✓"

            # 检查 NVIDIA Container Toolkit
            if docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi &> /dev/null; then
                log_info "NVIDIA Container Toolkit 可用 ✓"
                return 0
            else
                log_warn "NVIDIA Container Toolkit 未安装或配置错误"
                echo ""
                echo "请安装 NVIDIA Container Toolkit："
                echo "  distribution=\$(. /etc/os-release;echo \$ID\$VERSION_ID)"
                echo "  curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -"
                echo "  curl -s -L https://nvidia.github.io/nvidia-docker/\$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list"
                echo "  sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit"
                echo "  sudo systemctl restart docker"
                return 1
            fi
        fi
    fi

    log_error "未检测到 NVIDIA GPU 或 NVIDIA Container Toolkit 未就绪，无法启动"
    exit 1
}

# 初始化目录
init() {
    log_step "初始化数据目录 ($DATA_DIR)..."

    if [ ! -d "$DATA_DIR" ]; then
        log_error "/mnt/data 不存在，请先挂载数据盘（参考 README.md 第一步）"
        exit 1
    fi

    mkdir -p "$DATA_DIR/datasets"
    mkdir -p "$DATA_DIR/datasets_coco"
    mkdir -p "$DATA_DIR/models"
    mkdir -p "$DATA_DIR/output"
    mkdir -p "$DATA_DIR/logs"
    mkdir -p "$DATA_DIR/pretrained"
    mkdir -p "$DATA_DIR/.torch"
    mkdir -p "$DATA_DIR/.cache/huggingface"
    mkdir -p "$DATA_DIR/.ultralytics"

    # 设置权限
    chmod -R 777 "$DATA_DIR"

    log_info "数据目录初始化完成"
    echo ""
    echo "数据存储位置: $DATA_DIR"
    echo "  ├── datasets/      # 上传的图片"
    echo "  ├── datasets_coco/ # 训练中间数据集"
    echo "  ├── models/        # 训练 checkpoint 和日志"
    echo "  ├── output/        # 导出文件"
    echo "  ├── logs/          # 后端运行日志"
    echo "  ├── pretrained/    # 预训练权重（手动拷贝）"
    echo "  └── .cache/        # HuggingFace 模型缓存"
    echo ""
}

# 启动服务（GPU 模式）
start_gpu() {
    log_step "启动服务（GPU 模式）..."
    docker compose -f docker-compose.yml up -d --build
    show_access_info
}

# 自动检测并启动
start_auto() {
    check_gpu
    start_gpu
}

# 显示访问信息
show_access_info() {
    echo ""
    log_info "服务启动成功！"
    echo ""
    echo "============================================"
    echo "  访问地址: http://localhost"
    echo "  或: http://$(hostname -I | awk '{print $1}')"
    echo "============================================"
    echo ""
    echo "常用命令:"
    echo "  ./deploy.sh logs     # 查看日志"
    echo "  ./deploy.sh status   # 查看状态"
    echo "  ./deploy.sh stop     # 停止服务"
    echo ""
}

# 停止服务
stop() {
    log_step "停止服务..."
    docker compose -f docker-compose.yml down 2>/dev/null || true
    log_info "服务已停止"
}

# 重启服务
restart() {
    log_step "重启服务..."
    docker compose -f docker-compose.yml restart
    log_info "服务已重启"
}

# 查看日志
logs() {
    docker compose -f docker-compose.yml logs -f
}

# 查看后端日志
logs_backend() {
    docker logs -f yolo-backend
}

# 更新部署
update() {
    log_step "更新部署..."

    # 拉取最新代码（如果是 git 仓库）
    if [ -d "$PROJECT_DIR/.git" ]; then
        log_info "拉取最新代码..."
        cd "$PROJECT_DIR"
        git pull
        cd "$SCRIPT_DIR"
    fi

    # 重新构建并启动
    check_gpu
    docker compose -f docker-compose.yml up -d --build

    log_info "更新完成"
}

# 从压缩包更新代码（保留数据）
upgrade() {
    TARBALL="$1"

    if [ -z "$TARBALL" ]; then
        # 查找当前目录或上级目录的压缩包
        TARBALL=$(ls -t ../*.tar.gz 2>/dev/null | head -1)
        if [ -z "$TARBALL" ]; then
            TARBALL=$(ls -t ../../*.tar.gz 2>/dev/null | head -1)
        fi
    fi

    if [ -z "$TARBALL" ] || [ ! -f "$TARBALL" ]; then
        log_error "请指定压缩包路径: ./deploy.sh upgrade <tarball.tar.gz>"
        log_info "或将压缩包放在上级目录"
        exit 1
    fi

    # 转换为绝对路径
    TARBALL=$(cd "$(dirname "$TARBALL")" && pwd)/$(basename "$TARBALL")
    BACKUP_DIR="/tmp/yolo-data-backup-$$"
    PARENT_DIR="$(dirname "$PROJECT_DIR")"
    PROJECT_NAME="$(basename "$PROJECT_DIR")"
    NEW_DATA_DIR="$PARENT_DIR/$PROJECT_NAME/deploy/data"

    log_step "从 $TARBALL 更新代码（保留数据）..."
    log_info "项目目录: $PROJECT_DIR"
    log_info "数据目录: $DATA_DIR"

    # 停止服务
    stop

    # 备份数据目录
    if [ -d "$DATA_DIR" ]; then
        log_info "备份数据目录到 $BACKUP_DIR ..."
        cp -r "$DATA_DIR" "$BACKUP_DIR"
    fi

    # 删除旧代码目录
    log_info "删除旧代码..."
    rm -rf "$PROJECT_DIR"

    # 解压新代码
    log_info "解压新代码..."
    cd "$PARENT_DIR"
    tar -xzf "$TARBALL"

    # 恢复数据目录
    if [ -d "$BACKUP_DIR" ]; then
        log_info "恢复数据目录..."
        rm -rf "$NEW_DATA_DIR"
        mv "$BACKUP_DIR" "$NEW_DATA_DIR"
        log_info "数据已恢复到 $NEW_DATA_DIR"
    fi

    # 进入新的 deploy 目录
    cd "$NEW_DATA_DIR/.."

    # 重新启动
    init
    start_auto

    log_info "升级完成！数据已保留"
}

# 显示状态
status() {
    echo ""
    echo "=== 容器状态 ==="
    docker compose -f docker-compose.yml ps 2>/dev/null

    echo ""
    echo "=== GPU 状态 ==="
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
    else
        echo "未检测到 NVIDIA GPU"
    fi

    echo ""
    echo "=== 磁盘使用 ==="
    df -h "$DATA_DIR" 2>/dev/null || echo "数据盘未挂载"
    du -sh "$DATA_DIR"/* 2>/dev/null | sort -rh | head -10
}

# 清理（保留数据）
clean() {
    log_step "清理 Docker 资源（保留数据）..."
    stop
    docker compose -f docker-compose.yml down --rmi local 2>/dev/null || true
    docker system prune -f
    log_info "清理完成，数据已保留在: $DATA_DIR"
}

# 完全清理
clean_all() {
    log_warn "即将删除所有数据，包括："
    echo "  - 数据库"
    echo "  - 上传的图片"
    echo "  - 训练的模型"
    echo "  - 导出的数据集"
    echo ""
    read -p "确认删除？(输入 yes 确认): " confirm
    if [ "$confirm" = "yes" ]; then
        clean
        rm -rf "$DATA_DIR"
        log_info "所有数据已删除"
    else
        log_info "取消操作"
    fi
}

# 备份数据（仅备份数据库相关内容，图片/模型体积太大不在此备份）
backup() {
    BACKUP_FILE="$HOME/yolo-backup-$(date +%Y%m%d-%H%M%S).tar.gz"
    log_step "备份数据到 $BACKUP_FILE ..."
    log_warn "仅备份 models/ 目录（图片请单独备份数据盘）"
    tar -czvf "$BACKUP_FILE" -C "$DATA_DIR" models
    log_info "备份完成: $BACKUP_FILE"
}

# 恢复数据
restore() {
    if [ -z "$1" ]; then
        log_error "请指定备份文件: ./deploy.sh restore <backup-file.tar.gz>"
        exit 1
    fi

    if [ ! -f "$1" ]; then
        log_error "备份文件不存在: $1"
        exit 1
    fi

    log_step "从 $1 恢复数据..."
    stop
    tar -xzvf "$1" -C "$SCRIPT_DIR"
    log_info "恢复完成"
}

# 帮助信息
help() {
    echo ""
    echo "YOLO/RF-DETR 训练标注平台 - 部署脚本"
    echo ""
    echo "使用方法: ./deploy.sh <命令>"
    echo ""
    echo "命令:"
    echo "  start        启动服务（GPU 模式）"
    echo "  stop         停止服务"
    echo "  restart      重启服务"
    echo "  update       更新并重启服务（git pull）"
    echo "  upgrade      从压缩包更新代码（保留数据）"
    echo "  logs         查看所有日志"
    echo "  logs-backend 查看后端日志"
    echo "  status       查看服务状态"
    echo "  backup       备份数据"
    echo "  restore      恢复数据"
    echo "  clean        清理（保留数据）"
    echo "  clean-all    完全清理（删除数据）"
    echo "  help         显示帮助"
    echo ""
}

# 主函数
main() {
    case "${1:-help}" in
        start)
            check_docker
            init
            start_auto
            ;;
        stop)
            stop
            ;;
        restart)
            restart
            ;;
        logs)
            logs
            ;;
        logs-backend)
            logs_backend
            ;;
        update)
            check_docker
            update
            ;;
        upgrade)
            check_docker
            upgrade "$2"
            ;;
        status)
            status
            ;;
        init)
            init
            ;;
        clean)
            clean
            ;;
        clean-all)
            clean_all
            ;;
        backup)
            backup
            ;;
        restore)
            restore "$2"
            ;;
        help|--help|-h)
            help
            ;;
        *)
            log_error "未知命令: $1"
            help
            exit 1
            ;;
    esac
}

main "$@"
