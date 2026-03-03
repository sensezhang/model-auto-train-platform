#!/bin/bash
# ==========================================
#  YOLO Platform - 直接启动（不用 Docker）
#  用法:
#    ./start-direct.sh setup   # 首次安装依赖
#    ./start-direct.sh start   # 启动服务
#    ./start-direct.sh stop    # 停止服务
#    ./start-direct.sh restart # 重启服务
#    ./start-direct.sh status  # 查看状态
#    ./start-direct.sh logs    # 查看后端日志
# ==========================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BACKEND_DIR="$PROJECT_DIR/backend"
FRONTEND_DIR="$PROJECT_DIR/frontend"
VENV_DIR="$BACKEND_DIR/.venv"
PID_FILE="/tmp/yolo-backend.pid"
LOG_FILE="$BACKEND_DIR/logs/backend.log"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# ── 安装所有依赖（首次运行） ────────────────────────────────
setup() {
    info "=== 安装依赖 ==="

    # Python venv
    info "创建 Python 虚拟环境..."
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"

    info "安装 Python 依赖..."
    pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
    pip install -r "$BACKEND_DIR/requirements.txt" \
        -i https://pypi.tuna.tsinghua.edu.cn/simple \
        --trusted-host pypi.tuna.tsinghua.edu.cn

    # numpy 1.x（rfdetr 不兼容 2.x）
    pip install --force-reinstall "numpy>=1.24.0,<2.0" \
        -i https://pypi.tuna.tsinghua.edu.cn/simple

    # rfdetr
    info "安装 RF-DETR..."
    pip install rfdetr \
        -i https://pypi.tuna.tsinghua.edu.cn/simple || \
    pip install git+https://github.com/roboflow/rf-detr.git

    deactivate

    # Node / 前端
    info "安装前端依赖并构建..."
    cd "$FRONTEND_DIR"
    npm install --registry https://registry.npmmirror.com
    npm run build
    cd "$SCRIPT_DIR"

    # 配置 nginx
    setup_nginx

    # 创建必要目录
    mkdir -p "$BACKEND_DIR/logs" "$BACKEND_DIR/datasets" \
             "$BACKEND_DIR/datasets_coco" "$BACKEND_DIR/models" \
             "$BACKEND_DIR/output" "$BACKEND_DIR/pretrained"

    info "=== 安装完成，运行 ./start-direct.sh start 启动服务 ==="
}

# ── 生成并安装 nginx 配置 ───────────────────────────────────
setup_nginx() {
    info "配置 nginx..."
    DIST_DIR="$FRONTEND_DIR/dist"

    NGINX_CONF="/etc/nginx/conf.d/yolo-platform.conf"
    sudo tee "$NGINX_CONF" > /dev/null <<EOF
server {
    listen 80;
    server_name _;

    client_max_body_size 2G;
    client_body_timeout 600s;
    client_header_timeout 600s;

    # 前端静态文件
    location / {
        root $DIST_DIR;
        index index.html;
        try_files \$uri \$uri/ /index.html;
    }

    # API 代理到后端
    location /api {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_request_buffering off;
        proxy_buffering off;
        proxy_connect_timeout 600;
        proxy_send_timeout 600;
        proxy_read_timeout 600;
        client_max_body_size 2G;
    }

    # 静态文件代理
    location /files {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
    }
}
EOF

    sudo nginx -t && sudo systemctl reload nginx
    info "nginx 配置完成"
}

# ── 启动后端 ────────────────────────────────────────────────
start_backend() {
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        warn "后端已在运行（PID: $(cat "$PID_FILE")）"
        return
    fi

    mkdir -p "$BACKEND_DIR/logs"
    source "$VENV_DIR/bin/activate"
    cd "$BACKEND_DIR"

    nohup uvicorn app.main:app \
        --host 0.0.0.0 \
        --port 8000 \
        --workers 1 \
        >> "$LOG_FILE" 2>&1 &

    echo $! > "$PID_FILE"
    deactivate
    info "后端已启动（PID: $(cat "$PID_FILE")，日志: $LOG_FILE）"
}

# ── 停止后端 ────────────────────────────────────────────────
stop_backend() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            kill "$PID"
            info "后端已停止（PID: $PID）"
        else
            warn "进程 $PID 已不存在"
        fi
        rm -f "$PID_FILE"
    else
        warn "未找到 PID 文件，后端可能未运行"
    fi
}

# ── 查看状态 ────────────────────────────────────────────────
status() {
    echo ""
    echo "=== 后端状态 ==="
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        info "后端运行中（PID: $(cat "$PID_FILE")）"
    else
        warn "后端未运行"
    fi

    echo ""
    echo "=== nginx 状态 ==="
    systemctl is-active nginx && info "nginx 运行中" || warn "nginx 未运行"

    echo ""
    echo "=== GPU 状态 ==="
    if command -v nvidia-smi &>/dev/null; then
        nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader
    else
        warn "未检测到 NVIDIA GPU"
    fi

    echo ""
    SERVER_IP=$(hostname -I | awk '{print $1}')
    info "访问地址: http://$SERVER_IP"
}

# ── 主入口 ──────────────────────────────────────────────────
case "${1:-help}" in
    setup)
        setup
        ;;
    start)
        start_backend
        sudo systemctl start nginx 2>/dev/null || true
        status
        ;;
    stop)
        stop_backend
        ;;
    restart)
        stop_backend
        sleep 1
        start_backend
        ;;
    status)
        status
        ;;
    logs)
        tail -f "$LOG_FILE"
        ;;
    nginx)
        setup_nginx
        ;;
    *)
        echo "用法: $0 {setup|start|stop|restart|status|logs|nginx}"
        ;;
esac
