#!/bin/bash
# ==========================================
#  Model Auto Train Platform - Server Build & Push
#  用法: ./build-push.sh [版本号]
#  示例: ./build-push.sh v2
# ==========================================
set -e

VERSION=${1:-v1}
REGISTRY="hub.ikingtec.com/ai"
BACKEND_IMAGE="$REGISTRY/model-train-backend:$VERSION"
FRONTEND_IMAGE="$REGISTRY/model-train-frontend:$VERSION"

# 脚本所在目录的上一级即项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=========================================="
echo "  Model Auto Train Platform - Build and Push Images"
echo "=========================================="
echo ""
echo "Registry : $REGISTRY"
echo "Version  : $VERSION"
echo "Backend  : $BACKEND_IMAGE"
echo "Frontend : $FRONTEND_IMAGE"
echo "Project  : $PROJECT_DIR"
echo ""

cd "$PROJECT_DIR"

# [1/4] 构建 backend
echo "[1/4] Building backend image..."
docker build -f deploy/Dockerfile.backend -t "$BACKEND_IMAGE" .
echo "Backend build OK."
echo ""

# [2/4] 构建 frontend
echo "[2/4] Building frontend image..."
docker build -f deploy/Dockerfile.frontend -t "$FRONTEND_IMAGE" .
echo "Frontend build OK."
echo ""

# [3/4] 推送 backend
echo "[3/4] Pushing backend image..."
docker push "$BACKEND_IMAGE"
echo "Backend push OK."
echo ""

# [4/4] 推送 frontend
echo "[4/4] Pushing frontend image..."
docker push "$FRONTEND_IMAGE"
echo "Frontend push OK."
echo ""

# 推完删掉本地镜像，释放服务器磁盘
echo "Cleaning up local images..."
docker image rm "$BACKEND_IMAGE" "$FRONTEND_IMAGE" 2>/dev/null || true
docker builder prune -f
echo "Cleanup done."
echo ""

echo "=========================================="
echo "Done! Images pushed:"
echo "  $BACKEND_IMAGE"
echo "  $FRONTEND_IMAGE"
echo ""
echo "Deploy on target server:"
echo "  cd /opt/model-auto-train-platform/deploy"
echo "  docker compose pull"
echo "  docker compose up -d"
echo "=========================================="
