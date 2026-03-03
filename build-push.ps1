param(
    [string]$Version = "v1"
)

$REGISTRY = "hub.ikingtec.com/ai"
$PROJECT_DIR = "E:\PycharmProjects\yolo-train-test"
$BACKEND_IMAGE = "$REGISTRY/yolo-backend:$Version"
$FRONTEND_IMAGE = "$REGISTRY/yolo-frontend:$Version"

Write-Host "=========================================="
Write-Host "  YOLO Platform - Build and Push Images"
Write-Host "=========================================="
Write-Host ""
Write-Host "Registry : $REGISTRY"
Write-Host "Version  : $Version"
Write-Host "Backend  : $BACKEND_IMAGE"
Write-Host "Frontend : $FRONTEND_IMAGE"
Write-Host ""

Set-Location $PROJECT_DIR

# [1/4] Build backend
Write-Host "[1/4] Building backend image..."
docker build -f deploy/Dockerfile.backend -t $BACKEND_IMAGE .
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Backend build failed" -ForegroundColor Red
    exit 1
}
Write-Host "Backend build OK." -ForegroundColor Green
Write-Host ""

# [2/4] Build frontend
Write-Host "[2/4] Building frontend image..."
docker build -f deploy/Dockerfile.frontend -t $FRONTEND_IMAGE .
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Frontend build failed" -ForegroundColor Red
    exit 1
}
Write-Host "Frontend build OK." -ForegroundColor Green
Write-Host ""

# [3/4] Push backend
Write-Host "[3/4] Pushing backend image..."
docker push $BACKEND_IMAGE
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Backend push failed. Run: docker login hub.ikingtec.com" -ForegroundColor Red
    exit 1
}
Write-Host "Backend push OK." -ForegroundColor Green
Write-Host ""

# [4/4] Push frontend
Write-Host "[4/4] Pushing frontend image..."
docker push $FRONTEND_IMAGE
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Frontend push failed." -ForegroundColor Red
    exit 1
}
Write-Host "Frontend push OK." -ForegroundColor Green
Write-Host ""

Write-Host "=========================================="
Write-Host "Done!"
Write-Host ""
Write-Host "Server deploy:"
Write-Host "  cd /opt/yolo-train-test/deploy"
Write-Host "  docker compose pull"
Write-Host "  docker compose up -d"
Write-Host "=========================================="
