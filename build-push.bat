@echo off
setlocal EnableDelayedExpansion

echo ==========================================
echo   YOLO Platform - Build and Push Images
echo ==========================================
echo.

set "REGISTRY=hub.ikingtec.com/ai"
set "VERSION=v1"
set "PROJECT_DIR=E:\PycharmProjects\yolo-train-test"

if not "%1"=="" set "VERSION=%1"

set "BACKEND_IMAGE=%REGISTRY%/yolo-backend:%VERSION%"
set "FRONTEND_IMAGE=%REGISTRY%/yolo-frontend:%VERSION%"

echo Registry : %REGISTRY%
echo Version  : %VERSION%
echo Backend  : %BACKEND_IMAGE%
echo Frontend : %FRONTEND_IMAGE%
echo.

cd /d "%PROJECT_DIR%"

:: -- 确保 Docker 在 PATH 中（双击运行时可能未继承 Docker Desktop 路径）--
where docker >nul 2>&1
if %errorlevel% neq 0 (
    set "DOCKER_PATHS=C:\Program Files\Docker\Docker\resources\bin;C:\ProgramData\DockerDesktop\version-bin"
    set "PATH=%PATH%;!DOCKER_PATHS!"
    where docker >nul 2>&1
    if !errorlevel! neq 0 (
        echo ERROR: docker command not found.
        echo Please run this script from a CMD or PowerShell window where 'docker' works.
        pause
        exit /b 1
    )
)

echo [1/4] Building backend image...
docker build -f deploy/Dockerfile.backend -t %BACKEND_IMAGE% .
if %errorlevel% neq 0 (
    echo ERROR: Backend build failed
    pause
    exit /b 1
)
echo Backend build OK.
echo.

echo [2/4] Building frontend image...
docker build -f deploy/Dockerfile.frontend -t %FRONTEND_IMAGE% .
if %errorlevel% neq 0 (
    echo ERROR: Frontend build failed
    pause
    exit /b 1
)
echo Frontend build OK.
echo.

echo [3/4] Pushing backend image...
docker push %BACKEND_IMAGE%
if %errorlevel% neq 0 (
    echo ERROR: Backend push failed. Run: docker login hub.ikingtec.com
    pause
    exit /b 1
)
echo Backend push OK.
echo.

echo [4/4] Pushing frontend image...
docker push %FRONTEND_IMAGE%
if %errorlevel% neq 0 (
    echo ERROR: Frontend push failed.
    pause
    exit /b 1
)
echo Frontend push OK.
echo.

echo ==========================================
echo Done!
echo.
echo Server deploy:
echo   cd /opt/yolo-train-test/deploy
echo   docker compose pull
echo   docker compose up -d
echo ==========================================
echo.
pause
