@echo off
chcp 65001 >nul 2>&1
setlocal EnableDelayedExpansion

echo ========================================
echo   YOLO Training Platform - Pack Script
echo ========================================
echo.

set "PROJECT_DIR=E:\PycharmProjects"
set "PROJECT_NAME=yolo-train-test"
set "OUTPUT_FILE=%PROJECT_DIR%\%PROJECT_NAME%-deploy.tar.gz"

cd /d "%PROJECT_DIR%"

echo Packing project (code and config only)...
echo.

tar -czvf "%OUTPUT_FILE%" --exclude="node_modules" --exclude=".venv" --exclude="venv" --exclude="__pycache__" --exclude=".git" --exclude=".idea" --exclude=".claude" --exclude=".ultralytics" --exclude="datasets" --exclude="models" --exclude="data" --exclude="output" --exclude="deploy/data" --exclude="datasets_coco" --exclude="*.db" --exclude="*.pth" --exclude="*.pt" --exclude="*.onnx" --exclude="*.log" --exclude="*.tar.gz" --exclude="nul" --exclude=".cache" --exclude=".torch" "%PROJECT_NAME%"

echo.
echo ========================================
echo Done!
echo Output: %OUTPUT_FILE%
echo.
echo Deploy steps:
echo   1. Upload: scp %PROJECT_NAME%-deploy.tar.gz user@server:/opt/
echo   2. Extract: cd /opt ^&^& tar -xzvf %PROJECT_NAME%-deploy.tar.gz
echo   3. Start: cd %PROJECT_NAME%/deploy ^&^& ./deploy.sh start
echo ========================================
echo.
pause
