@echo off
echo ========================================
echo  YOLO Training Platform - Upgrade Script
echo ========================================
echo.

echo [Step 1/3] Running database migrations...
cd backend
call ..\.venv\Scripts\python migrate_add_framework.py
if errorlevel 1 (
    echo Migration failed! Please check the error above.
    pause
    exit /b 1
)

call ..\.venv\Scripts\python migrate_add_gpu_ids.py
if errorlevel 1 (
    echo Migration failed! Please check the error above.
    pause
    exit /b 1
)

echo.
echo [Step 2/3] Checking dependencies...
call ..\.venv\Scripts\pip show onnx >nul 2>&1
if errorlevel 1 (
    echo Installing ONNX...
    call ..\.venv\Scripts\pip install onnx onnxruntime
)

echo.
echo [Step 3/3] Checking PyTorch CUDA support...
call ..\.venv\Scripts\python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
if errorlevel 1 (
    echo WARNING: PyTorch CUDA not available!
    echo You can install CUDA version with:
    echo    .venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    echo.
    echo Continue with CPU version? (y/n^)
    set /p choice=
    if /i not "%choice%"=="y" exit /b 1
)

cd ..

echo.
echo ========================================
echo  Upgrade completed successfully!
echo ========================================
echo.
echo Next steps:
echo 1. Start backend:  cd backend ^&^& ..\.venv\Scripts\uvicorn app.main:app --reload
echo 2. Start frontend: cd frontend ^&^& npm run dev
echo 3. Open browser:   http://localhost:5173
echo.
echo For multi-GPU training, see: MULTI_GPU_SETUP.md
echo For deployment guide, see:   DEPLOYMENT_GUIDE.md
echo.
pause
