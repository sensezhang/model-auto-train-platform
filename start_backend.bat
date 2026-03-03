@echo off
REM Start backend service with CUDA-enabled PyTorch

REM Initialize conda
call D:\miniconda3\Scripts\activate.bat D:\miniconda3
call conda activate yolo-train-test-py311

REM Show environment info
echo ========================================
echo Current Python:
python --version
where python
echo PyTorch CUDA check:
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
echo ========================================

cd /d E:\PycharmProjects\yolo-train-test\backend

REM Start without reload to avoid subprocess Python version issues
C:\Users\YUHGT\.conda\envs\yolo-train-test-py311\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
