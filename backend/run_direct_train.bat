@echo off
echo ================================================
echo Direct YOLO Training Test
echo ================================================
echo.

set POLARS_SKIP_CPU_CHECK=1
set PYTHONUNBUFFERED=1
set CUDA_LAUNCH_BLOCKING=1

call conda activate yolo-train-test-py311

python direct_train.py
