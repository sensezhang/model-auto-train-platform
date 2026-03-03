@echo off
echo ================================================
echo Fixed YOLO Training for Windows
echo ================================================
echo.

set POLARS_SKIP_CPU_CHECK=1
set PYTHONUNBUFFERED=1

call conda activate yolo-train-test-py311

python fix_train.py

pause
