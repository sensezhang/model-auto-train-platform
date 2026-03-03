@echo off
echo ================================================
echo YOLO Training Diagnosis Tool
echo ================================================
echo.

set POLARS_SKIP_CPU_CHECK=1
set PYTHONUNBUFFERED=1

call conda activate yolo-train-test-py311
if errorlevel 1 (
    echo Failed to activate conda environment!
    pause
    exit /b 1
)

echo Running diagnosis...
echo.
python diagnose_training.py
