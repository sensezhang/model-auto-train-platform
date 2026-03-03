@echo off
set POLARS_SKIP_CPU_CHECK=1
set PYTHONUNBUFFERED=1
call conda activate yolo-train-test-py311
echo Starting YOLO training with unbuffered output...
python -u _train_worker3.py
echo Done with exit code %ERRORLEVEL%
pause
