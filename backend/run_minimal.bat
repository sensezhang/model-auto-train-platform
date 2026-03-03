@echo off
set POLARS_SKIP_CPU_CHECK=1
set PYTHONUNBUFFERED=1
set TQDM_DISABLE=1
call conda activate yolo-train-test-py311
python minimal_test.py
pause
