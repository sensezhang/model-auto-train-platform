@echo off
set POLARS_SKIP_CPU_CHECK=1
set PYTHONUNBUFFERED=1
call conda activate yolo-train-test-py311
python cpu_test.py
pause
