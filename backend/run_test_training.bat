@echo off
echo Activating conda environment...
call conda activate yolo-train-test-py311
set POLARS_SKIP_CPU_CHECK=1
echo Running YOLO training test...
python test_yolo_training.py
pause
