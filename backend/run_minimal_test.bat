@echo off
set POLARS_SKIP_CPU_CHECK=1
call conda activate yolo-train-test-py311
echo Starting minimal YOLO training test...
python _train_worker3.py > train3_output.log 2>&1
echo Done. Check train3_output.log for results.
