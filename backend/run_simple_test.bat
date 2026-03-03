@echo off
echo ================================================
echo Simple YOLO Training Test
echo ================================================
echo.

set POLARS_SKIP_CPU_CHECK=1
set PYTHONUNBUFFERED=1

call conda activate yolo-train-test-py311

echo Running test...
python simple_train_test.py

echo.
echo ================================================
echo Test completed. Check output above.
echo ================================================
pause
