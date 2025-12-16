"""测试训练脚本是否能独立运行"""
import subprocess
import sys
import os

# 获取 Python 可执行文件
python_exe = sys.executable
print(f"Python executable: {python_exe}")

# 测试运行 run_training.py
backend_dir = os.path.join(os.getcwd(), "backend")
script_path = os.path.join(backend_dir, "run_training.py")
print(f"Script path: {script_path}")
print(f"Script exists: {os.path.exists(script_path)}")

# 尝试启动一个测试进程（使用一个不存在的 job_id 来快速失败）
print("\n测试启动独立进程...")
try:
    result = subprocess.run(
        [python_exe, script_path, "999"],
        cwd=backend_dir,
        capture_output=True,
        text=True,
        timeout=5
    )
    print(f"返回码: {result.returncode}")
    print(f"输出: {result.stdout}")
    print(f"错误: {result.stderr}")
except subprocess.TimeoutExpired:
    print("进程超时（可能正常，如果训练开始了）")
except Exception as e:
    print(f"错误: {e}")
