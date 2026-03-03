"""独立进程运行训练任务，避免阻塞主服务器"""
import sys
import os
import traceback

# 设置环境变量，必须在导入任何库之前设置
os.environ['PYTHONUNBUFFERED'] = '1'  # 禁用输出缓冲

# 加载 .env 文件（子进程可能没有继承完整的环境变量）
try:
    from dotenv import load_dotenv
    _env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.isfile(_env_file):
        load_dotenv(_env_file, override=False)  # 不覆盖父进程传入的环境变量
except ImportError:
    pass

# Windows multiprocessing 修复 - 必须在导入torch/ultralytics之前设置
import multiprocessing
if sys.platform == 'win32':
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # 已经设置过了

# 确保能导入 app 模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.training_yolo import run_training_job

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_training.py <job_id>")
        sys.exit(1)

    job_id = int(sys.argv[1])
    print(f"Starting training job {job_id} in separate process...")

    try:
        run_training_job(job_id)
        print(f"Training job {job_id} completed successfully")
    except Exception as e:
        print(f"Training job {job_id} failed with error: {type(e).__name__}: {e}")
        print("Traceback:")
        traceback.print_exc()
        sys.exit(1)
