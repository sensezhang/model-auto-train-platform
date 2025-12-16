"""独立进程运行训练任务，避免阻塞主服务器"""
import sys
import os
import traceback

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
