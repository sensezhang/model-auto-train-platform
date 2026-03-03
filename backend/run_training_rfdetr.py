#!/usr/bin/env python
import sys
import os

# 禁用输出缓冲
os.environ['PYTHONUNBUFFERED'] = '1'

# 加载 .env 文件（子进程可能没有继承完整的环境变量）
try:
    from dotenv import load_dotenv
    _env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.isfile(_env_file):
        load_dotenv(_env_file, override=False)
except ImportError:
    pass

# 强制使用非交互式 matplotlib 后端，避免子进程中 tkinter 崩溃
# RF-DETR 训练结束后 GC 析构 matplotlib 图形时会触发 Tcl_AsyncDelete 错误
os.environ.setdefault('MPLBACKEND', 'Agg')

# 设置 CUDA 设备（必须在导入 torch 之前）
if 'CUDA_VISIBLE_DEVICES' in os.environ:
    print(f"Using GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")

# 添加backend目录到Python路径
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)

from app.services.training_rfdetr import run_training_job

if __name__ == '__main__':
    job_id = int(sys.argv[1])
    run_training_job(job_id)
