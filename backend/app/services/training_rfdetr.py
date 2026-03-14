"""
RF-DETR 训练服务
- 生成COCO格式数据集
- 调用rfdetr进行训练
- 保存模型和指标
"""

from __future__ import annotations

import os
import json
from typing import Optional, Dict
from datetime import datetime

from ..db import get_session
from ..models import TrainingJob, ModelArtifact
from .export_coco import export_dataset_to_coco


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _log(log_path: str, msg: str):
    ts = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    try:
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f"[{ts}] {msg}\n")
            f.flush()
    except Exception as e:
        try:
            backup_log = log_path + '.error'
            with open(backup_log, 'a', encoding='utf-8') as f:
                f.write(f"[{ts}] ERROR logging to {log_path}: {e}\n")
                f.write(f"[{ts}] Original message: {msg}\n")
        except Exception:
            pass


def _parse_config(job: TrainingJob) -> Dict:
    """解析训练配置参数"""
    config = {}

    # 从modelVariant中提取模型类型
    # 例如: "rfdetr-medium", "rfdetr-large"
    model_variant = job.modelVariant.lower()
    if 'large' in model_variant:
        config['model_type'] = 'large'
    elif 'small' in model_variant:
        config['model_type'] = 'small'
    else:
        config['model_type'] = 'medium'  # default

    # 基本训练参数
    config['epochs'] = job.epochs
    config['batch_size'] = job.batch if job.batch and job.batch > 0 else 16

    # GPU配置
    import torch
    gpu_ids_str = job.gpuIds
    if gpu_ids_str:
        gpu_ids = [int(x.strip()) for x in gpu_ids_str.split(',') if x.strip()]
        config['gpu_ids'] = gpu_ids
        config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        config['gpu_ids'] = None
        config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 学习率相关（可以从扩展配置中读取，这里先用默认值）
    config['lr'] = 0.0001
    config['weight_decay'] = 0.0001

    # 早停策略
    config['early_stopping'] = True
    config['early_stopping_patience'] = 20
    config['early_stopping_min_delta'] = 0.00001

    # 检查点和日志
    config['checkpoint_interval'] = 5
    config['tensorboard'] = True

    return config


def _patch_inference_mode():
    """
    Patch torch.inference_mode to use no_grad instead.
    This is necessary because RF-DETR's DINOv2 backbone uses inference_mode
    during model loading/initialization, which creates tensors that cannot
    be used in training with autograd.
    """
    import torch

    # Store original
    original_inference_mode = torch.inference_mode

    class NoOpInferenceMode:
        """Replace inference_mode with no_grad for training compatibility"""
        def __init__(self, mode=True):
            self.mode = mode
            self._no_grad = torch.no_grad() if mode else None

        def __enter__(self):
            if self._no_grad:
                self._no_grad.__enter__()
            return self

        def __exit__(self, *args):
            if self._no_grad:
                self._no_grad.__exit__(*args)

        def __call__(self, func):
            # Decorator usage - just use no_grad
            if self.mode:
                return torch.no_grad()(func)
            return func

    # Monkey patch
    torch.inference_mode = NoOpInferenceMode
    return original_inference_mode


def _restore_inference_mode(original):
    """Restore original inference_mode"""
    import torch
    torch.inference_mode = original


def _fix_inference_tensors(model):
    """
    Fix inference tensors in model parameters and buffers.
    RF-DETR's DINOv2 backbone may create tensors in inference mode
    which cannot be used in training.
    """
    import torch

    # Fix parameters
    for name, param in model.named_parameters():
        if hasattr(param, 'is_inference') and param.is_inference():
            # Create a new parameter with cloned data
            new_data = param.data.clone().detach()
            param.data = new_data

    # Fix buffers
    for name, buf in model.named_buffers():
        if buf is not None and hasattr(buf, 'is_inference') and buf.is_inference():
            # Clone the buffer
            new_buf = buf.clone().detach()
            # Re-register the buffer
            parts = name.split('.')
            module = model
            for part in parts[:-1]:
                module = getattr(module, part)
            setattr(module, parts[-1], new_buf)


def _train_with_rfdetr(
    dataset_dir: str,
    output_dir: str,
    config: Dict,
    log_path: Optional[str] = None,
    job_id: Optional[int] = None,
    resume_checkpoint: Optional[str] = None
) -> Dict:
    """使用RF-DETR进行训练"""
    import torch

    if log_path:
        _log(log_path, f"Entering _train_with_rfdetr function")
        _log(log_path, f"Dataset dir: {dataset_dir}")
        _log(log_path, f"Output dir: {output_dir}")
        _log(log_path, f"Config: {config}")

    # 设置GPU环境变量
    gpu_ids = config.get('gpu_ids')
    if gpu_ids:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in gpu_ids)
        if log_path:
            _log(log_path, f"Set CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

    # 在导入 rfdetr 之前应用补丁，避免 inference mode 问题
    if log_path:
        _log(log_path, "Patching torch.inference_mode for training compatibility...")
    original_inference_mode = _patch_inference_mode()

    try:
        if log_path:
            _log(log_path, "Importing rfdetr...")

        from rfdetr import RFDETRSmall, RFDETRMedium, RFDETRLarge

        if log_path:
            _log(log_path, "rfdetr imported successfully")
    except Exception as e:
        _restore_inference_mode(original_inference_mode)
        if log_path:
            _log(log_path, f"Failed to import rfdetr: {e}")
        raise RuntimeError("rfdetr is not installed. Please install 'rfdetr' package.") from e

    # 选择模型
    model_type = config.get('model_type', 'medium')
    if log_path:
        _log(log_path, f"Loading RF-DETR model: {model_type}")

    # 检查本地预训练权重
    pretrained_dir = "/app/pretrained"
    local_weights = {
        'small': os.path.join(pretrained_dir, 'rf-detr-small.pth'),
        'medium': os.path.join(pretrained_dir, 'rf-detr-medium.pth'),
        'large': os.path.join(pretrained_dir, 'rf-detr-large.pth'),
    }

    pretrain_weights = None
    if model_type in local_weights and os.path.exists(local_weights[model_type]):
        pretrain_weights = local_weights[model_type]
        if log_path:
            _log(log_path, f"Using local pretrained weights: {pretrain_weights}")
    else:
        if log_path:
            _log(log_path, f"Local weights not found, will download from HuggingFace")

    # 确保在非 inference mode 下创建模型
    # RF-DETR 的 DINOv2 backbone 可能在 inference mode 下创建张量
    with torch.no_grad():
        if model_type == 'small':
            model = RFDETRSmall(pretrain_weights=pretrain_weights)
        elif model_type == 'large':
            model = RFDETRLarge(pretrain_weights=pretrain_weights)
        else:
            model = RFDETRMedium(pretrain_weights=pretrain_weights)

    # 修复可能存在的 inference tensors
    if log_path:
        _log(log_path, "Fixing potential inference tensors for training...")
    try:
        _fix_inference_tensors(model.model)
        if log_path:
            _log(log_path, "Inference tensors fixed successfully")
    except Exception as e:
        if log_path:
            _log(log_path, f"Warning: Could not fix inference tensors: {e}")

    if log_path:
        _log(log_path, f"Model loaded: {model.__class__.__name__}")

    # 准备训练参数
    # 注意：在 Docker 容器中，设置 num_workers=0 避免多进程 DataLoader 问题
    # 多进程 DataLoader 可能导致 "Numpy is not available" 或共享内存不足错误
    train_kwargs = {
        'dataset_dir': dataset_dir,
        'device': config.get('device', 'cuda'),
        'output_dir': output_dir,
        'checkpoint_interval': config.get('checkpoint_interval', 5),
        'tensorboard': config.get('tensorboard', True),
        'num_workers': 0,  # Docker 容器内使用 0 避免多进程问题
    }

    # 添加可选参数
    if config.get('early_stopping'):
        train_kwargs['early_stopping'] = True
        train_kwargs['early_stopping_patience'] = config.get('early_stopping_patience', 15)
        train_kwargs['early_stopping_min_delta'] = config.get('early_stopping_min_delta', 0.0001)
        train_kwargs['early_stopping_use_ema'] = True

    # 恢复训练
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        train_kwargs['resume'] = resume_checkpoint
        if log_path:
            _log(log_path, f"Resuming from checkpoint: {resume_checkpoint}")

    if log_path:
        _log(log_path, f"Starting training with config: {train_kwargs}")

    # 开始训练
    try:
        results = model.train(**train_kwargs)

        if log_path:
            _log(log_path, "Training completed successfully")

        # 提取训练指标
        metrics = {}
        if hasattr(results, 'best_metrics'):
            metrics = results.best_metrics

        # 查找最佳权重文件
        best_checkpoint = None
        checkpoint_patterns = [
            os.path.join(output_dir, 'checkpoint_best.pth'),
            os.path.join(output_dir, 'checkpoint_best_ema.pth'),
            os.path.join(output_dir, 'best.pth'),
        ]
        for pattern in checkpoint_patterns:
            if os.path.exists(pattern):
                best_checkpoint = pattern
                break

        return {
            'metrics': metrics,
            'best_checkpoint': best_checkpoint,
            'model': model
        }

    except Exception as e:
        if log_path:
            _log(log_path, f"Training failed: {e}")
        raise
    finally:
        # 恢复原始的 inference_mode
        _restore_inference_mode(original_inference_mode)
        if log_path:
            _log(log_path, "Restored original torch.inference_mode")


def run_training_job(job_id: int):
    """执行RF-DETR训练任务"""
    # 加载任务
    with get_session() as session:
        job = session.get(TrainingJob, job_id)
        if not job:
            return
        project_id = job.projectId
        seed = job.seed
        job.status = 'running'
        job.startedAt = job.startedAt or datetime.utcnow()
        session.add(job)
        session.commit()

    # 准备输出目录
    out_dir = os.path.join(os.getcwd(), 'models', str(project_id), str(job_id))
    _ensure_dir(out_dir)
    log_path = os.path.join(out_dir, 'logs.txt')
    rel_logs = os.path.relpath(log_path, os.getcwd())

    with get_session() as session:
        job = session.get(TrainingJob, job_id)
        job.logsRef = rel_logs
        session.add(job)
        session.commit()

    _log(log_path, f"Job {job_id} started (RF-DETR training)")

    # 准备COCO格式数据集
    try:
        dataset_dir = os.path.join(os.getcwd(), 'datasets_coco', str(project_id), str(job_id))
        _ensure_dir(dataset_dir)

        _log(log_path, f"Exporting dataset to COCO format at {dataset_dir}")

        # 导出COCO格式数据集
        export_result = export_dataset_to_coco(
            project_id=project_id,
            output_dir=dataset_dir,
            seed=seed,
            train_ratio=0.8,
            val_ratio=0.2,
            test_ratio=0.0  # RF-DETR通常只用train和valid
        )

        _log(
            log_path,
            f"Dataset prepared: {export_result['total_images']} images, "
            f"{export_result['total_annotations']} annotations, "
            f"{export_result['total_classes']} classes"
        )

    except Exception as e:
        with get_session() as session:
            job = session.get(TrainingJob, job_id)
            job.status = 'failed'
            session.add(job)
            session.commit()
        _log(log_path, f"Dataset preparation failed: {e}")
        return

    # 解析配置
    with get_session() as session:
        job = session.get(TrainingJob, job_id)
        config = _parse_config(job)

    # 训练
    try:
        _log(log_path, f"Starting RF-DETR training with config: {config}")

        res = _train_with_rfdetr(
            dataset_dir=dataset_dir,
            output_dir=out_dir,
            config=config,
            log_path=log_path,
            job_id=job_id
        )

        _log(log_path, f"Training completed: {list(res.keys()) if res else 'None'}")

        metrics = res.get('metrics', {})
        best_checkpoint = res.get('best_checkpoint')

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        with get_session() as session:
            job = session.get(TrainingJob, job_id)
            job.status = 'failed'
            job.finishedAt = datetime.utcnow()
            session.add(job)
            session.commit()
        _log(log_path, f"Training failed: {type(e).__name__}: {e}")
        _log(log_path, f"Traceback:\n{tb}")
        return

    # 保存指标
    with get_session() as session:
        job = session.get(TrainingJob, job_id)
        # RF-DETR的指标名称可能不同，需要适配
        job.map50 = metrics.get('mAP50', metrics.get('map50'))
        job.map50_95 = metrics.get('mAP', metrics.get('map50_95'))
        job.precision = metrics.get('precision')
        job.recall = metrics.get('recall')
        session.add(job)
        session.commit()

    # 保存模型文件
    if best_checkpoint and os.path.isfile(best_checkpoint):
        try:
            size = os.path.getsize(best_checkpoint)
        except Exception:
            size = None
        rel = os.path.relpath(best_checkpoint, os.getcwd())
        artifact_path = rel
        with get_session() as session:
            session.add(ModelArtifact(
                trainingJobId=job_id,
                format='pth',
                path=artifact_path,
                size=size
            ))
            session.commit()
        _log(log_path, f"Saved best checkpoint {artifact_path}")

    # 完成
    with get_session() as session:
        job = session.get(TrainingJob, job_id)
        if job and job.status == 'canceled':
            job.finishedAt = datetime.utcnow()
            session.add(job)
            session.commit()
            _log(log_path, "Job canceled")
            return
        if job:
            job.status = 'succeeded'
            job.finishedAt = datetime.utcnow()
            session.add(job)
            session.commit()

    _log(log_path, "Job succeeded")


def start_training_async(job_id: int):
    """在独立进程中启动RF-DETR训练"""
    import subprocess
    import sys

    from ..db import DB_TYPE, DB_PATH, APP_DB_URL, get_session
    from ..models import TrainingJob

    # 获取 GPU 配置
    gpu_ids_str = None
    with get_session() as session:
        job = session.get(TrainingJob, job_id)
        if job and job.gpuIds:
            gpu_ids_str = job.gpuIds

    python_exe = sys.executable
    backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    script_path = os.path.join(backend_dir, "run_training_rfdetr.py")

    # 确保脚本存在
    if not os.path.exists(script_path):
        # 创建运行脚本
        script_content = """#!/usr/bin/env python
import sys
import os

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
"""
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)

    # 日志目录
    log_dir = os.path.join(backend_dir, "models", "training_logs")
    os.makedirs(log_dir, exist_ok=True)
    process_log = os.path.join(log_dir, f"rfdetr_process_{job_id}.log")

    # 启动进程
    creationflags = 0
    if sys.platform == 'win32':
        creationflags = subprocess.CREATE_NO_WINDOW

    env = os.environ.copy()
    if DB_TYPE == "mysql":
        env['APP_DB_URL'] = APP_DB_URL
    else:
        env['APP_DB_PATH'] = str(DB_PATH)

    # 在进程启动时就设置 GPU（必须在导入 torch 之前生效）
    if gpu_ids_str:
        env['CUDA_VISIBLE_DEVICES'] = gpu_ids_str

    log_f = open(process_log, 'w', encoding='utf-8', buffering=1)
    log_f.write(f"Starting RF-DETR training job {job_id} at {datetime.now()}\n")
    db_info = APP_DB_URL if DB_TYPE == "mysql" else f"sqlite:///{DB_PATH}"
    log_f.write(f"Database: {db_info}\n")
    log_f.write(f"Working directory: {backend_dir}\n")
    log_f.flush()

    process = subprocess.Popen(
        [python_exe, script_path, str(job_id)],
        stdout=log_f,
        stderr=subprocess.STDOUT,
        creationflags=creationflags,
        cwd=backend_dir,
        env=env,
        close_fds=False
    )

    return process
