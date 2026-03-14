from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
from urllib.parse import quote
import os
import asyncio

from ..db import get_session
from ..models import TrainingJob, Project, Image, Annotation, ModelArtifact
from ..services import training_yolo, training_rfdetr
from ..services.export_model import export_model_to_onnx
from ..services.metrics_parser import get_latest_metrics


router = APIRouter(tags=["training"])


class TrainJobCreate(BaseModel):
    projectId: int
    framework: str = "yolo"  # yolo | rfdetr
    modelVariant: str = "yolov11n"
    epochs: int = 50
    imgsz: int = 640
    batch: Optional[int] = None
    seed: int = 42
    gpuIds: Optional[str] = None  # 逗号分隔的GPU ID列表，如 "0,1,2"


@router.post("/training/jobs", response_model=TrainingJob)
def create_training_job(body: TrainJobCreate):
    with get_session() as session:
        proj = session.get(Project, body.projectId)
        if not proj:
            raise HTTPException(404, "Project not found")

        # Enforce single pending/running job per project to avoid resource contention
        existing = (
            session.query(TrainingJob)
            .filter(TrainingJob.projectId == body.projectId)
            .filter(TrainingJob.status.in_(["pending", "running"]))
            .first()
        )
        if existing:
            raise HTTPException(409, "This project already has a pending or running training job")
        # Check annotated images >= 50
        annotated_ids = [row.id for row in session.query(Image.id).filter(Image.projectId == body.projectId).all()]
        if not annotated_ids:
            raise HTTPException(400, "No images in project")
        count_imgs_with_anns = (
            session.query(Annotation.imageId)
            .filter(Annotation.imageId.in_(annotated_ids))
            .distinct()
            .count()
        )
        if count_imgs_with_anns < 50:
            raise HTTPException(400, f"Need at least 50 annotated images, got {count_imgs_with_anns}")

        job = TrainingJob(**body.dict(), status="pending")
        session.add(job)
        session.commit()
        session.refresh(job)
        # fire and forget - 根据framework选择训练服务
        if job.framework == "rfdetr":
            training_rfdetr.start_training_async(job.id)
        else:
            training_yolo.start_training_async(job.id)
        return job


@router.get("/training/jobs", response_model=List[TrainingJob])
def list_training_jobs():
    """获取所有训练任务"""
    with get_session() as session:
        jobs = session.query(TrainingJob).order_by(TrainingJob.id.desc()).all()
        return jobs


@router.get("/training/jobs/{job_id}", response_model=TrainingJob)
def get_training_job(job_id: int):
    with get_session() as session:
        job = session.get(TrainingJob, job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        return job


@router.get("/training/jobs/{job_id}/artifacts", response_model=List[ModelArtifact])
def list_artifacts(job_id: int):
    with get_session() as session:
        job = session.get(TrainingJob, job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        arts = session.query(ModelArtifact).filter(ModelArtifact.trainingJobId == job_id).all()
        return arts


@router.post("/training/jobs/{job_id}/cancel")
def cancel_training(job_id: int):
    with get_session() as session:
        job = session.get(TrainingJob, job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        if job.status in ("succeeded", "failed", "canceled"):
            return {"ok": True, "status": job.status}
        job.status = 'canceled'
        session.add(job)
        session.commit()
        return {"ok": True, "status": job.status}


@router.delete("/training/jobs/{job_id}")
def delete_training_job(job_id: int):
    """删除已完成/失败/已取消的训练任务及其关联数据"""
    with get_session() as session:
        job = session.get(TrainingJob, job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        if job.status in ("pending", "running"):
            raise HTTPException(400, "Cannot delete a pending or running job, cancel it first")
        # 删除关联 artifacts
        session.query(ModelArtifact).filter(ModelArtifact.trainingJobId == job_id).delete()
        session.delete(job)
        session.commit()
        return {"ok": True}


@router.post("/training/jobs/cleanup-stuck")
def cleanup_stuck_jobs():
    """清理卡住的训练任务（状态为 pending 或 running 但实际已停止的任务）"""
    with get_session() as session:
        stuck_jobs = (
            session.query(TrainingJob)
            .filter(TrainingJob.status.in_(["pending", "running"]))
            .all()
        )
        count = len(stuck_jobs)
        for job in stuck_jobs:
            job.status = "failed"
            job.finishedAt = datetime.utcnow()
            session.add(job)
        session.commit()
        return {"ok": True, "cleaned": count, "message": f"Cleaned up {count} stuck job(s)"}


@router.post("/training/jobs/{job_id}/export-onnx")
def export_job_to_onnx(job_id: int, artifact_id: int, simplify: bool = False):
    """
    导出训练好的模型为ONNX格式

    Args:
        job_id: 训练任务ID
        artifact_id: 模型文件的artifact ID (PT或PTH格式)
        simplify: 是否简化ONNX模型（仅RF-DETR，需要onnx-simplifier）

    Returns:
        导出结果
    """
    result = export_model_to_onnx(job_id, artifact_id, simplify)
    if result.get("success"):
        return result
    else:
        raise HTTPException(400, result.get("error", "Export failed"))


@router.get("/training/jobs/{job_id}/metrics")
def get_training_metrics(job_id: int, last_n: int = 50):
    """
    获取训练任务的实时指标

    Args:
        job_id: 训练任务ID
        last_n: 返回最近N个epoch的数据

    Returns:
        指标列表，包含epoch、loss、mAP等信息
    """
    with get_session() as session:
        job = session.get(TrainingJob, job_id)
        if not job:
            raise HTTPException(404, "Job not found")

        if not job.logsRef:
            return []

        log_path = os.path.join(os.getcwd(), job.logsRef)
        if not os.path.exists(log_path):
            return []

        metrics = get_latest_metrics(log_path, job.framework, last_n)
        return metrics


@router.get("/training/jobs/{job_id}/logs/stream")
async def stream_logs(job_id: int):
    with get_session() as session:
        job = session.get(TrainingJob, job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        if not job.logsRef:
            # no logs yet, still stream heartbeat
            log_path = None
        else:
            log_path = os.path.join(os.getcwd(), job.logsRef)

    async def event_gen():
        offset = 0
        while True:
            if log_path and os.path.exists(log_path):
                try:
                    with open(log_path, 'r', encoding='utf-8') as f:
                        f.seek(offset)
                        chunk = f.read()
                        offset = f.tell()
                        if chunk:
                            # SSE format
                            for line in chunk.splitlines():
                                yield f"data: {line}\n\n"
                except Exception:
                    pass
            else:
                yield "data: [waiting logs]\n\n"
            await asyncio.sleep(1.0)

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@router.get("/training/jobs/{job_id}/result-image")
def get_training_result_image(job_id: int):
    """
    获取训练结果图片（metrics_plot.png 或 results.png）

    Args:
        job_id: 训练任务ID

    Returns:
        训练结果图片
    """
    with get_session() as session:
        job = session.get(TrainingJob, job_id)
        if not job:
            raise HTTPException(404, "Job not found")

        # 只有训练成功的任务才有结果图片
        if job.status != 'succeeded':
            raise HTTPException(404, "Training not completed or failed")

        project_id = job.projectId

    # 构建可能的图片路径
    base_dir = os.path.join(os.getcwd(), 'models', str(project_id), str(job_id))

    # 检查多个可能的图片位置
    possible_paths = [
        # RF-DETR: 直接在训练目录下
        os.path.join(base_dir, 'metrics_plot.png'),
        # YOLO: 在train子目录下
        os.path.join(base_dir, 'train', 'results.png'),
        os.path.join(base_dir, 'train', 'confusion_matrix.png'),
    ]

    for img_path in possible_paths:
        if os.path.isfile(img_path):
            return FileResponse(
                path=img_path,
                media_type='image/png',
                filename=os.path.basename(img_path)
            )

    raise HTTPException(404, "Result image not found")


@router.get("/training/jobs/{job_id}/result-images")
def list_training_result_images(job_id: int):
    """
    列出训练任务的所有结果图片

    Args:
        job_id: 训练任务ID

    Returns:
        可用的图片列表
    """
    with get_session() as session:
        job = session.get(TrainingJob, job_id)
        if not job:
            raise HTTPException(404, "Job not found")

        project_id = job.projectId

    base_dir = os.path.join(os.getcwd(), 'models', str(project_id), str(job_id))

    # 查找所有PNG图片
    images = []
    search_dirs = [base_dir, os.path.join(base_dir, 'train')]

    for search_dir in search_dirs:
        if os.path.isdir(search_dir):
            for filename in os.listdir(search_dir):
                if filename.endswith('.png'):
                    images.append({
                        'name': filename,
                        'path': f'/api/training/jobs/{job_id}/image/{filename}'
                    })

    return images


@router.get("/training/jobs/{job_id}/image/{filename}")
def get_training_image(job_id: int, filename: str):
    """
    获取指定的训练图片

    Args:
        job_id: 训练任务ID
        filename: 图片文件名

    Returns:
        图片文件
    """
    # 安全检查：防止路径遍历
    if '..' in filename or '/' in filename or '\\' in filename:
        raise HTTPException(400, "Invalid filename")

    with get_session() as session:
        job = session.get(TrainingJob, job_id)
        if not job:
            raise HTTPException(404, "Job not found")

        project_id = job.projectId

    base_dir = os.path.join(os.getcwd(), 'models', str(project_id), str(job_id))

    # 检查多个可能的位置
    possible_paths = [
        os.path.join(base_dir, filename),
        os.path.join(base_dir, 'train', filename),
    ]

    for img_path in possible_paths:
        if os.path.isfile(img_path):
            return FileResponse(
                path=img_path,
                media_type='image/png',
                filename=filename
            )

    raise HTTPException(404, "Image not found")


@router.get("/training/artifacts/{artifact_id}/download")
def download_artifact(artifact_id: int):
    """
    下载模型文件（PT、PTH、ONNX等）

    Args:
        artifact_id: 模型文件的artifact ID

    Returns:
        模型文件
    """
    with get_session() as session:
        artifact = session.get(ModelArtifact, artifact_id)
        if not artifact:
            raise HTTPException(404, "Artifact not found")

        file_path = os.path.join(os.getcwd(), artifact.path)
        if not os.path.exists(file_path):
            raise HTTPException(404, "Model file not found on disk")

        # 获取文件名
        filename = os.path.basename(artifact.path)

        # 根据格式设置MIME类型
        mime_types = {
            'pt': 'application/octet-stream',
            'pth': 'application/octet-stream',
            'onnx': 'application/octet-stream',
        }
        media_type = mime_types.get(artifact.format, 'application/octet-stream')

        # 使用 RFC 5987 规范处理文件名
        encoded_filename = quote(filename, safe='')

        return FileResponse(
            path=file_path,
            media_type=media_type,
            filename=filename,
            headers={
                "Content-Disposition": f"attachment; filename*=UTF-8''{encoded_filename}"
            }
        )
