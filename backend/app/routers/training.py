from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
import os
import asyncio
from fastapi.responses import StreamingResponse

from ..db import get_session
from ..models import TrainingJob, Project, Image, Annotation, ModelArtifact
from ..services.training_yolo import start_training_async


router = APIRouter(tags=["training"])


class TrainJobCreate(BaseModel):
    projectId: int
    modelVariant: str = "yolov11n"
    epochs: int = 50
    imgsz: int = 640
    batch: Optional[int] = None
    seed: int = 42


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
        # fire and forget
        start_training_async(job.id)
        return job


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
