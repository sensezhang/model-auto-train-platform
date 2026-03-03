import os
import json
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

from ..db import get_session
from ..models import AutoLabelJob, Project, Image, JobStatus
from ..services.autolabel_worker import run_autolabel_job


router = APIRouter(tags=["autolabel"])


class AutoLabelJobCreate(BaseModel):
    projectId: int
    classId: int
    prompt: str
    imageIds: Optional[List[int]] = None   # None = 处理所有 labeled=false 的图片
    apiKey: Optional[str] = None           # 不填则读 ZHIPUAI_API_KEY 环境变量
    threshold: float = 0.3


class AutoLabelJobResponse(BaseModel):
    id: int
    projectId: int
    classId: Optional[int]
    prompt: Optional[str]
    status: str
    imagesCount: int
    processedCount: int
    boxesCount: int
    startedAt: Optional[datetime]
    finishedAt: Optional[datetime]

    class Config:
        from_attributes = True


@router.post("/autolabel/jobs", response_model=AutoLabelJobResponse)
def create_job(body: AutoLabelJobCreate, background_tasks: BackgroundTasks):
    # 解析 API Key：请求体 > 环境变量
    api_key = body.apiKey or os.getenv("AUTOLABEL_API_KEY", "")
    if not api_key:
        raise HTTPException(400, "未提供 API Key，请在请求中提供 apiKey 或在 .env 中设置 AUTOLABEL_API_KEY")

    with get_session() as session:
        proj = session.get(Project, body.projectId)
        if not proj:
            raise HTTPException(404, "Project not found")

        # 统计待处理图片数
        if body.imageIds:
            count = len(body.imageIds)
            logs_ref = json.dumps(body.imageIds)
        else:
            count = session.query(Image).filter(
                Image.projectId == body.projectId,
                Image.labeled == False  # noqa: E712
            ).count()
            logs_ref = None

        job = AutoLabelJob(
            projectId=body.projectId,
            classId=body.classId,
            prompt=body.prompt,
            status=JobStatus.pending,
            threshold=body.threshold,
            imagesCount=count,
            processedCount=0,
            boxesCount=0,
            logsRef=logs_ref,
        )
        session.add(job)
        session.commit()
        session.refresh(job)
        job_id = job.id

    # 后台运行
    background_tasks.add_task(run_autolabel_job, job_id, api_key)

    with get_session() as session:
        job = session.get(AutoLabelJob, job_id)
        return AutoLabelJobResponse(
            id=job.id,
            projectId=job.projectId,
            classId=job.classId,
            prompt=job.prompt,
            status=job.status,
            imagesCount=job.imagesCount,
            processedCount=job.processedCount,
            boxesCount=job.boxesCount,
            startedAt=job.startedAt,
            finishedAt=job.finishedAt,
        )


@router.get("/autolabel/jobs/{job_id}", response_model=AutoLabelJobResponse)
def get_job(job_id: int):
    with get_session() as session:
        job = session.get(AutoLabelJob, job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        return AutoLabelJobResponse(
            id=job.id,
            projectId=job.projectId,
            classId=job.classId,
            prompt=job.prompt,
            status=job.status,
            imagesCount=job.imagesCount,
            processedCount=job.processedCount,
            boxesCount=job.boxesCount,
            startedAt=job.startedAt,
            finishedAt=job.finishedAt,
        )
