import os
import json
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

from ..db import get_session
from ..models import AutoLabelJob, Project, Image, JobStatus, ModelArtifact
from ..services.autolabel_worker import run_autolabel_job
from ..services.autolabel_model_worker import run_model_autolabel_job


router = APIRouter(tags=["autolabel"])


class AutoLabelJobCreate(BaseModel):
    projectId: int
    classId: int
    prompt: str
    imageIds: Optional[List[int]] = None   # None = 按 scope 决定
    apiKey: Optional[str] = None           # 不填则读环境变量
    threshold: float = 0.3
    scope: str = 'unlabeled'               # 'unlabeled' | 'all'（imageIds 为空时生效）


class ModelAutoLabelJobCreate(BaseModel):
    projectId: int
    artifactId: int                        # 使用的模型产物 ID
    imageIds: Optional[List[int]] = None   # None = 按 scope 决定
    confidenceThreshold: float = 0.25
    iouThreshold: float = 0.45
    scope: str = 'unlabeled'               # 'unlabeled' | 'all'（imageIds 为空时生效）


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


def _build_job_response(job: AutoLabelJob) -> AutoLabelJobResponse:
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

        # 统计待处理图片数 & 确定 logsRef
        if body.imageIds:
            count = len(body.imageIds)
            logs_ref = json.dumps(body.imageIds)
        elif body.scope == 'all':
            # 全部图片：提前收集 ID，传给 worker
            all_ids = [img.id for img in session.query(Image).filter(
                Image.projectId == body.projectId
            ).all()]
            count = len(all_ids)
            logs_ref = json.dumps(all_ids)
        else:
            # 默认：仅未完成标注的图片
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
        return _build_job_response(job)


@router.post("/autolabel/model-jobs", response_model=AutoLabelJobResponse)
def create_model_job(body: ModelAutoLabelJobCreate, background_tasks: BackgroundTasks):
    """使用已训练模型批量推理生成标注框"""
    with get_session() as session:
        proj = session.get(Project, body.projectId)
        if not proj:
            raise HTTPException(404, "Project not found")

        # 检查模型产物是否存在
        artifact = session.get(ModelArtifact, body.artifactId)
        if not artifact:
            raise HTTPException(404, "Model artifact not found")

        # 统计待处理图片数
        if body.imageIds:
            count = len(body.imageIds)
        elif body.scope == 'all':
            count = session.query(Image).filter(
                Image.projectId == body.projectId
            ).count()
        else:
            count = session.query(Image).filter(
                Image.projectId == body.projectId,
                Image.labeled == False  # noqa: E712
            ).count()

        # 将模型参数存入 logsRef（JSON 元数据）
        meta: dict = {
            "type": "model",
            "artifactId": body.artifactId,
            "confidenceThreshold": body.confidenceThreshold,
            "iouThreshold": body.iouThreshold,
            "scope": body.scope,
        }
        if body.imageIds:
            meta["imageIds"] = body.imageIds

        job = AutoLabelJob(
            projectId=body.projectId,
            classId=None,
            prompt="__model__",
            status=JobStatus.pending,
            threshold=body.confidenceThreshold,
            imagesCount=count,
            processedCount=0,
            boxesCount=0,
            logsRef=json.dumps(meta),
        )
        session.add(job)
        session.commit()
        session.refresh(job)
        job_id = job.id

    # 后台运行
    background_tasks.add_task(run_model_autolabel_job, job_id)

    with get_session() as session:
        job = session.get(AutoLabelJob, job_id)
        return _build_job_response(job)


@router.get("/autolabel/jobs/{job_id}", response_model=AutoLabelJobResponse)
def get_job(job_id: int):
    with get_session() as session:
        job = session.get(AutoLabelJob, job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        return _build_job_response(job)
