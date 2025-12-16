from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from datetime import datetime

from ..db import get_session
from ..models import AutoLabelJob, Project, ProposedAnnotation, Image


router = APIRouter(tags=["autolabel"])


class AutoLabelJobCreate(BaseModel):
    projectId: int
    imageIds: List[int]
    threshold: float = 0.5


@router.post("/autolabel/jobs", response_model=AutoLabelJob)
def create_job(body: AutoLabelJobCreate):
    with get_session() as session:
        proj = session.get(Project, body.projectId)
        if not proj:
            raise HTTPException(404, "Project not found")
        job = AutoLabelJob(projectId=body.projectId, status="pending", threshold=body.threshold)
        job.imagesCount = len(body.imageIds)
        session.add(job)
        session.commit()
        session.refresh(job)
        # NOTE: MVP骨架：并未实际调用GLM，后续由services/worker接入
        return job


@router.get("/autolabel/jobs/{job_id}", response_model=AutoLabelJob)
def get_job(job_id: int):
    with get_session() as session:
        job = session.get(AutoLabelJob, job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        return job


class MergeRequest(BaseModel):
    imageId: int
    accepts: List[int]  # ProposedAnnotation ids to accept


@router.post("/autolabel/merge")
def merge_annotations(body: MergeRequest):
    with get_session() as session:
        img = session.get(Image, body.imageId)
        if not img:
            raise HTTPException(404, "Image not found")
        # 占位：实际合并逻辑由服务层完成
        return {"merged": len(body.accepts)}

