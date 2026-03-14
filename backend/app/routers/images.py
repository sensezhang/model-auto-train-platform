from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

from ..db import get_session
from ..models import Image, Project, Annotation, ProposedAnnotation


router = APIRouter(tags=["images"])


class PaginatedImages(BaseModel):
    items: List[Image]
    total: int
    page: int
    page_size: int
    has_more: bool


class DeleteImagesRequest(BaseModel):
    image_ids: List[int]


class DeleteImagesResponse(BaseModel):
    deleted_images: int
    deleted_annotations: int
    deleted_proposed_annotations: int


class LabeledPatch(BaseModel):
    labeled: bool


class MarkLabeledRequest(BaseModel):
    image_ids: List[int]
    labeled: bool = True


@router.get("/projects/{project_id}/images", response_model=PaginatedImages)
def list_images(
    project_id: int,
    status: Optional[str] = Query(default=None),
    labeled: Optional[bool] = Query(default=None),
    date_from: Optional[str] = Query(default=None, description="上传时间起始（ISO 格式，含）"),
    date_to: Optional[str] = Query(default=None, description="上传时间截止（ISO 格式，含）"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=200)
):
    """
    分页获取项目图片列表

    Args:
        project_id: 项目ID
        status: 状态过滤 (unannotated/ai_pending/annotated)
        labeled: 完成状态过滤 (true=已完成, false=未完成)
        date_from: 上传时间起始（ISO 格式，如 2024-01-01T00:00:00）
        date_to: 上传时间截止（ISO 格式，如 2024-12-31T23:59:59）
        page: 页码，从1开始
        page_size: 每页数量，默认50，最大200
    """
    with get_session() as session:
        proj = session.get(Project, project_id)
        if not proj:
            raise HTTPException(404, "Project not found")

        q = session.query(Image).filter(Image.projectId == project_id)
        if status:
            q = q.filter(Image.status == status)
        if labeled is not None:
            q = q.filter(Image.labeled == labeled)
        if date_from:
            try:
                df = datetime.fromisoformat(date_from.replace('Z', '').replace('+00:00', ''))
                q = q.filter(Image.createdAt >= df)
            except ValueError:
                pass
        if date_to:
            try:
                dt = datetime.fromisoformat(date_to.replace('Z', '').replace('+00:00', ''))
                q = q.filter(Image.createdAt <= dt)
            except ValueError:
                pass

        total = q.count()
        offset = (page - 1) * page_size
        # 按上传时间倒序（最新在前）
        items = q.order_by(Image.createdAt.desc()).offset(offset).limit(page_size).all()
        has_more = offset + len(items) < total

        return PaginatedImages(
            items=items,
            total=total,
            page=page,
            page_size=page_size,
            has_more=has_more
        )


@router.patch("/images/{image_id}/labeled", response_model=Image)
def set_image_labeled(image_id: int, body: LabeledPatch):
    """标记单张图片的完成状态"""
    with get_session() as session:
        img = session.get(Image, image_id)
        if not img:
            raise HTTPException(404, "Image not found")
        img.labeled = body.labeled
        session.add(img)
        session.commit()
        session.refresh(img)
        return img


@router.post("/projects/{project_id}/images/mark-labeled")
def bulk_mark_labeled(project_id: int, body: MarkLabeledRequest):
    """批量标记图片完成状态"""
    if not body.image_ids:
        raise HTTPException(400, "No image IDs provided")

    with get_session() as session:
        proj = session.get(Project, project_id)
        if not proj:
            raise HTTPException(404, "Project not found")

        images = session.query(Image).filter(
            Image.id.in_(body.image_ids),
            Image.projectId == project_id
        ).all()

        for img in images:
            img.labeled = body.labeled
            session.add(img)

        session.commit()
        return {"updated": len(images), "labeled": body.labeled}


@router.delete("/projects/{project_id}/images", response_model=DeleteImagesResponse)
def delete_images(project_id: int, body: DeleteImagesRequest):
    """
    批量删除图片及其关联的标注数据
    """
    if not body.image_ids:
        raise HTTPException(400, "No image IDs provided")

    with get_session() as session:
        proj = session.get(Project, project_id)
        if not proj:
            raise HTTPException(404, "Project not found")

        images = session.query(Image).filter(
            Image.id.in_(body.image_ids),
            Image.projectId == project_id
        ).all()

        if len(images) != len(body.image_ids):
            raise HTTPException(400, "Some images not found or don't belong to this project")

        valid_image_ids = [img.id for img in images]

        deleted_annotations = session.query(Annotation).filter(
            Annotation.imageId.in_(valid_image_ids)
        ).delete(synchronize_session=False)

        deleted_proposed = session.query(ProposedAnnotation).filter(
            ProposedAnnotation.imageId.in_(valid_image_ids)
        ).delete(synchronize_session=False)

        for img in images:
            session.delete(img)

        session.commit()

        return DeleteImagesResponse(
            deleted_images=len(images),
            deleted_annotations=deleted_annotations,
            deleted_proposed_annotations=deleted_proposed
        )
