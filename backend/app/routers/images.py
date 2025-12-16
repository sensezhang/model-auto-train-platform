from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional

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


@router.get("/projects/{project_id}/images", response_model=PaginatedImages)
def list_images(
    project_id: int,
    status: Optional[str] = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=200)
):
    """
    分页获取项目图片列表

    Args:
        project_id: 项目ID
        status: 状态过滤 (unannotated/ai_pending/annotated)
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

        # 获取总数
        total = q.count()

        # 分页查询
        offset = (page - 1) * page_size
        items = q.order_by(Image.id).offset(offset).limit(page_size).all()

        has_more = offset + len(items) < total

        return PaginatedImages(
            items=items,
            total=total,
            page=page,
            page_size=page_size,
            has_more=has_more
        )


@router.delete("/projects/{project_id}/images", response_model=DeleteImagesResponse)
def delete_images(project_id: int, body: DeleteImagesRequest):
    """
    批量删除图片及其关联的标注数据

    Args:
        project_id: 项目ID
        body: 包含要删除的图片ID列表

    Returns:
        删除统计信息
    """
    if not body.image_ids:
        raise HTTPException(400, "No image IDs provided")

    with get_session() as session:
        proj = session.get(Project, project_id)
        if not proj:
            raise HTTPException(404, "Project not found")

        # 验证所有图片都属于该项目
        images = session.query(Image).filter(
            Image.id.in_(body.image_ids),
            Image.projectId == project_id
        ).all()

        if len(images) != len(body.image_ids):
            raise HTTPException(400, "Some images not found or don't belong to this project")

        valid_image_ids = [img.id for img in images]

        # 删除关联的标注
        deleted_annotations = session.query(Annotation).filter(
            Annotation.imageId.in_(valid_image_ids)
        ).delete(synchronize_session=False)

        # 删除关联的待审核标注
        deleted_proposed = session.query(ProposedAnnotation).filter(
            ProposedAnnotation.imageId.in_(valid_image_ids)
        ).delete(synchronize_session=False)

        # 删除图片记录
        for img in images:
            session.delete(img)

        session.commit()

        return DeleteImagesResponse(
            deleted_images=len(images),
            deleted_annotations=deleted_annotations,
            deleted_proposed_annotations=deleted_proposed
        )

