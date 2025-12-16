from fastapi import APIRouter, HTTPException
from typing import List
from pydantic import BaseModel

from ..db import get_session
from ..models import Annotation, Image


router = APIRouter(tags=["annotations"])


class AnnotationCreate(BaseModel):
    imageId: int
    classId: int
    x: float
    y: float
    w: float
    h: float
    confidence: float | None = None
    source: str = "manual"


@router.get("/images/{image_id}/annotations", response_model=List[Annotation])
def list_annotations(image_id: int):
    with get_session() as session:
        return session.query(Annotation).filter(Annotation.imageId == image_id).all()


@router.post("/annotations", response_model=Annotation)
def create_annotation(body: AnnotationCreate):
    with get_session() as session:
        img = session.get(Image, body.imageId)
        if not img:
            raise HTTPException(404, "Image not found")
        ann = Annotation(**body.dict())
        session.add(ann)
        # 写入后将图片标记为已标注
        if img.status != "annotated":
            img.status = "annotated"
            session.add(img)
        session.commit()
        session.refresh(ann)
        return ann


class AnnotationUpdate(BaseModel):
    classId: int | None = None
    x: float | None = None
    y: float | None = None
    w: float | None = None
    h: float | None = None
    confidence: float | None = None


@router.put("/annotations/{annotation_id}", response_model=Annotation)
def update_annotation(annotation_id: int, body: AnnotationUpdate):
    with get_session() as session:
        ann = session.get(Annotation, annotation_id)
        if not ann:
            raise HTTPException(404, "Annotation not found")
        data = body.dict(exclude_unset=True)
        for k, v in data.items():
            setattr(ann, k, v)
        session.add(ann)
        # 保持图片状态
        img = session.get(Image, ann.imageId)
        if img and img.status != "annotated":
            img.status = "annotated"
            session.add(img)
        session.commit()
        session.refresh(ann)
        return ann


@router.delete("/annotations/{annotation_id}")
def delete_annotation(annotation_id: int):
    with get_session() as session:
        ann = session.get(Annotation, annotation_id)
        if not ann:
            raise HTTPException(404, "Annotation not found")
        image_id = ann.imageId
        session.delete(ann)
        session.commit()
        # 若图像无其他标注，将状态置回未标注
        remain = session.query(Annotation).filter(Annotation.imageId == image_id).count()
        if remain == 0:
            img = session.get(Image, image_id)
            if img and img.status != "unannotated":
                img.status = "unannotated"
                session.add(img)
                session.commit()
        return {"ok": True, "remaining": remain}
