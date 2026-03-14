"""
Inference API Router
- Run model inference on uploaded images
- Support YOLO and RF-DETR models
"""

from __future__ import annotations

import os
import base64
import io
from typing import Optional, List
from datetime import datetime

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlmodel import select
from PIL import Image, ImageDraw, ImageFont

from ..db import get_session
from ..models import TrainingJob, ModelArtifact, Project, Class as DBClass
from ..services.inference import run_inference, get_model_class_names
from ..utils.oss_storage import resolve_local_path


router = APIRouter(prefix="/inference", tags=["inference"])


class InferenceRequest(BaseModel):
    """Request body for inference"""
    artifact_id: int
    image_data: str  # Base64 encoded image
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    imgsz: int = 640


class DetectionResult(BaseModel):
    """Single detection result"""
    class_id: int
    class_name: str
    confidence: float
    x: float
    y: float
    width: float
    height: float


class InferenceResponse(BaseModel):
    """Response from inference"""
    model_config = {"protected_namespaces": ()}

    detections: List[DetectionResult]
    image_width: int
    image_height: int
    inference_time_ms: float
    model_info: dict


class AvailableModel(BaseModel):
    """Available model for inference"""
    model_config = {"protected_namespaces": ()}

    artifact_id: int
    job_id: int
    project_id: int
    project_name: str
    framework: str
    model_variant: str
    format: str
    path: str
    map50: Optional[float]
    map50_95: Optional[float]
    created_at: str


@router.get("/models", response_model=List[AvailableModel])
def list_available_models():
    """List all available models for inference (from succeeded training jobs)"""
    with get_session() as session:
        # Get all succeeded training jobs
        jobs = session.exec(
            select(TrainingJob).where(TrainingJob.status == 'succeeded')
        ).all()

        models = []
        for job in jobs:
            # Get project info
            project = session.get(Project, job.projectId)
            if not project:
                continue

            # Get model artifacts
            artifacts = session.exec(
                select(ModelArtifact).where(ModelArtifact.trainingJobId == job.id)
            ).all()

            for artifact in artifacts:
                # Skip ONNX for now, focus on PT/PTH
                if artifact.format.lower() in ('pt', 'pth'):
                    models.append(AvailableModel(
                        artifact_id=artifact.id,
                        job_id=job.id,
                        project_id=job.projectId,
                        project_name=project.name,
                        framework=job.framework,
                        model_variant=job.modelVariant,
                        format=artifact.format,
                        path=artifact.path,
                        map50=job.map50,
                        map50_95=job.map50_95,
                        created_at=artifact.createdAt.isoformat() if artifact.createdAt else '',
                    ))

        # Sort by creation date, newest first
        models.sort(key=lambda m: m.created_at, reverse=True)
        return models


@router.get("/models/{artifact_id}/classes")
def get_model_classes(artifact_id: int):
    """Get class names for a model"""
    with get_session() as session:
        artifact = session.get(ModelArtifact, artifact_id)
        if not artifact:
            raise HTTPException(status_code=404, detail="Model artifact not found")

        job = session.get(TrainingJob, artifact.trainingJobId)
        if not job:
            raise HTTPException(status_code=404, detail="Training job not found")

        # Get class names from project
        classes = session.exec(
            select(DBClass).where(DBClass.projectId == job.projectId).order_by(DBClass.id)
        ).all()

        class_names = [c.name for c in classes]

        # Also try to get from model if YOLO（支持本地路径和 OSS URL）
        if job.framework.lower() == 'yolo':
            model_path = resolve_local_path(artifact.path)
            if model_path:
                try:
                    model_class_names = get_model_class_names(model_path, job.framework)
                    if model_class_names:
                        class_names = model_class_names
                except Exception:
                    pass

        return {
            'class_names': class_names,
            'framework': job.framework,
            'model_variant': job.modelVariant,
        }


@router.post("/run", response_model=InferenceResponse)
def run_model_inference(request: InferenceRequest):
    """Run inference on an image using specified model"""
    with get_session() as session:
        # Get artifact
        artifact = session.get(ModelArtifact, request.artifact_id)
        if not artifact:
            raise HTTPException(status_code=404, detail="Model artifact not found")

        # Get training job info
        job = session.get(TrainingJob, artifact.trainingJobId)
        if not job:
            raise HTTPException(status_code=404, detail="Training job not found")

        # 解析模型路径（本地路径或 OSS URL → 本地文件，不存在则自动从 OSS 下载）
        model_path = resolve_local_path(artifact.path)
        if not model_path:
            raise HTTPException(status_code=404, detail=f"Model file not found: {artifact.path}")

        # Determine model type for RF-DETR
        model_type = 'medium'
        if job.framework.lower() == 'rfdetr':
            variant = job.modelVariant.lower()
            if 'small' in variant:
                model_type = 'small'
            elif 'large' in variant:
                model_type = 'large'

        # Run inference
        try:
            result = run_inference(
                model_path=model_path,
                framework=job.framework,
                image_data=request.image_data,
                confidence_threshold=request.confidence_threshold,
                iou_threshold=request.iou_threshold,
                imgsz=request.imgsz,
                model_type=model_type,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

        return InferenceResponse(
            detections=[DetectionResult(**d) for d in result['detections']],
            image_width=result['image_width'],
            image_height=result['image_height'],
            inference_time_ms=result['inference_time_ms'],
            model_info={
                'framework': job.framework,
                'model_variant': job.modelVariant,
                'artifact_id': artifact.id,
                'job_id': job.id,
            }
        )


@router.post("/run-file")
async def run_inference_with_file(
    artifact_id: int = Form(...),
    confidence_threshold: float = Form(0.25),
    iou_threshold: float = Form(0.45),
    imgsz: int = Form(640),
    file: UploadFile = File(...),
):
    """Run inference on an uploaded image file"""
    # Read file and convert to base64
    content = await file.read()
    base64_data = base64.b64encode(content).decode('utf-8')

    # Create request
    request = InferenceRequest(
        artifact_id=artifact_id,
        image_data=f"data:image/jpeg;base64,{base64_data}",
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        imgsz=imgsz,
    )

    return run_model_inference(request)


# Color palette for drawing boxes
COLORS = [
    (255, 0, 0),      # Red
    (0, 255, 0),      # Green
    (0, 0, 255),      # Blue
    (255, 255, 0),    # Yellow
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Cyan
    (255, 128, 0),    # Orange
    (128, 0, 255),    # Purple
    (0, 128, 255),    # Light Blue
    (255, 0, 128),    # Pink
    (128, 255, 0),    # Lime
    (0, 255, 128),    # Mint
]


@router.post("/visualize")
def visualize_inference(request: InferenceRequest):
    """
    Run inference and return an image with drawn bounding boxes
    Returns the image as base64
    """
    # First run inference
    response = run_model_inference(request)

    # Decode the original image
    image_data = request.image_data
    if ',' in image_data:
        image_data = image_data.split(',', 1)[1]

    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes))

    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Draw bounding boxes
    draw = ImageDraw.Draw(image)

    # Try to load a font, fallback to default
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except Exception:
            font = ImageFont.load_default()

    for det in response.detections:
        # Get color based on class_id
        color = COLORS[det.class_id % len(COLORS)]

        # Draw rectangle
        x1, y1 = det.x, det.y
        x2, y2 = det.x + det.width, det.y + det.height

        # Draw box with thicker line
        for i in range(3):
            draw.rectangle([x1-i, y1-i, x2+i, y2+i], outline=color)

        # Draw label background
        label = f"{det.class_name} {det.confidence:.2f}"
        bbox = draw.textbbox((x1, y1), label, font=font)
        label_width = bbox[2] - bbox[0]
        label_height = bbox[3] - bbox[1]

        draw.rectangle(
            [x1, y1 - label_height - 4, x1 + label_width + 4, y1],
            fill=color
        )

        # Draw label text
        draw.text((x1 + 2, y1 - label_height - 2), label, fill=(255, 255, 255), font=font)

    # Convert image to base64
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=95)
    buffer.seek(0)

    result_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return {
        'image_data': f"data:image/jpeg;base64,{result_base64}",
        'detections': [d.dict() for d in response.detections],
        'image_width': response.image_width,
        'image_height': response.image_height,
        'inference_time_ms': response.inference_time_ms,
        'model_info': response.model_info,
    }


@router.post("/batch")
def run_batch_inference(
    artifact_id: int,
    image_data_list: List[str],
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    imgsz: int = 640,
):
    """Run inference on multiple images"""
    results = []

    for i, image_data in enumerate(image_data_list):
        try:
            request = InferenceRequest(
                artifact_id=artifact_id,
                image_data=image_data,
                confidence_threshold=confidence_threshold,
                iou_threshold=iou_threshold,
                imgsz=imgsz,
            )
            result = run_model_inference(request)
            results.append({
                'index': i,
                'success': True,
                'result': result.dict(),
            })
        except Exception as e:
            results.append({
                'index': i,
                'success': False,
                'error': str(e),
            })

    return {'results': results}


# ──────────────────────────────────────────────
# SAM3 分割推理
# ──────────────────────────────────────────────

class SAM3Request(BaseModel):
    image_data: str            # base64 编码图片（支持 data:image/...;base64,xxx 和纯 base64）
    text_labels: List[str] = []  # ["car","person"]；空列表则分割所有对象
    conf: float = 0.4
    iou: float = 0.9
    imgsz: int = 512           # 推理图像尺寸，越小越快，GPU 下 512 通常 2-5s


@router.post("/sam3")
def sam3_inference(body: SAM3Request):
    """
    调用远程 SAM3 API 对单张图片进行分割推理。
    返回分割结果多边形列表（原图坐标系），前端负责叠加渲染。
    """
    import tempfile
    import requests as _requests
    from ..services.sam3_service import run_sam3, SAM3_API_URL

    # base64 解码 → 临时文件
    raw = body.image_data
    if "," in raw:
        raw = raw.split(",", 1)[1]
    try:
        img_bytes = base64.b64decode(raw)
    except Exception as e:
        raise HTTPException(400, f"图片 base64 解码失败: {e}")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f:
            f.write(img_bytes)
            tmp_path = f.name

        result = run_sam3(
            tmp_path,
            body.text_labels,
            conf=body.conf,
            iou=body.iou,
            imgsz=body.imgsz,
        )
    except _requests.exceptions.ConnectionError:
        raise HTTPException(503, f"无法连接 SAM3 API（{SAM3_API_URL}），请确认服务已启动")
    except _requests.exceptions.Timeout:
        raise HTTPException(504, "SAM3 API 请求超时")
    except _requests.exceptions.HTTPError as e:
        raise HTTPException(502, f"SAM3 API 返回错误: {e}")
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    return result
