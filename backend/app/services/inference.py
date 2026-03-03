"""
Model Inference Service
- Support YOLO and RF-DETR model inference
- Load model from training artifacts
- Return detection results with bounding boxes
"""

from __future__ import annotations

import os
import base64
import io
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from PIL import Image
import numpy as np


@dataclass
class Detection:
    """Single detection result"""
    class_id: int
    class_name: str
    confidence: float
    x: float  # top-left x
    y: float  # top-left y
    width: float
    height: float


@dataclass
class InferenceResult:
    """Inference result for a single image"""
    detections: List[Detection]
    image_width: int
    image_height: int
    inference_time_ms: float


def _load_image_from_base64(base64_data: str) -> Image.Image:
    """Load PIL Image from base64 data"""
    # Handle data URL format
    if ',' in base64_data:
        base64_data = base64_data.split(',', 1)[1]

    image_bytes = base64.b64decode(base64_data)
    image = Image.open(io.BytesIO(image_bytes))

    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')

    return image


def _load_image_from_path(image_path: str) -> Image.Image:
    """Load PIL Image from file path"""
    image = Image.open(image_path)

    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')

    return image


def infer_yolo(
    model_path: str,
    image: Image.Image,
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    imgsz: int = 640,
) -> InferenceResult:
    """
    Run YOLO inference on an image

    Args:
        model_path: Path to YOLO model (.pt file)
        image: PIL Image
        confidence_threshold: Minimum confidence for detections
        iou_threshold: IoU threshold for NMS
        imgsz: Input image size

    Returns:
        InferenceResult with detections
    """
    import time

    # Import YOLO
    try:
        from ultralytics import YOLO
    except ImportError:
        raise RuntimeError("Ultralytics is not installed. Please install 'ultralytics' package.")

    # Load model
    model = YOLO(model_path)

    # Get image dimensions
    img_width, img_height = image.size

    # Run inference
    start_time = time.time()
    results = model.predict(
        source=image,
        conf=confidence_threshold,
        iou=iou_threshold,
        imgsz=imgsz,
        verbose=False,
    )
    inference_time = (time.time() - start_time) * 1000  # ms

    # Parse results
    detections = []
    if results and len(results) > 0:
        result = results[0]
        boxes = result.boxes

        if boxes is not None:
            for i in range(len(boxes)):
                # Get box coordinates (xyxy format)
                box = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = box

                # Get confidence and class
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())

                # Get class name
                cls_name = model.names.get(cls_id, f"class_{cls_id}")

                detections.append(Detection(
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=conf,
                    x=float(x1),
                    y=float(y1),
                    width=float(x2 - x1),
                    height=float(y2 - y1),
                ))

    return InferenceResult(
        detections=detections,
        image_width=img_width,
        image_height=img_height,
        inference_time_ms=inference_time,
    )


def infer_rfdetr(
    model_path: str,
    image: Image.Image,
    confidence_threshold: float = 0.5,
    model_type: str = 'medium',
) -> InferenceResult:
    """
    Run RF-DETR inference on an image

    Args:
        model_path: Path to RF-DETR checkpoint (.pth file)
        image: PIL Image
        confidence_threshold: Minimum confidence for detections
        model_type: Model type ('small', 'medium', 'large')

    Returns:
        InferenceResult with detections
    """
    import time
    import torch

    # Import RF-DETR
    try:
        from rfdetr import RFDETRSmall, RFDETRMedium, RFDETRLarge
        from rfdetr.main import Model as RFDETRModel
    except ImportError:
        raise RuntimeError("rfdetr is not installed. Please install 'rfdetr' package.")

    # Get image dimensions
    img_width, img_height = image.size

    # Select model class
    if model_type == 'small':
        ModelClass = RFDETRSmall
    elif model_type == 'large':
        ModelClass = RFDETRLarge
    else:
        ModelClass = RFDETRMedium

    # Load model with checkpoint - RF-DETR uses pretrain_weights parameter
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # RF-DETR's Model class accepts pretrain_weights as the checkpoint path
    # This works for both pretrained weights and finetuned checkpoints
    model = ModelClass(pretrain_weights=model_path)

    # Run inference
    start_time = time.time()

    # Use model's predict method with threshold
    results = model.predict(image, threshold=confidence_threshold)

    inference_time = (time.time() - start_time) * 1000  # ms

    # Parse results - RF-DETR returns a list of Detections objects
    detections = []
    if results is not None:
        # RF-DETR predict returns a Detections object from supervision
        # It has .xyxy, .confidence, .class_id attributes
        if hasattr(results, 'xyxy') and hasattr(results, 'confidence'):
            # supervision Detections format
            boxes = results.xyxy  # numpy array of shape (N, 4)
            scores = results.confidence  # numpy array of shape (N,)
            class_ids = results.class_id if hasattr(results, 'class_id') and results.class_id is not None else np.zeros(len(scores), dtype=int)

            for i in range(len(boxes)):
                box = boxes[i]
                x1, y1, x2, y2 = box
                label_val = int(class_ids[i]) if i < len(class_ids) else 0
                detections.append(Detection(
                    class_id=label_val,
                    class_name=f"class_{label_val}",
                    confidence=float(scores[i]) if i < len(scores) else 0.0,
                    x=float(x1),
                    y=float(y1),
                    width=float(x2 - x1),
                    height=float(y2 - y1),
                ))
        elif isinstance(results, dict):
            # Dict format with 'boxes', 'labels', 'scores'
            boxes = results.get('boxes', [])
            labels = results.get('labels', [])
            scores = results.get('scores', [])

            # Convert tensors to numpy if needed
            if hasattr(boxes, 'cpu'):
                boxes = boxes.cpu().numpy()
            if hasattr(labels, 'cpu'):
                labels = labels.cpu().numpy()
            if hasattr(scores, 'cpu'):
                scores = scores.cpu().numpy()

            for i in range(len(boxes)):
                box = boxes[i]
                if len(box) == 4:
                    x1, y1, x2, y2 = box
                    label_val = int(labels[i]) if i < len(labels) else 0
                    detections.append(Detection(
                        class_id=label_val,
                        class_name=f"class_{label_val}",
                        confidence=float(scores[i]) if i < len(scores) else 0.0,
                        x=float(x1),
                        y=float(y1),
                        width=float(x2 - x1),
                        height=float(y2 - y1),
                    ))
        elif isinstance(results, (list, tuple)) and len(results) > 0:
            # Handle list format
            for res in results:
                if hasattr(res, 'xyxy'):
                    boxes = res.xyxy
                    scores = res.confidence if hasattr(res, 'confidence') else []
                    class_ids = res.class_id if hasattr(res, 'class_id') and res.class_id is not None else []

                    for i in range(len(boxes)):
                        box = boxes[i]
                        x1, y1, x2, y2 = box
                        label_val = int(class_ids[i]) if i < len(class_ids) else 0
                        detections.append(Detection(
                            class_id=label_val,
                            class_name=f"class_{label_val}",
                            confidence=float(scores[i]) if i < len(scores) else 0.0,
                            x=float(x1),
                            y=float(y1),
                            width=float(x2 - x1),
                            height=float(y2 - y1),
                        ))

    return InferenceResult(
        detections=detections,
        image_width=img_width,
        image_height=img_height,
        inference_time_ms=inference_time,
    )


def run_inference(
    model_path: str,
    framework: str,
    image_data: str | None = None,
    image_path: str | None = None,
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    imgsz: int = 640,
    model_type: str = 'medium',
) -> Dict[str, Any]:
    """
    Run inference on an image using specified model

    Args:
        model_path: Path to model file
        framework: 'yolo' or 'rfdetr'
        image_data: Base64 encoded image data
        image_path: Path to image file
        confidence_threshold: Minimum confidence for detections
        iou_threshold: IoU threshold for NMS (YOLO only)
        imgsz: Input image size (YOLO only)
        model_type: Model type for RF-DETR

    Returns:
        Dict with inference results
    """
    # Load image
    if image_data:
        image = _load_image_from_base64(image_data)
    elif image_path:
        image = _load_image_from_path(image_path)
    else:
        raise ValueError("Either image_data or image_path must be provided")

    # Run inference based on framework
    if framework.lower() == 'yolo':
        result = infer_yolo(
            model_path=model_path,
            image=image,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            imgsz=imgsz,
        )
    elif framework.lower() == 'rfdetr':
        result = infer_rfdetr(
            model_path=model_path,
            image=image,
            confidence_threshold=confidence_threshold,
            model_type=model_type,
        )
    else:
        raise ValueError(f"Unsupported framework: {framework}")

    # Convert to dict
    return {
        'detections': [
            {
                'class_id': d.class_id,
                'class_name': d.class_name,
                'confidence': d.confidence,
                'x': d.x,
                'y': d.y,
                'width': d.width,
                'height': d.height,
            }
            for d in result.detections
        ],
        'image_width': result.image_width,
        'image_height': result.image_height,
        'inference_time_ms': result.inference_time_ms,
    }


def get_model_class_names(model_path: str, framework: str) -> List[str]:
    """
    Get class names from a trained model

    Args:
        model_path: Path to model file
        framework: 'yolo' or 'rfdetr'

    Returns:
        List of class names
    """
    if framework.lower() == 'yolo':
        try:
            from ultralytics import YOLO
            model = YOLO(model_path)
            return list(model.names.values())
        except Exception:
            return []
    elif framework.lower() == 'rfdetr':
        # RF-DETR doesn't store class names in checkpoint directly
        # Would need to load from config or training data
        return []

    return []
