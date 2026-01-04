from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import os
import tempfile
import zipfile
import re
from datetime import datetime
from urllib.parse import quote

from ..services.export_coco import export_dataset_to_coco
from ..services.export_yolo import export_dataset_to_yolo
from ..services.augmentation import AugmentationConfig, parse_augmentation_config


router = APIRouter(tags=["export"])


class BrightnessConfig(BaseModel):
    enabled: bool = False
    min: float = 0.7
    max: float = 1.3


class ContrastConfig(BaseModel):
    enabled: bool = False
    min: float = 0.7
    max: float = 1.3


class SaturationConfig(BaseModel):
    enabled: bool = False
    min: float = 0.7
    max: float = 1.3


class NoiseConfig(BaseModel):
    enabled: bool = False
    type: str = "gaussian"  # gaussian, salt_pepper
    intensity: float = 0.02


class BlurConfig(BaseModel):
    enabled: bool = False
    radius: float = 1.0


class ResizeConfig(BaseModel):
    enabled: bool = False
    width: Optional[int] = None
    height: Optional[int] = None
    keepAspect: bool = True


class AugmentationRequest(BaseModel):
    enabled: bool = False
    brightness: Optional[BrightnessConfig] = None
    contrast: Optional[ContrastConfig] = None
    saturation: Optional[SaturationConfig] = None
    noise: Optional[NoiseConfig] = None
    blur: Optional[BlurConfig] = None
    resize: Optional[ResizeConfig] = None
    count: int = 1  # 每张图生成多少张增强图


class ExportRequest(BaseModel):
    format: str = "coco"  # coco 或 yolo
    seed: Optional[int] = 42
    train_ratio: Optional[float] = 0.8
    val_ratio: Optional[float] = 0.1
    test_ratio: Optional[float] = 0.1
    augmentation: Optional[AugmentationRequest] = None


class ExportStats(BaseModel):
    project_id: int
    project_name: str
    format: str
    seed: int
    total_images: int
    total_annotations: int
    total_classes: int
    splits: dict
    augmentation_enabled: bool = False


@router.post("/projects/{project_id}/export", response_model=ExportStats)
def export_dataset_info(project_id: int, body: ExportRequest):
    """
    获取数据集导出的统计信息（不实际导出文件）

    Args:
        project_id: 项目ID
        body: 导出配置

    Returns:
        导出统计信息
    """
    if body.format not in ["coco", "yolo"]:
        raise HTTPException(400, "Format must be 'coco' or 'yolo'")

    # 验证比例
    if abs(body.train_ratio + body.val_ratio + body.test_ratio - 1.0) > 0.01:
        raise HTTPException(400, "Train, val, and test ratios must sum to 1.0")

    # 解析增强配置（包括 resize，resize 独立于数据增强开关）
    aug_config = None
    if body.augmentation:
        aug_config = _parse_augmentation_request(body.augmentation)

    try:
        # 创建临时目录
        with tempfile.TemporaryDirectory() as tmp_dir:
            if body.format == "yolo":
                result = export_dataset_to_yolo(
                    project_id=project_id,
                    output_dir=tmp_dir,
                    seed=body.seed,
                    train_ratio=body.train_ratio,
                    val_ratio=body.val_ratio,
                    test_ratio=body.test_ratio,
                    augmentation_config=aug_config
                )
            else:
                result = export_dataset_to_coco(
                    project_id=project_id,
                    output_dir=tmp_dir,
                    seed=body.seed,
                    train_ratio=body.train_ratio,
                    val_ratio=body.val_ratio,
                    test_ratio=body.test_ratio,
                    augmentation_config=aug_config
                )
            result['format'] = body.format
            return result
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"Export failed: {e}")


@router.post("/projects/{project_id}/export/local")
def export_dataset_local(
    project_id: int,
    body: ExportRequest
):
    """
    导出数据集到服务器本地目录（不下载）

    Args:
        project_id: 项目ID
        body: 导出配置（包含增强选项）

    Returns:
        生成的压缩包路径信息
    """
    if body.format not in ["coco", "yolo"]:
        raise HTTPException(400, "Format must be 'coco' or 'yolo'")

    # 验证比例
    if abs(body.train_ratio + body.val_ratio + body.test_ratio - 1.0) > 0.01:
        raise HTTPException(400, "Train, val, and test ratios must sum to 1.0")

    # 解析增强配置（包括 resize，resize 独立于数据增强开关）
    aug_config = None
    if body.augmentation:
        aug_config = _parse_augmentation_request(body.augmentation)

    try:
        # 确保 backend/data 目录存在
        backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        data_dir = os.path.join(backend_dir, "data")
        os.makedirs(data_dir, exist_ok=True)

        # 创建临时目录用于导出
        tmp_dir = tempfile.mkdtemp()
        export_dir = os.path.join(tmp_dir, "export")

        # 根据格式导出数据集
        if body.format == "yolo":
            result = export_dataset_to_yolo(
                project_id=project_id,
                output_dir=export_dir,
                seed=body.seed,
                train_ratio=body.train_ratio,
                val_ratio=body.val_ratio,
                test_ratio=body.test_ratio,
                augmentation_config=aug_config
            )
        else:
            result = export_dataset_to_coco(
                project_id=project_id,
                output_dir=export_dir,
                seed=body.seed,
                train_ratio=body.train_ratio,
                val_ratio=body.val_ratio,
                test_ratio=body.test_ratio,
                augmentation_config=aug_config
            )

        # 创建ZIP文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_project_name = re.sub(r'[^\w\u4e00-\u9fff-]', '_', result['project_name'])
        safe_project_name = re.sub(r'_+', '_', safe_project_name).strip('_')
        aug_suffix = "_aug" if result.get('augmentation_enabled') else ""
        zip_filename = f"{safe_project_name}_{body.format}{aug_suffix}_{timestamp}.zip"
        zip_path = os.path.join(data_dir, zip_filename)

        # 创建ZIP文件
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(export_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, export_dir)
                    zipf.write(file_path, arcname)

        # 清理临时目录
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

        # 获取文件大小
        file_size = os.path.getsize(zip_path)

        return {
            "success": True,
            "path": zip_path,
            "filename": zip_filename,
            "size": file_size,
            "size_human": _format_file_size(file_size),
            "project_name": result['project_name'],
            "format": body.format,
            "total_images": result['total_images'],
            "augmentation_enabled": result.get('augmentation_enabled', False)
        }

    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"Export failed: {e}")


def _format_file_size(size_bytes: int) -> str:
    """将字节数转换为人类可读的格式"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


@router.post("/projects/{project_id}/export/download")
def download_dataset_post(
    project_id: int,
    body: ExportRequest
):
    """
    导出并下载数据集的压缩包（POST方法，支持增强配置）

    Args:
        project_id: 项目ID
        body: 导出配置（包含增强选项）

    Returns:
        数据集压缩包文件
    """
    if body.format not in ["coco", "yolo"]:
        raise HTTPException(400, "Format must be 'coco' or 'yolo'")

    # 验证比例
    if abs(body.train_ratio + body.val_ratio + body.test_ratio - 1.0) > 0.01:
        raise HTTPException(400, "Train, val, and test ratios must sum to 1.0")

    # 解析增强配置（包括 resize，resize 独立于数据增强开关）
    aug_config = None
    if body.augmentation:
        aug_config = _parse_augmentation_request(body.augmentation)

    try:
        # 创建临时目录
        tmp_dir = tempfile.mkdtemp()
        export_dir = os.path.join(tmp_dir, "export")

        # 根据格式导出数据集
        if body.format == "yolo":
            result = export_dataset_to_yolo(
                project_id=project_id,
                output_dir=export_dir,
                seed=body.seed,
                train_ratio=body.train_ratio,
                val_ratio=body.val_ratio,
                test_ratio=body.test_ratio,
                augmentation_config=aug_config
            )
        else:
            result = export_dataset_to_coco(
                project_id=project_id,
                output_dir=export_dir,
                seed=body.seed,
                train_ratio=body.train_ratio,
                val_ratio=body.val_ratio,
                test_ratio=body.test_ratio,
                augmentation_config=aug_config
            )

        # 创建ZIP文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 清理项目名称，移除特殊字符，只保留字母、数字、中文和连字符
        safe_project_name = re.sub(r'[^\w\u4e00-\u9fff-]', '_', result['project_name'])
        # 移除连续的下划线
        safe_project_name = re.sub(r'_+', '_', safe_project_name).strip('_')

        # 如果启用了增强，在文件名中标注
        aug_suffix = "_aug" if result.get('augmentation_enabled') else ""
        zip_filename = f"{safe_project_name}_{body.format}{aug_suffix}_{timestamp}.zip"
        zip_path = os.path.join(tmp_dir, zip_filename)

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(export_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, export_dir)
                    zipf.write(file_path, arcname)

        # 返回文件 - 使用 RFC 5987 规范处理中文文件名
        encoded_filename = quote(zip_filename, safe='')
        return FileResponse(
            path=zip_path,
            media_type="application/zip",
            filename=zip_filename,
            headers={
                "Content-Disposition": f"attachment; filename*=UTF-8''{encoded_filename}"
            }
        )

    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"Export failed: {e}")


@router.get("/projects/{project_id}/export/download")
def download_dataset(
    project_id: int,
    format: str = Query(default="coco"),
    seed: int = Query(default=42)
):
    """
    导出并下载数据集的压缩包（GET方法，简单导出无增强）

    Args:
        project_id: 项目ID
        format: 导出格式 (coco 或 yolo)
        seed: 随机种子

    Returns:
        数据集压缩包文件
    """
    if format not in ["coco", "yolo"]:
        raise HTTPException(400, "Format must be 'coco' or 'yolo'")

    try:
        # 创建临时目录
        tmp_dir = tempfile.mkdtemp()
        export_dir = os.path.join(tmp_dir, "export")

        # 根据格式导出数据集
        if format == "yolo":
            result = export_dataset_to_yolo(
                project_id=project_id,
                output_dir=export_dir,
                seed=seed
            )
        else:
            result = export_dataset_to_coco(
                project_id=project_id,
                output_dir=export_dir,
                seed=seed
            )

        # 创建ZIP文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 清理项目名称，移除特殊字符，只保留字母、数字、中文和连字符
        safe_project_name = re.sub(r'[^\w\u4e00-\u9fff-]', '_', result['project_name'])
        # 移除连续的下划线
        safe_project_name = re.sub(r'_+', '_', safe_project_name).strip('_')
        zip_filename = f"{safe_project_name}_{format}_{timestamp}.zip"
        zip_path = os.path.join(tmp_dir, zip_filename)

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(export_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, export_dir)
                    zipf.write(file_path, arcname)

        # 返回文件 - 使用 RFC 5987 规范处理中文文件名
        encoded_filename = quote(zip_filename, safe='')
        return FileResponse(
            path=zip_path,
            media_type="application/zip",
            filename=zip_filename,
            headers={
                "Content-Disposition": f"attachment; filename*=UTF-8''{encoded_filename}"
            }
        )

    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"Export failed: {e}")


def _parse_augmentation_request(aug_req: AugmentationRequest) -> AugmentationConfig:
    """
    将API请求的增强配置转换为内部配置对象
    """
    config = AugmentationConfig()
    config.enabled = aug_req.enabled

    # Resize 是独立于数据增强开关的，始终解析
    if aug_req.resize and aug_req.resize.enabled:
        config.resize_enabled = True
        config.resize_width = aug_req.resize.width
        config.resize_height = aug_req.resize.height
        config.resize_keep_aspect = aug_req.resize.keepAspect

    # 如果数据增强未启用，只返回 resize 配置
    if not config.enabled:
        return config

    # 亮度
    if aug_req.brightness and aug_req.brightness.enabled:
        config.brightness_enabled = True
        config.brightness_range = (aug_req.brightness.min, aug_req.brightness.max)

    # 对比度
    if aug_req.contrast and aug_req.contrast.enabled:
        config.contrast_enabled = True
        config.contrast_range = (aug_req.contrast.min, aug_req.contrast.max)

    # 饱和度
    if aug_req.saturation and aug_req.saturation.enabled:
        config.saturation_enabled = True
        config.saturation_range = (aug_req.saturation.min, aug_req.saturation.max)

    # 噪音
    if aug_req.noise and aug_req.noise.enabled:
        config.noise_enabled = True
        config.noise_type = aug_req.noise.type
        config.noise_intensity = aug_req.noise.intensity

    # 模糊
    if aug_req.blur and aug_req.blur.enabled:
        config.blur_enabled = True
        config.blur_radius = aug_req.blur.radius

    # 增强倍数
    config.augment_count = aug_req.count

    return config
