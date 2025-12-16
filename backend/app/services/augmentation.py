"""
数据增强服务模块
支持图片亮度变化、噪音添加、resize等增强操作
"""
import os
import random
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np


@dataclass
class AugmentationConfig:
    """数据增强配置"""
    # 是否启用增强
    enabled: bool = False

    # 亮度变化
    brightness_enabled: bool = False
    brightness_range: Tuple[float, float] = (0.7, 1.3)  # 亮度系数范围

    # 对比度变化
    contrast_enabled: bool = False
    contrast_range: Tuple[float, float] = (0.7, 1.3)

    # 饱和度变化
    saturation_enabled: bool = False
    saturation_range: Tuple[float, float] = (0.7, 1.3)

    # 噪音
    noise_enabled: bool = False
    noise_type: str = "gaussian"  # gaussian, salt_pepper
    noise_intensity: float = 0.02  # 噪音强度 0-1

    # 模糊
    blur_enabled: bool = False
    blur_radius: float = 1.0

    # 图片resize
    resize_enabled: bool = False
    resize_width: Optional[int] = None
    resize_height: Optional[int] = None
    resize_keep_aspect: bool = True  # 保持宽高比

    # 增强倍数（每张图生成多少张增强图）
    augment_count: int = 1


def add_gaussian_noise(image: Image.Image, intensity: float = 0.02) -> Image.Image:
    """
    添加高斯噪音

    Args:
        image: PIL图像
        intensity: 噪音强度 (0-1)

    Returns:
        添加噪音后的图像
    """
    img_array = np.array(image).astype(np.float32)

    # 生成高斯噪音
    noise = np.random.normal(0, intensity * 255, img_array.shape)

    # 添加噪音并裁剪到有效范围
    noisy_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)

    return Image.fromarray(noisy_array)


def add_salt_pepper_noise(image: Image.Image, intensity: float = 0.02) -> Image.Image:
    """
    添加椒盐噪音

    Args:
        image: PIL图像
        intensity: 噪音强度 (0-1)

    Returns:
        添加噪音后的图像
    """
    img_array = np.array(image).copy()

    # 计算噪点数量
    num_pixels = img_array.shape[0] * img_array.shape[1]
    num_noise = int(num_pixels * intensity)

    # 添加盐噪音（白点）
    for _ in range(num_noise // 2):
        y = random.randint(0, img_array.shape[0] - 1)
        x = random.randint(0, img_array.shape[1] - 1)
        img_array[y, x] = 255

    # 添加椒噪音（黑点）
    for _ in range(num_noise // 2):
        y = random.randint(0, img_array.shape[0] - 1)
        x = random.randint(0, img_array.shape[1] - 1)
        img_array[y, x] = 0

    return Image.fromarray(img_array)


def resize_image(
    image: Image.Image,
    target_width: Optional[int],
    target_height: Optional[int],
    keep_aspect: bool = True
) -> Tuple[Image.Image, float, float]:
    """
    调整图片尺寸

    Args:
        image: PIL图像
        target_width: 目标宽度
        target_height: 目标高度
        keep_aspect: 是否保持宽高比

    Returns:
        (调整后的图像, 宽度缩放比例, 高度缩放比例)
    """
    original_width, original_height = image.size

    if target_width is None and target_height is None:
        return image, 1.0, 1.0

    if keep_aspect:
        # 保持宽高比
        if target_width and target_height:
            # 以较小的缩放比例为准
            width_ratio = target_width / original_width
            height_ratio = target_height / original_height
            ratio = min(width_ratio, height_ratio)
            new_width = int(original_width * ratio)
            new_height = int(original_height * ratio)
        elif target_width:
            ratio = target_width / original_width
            new_width = target_width
            new_height = int(original_height * ratio)
        else:
            ratio = target_height / original_height
            new_width = int(original_width * ratio)
            new_height = target_height
    else:
        # 不保持宽高比
        new_width = target_width or original_width
        new_height = target_height or original_height

    resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    width_scale = new_width / original_width
    height_scale = new_height / original_height

    return resized, width_scale, height_scale


def apply_augmentation(
    image: Image.Image,
    config: AugmentationConfig,
    seed: Optional[int] = None
) -> Image.Image:
    """
    对图片应用数据增强

    Args:
        image: PIL图像
        config: 增强配置
        seed: 随机种子（用于可重复的增强）

    Returns:
        增强后的图像
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    result = image.copy()

    # 亮度变化
    if config.brightness_enabled:
        factor = random.uniform(*config.brightness_range)
        enhancer = ImageEnhance.Brightness(result)
        result = enhancer.enhance(factor)

    # 对比度变化
    if config.contrast_enabled:
        factor = random.uniform(*config.contrast_range)
        enhancer = ImageEnhance.Contrast(result)
        result = enhancer.enhance(factor)

    # 饱和度变化
    if config.saturation_enabled:
        factor = random.uniform(*config.saturation_range)
        enhancer = ImageEnhance.Color(result)
        result = enhancer.enhance(factor)

    # 添加噪音
    if config.noise_enabled:
        if config.noise_type == "gaussian":
            result = add_gaussian_noise(result, config.noise_intensity)
        elif config.noise_type == "salt_pepper":
            result = add_salt_pepper_noise(result, config.noise_intensity)

    # 添加模糊
    if config.blur_enabled:
        result = result.filter(ImageFilter.GaussianBlur(radius=config.blur_radius))

    return result


def augment_image_with_annotations(
    image_path: str,
    annotations: List[Dict],
    config: AugmentationConfig,
    output_dir: str,
    base_filename: str,
    aug_index: int = 0
) -> Tuple[str, List[Dict], int, int]:
    """
    对图片及其标注进行增强处理

    Args:
        image_path: 原始图片路径
        annotations: 标注列表（COCO格式的bbox）
        config: 增强配置
        output_dir: 输出目录
        base_filename: 基础文件名
        aug_index: 增强索引（用于区分多次增强）

    Returns:
        (新图片路径, 调整后的标注列表, 新宽度, 新高度)
    """
    # 打开图片
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    original_width, original_height = image.size

    # 应用数据增强（不含resize）
    if config.enabled:
        augmented = apply_augmentation(image, config, seed=aug_index)
    else:
        augmented = image.copy()

    # 处理resize
    width_scale = 1.0
    height_scale = 1.0
    if config.resize_enabled:
        augmented, width_scale, height_scale = resize_image(
            augmented,
            config.resize_width,
            config.resize_height,
            config.resize_keep_aspect
        )

    new_width, new_height = augmented.size

    # 调整标注坐标
    adjusted_annotations = []
    for ann in annotations:
        new_ann = ann.copy()
        bbox = ann['bbox']  # [x, y, w, h] COCO格式（绝对坐标）
        new_bbox = [
            bbox[0] * width_scale,
            bbox[1] * height_scale,
            bbox[2] * width_scale,
            bbox[3] * height_scale
        ]
        new_ann['bbox'] = new_bbox
        new_ann['area'] = new_bbox[2] * new_bbox[3]
        adjusted_annotations.append(new_ann)

    # 生成新文件名
    name, ext = os.path.splitext(base_filename)
    if aug_index > 0:
        new_filename = f"{name}_aug{aug_index}{ext}"
    else:
        new_filename = base_filename

    # 保存图片
    output_path = os.path.join(output_dir, new_filename)
    augmented.save(output_path, quality=95)

    return output_path, adjusted_annotations, new_width, new_height


def parse_augmentation_config(params: Dict) -> AugmentationConfig:
    """
    从API参数解析增强配置

    Args:
        params: API请求参数

    Returns:
        AugmentationConfig对象
    """
    config = AugmentationConfig()

    # 检查是否启用增强
    augmentation = params.get('augmentation', {})
    if not augmentation:
        return config

    config.enabled = augmentation.get('enabled', False)

    if not config.enabled:
        return config

    # 亮度
    brightness = augmentation.get('brightness', {})
    config.brightness_enabled = brightness.get('enabled', False)
    if config.brightness_enabled:
        config.brightness_range = (
            brightness.get('min', 0.7),
            brightness.get('max', 1.3)
        )

    # 对比度
    contrast = augmentation.get('contrast', {})
    config.contrast_enabled = contrast.get('enabled', False)
    if config.contrast_enabled:
        config.contrast_range = (
            contrast.get('min', 0.7),
            contrast.get('max', 1.3)
        )

    # 饱和度
    saturation = augmentation.get('saturation', {})
    config.saturation_enabled = saturation.get('enabled', False)
    if config.saturation_enabled:
        config.saturation_range = (
            saturation.get('min', 0.7),
            saturation.get('max', 1.3)
        )

    # 噪音
    noise = augmentation.get('noise', {})
    config.noise_enabled = noise.get('enabled', False)
    if config.noise_enabled:
        config.noise_type = noise.get('type', 'gaussian')
        config.noise_intensity = noise.get('intensity', 0.02)

    # 模糊
    blur = augmentation.get('blur', {})
    config.blur_enabled = blur.get('enabled', False)
    if config.blur_enabled:
        config.blur_radius = blur.get('radius', 1.0)

    # Resize
    resize = augmentation.get('resize', {})
    config.resize_enabled = resize.get('enabled', False)
    if config.resize_enabled:
        config.resize_width = resize.get('width')
        config.resize_height = resize.get('height')
        config.resize_keep_aspect = resize.get('keepAspect', True)

    # 增强倍数
    config.augment_count = augmentation.get('count', 1)

    return config
