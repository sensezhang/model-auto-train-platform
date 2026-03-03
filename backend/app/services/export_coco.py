import json
import os
import shutil
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import random

from ..db import get_session
from ..models import Project, Class, Image, Annotation
from .augmentation import AugmentationConfig, augment_image_with_annotations
from ..utils.oss_storage import resolve_local_path


def split_dataset(
    image_ids: List[int],
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1
) -> Tuple[List[int], List[int], List[int]]:
    """
    将图像ID列表按照指定比例划分训练集、验证集、测试集

    Args:
        image_ids: 所有图像ID列表
        seed: 随机种子，保证可重复性
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例

    Returns:
        (train_ids, val_ids, test_ids) 三个集合的图像ID列表
    """
    random.seed(seed)
    ids = image_ids.copy()
    random.shuffle(ids)

    total = len(ids)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_ids = ids[:train_end]
    val_ids = ids[train_end:val_end]
    test_ids = ids[val_end:]

    return train_ids, val_ids, test_ids


def build_coco_annotations(
    images: List[Image],
    annotations: List[Annotation],
    classes: List[Class],
    split_name: str,
    project_name: str,
    image_size_overrides: Optional[Dict[int, Tuple[int, int]]] = None
) -> Dict:
    """
    构建COCO格式的标注数据

    Args:
        images: 图像列表
        annotations: 标注列表
        classes: 类别列表
        split_name: 数据集划分名称 (train/val/test)
        project_name: 项目名称
        image_size_overrides: 图像尺寸覆盖 {image_id: (new_width, new_height)}

    Returns:
        COCO格式的字典
    """
    # 创建图像ID到索引的映射
    image_id_map = {img.id: idx for idx, img in enumerate(images)}

    # COCO格式的images列表
    coco_images = []
    for idx, img in enumerate(images):
        # 检查是否有尺寸覆盖
        if image_size_overrides and img.id in image_size_overrides:
            width, height = image_size_overrides[img.id]
        else:
            width = img.width or 0
            height = img.height or 0

        coco_images.append({
            "id": idx,
            "file_name": os.path.basename(img.path),
            "width": width,
            "height": height,
        })

    # 创建类别ID到索引的映射
    class_id_map = {cls.id: idx for idx, cls in enumerate(classes)}

    # COCO格式的categories列表
    coco_categories = []
    for idx, cls in enumerate(classes):
        coco_categories.append({
            "id": idx,
            "name": cls.name,
            "supercategory": "object"
        })

    # COCO格式的annotations列表
    coco_annotations = []
    ann_id = 0
    for ann in annotations:
        if ann.imageId not in image_id_map:
            continue
        if ann.classId not in class_id_map:
            continue

        image_idx = image_id_map[ann.imageId]
        category_idx = class_id_map[ann.classId]

        # 获取对应的图像以计算绝对坐标
        img = next((i for i in images if i.id == ann.imageId), None)
        if not img or not img.width or not img.height:
            continue

        # 检查是否有尺寸覆盖（用于resize后的坐标计算）
        if image_size_overrides and img.id in image_size_overrides:
            target_width, target_height = image_size_overrides[img.id]
            width_scale = target_width / img.width
            height_scale = target_height / img.height
        else:
            target_width, target_height = img.width, img.height
            width_scale = 1.0
            height_scale = 1.0

        # 数据库存储的是像素坐标 (left_x, top_y, width, height) - 左上角坐标
        # COCO格式需要 (x_min, y_min, width, height) 绝对像素坐标
        # 如果有resize，需要按比例缩放
        abs_x = ann.x * width_scale
        abs_y = ann.y * height_scale
        abs_w = ann.w * width_scale
        abs_h = ann.h * height_scale

        coco_annotations.append({
            "id": ann_id,
            "image_id": image_idx,
            "category_id": category_idx,
            "bbox": [abs_x, abs_y, abs_w, abs_h],
            "area": abs_w * abs_h,
            "iscrowd": 0,
            "segmentation": []  # COCO格式要求，目标检测任务留空
        })
        ann_id += 1

    # 构建完整的COCO格式数据
    coco_data = {
        "info": {
            "description": f"{project_name} {split_name} dataset",
            "version": "1.0",
            "year": datetime.now().year,
            "date_created": datetime.now().isoformat(),
        },
        "licenses": [],
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": coco_categories,
    }

    return coco_data


def export_dataset_to_coco(
    project_id: int,
    output_dir: str,
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    augmentation_config: Optional[AugmentationConfig] = None
) -> Dict:
    """
    导出项目数据集为COCO格式

    Args:
        project_id: 项目ID
        output_dir: 输出目录路径
        seed: 随机种子，用于数据集划分
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        augmentation_config: 数据增强配置

    Returns:
        导出统计信息
    """
    with get_session() as session:
        # 获取项目信息
        project = session.get(Project, project_id)
        if not project:
            raise ValueError(f"Project {project_id} not found")

        # 获取所有类别
        classes = session.query(Class).filter(Class.projectId == project_id).all()
        if not classes:
            raise ValueError(f"No classes found for project {project_id}")

        # 获取所有图像
        images = session.query(Image).filter(Image.projectId == project_id).all()
        if not images:
            raise ValueError(f"No images found for project {project_id}")

        # 获取所有标注
        image_ids = [img.id for img in images]
        annotations = session.query(Annotation).filter(
            Annotation.imageId.in_(image_ids)
        ).all()

        # 划分数据集
        train_ids, val_ids, test_ids = split_dataset(
            image_ids, seed, train_ratio, val_ratio, test_ratio
        )

        # 创建输出目录结构
        os.makedirs(output_dir, exist_ok=True)
        for split in ['train', 'valid', 'test']:
            os.makedirs(os.path.join(output_dir, split), exist_ok=True)

        # 统计信息
        stats = {
            'train': {'images': 0, 'annotations': 0},
            'valid': {'images': 0, 'annotations': 0},
            'test': {'images': 0, 'annotations': 0}
        }

        # 处理每个数据集划分
        splits = {
            'train': train_ids,
            'valid': val_ids,
            'test': test_ids
        }

        # 确定是否需要增强（仅对训练集应用增强）
        apply_augmentation = (
            augmentation_config is not None and
            augmentation_config.enabled
        )

        for split_name, split_image_ids in splits.items():
            # 过滤该划分的图像和标注
            split_images = [img for img in images if img.id in split_image_ids]
            split_annotations = [ann for ann in annotations if ann.imageId in split_image_ids]

            split_dir = os.path.join(output_dir, split_name)

            # 用于存储增强后的图像信息
            augmented_images_info = []
            augmented_annotations_info = []
            image_size_overrides = {}

            for img in split_images:
                img_annotations = [ann for ann in split_annotations if ann.imageId == img.id]
                src_path = resolve_local_path(img.path)

                if not src_path:
                    continue

                base_filename = os.path.basename(img.path)

                # 是否对此划分应用增强（仅训练集）
                should_augment = apply_augmentation and split_name == 'train'

                if should_augment or (augmentation_config and augmentation_config.resize_enabled):
                    # 需要应用增强或resize
                    from PIL import Image as PILImage

                    # 构建原始标注的COCO格式bbox
                    # 数据库存储的是像素坐标 (left_x, top_y, width, height) - 左上角坐标
                    original_coco_anns = []
                    for ann in img_annotations:
                        if not img.width or not img.height:
                            continue
                        # 已经是左上角坐标，直接使用
                        abs_x = ann.x
                        abs_y = ann.y
                        abs_w = ann.w
                        abs_h = ann.h
                        original_coco_anns.append({
                            'bbox': [abs_x, abs_y, abs_w, abs_h],
                            'area': abs_w * abs_h,
                            'imageId': img.id,
                            'classId': ann.classId,
                            'ann_id': ann.id
                        })

                    # 处理原始图片（可能只有resize）
                    resize_only_config = AugmentationConfig(
                        enabled=False,
                        resize_enabled=augmentation_config.resize_enabled if augmentation_config else False,
                        resize_width=augmentation_config.resize_width if augmentation_config else None,
                        resize_height=augmentation_config.resize_height if augmentation_config else None,
                        resize_keep_aspect=augmentation_config.resize_keep_aspect if augmentation_config else True
                    )

                    output_path, adjusted_anns, new_width, new_height = augment_image_with_annotations(
                        src_path,
                        original_coco_anns,
                        resize_only_config,
                        split_dir,
                        base_filename,
                        aug_index=0
                    )

                    # 记录尺寸变化
                    if augmentation_config and augmentation_config.resize_enabled:
                        image_size_overrides[img.id] = (new_width, new_height)

                    # 如果需要增强，生成额外的增强图片
                    if should_augment and augmentation_config.augment_count > 0:
                        for aug_idx in range(1, augmentation_config.augment_count + 1):
                            aug_output_path, aug_anns, aug_width, aug_height = augment_image_with_annotations(
                                src_path,
                                original_coco_anns,
                                augmentation_config,
                                split_dir,
                                base_filename,
                                aug_index=aug_idx
                            )

                            # 创建虚拟图像记录用于增强图片
                            aug_filename = os.path.basename(aug_output_path)
                            augmented_images_info.append({
                                'original_id': img.id,
                                'filename': aug_filename,
                                'width': aug_width,
                                'height': aug_height,
                                'annotations': aug_anns
                            })

                else:
                    # 不需要增强，直接复制
                    dst_path = os.path.join(split_dir, base_filename)
                    shutil.copy2(src_path, dst_path)

            # 构建COCO格式数据
            coco_data = build_coco_annotations(
                split_images,
                split_annotations,
                classes,
                split_name,
                project.name,
                image_size_overrides if image_size_overrides else None
            )

            # 添加增强图片的信息到COCO数据
            if augmented_images_info:
                # 获取当前最大的image_id和annotation_id
                max_image_id = max([img['id'] for img in coco_data['images']], default=-1)
                max_ann_id = max([ann['id'] for ann in coco_data['annotations']], default=-1)

                # 创建类别ID映射
                class_id_map = {cls.id: idx for idx, cls in enumerate(classes)}

                for aug_info in augmented_images_info:
                    max_image_id += 1
                    new_image = {
                        'id': max_image_id,
                        'file_name': aug_info['filename'],
                        'width': aug_info['width'],
                        'height': aug_info['height']
                    }
                    coco_data['images'].append(new_image)

                    # 添加增强图片的标注
                    for aug_ann in aug_info['annotations']:
                        if aug_ann['classId'] not in class_id_map:
                            continue
                        max_ann_id += 1
                        new_ann = {
                            'id': max_ann_id,
                            'image_id': max_image_id,
                            'category_id': class_id_map[aug_ann['classId']],
                            'bbox': aug_ann['bbox'],
                            'area': aug_ann['area'],
                            'iscrowd': 0,
                            'segmentation': []
                        }
                        coco_data['annotations'].append(new_ann)

            # 保存标注文件
            # 使用ensure_ascii=True避免Windows上的gbk编码问题
            annotations_file = os.path.join(output_dir, split_name, '_annotations.coco.json')
            with open(annotations_file, 'w', encoding='utf-8') as f:
                json.dump(coco_data, f, indent=2, ensure_ascii=True)

            # 更新统计信息
            stats[split_name]['images'] = len(coco_data['images'])
            stats[split_name]['annotations'] = len(coco_data['annotations'])

        return {
            'project_id': project_id,
            'project_name': project.name,
            'output_dir': output_dir,
            'seed': seed,
            'total_images': sum(s['images'] for s in stats.values()),
            'total_annotations': sum(s['annotations'] for s in stats.values()),
            'total_classes': len(classes),
            'splits': stats,
            'augmentation_enabled': apply_augmentation
        }
