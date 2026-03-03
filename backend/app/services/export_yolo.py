import os
import shutil
import yaml
from typing import List, Dict, Optional

from ..db import get_session
from ..models import Project, Class, Image, Annotation
from .export_coco import split_dataset
from .augmentation import AugmentationConfig, augment_image_with_annotations
from ..utils.oss_storage import resolve_local_path, get_basename


def export_dataset_to_yolo(
    project_id: int,
    output_dir: str,
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    augmentation_config: Optional[AugmentationConfig] = None
) -> Dict:
    """
    导出项目数据集为YOLO格式

    YOLO格式结构:
    output_dir/
    ├── data.yaml
    ├── train/
    │   ├── images/
    │   │   └── *.jpg
    │   └── labels/
    │       └── *.txt
    ├── valid/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/

    YOLO标注格式 (每行一个目标):
    <class_id> <x_center> <y_center> <width> <height>
    所有坐标都是相对于图像尺寸的归一化值 (0-1)

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
            os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
            os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)

        # 创建类别ID到索引的映射
        class_id_map = {cls.id: idx for idx, cls in enumerate(classes)}

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

            images_dir = os.path.join(output_dir, split_name, 'images')
            labels_dir = os.path.join(output_dir, split_name, 'labels')

            image_count = 0
            annotation_count = 0

            for img in split_images:
                img_annotations = [ann for ann in split_annotations if ann.imageId == img.id]

                # 支持 OSS URL 和本地路径，自动下载缓存
                src_path = resolve_local_path(img.path)
                if src_path is None:
                    print(f"[export_yolo] 跳过无法定位的图片: {img.path}")
                    continue

                base_filename = get_basename(img.path)
                name_without_ext = os.path.splitext(base_filename)[0]

                # 是否对此划分应用增强（仅训练集）
                should_augment = apply_augmentation and split_name == 'train'

                if should_augment or (augmentation_config and augmentation_config.resize_enabled):
                    # 需要应用增强或resize
                    # 构建原始标注
                    original_anns = []
                    for ann in img_annotations:
                        if not img.width or not img.height:
                            continue
                        original_anns.append({
                            'bbox': [ann.x, ann.y, ann.w, ann.h],
                            'area': ann.w * ann.h,
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
                        original_anns,
                        resize_only_config,
                        images_dir,
                        base_filename,
                        aug_index=0
                    )

                    # 写入YOLO格式标注文件
                    label_path = os.path.join(labels_dir, f"{name_without_ext}.txt")
                    ann_count = _write_yolo_labels(label_path, adjusted_anns, new_width, new_height, class_id_map)
                    image_count += 1
                    annotation_count += ann_count

                    # 如果需要增强，生成额外的增强图片
                    if should_augment and augmentation_config.augment_count > 0:
                        for aug_idx in range(1, augmentation_config.augment_count + 1):
                            aug_output_path, aug_anns, aug_width, aug_height = augment_image_with_annotations(
                                src_path,
                                original_anns,
                                augmentation_config,
                                images_dir,
                                base_filename,
                                aug_index=aug_idx
                            )

                            # 写入增强图片的标注
                            aug_label_name = f"{name_without_ext}_aug{aug_idx}.txt"
                            aug_label_path = os.path.join(labels_dir, aug_label_name)
                            aug_ann_count = _write_yolo_labels(aug_label_path, aug_anns, aug_width, aug_height, class_id_map)
                            image_count += 1
                            annotation_count += aug_ann_count

                else:
                    # 不需要增强，直接复制图片
                    dst_path = os.path.join(images_dir, base_filename)
                    shutil.copy2(src_path, dst_path)

                    # 写入YOLO格式标注文件
                    label_path = os.path.join(labels_dir, f"{name_without_ext}.txt")
                    ann_count = _write_yolo_labels_from_db(
                        label_path, img_annotations, img.width, img.height, class_id_map
                    )
                    image_count += 1
                    annotation_count += ann_count

            stats[split_name]['images'] = image_count
            stats[split_name]['annotations'] = annotation_count

        # 生成 data.yaml 配置文件
        data_yaml = {
            'path': '.',
            'train': 'train/images',
            'val': 'valid/images',
            'test': 'test/images',
            'nc': len(classes),
            'names': [cls.name for cls in classes]
        }

        yaml_path = os.path.join(output_dir, 'data.yaml')
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)

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


def _write_yolo_labels(
    label_path: str,
    annotations: List[Dict],
    img_width: int,
    img_height: int,
    class_id_map: Dict[int, int]
) -> int:
    """
    将标注写入YOLO格式的txt文件

    Args:
        label_path: 标注文件路径
        annotations: 标注列表 (包含 bbox, classId)
        img_width: 图像宽度
        img_height: 图像高度
        class_id_map: 类别ID到索引的映射

    Returns:
        写入的标注数量
    """
    lines = []
    for ann in annotations:
        class_id = ann.get('classId')
        if class_id not in class_id_map:
            continue

        class_idx = class_id_map[class_id]
        bbox = ann['bbox']  # [x, y, w, h] 像素坐标

        # 转换为YOLO格式 (归一化的中心点坐标和宽高)
        x_center = (bbox[0] + bbox[2] / 2) / img_width
        y_center = (bbox[1] + bbox[3] / 2) / img_height
        width = bbox[2] / img_width
        height = bbox[3] / img_height

        # 确保值在0-1范围内
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        width = max(0, min(1, width))
        height = max(0, min(1, height))

        lines.append(f"{class_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    with open(label_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    return len(lines)


def _write_yolo_labels_from_db(
    label_path: str,
    annotations: List[Annotation],
    img_width: int,
    img_height: int,
    class_id_map: Dict[int, int]
) -> int:
    """
    从数据库标注对象写入YOLO格式的txt文件

    Args:
        label_path: 标注文件路径
        annotations: 数据库标注对象列表
        img_width: 图像宽度
        img_height: 图像高度
        class_id_map: 类别ID到索引的映射

    Returns:
        写入的标注数量
    """
    if not img_width or not img_height:
        with open(label_path, 'w', encoding='utf-8') as f:
            f.write('')
        return 0

    lines = []
    for ann in annotations:
        if ann.classId not in class_id_map:
            continue

        class_idx = class_id_map[ann.classId]

        # 数据库存储的是像素坐标 (left_x, top_y, width, height)
        # 转换为YOLO格式 (归一化的中心点坐标和宽高)
        x_center = (ann.x + ann.w / 2) / img_width
        y_center = (ann.y + ann.h / 2) / img_height
        width = ann.w / img_width
        height = ann.h / img_height

        # 确保值在0-1范围内
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        width = max(0, min(1, width))
        height = max(0, min(1, height))

        lines.append(f"{class_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    with open(label_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    return len(lines)
