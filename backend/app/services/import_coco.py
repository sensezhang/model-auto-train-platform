"""
COCO格式数据集导入服务
与 export_coco.py 的导出格式完全匹配
"""
import json
import os
import shutil
import tempfile
import zipfile
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from ..db import get_session
from ..models import Project, Class, Image, Annotation
from ..utils.files import sha1_of_bytes, project_images_dir


def import_coco_dataset(
    project_id: int,
    zip_path: str,
    progress_callback: Optional[callable] = None
) -> Dict:
    """
    导入COCO格式的数据集ZIP文件

    ZIP文件结构应该是:
    dataset.zip/
        train/
            image1.jpg
            image2.jpg
            _annotations.coco.json
        valid/
            image3.jpg
            _annotations.coco.json
        test/
            image4.jpg
            _annotations.coco.json

    Args:
        project_id: 项目ID
        zip_path: ZIP文件路径
        progress_callback: 进度回调函数 callback(current, total, message)

    Returns:
        导入结果字典
    """
    with get_session() as session:
        project = session.get(Project, project_id)
        if not project:
            raise ValueError(f"Project {project_id} not found")

        # 获取项目的类别
        classes = session.query(Class).filter(Class.projectId == project_id).all()
        class_name_to_id = {cls.name: cls.id for cls in classes}

    # 创建临时目录解压ZIP
    with tempfile.TemporaryDirectory() as temp_dir:
        # 解压ZIP文件
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        # 查找根目录（可能有额外的嵌套目录）
        root_dir = temp_dir
        items = os.listdir(temp_dir)

        # 如果只有一个目录，进入它
        if len(items) == 1 and os.path.isdir(os.path.join(temp_dir, items[0])):
            root_dir = os.path.join(temp_dir, items[0])

        # 统计信息
        stats = {
            'total_images': 0,
            'imported_images': 0,
            'duplicate_images': 0,
            'error_images': 0,
            'total_annotations': 0,
            'imported_annotations': 0,
            'skipped_annotations': 0,
            'new_classes': [],
            'splits': {}
        }

        # 处理每个数据集划分 (train, valid, test)
        splits = ['train', 'valid', 'test']

        for split in splits:
            split_dir = os.path.join(root_dir, split)
            if not os.path.exists(split_dir):
                continue

            split_stats = import_split(
                project_id=project_id,
                split_dir=split_dir,
                split_name=split,
                class_name_to_id=class_name_to_id,
                progress_callback=progress_callback
            )

            # 更新统计信息
            stats['total_images'] += split_stats['total_images']
            stats['imported_images'] += split_stats['imported_images']
            stats['duplicate_images'] += split_stats['duplicate_images']
            stats['error_images'] += split_stats['error_images']
            stats['total_annotations'] += split_stats['total_annotations']
            stats['imported_annotations'] += split_stats['imported_annotations']
            stats['skipped_annotations'] += split_stats['skipped_annotations']
            stats['splits'][split] = split_stats

            # 收集新创建的类别
            for cls_name in split_stats['new_classes']:
                if cls_name not in stats['new_classes']:
                    stats['new_classes'].append(cls_name)

        return stats


def import_split(
    project_id: int,
    split_dir: str,
    split_name: str,
    class_name_to_id: Dict[str, int],
    progress_callback: Optional[callable] = None
) -> Dict:
    """
    导入单个数据集划分 (train/valid/test)

    Args:
        project_id: 项目ID
        split_dir: 划分目录路径
        split_name: 划分名称
        class_name_to_id: 类别名称到ID的映射（可修改）
        progress_callback: 进度回调

    Returns:
        导入统计信息
    """
    stats = {
        'total_images': 0,
        'imported_images': 0,
        'duplicate_images': 0,
        'error_images': 0,
        'total_annotations': 0,
        'imported_annotations': 0,
        'skipped_annotations': 0,
        'new_classes': []
    }

    # 读取COCO标注文件
    annotations_file = os.path.join(split_dir, '_annotations.coco.json')
    if not os.path.exists(annotations_file):
        print(f"Warning: {split_name}/_annotations.coco.json not found, skipping")
        return stats

    with open(annotations_file, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    # 解析COCO数据
    coco_images = {img['id']: img for img in coco_data.get('images', [])}
    coco_categories = {cat['id']: cat for cat in coco_data.get('categories', [])}
    coco_annotations = coco_data.get('annotations', [])

    stats['total_images'] = len(coco_images)
    stats['total_annotations'] = len(coco_annotations)

    # 检查并创建缺失的类别
    with get_session() as session:
        for cat_id, cat_info in coco_categories.items():
            cat_name = cat_info['name']
            if cat_name not in class_name_to_id:
                # 创建新类别
                new_class = Class(
                    projectId=project_id,
                    name=cat_name
                )
                session.add(new_class)
                session.flush()  # 获取ID
                class_name_to_id[cat_name] = new_class.id
                stats['new_classes'].append(cat_name)
                print(f"Created new class: {cat_name} (ID: {new_class.id})")
        session.commit()

    # 创建COCO category_id到项目class_id的映射
    coco_cat_to_class_id = {}
    for cat_id, cat_info in coco_categories.items():
        cat_name = cat_info['name']
        if cat_name in class_name_to_id:
            coco_cat_to_class_id[cat_id] = class_name_to_id[cat_name]

    # 导入图片和标注
    for idx, (coco_img_id, coco_img_info) in enumerate(coco_images.items()):
        if progress_callback:
            progress_callback(
                idx + 1,
                len(coco_images),
                f"导入 {split_name} 中的图片: {coco_img_info['file_name']}"
            )

        file_name = coco_img_info['file_name']
        img_path = os.path.join(split_dir, file_name)

        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path}")
            stats['error_images'] += 1
            continue

        # 读取文件并计算哈希
        with open(img_path, 'rb') as f:
            image_data = f.read()
        file_hash = sha1_of_bytes(image_data)

        with get_session() as session:
            # 检查是否已存在
            existing = (
                session.query(Image)
                .filter(Image.projectId == project_id)
                .filter(Image.checksum == file_hash)
                .first()
            )

            if existing:
                stats['duplicate_images'] += 1
                # 跳过标注导入（图片已存在）
                continue

            # 复制图片到项目目录
            images_dir = project_images_dir(project_id)

            # 生成唯一文件名
            base_name, ext = os.path.splitext(file_name)
            dest_filename = f"{base_name}_{file_hash[:8]}{ext}"
            dest_path = os.path.join(images_dir, dest_filename)

            shutil.copy2(img_path, dest_path)

            # 创建图片记录 - 保存相对路径
            rel_path = os.path.relpath(dest_path, os.getcwd())
            new_image = Image(
                projectId=project_id,
                path=rel_path,
                checksum=file_hash,
                width=coco_img_info.get('width'),
                height=coco_img_info.get('height'),
                status="unannotated"
            )
            session.add(new_image)
            session.flush()  # 获取image_id

            image_id = new_image.id
            stats['imported_images'] += 1

            # 导入该图片的标注
            img_annotations = [
                ann for ann in coco_annotations
                if ann['image_id'] == coco_img_id
            ]

            for ann in img_annotations:
                category_id = ann.get('category_id')
                if category_id not in coco_cat_to_class_id:
                    stats['skipped_annotations'] += 1
                    continue

                bbox = ann.get('bbox')  # [x, y, width, height]
                if not bbox or len(bbox) != 4:
                    stats['skipped_annotations'] += 1
                    continue

                # COCO bbox格式: [x_min, y_min, width, height] (左上角坐标)
                # 数据库格式也是: (x, y, w, h) - 左上角坐标
                x, y, w, h = bbox

                # 创建标注记录
                new_annotation = Annotation(
                    imageId=image_id,
                    classId=coco_cat_to_class_id[category_id],
                    x=x,
                    y=y,
                    w=w,
                    h=h
                )
                session.add(new_annotation)
                stats['imported_annotations'] += 1

            session.commit()

    return stats
