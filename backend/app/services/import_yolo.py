"""
YOLO格式数据集导入服务
支持导入包含train/valid/test文件夹和data.yaml的YOLO格式zip包
"""
import io
import os
import zipfile
import yaml
from typing import Dict, List, Optional, Tuple
from PIL import Image as PILImage

from ..db import get_session
from ..models import Image as DBImage, Annotation, Class
from ..utils.files import (
    project_images_dir,
    sha1_of_bytes,
    safe_join,
    read_image_size,
    ALLOWED_EXTS
)


def parse_data_yaml(content: str) -> Dict:
    """
    解析YOLO data.yaml文件

    Args:
        content: yaml文件内容

    Returns:
        解析后的字典，包含names等信息
    """
    try:
        data = yaml.safe_load(content)
        return data or {}
    except Exception as e:
        print(f"[YOLO Import] Failed to parse data.yaml: {e}")
        return {}


def parse_yolo_label(label_content: str, image_width: int, image_height: int) -> List[Dict]:
    """
    解析YOLO格式的标注文件

    支持两种格式:
    1. 标准YOLO格式: class_id center_x center_y width height (归一化坐标)
    2. x1y1x2y2格式: class_id x1 y1 x2 y2 (归一化坐标)

    自动检测格式：如果 x2 > x1 且 y2 > y1 且值都在合理范围，则认为是x1y1x2y2格式

    Args:
        label_content: 标注文件内容
        image_width: 图像宽度
        image_height: 图像高度

    Returns:
        标注列表 [{"class_idx": int, "x": float, "y": float, "w": float, "h": float}, ...]
        返回的坐标是归一化的中心点坐标格式 (center_x, center_y, width, height)
    """
    annotations = []
    for line in label_content.strip().split('\n'):
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 5:
            continue

        try:
            class_idx = int(parts[0])
            v1 = float(parts[1])
            v2 = float(parts[2])
            v3 = float(parts[3])
            v4 = float(parts[4])

            # 检测是哪种格式
            # x1y1x2y2格式: v3 > v1 (x2 > x1) 且 v4 > v2 (y2 > y1)
            # 标准YOLO格式: v3和v4是宽高，通常较小
            is_xyxy_format = (v3 > v1 and v4 > v2 and
                             v1 >= 0 and v1 <= 1 and
                             v2 >= 0 and v2 <= 1 and
                             v3 >= 0 and v3 <= 1 and
                             v4 >= 0 and v4 <= 1)

            if is_xyxy_format:
                # x1, y1, x2, y2 格式 -> 转换为 center_x, center_y, width, height
                x1, y1, x2, y2 = v1, v2, v3, v4
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                width = x2 - x1
                height = y2 - y1
            else:
                # 标准YOLO格式: center_x, center_y, width, height
                center_x = v1
                center_y = v2
                width = v3
                height = v4

            # 验证坐标有效性
            if width <= 0 or height <= 0:
                continue
            if center_x < 0 or center_x > 1 or center_y < 0 or center_y > 1:
                continue

            annotations.append({
                "class_idx": class_idx,
                "x": center_x,
                "y": center_y,
                "w": width,
                "h": height
            })
        except (ValueError, IndexError):
            continue

    return annotations


def find_label_file(zf: zipfile.ZipFile, image_path: str, all_files: List[str]) -> Optional[str]:
    """
    根据图片路径查找对应的标注文件

    YOLO数据集结构通常为:
    - train/images/xxx.jpg -> train/labels/xxx.txt
    - valid/images/xxx.jpg -> valid/labels/xxx.txt

    Args:
        zf: ZipFile对象
        image_path: 图片在zip中的路径
        all_files: zip中所有文件列表

    Returns:
        标注文件路径，如果不存在则返回None
    """
    # 获取图片文件名（不含扩展名）
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    label_filename = base_name + '.txt'

    # 标准化为正斜杠
    image_path_normalized = image_path.replace('\\', '/')

    # 构建所有可能的标注文件路径
    possible_label_paths = []

    # 1. 直接替换 images -> labels
    if '/images/' in image_path_normalized:
        label_path = image_path_normalized.replace('/images/', '/labels/')
        label_path = os.path.splitext(label_path)[0] + '.txt'
        possible_label_paths.append(label_path)

    # 2. 在所有 .txt 文件中查找同名文件
    # 这是最可靠的方法：直接在所有文件中找同名的 .txt
    all_files_normalized = {}
    all_label_files = []
    for f in all_files:
        normalized = f.replace('\\', '/')
        all_files_normalized[normalized] = f
        all_files_normalized[normalized.lower()] = f
        if normalized.lower().endswith('.txt'):
            all_label_files.append((normalized, f))

    # 先尝试精确路径匹配
    for label_path in possible_label_paths:
        if label_path in all_files_normalized:
            return all_files_normalized[label_path]
        if label_path.lower() in all_files_normalized:
            return all_files_normalized[label_path.lower()]

    # 再尝试在所有 .txt 文件中找同名文件
    for normalized, original in all_label_files:
        if normalized.endswith('/' + label_filename) or normalized.endswith('/' + label_filename.lower()):
            return original
        if normalized == label_filename or normalized == label_filename.lower():
            return original
        # 检查文件名部分是否匹配
        txt_basename = os.path.basename(normalized)
        if txt_basename.lower() == label_filename.lower():
            # 进一步检查是否在 labels 目录下
            if '/labels/' in normalized.lower():
                return original

    # 最后尝试：任何同名的 .txt 文件
    for normalized, original in all_label_files:
        txt_basename = os.path.basename(normalized)
        if txt_basename.lower() == label_filename.lower():
            return original

    return None


def build_class_mapping(
    yaml_names: List[str],
    project_classes: List
) -> Dict[int, int]:
    """
    构建YOLO类别索引到项目类别ID的映射

    Args:
        yaml_names: data.yaml中的类别名称列表
        project_classes: 项目中的类别列表

    Returns:
        映射字典 {yolo_class_idx: project_class_id}
    """
    # 构建项目类别名称到ID的映射（忽略大小写）
    project_class_map = {cls.name.lower(): cls.id for cls in project_classes}

    print(f"[YOLO Import] Project classes: {project_class_map}")
    print(f"[YOLO Import] YAML names: {yaml_names}")

    # 构建YOLO索引到项目类别ID的映射
    mapping = {}
    for idx, name in enumerate(yaml_names):
        name_lower = name.lower()
        if name_lower in project_class_map:
            mapping[idx] = project_class_map[name_lower]
            print(f"[YOLO Import] Mapped class {idx} ({name}) -> {project_class_map[name_lower]}")
        else:
            print(f"[YOLO Import] Class {idx} ({name}) not found in project classes")

    return mapping


async def import_yolo_dataset(
    project_id: int,
    upload_file,
    import_annotations: bool = True,
    progress_callback=None
) -> Dict[str, int]:
    """
    导入YOLO格式数据集

    Args:
        project_id: 项目ID
        upload_file: 上传的zip文件
        import_annotations: 是否导入标注
        progress_callback: 进度回调函数 (current, total, status_message) -> None

    Returns:
        导入统计信息
    """
    total_images = 0
    imported_images = 0
    duplicate_images = 0
    error_images = 0
    imported_annotations = 0
    skipped_annotations = 0  # 类别不匹配跳过的标注
    labels_not_found = 0  # 找不到标注文件的图片数

    def report_progress(current, total, message=""):
        if progress_callback:
            progress_callback(current, total, message)

    images_dir = project_images_dir(project_id)

    # 获取项目类别
    with get_session() as session:
        project_classes = session.query(Class).filter(Class.projectId == project_id).all()
        # 需要在session内复制数据，避免detached instance问题
        project_classes_data = [(cls.id, cls.name) for cls in project_classes]

    # 重建类别对象用于映射
    class ProjectClassItem:
        def __init__(self, id, name):
            self.id = id
            self.name = name

    project_classes = [ProjectClassItem(id, name) for id, name in project_classes_data]

    print(f"[YOLO Import] Project {project_id} has {len(project_classes)} classes")

    fileobj = upload_file.file
    fileobj.seek(0)

    with zipfile.ZipFile(fileobj) as zf:
        all_files = zf.namelist()
        print(f"[YOLO Import] ZIP contains {len(all_files)} files")

        # 打印前20个文件用于调试
        print(f"[YOLO Import] First 20 files: {all_files[:20]}")

        # 查找并解析data.yaml
        yaml_content = None
        yaml_names = []
        yaml_file_found = None
        for f in all_files:
            fname = os.path.basename(f).lower()
            if fname == 'data.yaml' or fname == 'data.yml':
                yaml_file_found = f
                try:
                    yaml_content = zf.read(f).decode('utf-8')
                    print(f"[YOLO Import] Found data.yaml at: {f}")
                    print(f"[YOLO Import] data.yaml content:\n{yaml_content[:500]}")
                    yaml_data = parse_data_yaml(yaml_content)
                    yaml_names = yaml_data.get('names', [])
                    # names可能是列表或字典
                    if isinstance(yaml_names, dict):
                        # {0: 'class1', 1: 'class2'} -> ['class1', 'class2']
                        max_idx = max(yaml_names.keys()) if yaml_names else -1
                        yaml_names = [yaml_names.get(i, f'class_{i}') for i in range(max_idx + 1)]
                    print(f"[YOLO Import] Parsed names: {yaml_names}")
                    break
                except Exception as e:
                    print(f"[YOLO Import] Error reading data.yaml: {e}")

        if not yaml_file_found:
            print("[YOLO Import] WARNING: data.yaml not found in ZIP!")

        # 构建类别映射
        class_mapping = build_class_mapping(yaml_names, project_classes) if yaml_names else {}
        print(f"[YOLO Import] Class mapping: {class_mapping}")

        # 如果没有yaml但有项目类别，尝试直接使用索引映射
        if not class_mapping and project_classes:
            print("[YOLO Import] No class mapping from YAML, trying direct index mapping")
            # 假设标注文件中的类别索引直接对应项目类别的顺序
            for idx, cls in enumerate(project_classes):
                class_mapping[idx] = cls.id
            print(f"[YOLO Import] Direct index mapping: {class_mapping}")

        # 查找所有图片文件
        image_files = []
        for f in all_files:
            if f.endswith('/') or '/__MACOSX' in f or f.startswith('__MACOSX'):
                continue
            ext = os.path.splitext(f.lower())[1]
            if ext in ALLOWED_EXTS:
                image_files.append(f)

        total_images = len(image_files)
        print(f"[YOLO Import] Found {total_images} image files")

        # 报告初始进度
        report_progress(0, total_images, "开始导入...")

        # 查找所有标注文件
        label_files = [f for f in all_files if f.lower().endswith('.txt') and 'readme' not in f.lower()]
        print(f"[YOLO Import] Found {len(label_files)} potential label files")
        if label_files:
            print(f"[YOLO Import] Sample label files: {label_files[:5]}")

        # 处理每个图片
        processed_count = 0
        for image_path in image_files:
            processed_count += 1
            # 每处理一张图片报告一次进度
            if processed_count % 10 == 0 or processed_count == total_images:
                report_progress(
                    processed_count,
                    total_images,
                    f"正在处理: {os.path.basename(image_path)}"
                )

            try:
                # 读取图片数据
                image_data = zf.read(image_path)
                checksum = sha1_of_bytes(image_data)

                # 检查重复
                with get_session() as session:
                    exists = (
                        session.query(DBImage)
                        .filter(DBImage.projectId == project_id, DBImage.checksum == checksum)
                        .first()
                    )
                    if exists:
                        duplicate_images += 1
                        continue

                # 读取图片尺寸
                try:
                    width, height = read_image_size(image_data)
                except Exception:
                    error_images += 1
                    continue

                # 生成安全文件名
                base_name = os.path.basename(image_path)
                if not base_name:
                    base_name = checksum + os.path.splitext(image_path)[1]

                dest_path = safe_join(images_dir, base_name)

                # 若重名，使用checksum前缀
                if os.path.exists(dest_path):
                    ext = os.path.splitext(base_name)[1]
                    dest_path = safe_join(images_dir, f"{checksum}{ext}")

                # 保存图片
                with open(dest_path, 'wb') as f:
                    f.write(image_data)

                # 写入数据库
                rel_path = os.path.relpath(dest_path, os.getcwd())
                with get_session() as session:
                    db_img = DBImage(
                        projectId=project_id,
                        path=rel_path,
                        width=width,
                        height=height,
                        checksum=checksum,
                        status="unannotated"
                    )
                    session.add(db_img)
                    session.commit()
                    session.refresh(db_img)
                    image_id = db_img.id

                imported_images += 1

                # 导入标注
                if import_annotations and class_mapping:
                    label_path = find_label_file(zf, image_path, all_files)
                    if label_path:
                        try:
                            label_content = zf.read(label_path).decode('utf-8')
                            annotations = parse_yolo_label(label_content, width, height)

                            if imported_images <= 3:  # 只打印前几个用于调试
                                print(f"[YOLO Import] Image: {image_path}")
                                print(f"[YOLO Import] Label: {label_path}")
                                print(f"[YOLO Import] Annotations: {annotations}")

                            has_valid_annotation = False
                            for ann in annotations:
                                class_idx = ann['class_idx']
                                if class_idx not in class_mapping:
                                    skipped_annotations += 1
                                    if skipped_annotations <= 5:
                                        print(f"[YOLO Import] Skipped annotation with class_idx {class_idx} (not in mapping)")
                                    continue

                                project_class_id = class_mapping[class_idx]

                                # 将归一化坐标转换为像素坐标
                                # YOLO格式: center_x, center_y, width, height (归一化 0-1)
                                # 数据库/前端格式: x, y, w, h (像素坐标，x/y是左上角)
                                pixel_w = ann['w'] * width
                                pixel_h = ann['h'] * height
                                # 中心点转左上角
                                pixel_x = ann['x'] * width - pixel_w / 2
                                pixel_y = ann['y'] * height - pixel_h / 2

                                with get_session() as session:
                                    db_ann = Annotation(
                                        imageId=image_id,
                                        classId=project_class_id,
                                        x=pixel_x,
                                        y=pixel_y,
                                        w=pixel_w,
                                        h=pixel_h,
                                        source='manual'
                                    )
                                    session.add(db_ann)
                                    session.commit()

                                imported_annotations += 1
                                has_valid_annotation = True

                            # 如果有有效标注，更新图片状态
                            if has_valid_annotation:
                                with get_session() as session:
                                    db_img = session.get(DBImage, image_id)
                                    if db_img:
                                        db_img.status = "annotated"
                                        session.commit()

                        except Exception as e:
                            print(f"[YOLO Import] Error processing label {label_path}: {e}")
                    else:
                        labels_not_found += 1
                        if labels_not_found <= 5:
                            print(f"[YOLO Import] Label not found for image: {image_path}")

            except Exception as e:
                print(f"[YOLO Import] Error processing image {image_path}: {e}")
                error_images += 1
                continue

    print(f"[YOLO Import] Summary: images={imported_images}, annotations={imported_annotations}, skipped={skipped_annotations}, labels_not_found={labels_not_found}")

    return {
        "total": total_images,
        "imported": imported_images,
        "duplicates": duplicate_images,
        "errors": error_images,
        "annotations_imported": imported_annotations,
        "annotations_skipped": skipped_annotations,
        "class_mapping_found": len(class_mapping) > 0
    }
