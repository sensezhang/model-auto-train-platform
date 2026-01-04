import io
import os
import zipfile
import hashlib
from typing import Tuple, Dict, Optional
from PIL import Image as PILImage

from ..db import get_session
from ..models import Image as DBImage


ALLOWED_EXTS = {".jpg", ".jpeg", ".png"}

# 缩略图配置
THUMBNAIL_WIDTH = 200  # 缩略图宽度
DISPLAY_WIDTH = 1920   # 标注用图宽度
JPEG_QUALITY = 85      # JPEG 压缩质量


def datasets_root() -> str:
    root = os.path.join(os.getcwd(), "datasets")
    os.makedirs(root, exist_ok=True)
    return root


def project_images_dir(project_id: int) -> str:
    path = os.path.join(datasets_root(), str(project_id), "images")
    os.makedirs(path, exist_ok=True)
    return path


def project_thumbnails_dir(project_id: int) -> str:
    """缩略图目录"""
    path = os.path.join(datasets_root(), str(project_id), "thumbnails")
    os.makedirs(path, exist_ok=True)
    return path


def project_display_dir(project_id: int) -> str:
    """标注用图目录"""
    path = os.path.join(datasets_root(), str(project_id), "display")
    os.makedirs(path, exist_ok=True)
    return path


def generate_thumbnail_and_display(
    original_path: str,
    project_id: int,
    filename: str
) -> Tuple[Optional[str], Optional[str]]:
    """
    根据原图生成缩略图和标注用图

    Args:
        original_path: 原图路径
        project_id: 项目ID
        filename: 文件名

    Returns:
        (thumbnail_path, display_path) 相对路径元组
    """
    try:
        with PILImage.open(original_path) as img:
            # 转换为 RGB（处理 RGBA 等格式）
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')

            orig_width, orig_height = img.size

            # 生成缩略图
            thumb_dir = project_thumbnails_dir(project_id)
            thumb_path = os.path.join(thumb_dir, filename)
            if orig_width > THUMBNAIL_WIDTH:
                ratio = THUMBNAIL_WIDTH / orig_width
                thumb_size = (THUMBNAIL_WIDTH, int(orig_height * ratio))
                thumb_img = img.resize(thumb_size, PILImage.LANCZOS)
            else:
                thumb_img = img.copy()
            thumb_img.save(thumb_path, 'JPEG', quality=JPEG_QUALITY)
            thumb_rel = os.path.relpath(thumb_path, os.getcwd())

            # 生成标注用图
            display_dir = project_display_dir(project_id)
            display_path = os.path.join(display_dir, filename)
            if orig_width > DISPLAY_WIDTH:
                ratio = DISPLAY_WIDTH / orig_width
                display_size = (DISPLAY_WIDTH, int(orig_height * ratio))
                display_img = img.resize(display_size, PILImage.LANCZOS)
            else:
                display_img = img.copy()
            display_img.save(display_path, 'JPEG', quality=JPEG_QUALITY)
            display_rel = os.path.relpath(display_path, os.getcwd())

            return thumb_rel, display_rel

    except Exception as e:
        print(f"生成缩略图失败: {e}")
        return None, None


def sha1_of_bytes(b: bytes) -> str:
    h = hashlib.sha1()
    h.update(b)
    return h.hexdigest()


def safe_join(base: str, *paths: str) -> str:
    joined = os.path.normpath(os.path.join(base, *paths))
    if not joined.startswith(os.path.abspath(base)):
        raise ValueError("Unsafe path detected")
    return joined


def read_image_size(data: bytes) -> Tuple[int, int]:
    with PILImage.open(io.BytesIO(data)) as im:
        return im.width, im.height


async def extract_images_from_zip(project_id: int, upload_file) -> Dict[str, int]:
    total = 0
    imported = 0
    duplicates = 0
    errors = 0

    images_dir = project_images_dir(project_id)

    # 使用UploadFile.file (SpooledTemporaryFile) 直接读
    fileobj = upload_file.file
    fileobj.seek(0)
    with zipfile.ZipFile(fileobj) as zf:
        for info in zf.infolist():
            # 跳过目录或隐藏文件
            if info.is_dir():
                continue
            name_lower = info.filename.lower()
            ext = os.path.splitext(name_lower)[1]
            if ext not in ALLOWED_EXTS:
                continue
            total += 1

            try:
                data = zf.read(info)
            except Exception:
                errors += 1
                continue

            checksum = sha1_of_bytes(data)

            # 去重：同一project下checksum重复则跳过
            with get_session() as session:
                exists = (
                    session.query(DBImage)
                    .filter(DBImage.projectId == project_id, DBImage.checksum == checksum)
                    .first()
                )
                if exists:
                    duplicates += 1
                    continue

            # 安全文件名
            base_name = os.path.basename(info.filename)
            if not base_name:
                base_name = checksum + ext
            dest_path = safe_join(images_dir, base_name)

            # 若重名，使用checksum前缀避免覆盖
            if os.path.exists(dest_path):
                dest_path = safe_join(images_dir, f"{checksum}{ext}")

            try:
                with open(dest_path, "wb") as f:
                    f.write(data)
                width, height = read_image_size(data)
            except Exception:
                errors += 1
                # 清理失败文件
                try:
                    if os.path.exists(dest_path):
                        os.remove(dest_path)
                except Exception:
                    pass
                continue

            # 写入DB
            rel_path = os.path.relpath(dest_path, os.getcwd())
            with get_session() as session:
                db_img = DBImage(
                    projectId=project_id,
                    path=rel_path,
                    width=width,
                    height=height,
                    checksum=checksum,
                    status="unannotated",
                )
                session.add(db_img)
                session.commit()

            imported += 1

    return {"total": total, "imported": imported, "duplicates": duplicates, "errors": errors}


async def import_single_image(project_id: int, upload_file) -> Dict[str, any]:
    """
    导入单张图片

    Args:
        project_id: 项目ID
        upload_file: 上传的文件对象

    Returns:
        导入结果，包含 success, message, image_id 等信息
    """
    filename = upload_file.filename or ""
    ext = os.path.splitext(filename.lower())[1]

    if ext not in ALLOWED_EXTS:
        return {"success": False, "message": f"不支持的文件格式: {ext}，仅支持 jpg, jpeg, png"}

    images_dir = project_images_dir(project_id)

    try:
        # 读取文件数据
        data = await upload_file.read()
    except Exception as e:
        return {"success": False, "message": f"读取文件失败: {e}"}

    checksum = sha1_of_bytes(data)

    # 去重检查
    with get_session() as session:
        exists = (
            session.query(DBImage)
            .filter(DBImage.projectId == project_id, DBImage.checksum == checksum)
            .first()
        )
        if exists:
            return {"success": False, "message": "图片已存在（重复）", "duplicate": True, "image_id": exists.id}

    # 安全文件名
    base_name = os.path.basename(filename)
    if not base_name:
        base_name = checksum + ext
    dest_path = safe_join(images_dir, base_name)

    # 若重名，使用checksum前缀避免覆盖
    if os.path.exists(dest_path):
        dest_path = safe_join(images_dir, f"{checksum}{ext}")

    try:
        with open(dest_path, "wb") as f:
            f.write(data)
        width, height = read_image_size(data)
    except Exception as e:
        # 清理失败文件
        try:
            if os.path.exists(dest_path):
                os.remove(dest_path)
        except Exception:
            pass
        return {"success": False, "message": f"保存图片失败: {e}"}

    # 生成缩略图和标注用图
    final_filename = os.path.basename(dest_path)
    # 确保文件名是 .jpg 格式（缩略图统一用 jpg）
    thumb_filename = os.path.splitext(final_filename)[0] + ".jpg"
    thumb_path, display_path = generate_thumbnail_and_display(dest_path, project_id, thumb_filename)

    # 写入DB
    rel_path = os.path.relpath(dest_path, os.getcwd())
    with get_session() as session:
        db_img = DBImage(
            projectId=project_id,
            path=rel_path,
            thumbnailPath=thumb_path,
            displayPath=display_path,
            width=width,
            height=height,
            checksum=checksum,
            status="unannotated",
        )
        session.add(db_img)
        session.commit()
        session.refresh(db_img)
        image_id = db_img.id

    return {"success": True, "message": "导入成功", "image_id": image_id}


def import_image_from_base64(project_id: int, filename: str, base64_data: str) -> Dict[str, any]:
    """
    从 Base64 数据导入图片（避免 multipart/form-data）

    Args:
        project_id: 项目ID
        filename: 文件名
        base64_data: Base64 编码的图片数据

    Returns:
        导入结果
    """
    import base64

    ext = os.path.splitext(filename.lower())[1]
    if ext not in ALLOWED_EXTS:
        return {"success": False, "message": f"不支持的文件格式: {ext}，仅支持 jpg, jpeg, png"}

    images_dir = project_images_dir(project_id)

    try:
        # 解码 Base64 数据
        data = base64.b64decode(base64_data)
    except Exception as e:
        return {"success": False, "message": f"Base64 解码失败: {e}"}

    checksum = sha1_of_bytes(data)

    # 去重检查
    with get_session() as session:
        exists = (
            session.query(DBImage)
            .filter(DBImage.projectId == project_id, DBImage.checksum == checksum)
            .first()
        )
        if exists:
            return {"success": False, "message": "图片已存在（重复）", "duplicate": True, "image_id": exists.id}

    # 安全文件名
    base_name = os.path.basename(filename)
    if not base_name:
        base_name = checksum + ext
    dest_path = safe_join(images_dir, base_name)

    # 若重名，使用checksum前缀避免覆盖
    if os.path.exists(dest_path):
        dest_path = safe_join(images_dir, f"{checksum}{ext}")

    try:
        with open(dest_path, "wb") as f:
            f.write(data)
        width, height = read_image_size(data)
    except Exception as e:
        try:
            if os.path.exists(dest_path):
                os.remove(dest_path)
        except Exception:
            pass
        return {"success": False, "message": f"保存图片失败: {e}"}

    # 生成缩略图和标注用图
    final_filename = os.path.basename(dest_path)
    thumb_filename = os.path.splitext(final_filename)[0] + ".jpg"
    thumb_path, display_path = generate_thumbnail_and_display(dest_path, project_id, thumb_filename)

    # 写入DB
    rel_path = os.path.relpath(dest_path, os.getcwd())
    with get_session() as session:
        db_img = DBImage(
            projectId=project_id,
            path=rel_path,
            thumbnailPath=thumb_path,
            displayPath=display_path,
            width=width,
            height=height,
            checksum=checksum,
            status="unannotated",
        )
        session.add(db_img)
        session.commit()
        session.refresh(db_img)
        image_id = db_img.id

    return {"success": True, "message": "导入成功", "image_id": image_id}

