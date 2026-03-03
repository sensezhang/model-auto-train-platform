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
DISPLAY_WIDTH   = 1920  # 标注用图宽度
JPEG_QUALITY    = 85   # JPEG 压缩质量


# ──────────────────────────────────────────────────────────────
# 本地目录辅助
# ──────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────
# 路径 / URL 工具
# ──────────────────────────────────────────────────────────────

def _rel(abs_path: str) -> str:
    """将绝对路径转换为相对于 CWD 的相对路径（统一正斜杠）"""
    return os.path.relpath(abs_path, os.getcwd()).replace("\\", "/")


def _to_storage_path(rel_path: str, local_abs_path: str) -> str:
    """返回文件的相对路径（本地文件服务）"""
    return rel_path


# ──────────────────────────────────────────────────────────────
# 缩略图 / 标注用图生成
# ──────────────────────────────────────────────────────────────

def generate_thumbnail_and_display(
    original_path: str,
    project_id: int,
    filename: str,
) -> Tuple[Optional[str], Optional[str]]:
    """
    根据原图生成缩略图和标注用图。

    Args:
        original_path : 原图本地绝对路径
        project_id    : 项目 ID
        filename      : 目标文件名（统一存为 .jpg）

    Returns:
        (thumbnail_storage_path, display_storage_path) 均为相对路径
    """
    try:
        with PILImage.open(original_path) as img:
            # 转换为 RGB（处理 RGBA/P 等格式）
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")

            orig_width, orig_height = img.size

            # ── 缩略图 ──────────────────────────────────────────
            thumb_dir  = project_thumbnails_dir(project_id)
            thumb_abs  = os.path.join(thumb_dir, filename)
            if orig_width > THUMBNAIL_WIDTH:
                ratio = THUMBNAIL_WIDTH / orig_width
                thumb_img = img.resize((THUMBNAIL_WIDTH, int(orig_height * ratio)), PILImage.LANCZOS)
            else:
                thumb_img = img.copy()
            thumb_img.save(thumb_abs, "JPEG", quality=JPEG_QUALITY)
            thumb_rel  = _rel(thumb_abs)
            thumb_path = _to_storage_path(thumb_rel, thumb_abs)

            # ── 标注用图 ────────────────────────────────────────
            display_dir  = project_display_dir(project_id)
            display_abs  = os.path.join(display_dir, filename)
            if orig_width > DISPLAY_WIDTH:
                ratio = DISPLAY_WIDTH / orig_width
                display_img = img.resize((DISPLAY_WIDTH, int(orig_height * ratio)), PILImage.LANCZOS)
            else:
                display_img = img.copy()
            display_img.save(display_abs, "JPEG", quality=JPEG_QUALITY)
            display_rel  = _rel(display_abs)
            display_path = _to_storage_path(display_rel, display_abs)

            return thumb_path, display_path

    except Exception as e:
        print(f"生成缩略图失败: {e}")
        return None, None


# ──────────────────────────────────────────────────────────────
# 基础工具
# ──────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────
# ZIP 批量导入
# ──────────────────────────────────────────────────────────────

async def extract_images_from_zip(project_id: int, upload_file) -> Dict[str, int]:
    total      = 0
    imported   = 0
    duplicates = 0
    errors     = 0

    images_dir = project_images_dir(project_id)

    fileobj = upload_file.file
    fileobj.seek(0)
    with zipfile.ZipFile(fileobj) as zf:
        for info in zf.infolist():
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

            with get_session() as session:
                exists = (
                    session.query(DBImage)
                    .filter(DBImage.projectId == project_id, DBImage.checksum == checksum)
                    .first()
                )
                if exists:
                    duplicates += 1
                    continue

            base_name = os.path.basename(info.filename) or (checksum + ext)
            dest_abs  = safe_join(images_dir, base_name)
            if os.path.exists(dest_abs):
                dest_abs = safe_join(images_dir, f"{checksum}{ext}")

            try:
                with open(dest_abs, "wb") as f:
                    f.write(data)
                width, height = read_image_size(data)
            except Exception:
                errors += 1
                try:
                    if os.path.exists(dest_abs):
                        os.remove(dest_abs)
                except Exception:
                    pass
                continue

            dest_rel  = _rel(dest_abs)
            img_path  = _to_storage_path(dest_rel, dest_abs)

            with get_session() as session:
                db_img = DBImage(
                    projectId=project_id,
                    path=img_path,
                    width=width,
                    height=height,
                    checksum=checksum,
                    status="unannotated",
                )
                session.add(db_img)
                session.commit()

            imported += 1

    return {"total": total, "imported": imported, "duplicates": duplicates, "errors": errors}


# ──────────────────────────────────────────────────────────────
# 单张图片导入（multipart/form-data）
# ──────────────────────────────────────────────────────────────

async def import_single_image(project_id: int, upload_file) -> Dict[str, any]:
    filename = upload_file.filename or ""
    ext = os.path.splitext(filename.lower())[1]

    if ext not in ALLOWED_EXTS:
        return {"success": False, "message": f"不支持的文件格式: {ext}，仅支持 jpg, jpeg, png"}

    images_dir = project_images_dir(project_id)

    try:
        data = await upload_file.read()
    except Exception as e:
        return {"success": False, "message": f"读取文件失败: {e}"}

    checksum = sha1_of_bytes(data)

    with get_session() as session:
        exists = (
            session.query(DBImage)
            .filter(DBImage.projectId == project_id, DBImage.checksum == checksum)
            .first()
        )
        if exists:
            return {"success": False, "message": "图片已存在（重复）", "duplicate": True, "image_id": exists.id}

    base_name = os.path.basename(filename) or (checksum + ext)
    dest_abs  = safe_join(images_dir, base_name)
    if os.path.exists(dest_abs):
        dest_abs = safe_join(images_dir, f"{checksum}{ext}")

    try:
        with open(dest_abs, "wb") as f:
            f.write(data)
        width, height = read_image_size(data)
    except Exception as e:
        try:
            if os.path.exists(dest_abs):
                os.remove(dest_abs)
        except Exception:
            pass
        return {"success": False, "message": f"保存图片失败: {e}"}

    final_filename = os.path.basename(dest_abs)
    thumb_filename = os.path.splitext(final_filename)[0] + ".jpg"
    thumb_path, display_path = generate_thumbnail_and_display(dest_abs, project_id, thumb_filename)

    dest_rel  = _rel(dest_abs)
    img_path  = _to_storage_path(dest_rel, dest_abs)

    with get_session() as session:
        db_img = DBImage(
            projectId=project_id,
            path=img_path,
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


# ──────────────────────────────────────────────────────────────
# 单张图片导入（Base64）
# ──────────────────────────────────────────────────────────────

def import_image_from_base64(project_id: int, filename: str, base64_data: str) -> Dict[str, any]:
    import base64

    ext = os.path.splitext(filename.lower())[1]
    if ext not in ALLOWED_EXTS:
        return {"success": False, "message": f"不支持的文件格式: {ext}，仅支持 jpg, jpeg, png"}

    images_dir = project_images_dir(project_id)

    try:
        data = base64.b64decode(base64_data)
    except Exception as e:
        return {"success": False, "message": f"Base64 解码失败: {e}"}

    checksum = sha1_of_bytes(data)

    with get_session() as session:
        exists = (
            session.query(DBImage)
            .filter(DBImage.projectId == project_id, DBImage.checksum == checksum)
            .first()
        )
        if exists:
            return {"success": False, "message": "图片已存在（重复）", "duplicate": True, "image_id": exists.id}

    base_name = os.path.basename(filename) or (checksum + ext)
    dest_abs  = safe_join(images_dir, base_name)
    if os.path.exists(dest_abs):
        dest_abs = safe_join(images_dir, f"{checksum}{ext}")

    try:
        with open(dest_abs, "wb") as f:
            f.write(data)
        width, height = read_image_size(data)
    except Exception as e:
        try:
            if os.path.exists(dest_abs):
                os.remove(dest_abs)
        except Exception:
            pass
        return {"success": False, "message": f"保存图片失败: {e}"}

    final_filename = os.path.basename(dest_abs)
    thumb_filename = os.path.splitext(final_filename)[0] + ".jpg"
    thumb_path, display_path = generate_thumbnail_and_display(dest_abs, project_id, thumb_filename)

    dest_rel  = _rel(dest_abs)
    img_path  = _to_storage_path(dest_rel, dest_abs)

    with get_session() as session:
        db_img = DBImage(
            projectId=project_id,
            path=img_path,
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

