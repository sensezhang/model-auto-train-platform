import io
import os
import zipfile
import hashlib
from typing import Tuple, Dict
from PIL import Image as PILImage

from ..db import get_session
from ..models import Image as DBImage


ALLOWED_EXTS = {".jpg", ".jpeg", ".png"}


def datasets_root() -> str:
    root = os.path.join(os.getcwd(), "datasets")
    os.makedirs(root, exist_ok=True)
    return root


def project_images_dir(project_id: int) -> str:
    path = os.path.join(datasets_root(), str(project_id), "images")
    os.makedirs(path, exist_ok=True)
    return path


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
        session.refresh(db_img)
        image_id = db_img.id

    return {"success": True, "message": "导入成功", "image_id": image_id}

