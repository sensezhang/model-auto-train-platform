from fastapi import APIRouter, HTTPException, UploadFile, File, Query, BackgroundTasks, Request
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import asyncio
import tempfile
import shutil
import os
from sqlalchemy import func

from ..db import get_session, init_db
from ..models import (
    Project,
    Class,
    Image,
    Annotation,
    ProposedAnnotation,
    AutoLabelJob,
    TrainingJob,
    ModelArtifact,
    DatasetVersion,
    ActivityLog,
    ImportJob,
)
from ..utils.files import extract_images_from_zip, extract_images_from_zip_sync_with_progress, import_single_image, import_image_from_base64, generate_thumbnail_and_display
from ..utils.project_cleanup import remove_project_files
from ..services.import_yolo import import_yolo_dataset
from ..services.import_coco import import_coco_dataset

# 上传临时文件优先写入已挂载的数据卷（/app/datasets → /mnt/data/datasets）
# 避免大文件撑爆容器 overlay 文件系统的 /tmp
_UPLOAD_TMP_DIR: str | None = "/app/datasets" if os.path.isdir("/app/datasets") else None


router = APIRouter(tags=["projects"])


class ClassInfo(BaseModel):
    id: int
    name: str
    color: Optional[str] = None
    hotkey: Optional[str] = None


class ProjectResponse(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    status: Optional[str] = None
    createdAt: datetime
    classes: List[ClassInfo] = []
    imageCount: int = 0


class ProjectCreate(BaseModel):
    name: str
    description: str | None = None
    classes: List[str] = []


@router.on_event("startup")
def _init():
    init_db()
    _recover_stale_jobs()


def _recover_stale_jobs():
    """服务启动时，将上次异常退出遗留的 running 任务标记为 failed"""
    from datetime import datetime
    from ..models import TrainingJob, AutoLabelJob, ImportJob
    with get_session() as session:
        for model_cls in (TrainingJob, AutoLabelJob, ImportJob):
            stale = session.query(model_cls).filter(model_cls.status == 'running').all()
            for job in stale:
                job.status = 'failed'
                if hasattr(job, 'finishedAt'):
                    job.finishedAt = datetime.utcnow()
                if hasattr(job, 'message'):
                    job.message = '服务重启时进程已中断'
                session.add(job)
        session.commit()


@router.post("/projects", response_model=Project)
def create_project(body: ProjectCreate):
    with get_session() as session:
        proj = Project(name=body.name, description=body.description)
        session.add(proj)
        session.flush()
        for cname in body.classes:
            session.add(Class(projectId=proj.id, name=cname))
        session.commit()
        session.refresh(proj)
        return proj


@router.get("/projects", response_model=List[ProjectResponse])
def list_projects(includeDeleted: bool = Query(default=False)):
    with get_session() as session:
        q = session.query(Project)
        if not includeDeleted:
            q = q.filter(Project.status != "deleted")
        projects = q.all()
        result = []
        for proj in projects:
            count = session.query(func.count(Image.id)).filter(Image.projectId == proj.id).scalar() or 0
            classes = session.query(Class).filter(Class.projectId == proj.id).all()
            result.append(ProjectResponse(
                id=proj.id,
                name=proj.name,
                description=proj.description,
                status=proj.status,
                createdAt=proj.createdAt,
                classes=[ClassInfo(id=c.id, name=c.name, color=c.color, hotkey=c.hotkey) for c in classes],
                imageCount=count,
            ))
        return result


@router.get("/projects/{project_id}", response_model=ProjectResponse)
def get_project(project_id: int):
    with get_session() as session:
        proj = session.get(Project, project_id)
        if not proj:
            raise HTTPException(404, "Project not found")
        count = session.query(func.count(Image.id)).filter(Image.projectId == proj.id).scalar() or 0
        classes = session.query(Class).filter(Class.projectId == proj.id).all()
        return ProjectResponse(
            id=proj.id,
            name=proj.name,
            description=proj.description,
            status=proj.status,
            createdAt=proj.createdAt,
            classes=[ClassInfo(id=c.id, name=c.name, color=c.color, hotkey=c.hotkey) for c in classes],
            imageCount=count,
        )


class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None


@router.patch("/projects/{project_id}", response_model=Project)
def update_project(project_id: int, body: ProjectUpdate):
    with get_session() as session:
        proj = session.get(Project, project_id)
        if not proj:
            raise HTTPException(404, "Project not found")
        if body.name is not None:
            if not body.name.strip():
                raise HTTPException(400, "项目名称不能为空")
            proj.name = body.name.strip()
        if body.description is not None:
            proj.description = body.description
        session.add(proj)
        session.commit()
        session.refresh(proj)
        return proj


class ImportSummary(BaseModel):
    total: int
    imported: int
    duplicates: int
    errors: int


class YoloImportSummary(BaseModel):
    total: int
    imported: int
    duplicates: int
    errors: int
    annotations_imported: int
    annotations_skipped: int
    class_mapping_found: bool


@router.post("/projects/{project_id}/import")
async def import_zip(
    project_id: int,
    background_tasks: BackgroundTasks,
    request: Request,
    filename: str = Query(default="upload.zip"),
):
    """导入普通图片zip包（仅图片，无标注），异步执行，返回任务ID。
    文件以 application/octet-stream 直接流式发送，无 multipart 限制。"""
    with get_session() as session:
        proj = session.get(Project, project_id)
        if not proj:
            raise HTTPException(404, "Project not found")

        import_job = ImportJob(
            projectId=project_id,
            format="images",
            status="pending",
            message="准备导入...",
        )
        session.add(import_job)
        session.commit()
        session.refresh(import_job)
        job_id = import_job.id

    # 直接从请求流读取，不经过 multipart 解析，支持任意大小
    # 写入数据卷避免撑爆容器 /tmp（overlay 文件系统空间有限）
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip", dir=_UPLOAD_TMP_DIR)
    tmp_path = tmp.name
    tmp.close()
    print(f"[Images Import] Saving to {tmp_path} (tmp_dir={_UPLOAD_TMP_DIR})")
    try:
        total_size = 0
        logged_mb = 0
        with open(tmp_path, "wb") as f:
            async for chunk in request.stream():
                f.write(chunk)
                total_size += len(chunk)
                cur_256mb = total_size // (256 * 1024 * 1024)
                if cur_256mb > logged_mb:
                    logged_mb = cur_256mb
                    print(f"[Images Import] Received {total_size / 1024 / 1024:.0f} MB...")
        print(f"[Images Import] File saved ({filename}), size: {total_size / 1024 / 1024:.1f} MB")
    except Exception as e:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        raise HTTPException(500, f"Failed to save uploaded file: {e}")

    background_tasks.add_task(run_images_import_sync, job_id, project_id, tmp_path)
    return {"job_id": job_id, "status": "pending", "message": "导入任务已创建"}


class SingleImageImportResult(BaseModel):
    success: bool
    message: str
    image_id: Optional[int] = None
    duplicate: Optional[bool] = None


@router.post("/projects/{project_id}/import/image", response_model=SingleImageImportResult)
async def import_image(project_id: int, file: UploadFile = File(...)):
    """导入单张图片"""
    with get_session() as session:
        proj = session.get(Project, project_id)
        if not proj:
            raise HTTPException(404, "Project not found")

    result = await import_single_image(project_id, file)
    return SingleImageImportResult(**result)


class Base64ImageImport(BaseModel):
    filename: str
    data: str  # Base64 编码的图片数据


@router.post("/projects/{project_id}/import/image-base64", response_model=SingleImageImportResult)
def import_image_base64(project_id: int, body: Base64ImageImport):
    """
    导入单张图片（Base64 JSON 格式，避免 multipart/form-data）

    前端使用 FileReader.readAsDataURL() 获取 Base64 数据
    """
    with get_session() as session:
        proj = session.get(Project, project_id)
        if not proj:
            raise HTTPException(404, "Project not found")

    # 移除可能的 data URL 前缀 (data:image/jpeg;base64,)
    base64_data = body.data
    if "," in base64_data:
        base64_data = base64_data.split(",", 1)[1]

    result = import_image_from_base64(project_id, body.filename, base64_data)
    return SingleImageImportResult(**result)


@router.post("/projects/{project_id}/import/yolo")
async def import_yolo_zip(
    project_id: int,
    background_tasks: BackgroundTasks,
    request: Request,
    import_annotations: bool = Query(default=True, description="是否导入标注"),
    filename: str = Query(default="upload.zip"),
):
    """
    导入YOLO格式数据集zip包（异步，返回任务ID）。
    文件以 application/octet-stream 直接流式发送，无 multipart 限制。
    """
    with get_session() as session:
        proj = session.get(Project, project_id)
        if not proj:
            raise HTTPException(404, "Project not found")

        # 创建导入任务
        import_job = ImportJob(
            projectId=project_id,
            format="yolo",
            status="pending",
            message="准备导入..."
        )
        session.add(import_job)
        session.commit()
        session.refresh(import_job)
        job_id = import_job.id

    # 直接从请求流读取，不经过 multipart 解析，支持任意大小
    # 写入数据卷避免撑爆容器 /tmp（overlay 文件系统空间有限）
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip", dir=_UPLOAD_TMP_DIR)
    tmp_path = tmp.name
    tmp.close()
    print(f"[YOLO Import] Saving to {tmp_path} (tmp_dir={_UPLOAD_TMP_DIR})")
    try:
        total_size = 0
        logged_mb = 0
        with open(tmp_path, "wb") as f:
            async for chunk in request.stream():
                f.write(chunk)
                total_size += len(chunk)
                cur_256mb = total_size // (256 * 1024 * 1024)
                if cur_256mb > logged_mb:
                    logged_mb = cur_256mb
                    print(f"[YOLO Import] Received {total_size / 1024 / 1024:.0f} MB...")
        print(f"[YOLO Import] File saved ({filename}), size: {total_size / 1024 / 1024:.1f} MB")
    except Exception as e:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        raise HTTPException(500, f"Failed to save uploaded file: {e}")

    # 在后台运行导入任务
    background_tasks.add_task(
        run_yolo_import_sync,
        job_id,
        project_id,
        tmp_path,
        import_annotations
    )

    return {"job_id": job_id, "status": "pending", "message": "导入任务已创建"}


@router.post("/projects/{project_id}/import/coco")
async def import_coco_zip(
    project_id: int,
    background_tasks: BackgroundTasks,
    request: Request,
    filename: str = Query(default="upload.zip"),
):
    """
    导入COCO格式数据集zip包（异步，返回任务ID）

    支持的结构:
    - train/_annotations.coco.json + images
    - valid/_annotations.coco.json + images (可选)
    - test/_annotations.coco.json + images (可选)

    类别映射规则:
    - 根据COCO JSON中的categories与项目类别名称进行匹配
    - 如果项目中不存在某个类别，会自动创建
    """
    print(f"[COCO Import] Received file: {filename}")

    with get_session() as session:
        proj = session.get(Project, project_id)
        if not proj:
            raise HTTPException(404, "Project not found")

        # 创建导入任务
        import_job = ImportJob(
            projectId=project_id,
            format="coco",
            status="pending",
            message="准备导入..."
        )
        session.add(import_job)
        session.commit()
        session.refresh(import_job)
        job_id = import_job.id
        print(f"[COCO Import] Created job {job_id} for project {project_id}")

    # 保存上传文件到临时位置（原始流写入，无 multipart 解析，支持超大文件）
    # 写入数据卷避免撑爆容器 /tmp（overlay 文件系统空间有限）
    temp_dir = tempfile.mkdtemp(dir=_UPLOAD_TMP_DIR)
    temp_file_path = os.path.join(temp_dir, "upload.zip")

    try:
        print(f"[COCO Import] Saving file to {temp_file_path} (tmp_dir={_UPLOAD_TMP_DIR})")
        total_size = 0
        logged_mb = 0

        with open(temp_file_path, "wb") as f:
            async for chunk in request.stream():
                f.write(chunk)
                total_size += len(chunk)
                cur_256mb = total_size // (256 * 1024 * 1024)
                if cur_256mb > logged_mb:
                    logged_mb = cur_256mb
                    print(f"[COCO Import] Received {total_size / 1024 / 1024:.0f} MB...")

        print(f"[COCO Import] File saved ({filename}), size: {total_size / 1024 / 1024:.1f} MB")

    except Exception as e:
        print(f"[COCO Import] Failed to save file: {e}")
        import traceback
        traceback.print_exc()
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(500, f"Failed to save uploaded file: {e}")

    # 在后台运行导入任务
    background_tasks.add_task(
        run_coco_import_sync,
        job_id,
        project_id,
        temp_file_path
    )

    return {"job_id": job_id, "status": "pending", "message": "导入任务已创建"}


def run_images_import_sync(job_id: int, project_id: int, temp_file_path: str):
    """后台运行纯图片 ZIP 导入任务"""
    try:
        with get_session() as session:
            job = session.get(ImportJob, job_id)
            if job:
                job.status = "running"
                job.startedAt = datetime.utcnow()
                job.message = "正在解析 ZIP 文件..."
                session.commit()

        def progress_callback(current: int, total: int, message: str):
            try:
                with get_session() as session:
                    job = session.get(ImportJob, job_id)
                    if job:
                        job.current = current
                        job.total = total
                        job.message = message
                        session.commit()
            except Exception as e:
                print(f"[Images Import Job {job_id}] Progress callback error: {e}")

        result = extract_images_from_zip_sync_with_progress(
            project_id, temp_file_path, progress_callback
        )

        with get_session() as session:
            job = session.get(ImportJob, job_id)
            if job:
                job.status = "succeeded"
                job.current = result["total"]
                job.total = result["total"]
                job.imported = result["imported"]
                job.duplicates = result["duplicates"]
                job.errors = result["errors"]
                job.finishedAt = datetime.utcnow()
                job.message = f"导入完成：{result['imported']} 张新增，{result['duplicates']} 张重复，{result['errors']} 张错误"
                session.commit()

    except Exception as e:
        import traceback
        traceback.print_exc()
        try:
            with get_session() as session:
                job = session.get(ImportJob, job_id)
                if job:
                    job.status = "failed"
                    job.finishedAt = datetime.utcnow()
                    job.message = f"导入失败：{str(e)}"
                    session.commit()
        except Exception:
            pass
    finally:
        try:
            os.unlink(temp_file_path)
        except Exception:
            pass


def run_yolo_import_sync(job_id: int, project_id: int, temp_file_path: str, import_annotations: bool):
    """后台运行YOLO导入任务（同步包装器）"""
    import asyncio
    try:
        # 创建新的事件循环
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(
            run_yolo_import(job_id, project_id, temp_file_path, import_annotations)
        )
    finally:
        loop.close()


async def run_yolo_import(job_id: int, project_id: int, temp_file_path: str, import_annotations: bool):
    """后台运行YOLO导入任务"""
    import traceback
    try:
        print(f"[YOLO Import Job {job_id}] Starting import for project {project_id}")

        # 更新任务状态为运行中
        with get_session() as session:
            job = session.get(ImportJob, job_id)
            if job:
                job.status = "running"
                job.startedAt = datetime.utcnow()
                job.message = "正在解析文件..."
                session.commit()
                print(f"[YOLO Import Job {job_id}] Status updated to running")

        # 创建进度回调
        def progress_callback(current: int, total: int, message: str):
            try:
                with get_session() as session:
                    job = session.get(ImportJob, job_id)
                    if job:
                        job.current = current
                        job.total = total
                        job.message = message
                        session.commit()
            except Exception as e:
                print(f"[YOLO Import Job {job_id}] Progress callback error: {e}")

        # 创建一个类似文件对象的包装器
        class FileWrapper:
            def __init__(self, path):
                self.file = open(path, 'rb')

            def close(self):
                self.file.close()

        file_wrapper = FileWrapper(temp_file_path)
        try:
            print(f"[YOLO Import Job {job_id}] Calling import_yolo_dataset")
            result = await import_yolo_dataset(
                project_id,
                file_wrapper,
                import_annotations,
                progress_callback
            )
            print(f"[YOLO Import Job {job_id}] Import completed: {result}")
        finally:
            file_wrapper.close()

        # 更新任务状态为完成
        with get_session() as session:
            job = session.get(ImportJob, job_id)
            if job:
                job.status = "succeeded"
                job.finishedAt = datetime.utcnow()
                job.total = result.get("total", 0)
                job.current = result.get("total", 0)
                job.imported = result.get("imported", 0)
                job.duplicates = result.get("duplicates", 0)
                job.errors = result.get("errors", 0)
                job.annotations_imported = result.get("annotations_imported", 0)
                job.annotations_skipped = result.get("annotations_skipped", 0)
                job.message = "导入完成"
                session.commit()
                print(f"[YOLO Import Job {job_id}] Status updated to succeeded")

    except Exception as e:
        # 更新任务状态为失败
        print(f"[YOLO Import Job {job_id}] Error: {e}")
        traceback.print_exc()
        with get_session() as session:
            job = session.get(ImportJob, job_id)
            if job:
                job.status = "failed"
                job.finishedAt = datetime.utcnow()
                job.message = f"导入失败: {str(e)}"
                session.commit()
    finally:
        # 清理临时文件
        try:
            os.unlink(temp_file_path)
            print(f"[YOLO Import Job {job_id}] Temp file cleaned up")
        except Exception:
            pass


def run_coco_import_sync(job_id: int, project_id: int, temp_file_path: str):
    """后台运行COCO导入任务（同步包装器）"""
    import asyncio
    try:
        # 创建新的事件循环
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(
            run_coco_import(job_id, project_id, temp_file_path)
        )
    finally:
        loop.close()


async def run_coco_import(job_id: int, project_id: int, temp_file_path: str):
    """后台运行COCO导入任务"""
    import traceback
    try:
        print(f"[COCO Import Job {job_id}] Starting import for project {project_id}")

        # 更新任务状态为运行中
        with get_session() as session:
            job = session.get(ImportJob, job_id)
            if job:
                job.status = "running"
                job.startedAt = datetime.utcnow()
                job.message = "正在解析COCO数据..."
                session.commit()
                print(f"[COCO Import Job {job_id}] Status updated to running")

        # 创建进度回调
        def progress_callback(current: int, total: int, message: str):
            try:
                with get_session() as session:
                    job = session.get(ImportJob, job_id)
                    if job:
                        job.current = current
                        job.total = total
                        job.message = message
                        session.commit()
            except Exception as e:
                print(f"[COCO Import Job {job_id}] Progress callback error: {e}")

        # 执行导入
        print(f"[COCO Import Job {job_id}] Calling import_coco_dataset")
        result = import_coco_dataset(
            project_id,
            temp_file_path,
            progress_callback
        )
        print(f"[COCO Import Job {job_id}] Import completed: {result}")

        # 更新任务状态为完成
        with get_session() as session:
            job = session.get(ImportJob, job_id)
            if job:
                job.status = "succeeded"
                job.finishedAt = datetime.utcnow()
                job.total = result.get("total_images", 0)
                job.current = result.get("total_images", 0)
                job.imported = result.get("imported_images", 0)
                job.duplicates = result.get("duplicate_images", 0)
                job.errors = result.get("error_images", 0)
                job.annotations_imported = result.get("imported_annotations", 0)
                job.annotations_skipped = result.get("skipped_annotations", 0)
                job.message = "导入完成"
                session.commit()
                print(f"[COCO Import Job {job_id}] Status updated to succeeded")

    except Exception as e:
        # 更新任务状态为失败
        print(f"[COCO Import Job {job_id}] Error: {e}")
        traceback.print_exc()
        with get_session() as session:
            job = session.get(ImportJob, job_id)
            if job:
                job.status = "failed"
                job.finishedAt = datetime.utcnow()
                job.message = f"导入失败: {str(e)}"
                session.commit()
    finally:
        # 清理临时文件
        try:
            os.unlink(temp_file_path)
            print(f"[COCO Import Job {job_id}] Temp file cleaned up")
        except Exception:
            pass


@router.get("/projects/{project_id}/import/jobs/{job_id}")
def get_import_job(project_id: int, job_id: int):
    """获取导入任务状态和进度"""
    with get_session() as session:
        job = session.get(ImportJob, job_id)
        if not job or job.projectId != project_id:
            raise HTTPException(404, "Import job not found")

        return {
            "id": job.id,
            "status": job.status,
            "format": job.format,
            "total": job.total,
            "current": job.current,
            "imported": job.imported,
            "duplicates": job.duplicates,
            "errors": job.errors,
            "annotations_imported": job.annotations_imported,
            "annotations_skipped": job.annotations_skipped,
            "message": job.message,
            "progress": round(job.current / job.total * 100, 1) if job.total > 0 else 0
        }


@router.post("/projects/{project_id}/generate-thumbnails")
def generate_thumbnails(project_id: int, background_tasks: BackgroundTasks):
    """
    为项目中没有缩略图的图片批量生成缩略图和标注用图
    """
    with get_session() as session:
        proj = session.get(Project, project_id)
        if not proj:
            raise HTTPException(404, "Project not found")

        # 查找没有缩略图的图片数量
        images_without_thumb = (
            session.query(Image)
            .filter(Image.projectId == project_id)
            .filter((Image.thumbnailPath == None) | (Image.thumbnailPath == ""))
            .count()
        )

    if images_without_thumb == 0:
        return {"message": "所有图片都已有缩略图", "total": 0, "status": "completed"}

    # 在后台执行生成任务
    background_tasks.add_task(run_generate_thumbnails, project_id)

    return {
        "message": f"开始为 {images_without_thumb} 张图片生成缩略图",
        "total": images_without_thumb,
        "status": "started"
    }


def run_generate_thumbnails(project_id: int):
    """后台任务：为项目图片生成缩略图"""
    import os
    from ..utils.oss_storage import resolve_local_path, get_basename

    with get_session() as session:
        images = (
            session.query(Image)
            .filter(Image.projectId == project_id)
            .filter((Image.thumbnailPath == None) | (Image.thumbnailPath == ""))
            .all()
        )

        total = len(images)
        success = 0
        failed = 0

        for i, img in enumerate(images):
            try:
                original_path = resolve_local_path(img.path)
                if not original_path:
                    print(f"[Thumbnail] 文件无法定位: {img.path}")
                    failed += 1
                    continue

                filename = get_basename(img.path)
                thumb_filename = os.path.splitext(filename)[0] + ".jpg"

                thumb_path, display_path = generate_thumbnail_and_display(
                    original_path, project_id, thumb_filename
                )

                if thumb_path and display_path:
                    img.thumbnailPath = thumb_path
                    img.displayPath = display_path
                    session.add(img)
                    success += 1
                else:
                    failed += 1

                # 每处理 10 张提交一次
                if (i + 1) % 10 == 0:
                    session.commit()
                    print(f"[Thumbnail] 进度: {i + 1}/{total}")

            except Exception as e:
                print(f"[Thumbnail] 生成失败: {img.path}, 错误: {e}")
                failed += 1

        session.commit()
        print(f"[Thumbnail] 完成: 成功 {success}, 失败 {failed}, 总计 {total}")


@router.get("/projects/{project_id}/thumbnail-status")
def get_thumbnail_status(project_id: int):
    """获取项目缩略图生成状态"""
    with get_session() as session:
        proj = session.get(Project, project_id)
        if not proj:
            raise HTTPException(404, "Project not found")

        total = session.query(Image).filter(Image.projectId == project_id).count()
        with_thumbnail = (
            session.query(Image)
            .filter(Image.projectId == project_id)
            .filter(Image.thumbnailPath != None)
            .filter(Image.thumbnailPath != "")
            .count()
        )

        return {
            "total": total,
            "with_thumbnail": with_thumbnail,
            "without_thumbnail": total - with_thumbnail,
            "progress": round(with_thumbnail / total * 100, 1) if total > 0 else 100
        }


@router.get("/projects/{project_id}/classes", response_model=list[Class])
def list_classes(project_id: int):
    with get_session() as session:
        proj = session.get(Project, project_id)
        if not proj:
            raise HTTPException(404, "Project not found")
        return session.query(Class).filter(Class.projectId == project_id).all()


class ClassCreate(BaseModel):
    name: str
    color: Optional[str] = None
    hotkey: Optional[str] = None


@router.post("/projects/{project_id}/classes", response_model=Class)
def create_class(project_id: int, body: ClassCreate):
    with get_session() as session:
        proj = session.get(Project, project_id)
        if not proj:
            raise HTTPException(404, "Project not found")
        c = Class(projectId=project_id, name=body.name, color=body.color, hotkey=body.hotkey)
        session.add(c)
        session.commit()
        session.refresh(c)
        return c


@router.delete("/projects/{project_id}")
def delete_project(project_id: int):
    """
    物理删除项目及其所有关联数据
    - 删除项目下所有标注
    - 删除项目下所有图片记录
    - 删除项目下所有类别
    - 删除训练任务和模型产物
    - 删除磁盘上的图片文件和模型文件
    """
    with get_session() as session:
        proj = session.get(Project, project_id)
        if not proj:
            raise HTTPException(404, "Project not found")

        # 删除标注（先删除，因为有外键约束）
        image_ids = [row.id for row in session.query(Image.id).filter(Image.projectId == project_id).all()]
        if image_ids:
            session.query(Annotation).filter(Annotation.imageId.in_(image_ids)).delete(synchronize_session=False)
            session.query(ProposedAnnotation).filter(ProposedAnnotation.imageId.in_(image_ids)).delete(synchronize_session=False)

        # 删除图片记录
        session.query(Image).filter(Image.projectId == project_id).delete(synchronize_session=False)

        # 删除类别
        session.query(Class).filter(Class.projectId == project_id).delete(synchronize_session=False)

        # 删除训练任务和模型产物
        train_ids = [row.id for row in session.query(TrainingJob.id).filter(TrainingJob.projectId == project_id).all()]
        if train_ids:
            session.query(ModelArtifact).filter(ModelArtifact.trainingJobId.in_(train_ids)).delete(synchronize_session=False)
        session.query(TrainingJob).filter(TrainingJob.projectId == project_id).delete(synchronize_session=False)

        # 删除其他关联数据
        session.query(AutoLabelJob).filter(AutoLabelJob.projectId == project_id).delete(synchronize_session=False)
        session.query(DatasetVersion).filter(DatasetVersion.projectId == project_id).delete(synchronize_session=False)
        session.query(ActivityLog).filter(ActivityLog.projectId == project_id).delete(synchronize_session=False)

        # 删除项目本身
        session.delete(proj)
        session.commit()

    # 删除磁盘上的文件（图片、数据集、模型）
    try:
        remove_project_files(project_id)
    except Exception as e:
        # 文件删除失败不影响数据库删除结果
        print(f"Warning: Failed to remove project files: {e}")

    return {"ok": True}
