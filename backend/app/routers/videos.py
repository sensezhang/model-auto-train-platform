"""
视频导入路由：接受视频文件，后台异步抽帧，返回 ImportJob ID。
"""

import os
import tempfile
from datetime import datetime

from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks

from ..db import get_session
from ..models import Project, ImportJob, JobStatus
from ..services.video_extractor import extract_frames


router = APIRouter(tags=["videos"])

ALLOWED_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}


@router.post("/projects/{project_id}/import/video")
async def import_video(
    project_id: int,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    上传视频文件，后台按 1fps 抽帧并导入到项目。

    支持格式：mp4, avi, mov, mkv, wmv, flv
    返回 { jobId, message }，通过 /api/projects/{id}/import/jobs/{jobId} 轮询进度。
    """
    if not file.filename:
        raise HTTPException(400, "未提供文件名")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_VIDEO_EXTS:
        raise HTTPException(400, f"不支持的视频格式 {ext}，支持：{', '.join(ALLOWED_VIDEO_EXTS)}")

    with get_session() as session:
        proj = session.get(Project, project_id)
        if not proj:
            raise HTTPException(404, "Project not found")

        # 创建导入任务
        job = ImportJob(
            projectId=project_id,
            format="video",
            status=JobStatus.pending,
            message="等待开始抽帧...",
            startedAt=None,
        )
        session.add(job)
        session.commit()
        session.refresh(job)
        job_id = job.id

    # 保存上传文件到临时目录（异步分块读取，支持大文件）
    suffix = ext
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_path = tmp.name
    tmp.close()
    try:
        chunk_size = 8 * 1024 * 1024  # 8MB
        with open(tmp_path, "wb") as f:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
    except Exception as e:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        raise HTTPException(500, f"保存上传文件失败: {e}")

    # 后台抽帧
    background_tasks.add_task(extract_frames, tmp_path, project_id, job_id)

    return {"jobId": job_id, "message": "视频上传成功，正在后台抽帧..."}
