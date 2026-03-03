"""
视频抽帧服务：使用 ffmpeg（imageio-ffmpeg 内置）按 1fps 提取帧，保存为 JPEG，
并写入 Image 数据库记录（含缩略图和标注用图）。

使用 ffmpeg 替代 OpenCV 读取视频，支持 H.264 High Profile 等 OpenCV 无法解码的格式。
"""

import os
import re
import hashlib
import logging
import subprocess
from datetime import datetime

import cv2
import numpy as np

try:
    import imageio_ffmpeg
    _FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    _FFMPEG_EXE = "ffmpeg"  # fallback: 系统 PATH 中的 ffmpeg

from ..db import get_session
from ..models import Image as DBImage, ImportJob, JobStatus
from ..utils.files import (
    project_images_dir,
    generate_thumbnail_and_display,
)

logger = logging.getLogger(__name__)


def _sha1_file(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _get_video_info(video_path: str):
    """
    通过 ffmpeg 解析视频元数据，返回 (width, height, fps, duration_sec)。
    解析失败返回 None。
    """
    try:
        result = subprocess.run(
            [_FFMPEG_EXE, "-i", video_path],
            capture_output=True, text=True, errors="replace", timeout=30
        )
        stderr = result.stderr

        dur_m = re.search(r"Duration:\s*(\d+):(\d+):(\d+)\.(\d+)", stderr)
        vid_m = re.search(r"Stream.*Video:.* (\d{3,5})x(\d{3,5})", stderr)
        fps_m = re.search(r"(\d+(?:\.\d+)?)\s*fps", stderr)

        if not (dur_m and vid_m):
            return None

        h, m, s, cs = int(dur_m[1]), int(dur_m[2]), int(dur_m[3]), int(dur_m[4])
        duration_sec = h * 3600 + m * 60 + s + cs / 100
        width = int(vid_m[1])
        height = int(vid_m[2])
        fps = float(fps_m[1]) if fps_m else 30.0

        return width, height, fps, duration_sec
    except Exception as e:
        logger.warning("_get_video_info 失败: %s", e)
        return None


def extract_frames(video_path: str, project_id: int, job_id: int):
    """
    从视频中按 1fps 提取帧，保存并写入数据库。
    使用 ffmpeg 管道读取，支持 H.264 High Profile 等格式。

    Args:
        video_path: 视频文件的绝对路径
        project_id: 目标项目 ID
        job_id: ImportJob ID，用于更新进度
    """
    file_size = os.path.getsize(video_path) if os.path.exists(video_path) else -1
    print(f"[VideoExtractor] 开始: path={video_path}, size={file_size} bytes, "
          f"project={project_id}, job={job_id}", flush=True)

    # 获取视频元数据
    info = _get_video_info(video_path)
    if info is None:
        _fail_job(job_id, f"无法解析视频文件: {video_path}")
        return

    width, height, fps, duration_sec = info
    estimated_total = max(1, int(duration_sec))
    print(f"[VideoExtractor] width={width}, height={height}, fps={fps}, "
          f"duration={duration_sec:.1f}s, estimated_frames={estimated_total}", flush=True)

    # 更新 job 为 running
    with get_session() as session:
        job = session.get(ImportJob, job_id)
        if job:
            job.status = JobStatus.running
            job.total = estimated_total
            job.startedAt = datetime.utcnow()
            job.message = "开始抽帧..."
            session.add(job)
            session.commit()

    images_dir = project_images_dir(project_id)
    video_stem = os.path.splitext(os.path.basename(video_path))[0]

    frame_size = width * height * 3  # BGR raw bytes per frame

    # ffmpeg 命令：以 1fps 输出原始 BGR 帧到 stdout
    cmd = [
        _FFMPEG_EXE,
        "-i", video_path,
        "-vf", "fps=1",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "pipe:1",
    ]

    saved = 0
    duplicates = 0
    errors = 0
    frame_idx = 0

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        _fail_job(job_id, f"启动 ffmpeg 失败: {e}")
        return

    try:
        while True:
            raw = proc.stdout.read(frame_size)
            if len(raw) < frame_size:
                break  # 视频结束或读取失败

            frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))

            # 构造文件名：视频名_秒数.jpg
            filename = f"{video_stem}_f{frame_idx:06d}.jpg"
            dest_path = os.path.join(images_dir, filename)

            # 如果文件名已存在则加 _dup 避免覆盖
            if os.path.exists(dest_path):
                dest_path = os.path.join(images_dir, f"{video_stem}_f{frame_idx:06d}_dup{saved}.jpg")
                filename = os.path.basename(dest_path)

            try:
                cv2.imwrite(dest_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

                checksum = _sha1_file(dest_path)

                # 去重检查
                with get_session() as session:
                    existing = session.query(DBImage).filter(
                        DBImage.projectId == project_id,
                        DBImage.checksum == checksum
                    ).first()

                    if existing:
                        duplicates += 1
                        os.remove(dest_path)
                    else:
                        rel_path = os.path.relpath(dest_path, os.getcwd()).replace("\\", "/")

                        thumb_rel, display_rel = generate_thumbnail_and_display(
                            dest_path, project_id, filename
                        )

                        db_img = DBImage(
                            projectId=project_id,
                            path=rel_path,
                            thumbnailPath=thumb_rel.replace("\\", "/") if thumb_rel else None,
                            displayPath=display_rel.replace("\\", "/") if display_rel else None,
                            width=width,
                            height=height,
                            checksum=checksum,
                        )
                        session.add(db_img)
                        session.commit()
                        saved += 1

            except Exception as e:
                import traceback
                logger.error("保存帧 %d 失败: %s", frame_idx, e)
                print(f"[VideoExtractor] 保存帧 {frame_idx} 异常: {e}\n{traceback.format_exc()}", flush=True)
                errors += 1

            # 更新进度
            with get_session() as session:
                job = session.get(ImportJob, job_id)
                if job:
                    job.current = saved + duplicates + errors
                    job.imported = saved
                    job.duplicates = duplicates
                    job.errors = errors
                    job.message = f"已抽取 {saved} 帧..."
                    session.add(job)
                    session.commit()

            frame_idx += 1

    finally:
        proc.kill()
        proc.wait()

    print(f"[VideoExtractor] 循环结束: frame_idx={frame_idx}, saved={saved}, "
          f"duplicates={duplicates}, errors={errors}", flush=True)

    # 清理临时视频文件
    try:
        os.remove(video_path)
    except Exception:
        pass

    # 如果一帧都没读取到，视为失败
    if frame_idx == 0:
        _fail_job(job_id, "无法从视频中读取帧，请检查视频格式是否支持")
        logger.error("视频抽帧失败：无法读取任何帧 (project=%d)", project_id)
        return

    # 完成
    with get_session() as session:
        job = session.get(ImportJob, job_id)
        if job:
            job.status = JobStatus.succeeded
            job.total = saved + duplicates + errors
            job.current = job.total
            job.imported = saved
            job.duplicates = duplicates
            job.errors = errors
            job.finishedAt = datetime.utcnow()
            job.message = f"抽帧完成：新增 {saved} 帧，重复 {duplicates}，错误 {errors}"
            session.add(job)
            session.commit()

    logger.info("视频抽帧完成：project=%d, saved=%d, dup=%d, err=%d", project_id, saved, duplicates, errors)


def _fail_job(job_id: int, message: str):
    with get_session() as session:
        job = session.get(ImportJob, job_id)
        if job:
            job.status = JobStatus.failed
            job.message = message
            job.finishedAt = datetime.utcnow()
            session.add(job)
            session.commit()
    logger.error("视频抽帧失败 (job=%d): %s", job_id, message)
