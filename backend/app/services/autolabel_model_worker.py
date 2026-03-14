"""
模型推理自动标注工作线程

使用已训练的 YOLO / RF-DETR 模型对图片批量推理，
将检测结果写入 Annotation 表（source='ai'），并将有结果的图片状态标记为 'annotated'。

调用方式（由 autolabel.py 通过 BackgroundTasks 触发）：
    background_tasks.add_task(run_model_autolabel_job, job_id)

logsRef JSON 格式（由 autolabel.py 创建 job 时写入）：
{
    "type": "model",
    "artifactId": <int>,
    "confidenceThreshold": <float>,
    "iouThreshold": <float>,
    "scope": "unlabeled" | "all",
    "imageIds": [<int>, ...]   // 可选，有则优先使用
}
"""

import os
import base64
import json
import logging
import traceback
from datetime import datetime

from sqlmodel import select

from ..db import get_session
from ..models import (
    AutoLabelJob, Image, Annotation,
    ModelArtifact, TrainingJob,
    Class as DBClass, JobStatus,
)
from ..services.inference import run_inference
from ..utils.oss_storage import resolve_local_path

logger = logging.getLogger(__name__)


def run_model_autolabel_job(job_id: int) -> None:
    """后台工作线程入口：使用训练模型对图片批量推理并写入标注"""

    # ── Step 1: 读取 job 元数据 ────────────────────────────────────────────
    with get_session() as session:
        job = session.get(AutoLabelJob, job_id)
        if not job:
            logger.error("AutoLabelJob %d not found", job_id)
            return

        try:
            meta = json.loads(job.logsRef or "{}")
        except Exception:
            meta = {}

        artifact_id         = meta.get("artifactId")
        confidence_threshold = float(meta.get("confidenceThreshold", 0.25))
        iou_threshold        = float(meta.get("iouThreshold", 0.45))
        image_ids            = meta.get("imageIds")   # None 或 list
        scope                = meta.get("scope", "unlabeled")
        project_id           = job.projectId

        # 加载模型产物信息
        artifact = session.get(ModelArtifact, artifact_id)
        if not artifact:
            job.status = JobStatus.failed
            session.add(job)
            session.commit()
            logger.error("ModelArtifact %d not found for job %d", artifact_id, job_id)
            return

        training_job = session.get(TrainingJob, artifact.trainingJobId)
        if not training_job:
            job.status = JobStatus.failed
            session.add(job)
            session.commit()
            logger.error("TrainingJob not found for artifact %d", artifact_id)
            return

        framework   = training_job.framework
        model_type  = "medium"
        if framework.lower() == "rfdetr":
            variant = training_job.modelVariant.lower()
            if "small" in variant:
                model_type = "small"
            elif "large" in variant:
                model_type = "large"

        # 解析模型文件路径
        model_path = resolve_local_path(artifact.path)
        if not model_path:
            job.status = JobStatus.failed
            session.add(job)
            session.commit()
            logger.error("Model file not found: %s (job %d)", artifact.path, job_id)
            return

        # 构建 class_index → DB class_id 映射（按 id 排序与训练数据顺序一致）
        classes = session.exec(
            select(DBClass)
            .where(DBClass.projectId == project_id)
            .order_by(DBClass.id)
        ).all()
        class_id_map = {i: cls.id for i, cls in enumerate(classes)}

        # 确定待处理图片列表
        if image_ids:
            images = session.query(Image).filter(
                Image.id.in_(image_ids),
                Image.projectId == project_id,
            ).all()
        elif scope == "all":
            images = session.query(Image).filter(
                Image.projectId == project_id,
            ).all()
        else:
            images = session.query(Image).filter(
                Image.projectId == project_id,
                Image.labeled == False,  # noqa: E712
            ).all()

        image_list = [
            (img.id, img.path, img.displayPath, img.width or 0, img.height or 0)
            for img in images
        ]

        # 更新 job 状态为 running
        job.status     = JobStatus.running
        job.startedAt  = datetime.utcnow()
        job.imagesCount = len(image_list)
        session.add(job)
        session.commit()

    total       = len(image_list)
    boxes_count = 0

    # ── Step 2: 逐图推理 ───────────────────────────────────────────────────
    for idx, (img_id, img_path, display_path, img_w, img_h) in enumerate(image_list):
        # 优先使用标注用图（分辨率更高），其次使用原图
        actual_path = display_path or img_path
        abs_path    = resolve_local_path(actual_path) or resolve_local_path(img_path)

        if not abs_path:
            logger.warning("[ModelAutoLabel] 图片文件无法定位，跳过: %s (job %d)", actual_path, job_id)
        else:
            try:
                # 读取图片并编码为 base64
                with open(abs_path, "rb") as f:
                    img_bytes = f.read()

                ext  = os.path.splitext(abs_path)[1].lower()
                mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
                img_b64 = f"data:{mime};base64,{base64.b64encode(img_bytes).decode()}"

                # 执行模型推理
                result = run_inference(
                    model_path=model_path,
                    framework=framework,
                    image_data=img_b64,
                    confidence_threshold=confidence_threshold,
                    iou_threshold=iou_threshold,
                    imgsz=640,
                    model_type=model_type,
                )

                detections = result.get("detections", [])

                # 写入标注
                with get_session() as session:
                    saved = 0
                    for det in detections:
                        class_index = det["class_id"]
                        class_id    = class_id_map.get(class_index)
                        if class_id is None:
                            # 模型输出了训练数据中不存在的类别索引，跳过
                            continue

                        ann = Annotation(
                            imageId    = img_id,
                            classId    = class_id,
                            x          = float(det["x"]),
                            y          = float(det["y"]),
                            w          = float(det["width"]),
                            h          = float(det["height"]),
                            confidence = float(det["confidence"]),
                            source     = "ai",
                        )
                        session.add(ann)
                        saved += 1

                    boxes_count += saved

                    # 若有检测结果，更新图片状态为 annotated
                    if saved > 0:
                        img_obj = session.get(Image, img_id)
                        if img_obj:
                            img_obj.status = "annotated"
                            session.add(img_obj)

                    session.commit()

                print(
                    f"[ModelAutoLabel] 图片 {img_id}：推理完成，生成 {saved} 个框 "
                    f"(job {job_id})",
                    flush=True,
                )

            except Exception as e:
                print(
                    f"[ModelAutoLabel] 处理图片 {img_id} 异常 (job {job_id}): {e}\n"
                    f"{traceback.format_exc()}",
                    flush=True,
                )
                logger.error("处理图片 %d 失败 (job %d): %s", img_id, job_id, e)

        # 每张图处理完成后更新进度
        with get_session() as session:
            job_db = session.get(AutoLabelJob, job_id)
            if job_db:
                job_db.processedCount = idx + 1
                job_db.boxesCount     = boxes_count
                session.add(job_db)
                session.commit()

    # ── Step 3: 标记完成 ──────────────────────────────────────────────────
    with get_session() as session:
        job_db = session.get(AutoLabelJob, job_id)
        if job_db:
            job_db.status         = JobStatus.succeeded
            job_db.finishedAt     = datetime.utcnow()
            job_db.processedCount = total
            job_db.boxesCount     = boxes_count
            session.add(job_db)
            session.commit()

    logger.info(
        "ModelAutoLabelJob %d 完成：处理 %d 张图，生成 %d 个框",
        job_id, total, boxes_count,
    )
