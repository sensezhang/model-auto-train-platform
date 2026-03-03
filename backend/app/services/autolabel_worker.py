"""
VLM 自动标注工作线程

调用自部署的视觉大模型（OpenAI 兼容接口）对图片进行目标检测，
将检测结果存入 Annotation 表并将图片标记为 labeled=True。

环境变量（.env 中配置）：
    AUTOLABEL_BASE_URL   模型服务地址，如 http://10.0.0.1:8000/v1
    AUTOLABEL_MODEL      模型名称，如 glm-4v-9b
    AUTOLABEL_API_KEY    API Key（自部署可填任意字符串）
"""

import os
import base64
import json
import re
import logging
from datetime import datetime
from typing import List, Optional

from ..db import get_session
from ..models import AutoLabelJob, Image, Annotation, JobStatus
from ..utils.oss_storage import resolve_local_path

logger = logging.getLogger(__name__)

# ── 从环境变量读取模型配置（main.py 已在 import 前完成 load_dotenv）────────
AUTOLABEL_BASE_URL = os.getenv("AUTOLABEL_BASE_URL", "")
AUTOLABEL_MODEL    = os.getenv("AUTOLABEL_MODEL",    "glm-4v-9b")
AUTOLABEL_API_KEY  = os.getenv("AUTOLABEL_API_KEY",  "")


def _call_vlm(image_path: str, prompt: str, api_key: str) -> List[dict]:
    """
    调用自部署视觉大模型检测图片中的目标（OpenAI 兼容接口）。

    Returns:
        list of {"x1", "y1", "x2", "y2"} 像素坐标字典
    """
    try:
        from openai import OpenAI
        import httpx
    except ImportError:
        raise RuntimeError("openai 包未安装，请运行: pip install openai>=1.0.0")

    base_url = AUTOLABEL_BASE_URL
    model    = AUTOLABEL_MODEL
    key      = api_key or AUTOLABEL_API_KEY

    if not base_url:
        raise RuntimeError("AUTOLABEL_BASE_URL 未配置，请在 .env 中设置")
    if not key:
        raise RuntimeError("AUTOLABEL_API_KEY 未配置，请在 .env 中设置")

    # trust_env=False：禁止 httpx 读取系统代理环境变量（HTTP_PROXY/HTTPS_PROXY），
    # 避免内网模型请求被 Privoxy/Clash 等代理拦截导致 500 错误
    client = OpenAI(
        base_url=base_url,
        api_key=key,
        http_client=httpx.Client(trust_env=False),
    )

    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    # 根据文件扩展名确定 MIME 类型
    ext = os.path.splitext(image_path)[1].lower()
    mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"

    system_msg = (
        "你是目标检测助手。分析图片，仅返回纯JSON，格式为："
        '{"detections":[{"bbox":[x1,y1,x2,y2]},...]} '
        "bbox 为像素坐标，x1y1 左上角，x2y2 右下角。不要输出任何其他文字。"
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system_msg,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{img_b64}"},
                    },
                    {
                        "type": "text",
                        "text": f"请{prompt}，返回所有目标的检测结果JSON。",
                    },
                ],
            }
        ],
        temperature=0.1,
    )

    raw = resp.choices[0].message.content.strip()
    print(f"[AutoLabel] VLM raw response: {raw[:500]}", flush=True)

    # 提取 JSON（模型有时会在 JSON 前后附加文字）
    json_match = re.search(r'\{.*\}', raw, re.DOTALL)
    if not json_match:
        print(f"[AutoLabel] 未找到 JSON，完整返回: {raw}", flush=True)
        logger.warning("VLM 未返回有效 JSON: %s", raw[:200])
        return []

    try:
        data = json.loads(json_match.group())
    except json.JSONDecodeError as e:
        print(f"[AutoLabel] JSON 解析失败: {e} | raw: {raw[:300]}", flush=True)
        logger.warning("JSON 解析失败: %s | raw: %s", e, raw[:200])
        return []

    detections = data.get("detections", [])
    results = []
    for det in detections:
        bbox = det.get("bbox", [])
        if len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            results.append({"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)})
    return results


def _bbox_to_xywh(x1: float, y1: float, x2: float, y2: float,
                   img_w: int, img_h: int) -> Optional[dict]:
    """
    将 VLM 返回的 0-1000 归一化坐标 (x1,y1,x2,y2) 转换为像素坐标的
    中心点格式 (cx, cy, w, h)，供前端直接使用。

    GLM 等视觉模型以 0-1000 表示相对位置（对应图片宽/高的千分之一），
    转换步骤：
        1. 除以 1000 → 归一化到 [0, 1]
        2. 乘以原图实际宽/高 → 像素坐标
        3. 转为左上角格式 (x, y, w, h)，x/y 为左上角顶点像素坐标
    """
    if img_w <= 0 or img_h <= 0:
        return None

    # 归一化（0-1000 → 像素）
    x1_px = (float(x1) / 1000.0) * img_w
    y1_px = (float(y1) / 1000.0) * img_h
    x2_px = (float(x2) / 1000.0) * img_w
    y2_px = (float(y2) / 1000.0) * img_h

    # 边界夹紧
    x1_px = max(0.0, min(x1_px, img_w))
    y1_px = max(0.0, min(y1_px, img_h))
    x2_px = max(0.0, min(x2_px, img_w))
    y2_px = max(0.0, min(y2_px, img_h))

    if x2_px <= x1_px or y2_px <= y1_px:
        return None

    w = x2_px - x1_px
    h = y2_px - y1_px
    return {"x": x1_px, "y": y1_px, "w": w, "h": h}


def run_autolabel_job(job_id: int, api_key: str):
    """
    后台工作线程入口：处理单个 AutoLabelJob。

    每张图调用一次 GLM-4V，解析结果后写入 Annotation 表，
    并将图片标记为 labeled=True。
    """
    # 加载 job 信息
    with get_session() as session:
        job = session.get(AutoLabelJob, job_id)
        if not job:
            logger.error("AutoLabelJob %d not found", job_id)
            return

        job.status = JobStatus.running
        job.startedAt = datetime.utcnow()
        session.add(job)
        session.commit()

        project_id = job.projectId
        class_id = job.classId
        prompt = job.prompt or "检测图中所有目标"
        image_ids_stored = job.logsRef  # 用 logsRef 暂存 JSON 格式的 imageIds

    # 确定要处理的图片列表
    with get_session() as session:
        if image_ids_stored:
            try:
                ids = json.loads(image_ids_stored)
                images = session.query(Image).filter(Image.id.in_(ids)).all()
            except Exception:
                images = []
        else:
            images = session.query(Image).filter(
                Image.projectId == project_id,
                Image.labeled == False  # noqa: E712
            ).all()

        image_list = [(img.id, img.path, img.displayPath, img.width or 0, img.height or 0)
                      for img in images]

    total = len(image_list)

    # 更新总数
    with get_session() as session:
        job = session.get(AutoLabelJob, job_id)
        if job:
            job.imagesCount = total
            session.add(job)
            session.commit()

    boxes_count = 0

    for idx, (img_id, img_path, display_path, img_w, img_h) in enumerate(image_list):
        # 优先使用 displayPath（标注用图），其次用原图路径
        # resolve_local_path 同时支持本地相对路径和 OSS URL
        actual_path = display_path or img_path
        abs_path = resolve_local_path(actual_path)

        if not abs_path:
            logger.warning("图片文件无法定位，跳过: %s", actual_path)
        else:
            try:
                detections = _call_vlm(abs_path, prompt, api_key)

                with get_session() as session:
                    for det in detections:
                        coords = _bbox_to_xywh(
                            det["x1"], det["y1"], det["x2"], det["y2"],
                            img_w or 1, img_h or 1
                        )
                        if coords is None:
                            continue
                        ann = Annotation(
                            imageId=img_id,
                            classId=class_id,
                            x=coords["x"],
                            y=coords["y"],
                            w=coords["w"],
                            h=coords["h"],
                            source="ai",
                        )
                        session.add(ann)
                        boxes_count += 1

                    # 更新图片状态（有检测结果则标为 annotated，不设置 labeled）
                    if detections:
                        img_obj = session.get(Image, img_id)
                        if img_obj:
                            img_obj.status = "annotated"
                            session.add(img_obj)

                    session.commit()

            except Exception as e:
                import traceback
                print(f"[AutoLabel] 处理图片 {img_id} 异常: {e}\n{traceback.format_exc()}", flush=True)
                logger.error("处理图片 %d 失败: %s", img_id, e)

        # 更新进度
        with get_session() as session:
            job = session.get(AutoLabelJob, job_id)
            if job:
                job.processedCount = idx + 1
                job.boxesCount = boxes_count
                session.add(job)
                session.commit()

    # 完成
    with get_session() as session:
        job = session.get(AutoLabelJob, job_id)
        if job:
            job.status = JobStatus.succeeded
            job.finishedAt = datetime.utcnow()
            job.processedCount = total
            job.boxesCount = boxes_count
            session.add(job)
            session.commit()

    logger.info("AutoLabelJob %d 完成：处理 %d 张图，生成 %d 个框", job_id, total, boxes_count)
