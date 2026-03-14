"""SAM3 分割推理服务（远程 API 版本）

调用远程 SAM3 服务（默认 http://192.168.2.200:30002），坐标全程使用原图像素空间，
无需本地加载模型，也无 letterbox 坐标映射问题。

API 格式（参考 illegal-parking-detect skill）：
  POST /api/segment
  multipart/form-data:
    image          = <图片文件>
    prompts        = "car,truck,road"   # 逗号分隔
    score_threshold = "0.4"
  Response:
  {
    "total_inference_time": 1.23,
    "results": {
      "<prompt>": {
        "count": N,
        "boxes":     [[x1,y1,x2,y2], ...],   # 原图像素坐标
        "scores":    [0.95, ...],
        "masks_rle": [{"size":[H,W],"counts":[...],"start_value":0}, ...]
      }
    }
  }
"""
import os
from typing import List, Dict, Any

import cv2
import numpy as np
import requests
from PIL import Image as PILImage

# 远程 API 地址，可通过环境变量覆盖
SAM3_API_URL = os.environ.get("SAM3_API_URL", "http://192.168.2.200:30002")

# 兼容旧代码引用的 SAM3_MODEL_PATH —— 远程模式下始终视为"已就绪"
SAM3_MODEL_PATH = "remote"


# ──────────────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────────────

def _rle_to_mask(rle: dict) -> np.ndarray:
    """将 RLE 解码为二值 mask，shape=(H, W), dtype=uint8，值 0/1。"""
    h, w = rle["size"]          # size = [height, width]
    counts = rle["counts"]
    start_value = rle.get("start_value", 0)

    flat = np.zeros(h * w, dtype=np.uint8)
    pos = 0
    cur = start_value
    for length in counts:
        flat[pos: pos + length] = cur
        pos += length
        cur = 1 - cur

    return flat.reshape(h, w)


def _mask_to_polygon(mask: np.ndarray, img_w: int, img_h: int) -> List[float]:
    """
    从二值 mask 中提取最大轮廓，简化后返回 flat [x1,y1,x2,y2,...] 格式。
    坐标已在原图像素空间，clip 到图像边界。
    """
    binary = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 3:
        return []

    epsilon = 0.005 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    if len(approx) < 3:
        return []

    flat: List[float] = []
    for pt in approx:
        flat.append(round(float(np.clip(pt[0][0], 0, img_w)), 1))
        flat.append(round(float(np.clip(pt[0][1], 0, img_h)), 1))
    return flat


# ──────────────────────────────────────────────
# 主推理函数（对外接口与原本地版本保持一致）
# ──────────────────────────────────────────────

def run_sam3(
    image_path: str,
    text_labels: List[str],
    conf: float = 0.4,
    iou: float = 0.9,       # 保留参数兼容调用方，远程 API 不使用
    imgsz: int = 512,       # 保留参数兼容调用方，远程 API 不使用
) -> Dict[str, Any]:
    """
    调用远程 SAM3 API 对图片进行分割推理。

    Args:
        image_path:   图片本地路径（由路由层临时保存）
        text_labels:  识别类别列表，如 ["car", "person"]；空列表则使用 "object"
        conf:         置信度阈值（传给远程 API 的 score_threshold）

    Returns:
        {
          "image_width":  int,           # 原图宽
          "image_height": int,           # 原图高
          "segments": [
            {
              "class_name": str,
              "polygon":    [x1,y1,...], # 原图像素坐标，flat 格式
              "bbox":       [x,y,w,h],  # 原图像素坐标
              "confidence": float
            }, ...
          ]
        }
    """
    # 读取原图尺寸（以 PIL 为准，与前端 result.image_width/height 一致）
    img = PILImage.open(image_path).convert("RGB")
    img_w, img_h = img.size

    prompts = text_labels if text_labels else ["object"]
    prompts_str = ",".join(prompts)

    print(f"[SAM3] → {SAM3_API_URL}/api/segment  prompts=[{prompts_str}]  conf={conf}  图片={img_w}x{img_h}")

    # 清除可能影响 requests 的代理环境变量，确保直连内网地址
    for _k in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'all_proxy']:
        os.environ.pop(_k, None)

    # 调用远程 API（同时通过 proxies 参数显式禁用代理）
    _no_proxy = {"http": None, "https": None}
    with open(image_path, "rb") as f:
        resp = requests.post(
            f"{SAM3_API_URL}/api/segment",
            files={"image": (os.path.basename(image_path), f, "image/jpeg")},
            data={"prompts": prompts_str, "score_threshold": str(conf)},
            timeout=120,
            proxies=_no_proxy,
        )
    resp.raise_for_status()
    api_result = resp.json()

    results_by_prompt: dict = api_result.get("results", {})
    segments: List[Dict[str, Any]] = []

    for class_name in prompts:
        prompt_data = results_by_prompt.get(class_name, {})
        count = prompt_data.get("count", 0)
        if count == 0:
            continue

        boxes     = prompt_data.get("boxes", [])      # [[x1,y1,x2,y2], ...]  原图坐标
        scores    = prompt_data.get("scores", [])
        masks_rle = prompt_data.get("masks_rle", [])

        for i in range(count):
            score = float(scores[i]) if i < len(scores) else conf

            # ── bbox：API 给 xyxy → 转成 xywh，clip 到图像范围 ──
            if i >= len(boxes):
                continue
            b = boxes[i]
            bx = round(float(np.clip(b[0], 0, img_w)), 1)
            by = round(float(np.clip(b[1], 0, img_h)), 1)
            bw = round(float(np.clip(b[2] - b[0], 0, img_w - bx)), 1)
            bh = round(float(np.clip(b[3] - b[1], 0, img_h - by)), 1)

            # ── polygon：从 RLE mask 提取最大轮廓 ──
            flat: List[float] = []
            if i < len(masks_rle):
                mask = _rle_to_mask(masks_rle[i])
                flat = _mask_to_polygon(mask, img_w, img_h)

            # fallback：无 polygon 时用 bbox 四角
            if not flat:
                flat = [bx, by, bx + bw, by, bx + bw, by + bh, bx, by + bh]

            segments.append({
                "class_name": class_name,
                "polygon":    flat,
                "bbox":       [bx, by, bw, bh],
                "confidence": round(score, 3),
            })

    print(f"[SAM3] ← 完成: {len(segments)} 个目标")
    return {"image_width": img_w, "image_height": img_h, "segments": segments}
