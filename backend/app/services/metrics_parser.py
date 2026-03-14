"""
训练指标解析服务 - 从日志文件中提取训练指标
"""
import os
import re
import csv
from typing import List, Dict, Optional
from datetime import datetime


def parse_yolo_results_csv(csv_path: str) -> List[Dict]:
    """
    解析 Ultralytics 自动生成的 results.csv，提取每个 epoch 的指标。
    比日志解析更可靠，数据直接来自 Ultralytics 内部统计。

    CSV 列名示例（带前缀/后缀）：
        epoch, train/box_loss, train/cls_loss, train/dfl_loss,
        metrics/precision(B), metrics/recall(B),
        metrics/mAP50(B), metrics/mAP50-95(B),
        val/box_loss, val/cls_loss, val/dfl_loss
    """
    if not os.path.exists(csv_path):
        return []

    metrics = []
    try:
        with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 去掉列名前后空格（Ultralytics CSV 列名有前导空格）
                row = {k.strip(): v.strip() for k, v in row.items() if k}

                def get_float(row: dict, *keys) -> Optional[float]:
                    for k in keys:
                        v = row.get(k)
                        if v is not None:
                            try:
                                return float(v)
                            except (ValueError, TypeError):
                                pass
                    return None

                epoch_raw = get_float(row, 'epoch')
                if epoch_raw is None:
                    continue

                # Ultralytics epoch 从 0 开始，转成 1-based 显示
                epoch = int(epoch_raw) + 1

                # 训练损失：优先 box_loss，作为总 loss 的代表
                train_box  = get_float(row, 'train/box_loss')
                train_cls  = get_float(row, 'train/cls_loss')
                train_dfl  = get_float(row, 'train/dfl_loss')
                loss = None
                if train_box is not None and train_cls is not None and train_dfl is not None:
                    loss = round(train_box + train_cls + train_dfl, 4)
                elif train_box is not None:
                    loss = train_box

                # 验证损失
                val_box = get_float(row, 'val/box_loss')
                val_cls = get_float(row, 'val/cls_loss')
                val_dfl = get_float(row, 'val/dfl_loss')
                val_loss = None
                if val_box is not None and val_cls is not None and val_dfl is not None:
                    val_loss = round(val_box + val_cls + val_dfl, 4)

                # mAP — 列名可能是 metrics/mAP50(B) 或 metrics/mAP50
                map50 = get_float(
                    row,
                    'metrics/mAP50(B)', 'metrics/mAP50',
                    'mAP50(B)', 'mAP50',
                )
                map50_95 = get_float(
                    row,
                    'metrics/mAP50-95(B)', 'metrics/mAP50-95',
                    'mAP50-95(B)', 'mAP50-95',
                )
                precision = get_float(
                    row,
                    'metrics/precision(B)', 'metrics/precision',
                    'precision(B)', 'precision',
                )
                recall = get_float(
                    row,
                    'metrics/recall(B)', 'metrics/recall',
                    'recall(B)', 'recall',
                )

                metrics.append({
                    'epoch':      epoch,
                    'loss':       loss,
                    'val_loss':   val_loss,
                    'mAP50':      map50,
                    'mAP50-95':   map50_95,
                    'precision':  precision,
                    'recall':     recall,
                    'timestamp':  datetime.now().isoformat(),
                })
    except Exception as e:
        print(f"Error parsing results.csv: {e}")

    return metrics


def parse_yolo_logs(log_path: str) -> List[Dict]:
    """
    解析YOLO训练日志，提取每个epoch的指标。
    优先使用同级目录下 train/results.csv（更准确），
    若 CSV 不存在则回退到日志解析。

    Returns:
        指标列表，每个元素包含: epoch, loss, mAP50, mAP50-95等
    """
    # ── 优先解析 results.csv ──────────────────────────────────
    if log_path and os.path.exists(log_path):
        csv_path = os.path.join(os.path.dirname(log_path), 'train', 'results.csv')
        csv_metrics = parse_yolo_results_csv(csv_path)
        if csv_metrics:
            return csv_metrics

    # ── 回退：解析自定义日志文件 ──────────────────────────────
    if not os.path.exists(log_path):
        return []

    metrics = []
    current_epoch = None

    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        for line in lines:
            # 格式1: "Starting epoch 1"
            m = re.search(r'[Ss]tarting epoch\s+(\d+)', line)
            if m:
                current_epoch = int(m.group(1))

            # 格式2: "Epoch 1/100 completed, loss=..."
            m = re.search(r'[Ee]poch[:\s]+(\d+)', line)
            if m:
                current_epoch = int(m.group(1))

            # 格式3: "Completed epoch 1"
            m = re.search(r'[Cc]ompleted epoch\s+(\d+)', line)
            if m:
                current_epoch = int(m.group(1))

            # 提取 loss
            loss = None
            for pattern in [
                r'(?:train_)?loss[=:\s]+([\d.]+)',
                r'box_loss[=:\s]+([\d.]+)',
                r'total.*?loss[=:\s]+([\d.]+)',
            ]:
                lm = re.search(pattern, line, re.IGNORECASE)
                if lm:
                    loss = float(lm.group(1))
                    break

            # 提取 mAP50（修复：不用 [^-] 消耗分隔符，改用负向先行断言）
            # 匹配 "mAP50=0.xx" 或 "mAP50: 0.xx"，但不匹配 "mAP50-95"
            map50_m = re.search(r'mAP50(?![-\d(])[=:\s]+([\d.]+)', line, re.IGNORECASE)
            map50 = float(map50_m.group(1)) if map50_m else None

            map5095_m = re.search(r'mAP50-95[=:\s]+([\d.]+)', line, re.IGNORECASE)
            map50_95 = float(map5095_m.group(1)) if map5095_m else None

            if current_epoch and (loss is not None or map50 is not None):
                existing = next((e for e in metrics if e['epoch'] == current_epoch), None)
                if existing:
                    if loss is not None:
                        existing['loss'] = loss
                    if map50 is not None:
                        existing['mAP50'] = map50
                    if map50_95 is not None:
                        existing['mAP50-95'] = map50_95
                else:
                    metrics.append({
                        'epoch':     current_epoch,
                        'loss':      loss,
                        'mAP50':     map50,
                        'mAP50-95':  map50_95,
                        'timestamp': datetime.now().isoformat(),
                    })
    except Exception as e:
        print(f"Error parsing YOLO logs: {e}")

    return metrics


def parse_rfdetr_logs(log_path: str) -> List[Dict]:
    """
    解析RF-DETR训练日志，提取每个epoch的指标

    Returns:
        指标列表
    """
    if not os.path.exists(log_path):
        return []

    metrics = []
    current_epoch = None

    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        for line in lines:
            # 匹配多种RF-DETR日志格式

            # 格式1: "Epoch [1/100]" 或 "Epoch: 1"
            epoch_match = re.search(r'[Ee]poch\s*\[?(\d+)', line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))

            # 格式2: "Starting epoch 1"
            epoch_start = re.search(r'[Ss]tarting epoch\s+(\d+)', line)
            if epoch_start:
                current_epoch = int(epoch_start.group(1))

            # 提取loss（多种格式）
            loss_patterns = [
                r'[Ll]oss[=:\s]+([\d.]+)',
                r'[Tt]rain.*?loss[=:\s]+([\d.]+)',
                r'[Tt]otal.*?loss[=:\s]+([\d.]+)',
            ]

            loss = None
            for pattern in loss_patterns:
                loss_match = re.search(pattern, line)
                if loss_match:
                    loss = float(loss_match.group(1))
                    break

            # 提取mAP指标
            map_match = re.search(r'(?<!50-)mAP[=:\s]+([\d.]+)', line)  # 排除mAP50-95
            map_val = float(map_match.group(1)) if map_match else None

            map50_match = re.search(r'mAP.*?50[^-][=:\s]+([\d.]+)', line)
            map50 = float(map50_match.group(1)) if map50_match else None

            map5095_match = re.search(r'mAP.*?50-95[=:\s]+([\d.]+)', line)
            map50_95 = float(map5095_match.group(1)) if map5095_match else None

            # 如果找到了epoch和任何指标，添加到结果中
            if current_epoch and (loss is not None or map_val is not None or map50 is not None):
                # 检查是否已存在该epoch的记录
                existing = next((m for m in metrics if m['epoch'] == current_epoch), None)

                if existing:
                    # 更新现有记录
                    if loss is not None:
                        existing['loss'] = loss
                    if map_val is not None:
                        existing['mAP'] = map_val
                    if map50 is not None:
                        existing['mAP50'] = map50
                    if map50_95 is not None:
                        existing['mAP50-95'] = map50_95
                else:
                    # 创建新记录
                    metrics.append({
                        'epoch': current_epoch,
                        'loss': loss,
                        'mAP': map_val,
                        'mAP50': map50,
                        'mAP50-95': map50_95,
                        'timestamp': datetime.now().isoformat()
                    })
    except Exception as e:
        print(f"Error parsing RF-DETR logs: {e}")

    return metrics


def get_training_metrics(log_path: str, framework: str = 'yolo') -> List[Dict]:
    """
    根据训练框架选择对应的日志解析器

    Args:
        log_path: 日志文件路径
        framework: 训练框架 (yolo/rfdetr)

    Returns:
        指标列表
    """
    if framework == 'rfdetr':
        return parse_rfdetr_logs(log_path)
    else:
        return parse_yolo_logs(log_path)


def get_latest_metrics(log_path: str, framework: str = 'yolo', last_n: int = 50) -> List[Dict]:
    """
    获取最近N个epoch的指标（用于图表显示）

    Args:
        log_path: 日志文件路径
        framework: 训练框架
        last_n: 返回最近N个epoch的数据

    Returns:
        最近的指标列表
    """
    all_metrics = get_training_metrics(log_path, framework)
    return all_metrics[-last_n:] if len(all_metrics) > last_n else all_metrics
