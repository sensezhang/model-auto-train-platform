"""
训练指标解析服务 - 从日志文件中提取训练指标
"""
import os
import re
from typing import List, Dict, Optional
from datetime import datetime


def parse_yolo_logs(log_path: str) -> List[Dict]:
    """
    解析YOLO训练日志，提取每个epoch的指标

    Returns:
        指标列表，每个元素包含: epoch, loss, mAP50, mAP50-95等
    """
    if not os.path.exists(log_path):
        return []

    metrics = []
    current_epoch = None

    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        for line in lines:
            # 匹配多种YOLO日志格式

            # 格式1: "Starting epoch 1"
            epoch_start = re.search(r'[Ss]tarting epoch\s+(\d+)', line)
            if epoch_start:
                current_epoch = int(epoch_start.group(1))

            # 格式2: "Epoch 1/100" 或 "epoch: 1"
            epoch_match = re.search(r'[Ee]poch[:\s]+(\d+)', line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))

            # 格式3: "Completed epoch 1"
            completed_match = re.search(r'[Cc]ompleted epoch\s+(\d+)', line)
            if completed_match:
                current_epoch = int(completed_match.group(1))

            # 提取loss（多种格式）
            # "loss: 2.345" 或 "loss=2.345" 或 "train_loss: 2.345"
            loss_patterns = [
                r'(?:train_)?loss[=:\s]+([\d.]+)',
                r'box_loss[=:\s]+([\d.]+)',  # YOLO的box loss
                r'total.*?loss[=:\s]+([\d.]+)',
            ]

            loss = None
            for pattern in loss_patterns:
                loss_match = re.search(pattern, line, re.IGNORECASE)
                if loss_match:
                    loss = float(loss_match.group(1))
                    break

            # 提取mAP指标
            map50_match = re.search(r'mAP.*?50[^-][=:\s]+([\d.]+)', line, re.IGNORECASE)
            map50 = float(map50_match.group(1)) if map50_match else None

            map5095_match = re.search(r'mAP.*?50-95[=:\s]+([\d.]+)', line, re.IGNORECASE)
            map50_95 = float(map5095_match.group(1)) if map5095_match else None

            # 如果找到了epoch和任何指标，添加到结果中
            if current_epoch and (loss is not None or map50 is not None):
                # 检查是否已存在该epoch的记录
                existing = next((m for m in metrics if m['epoch'] == current_epoch), None)

                if existing:
                    # 更新现有记录
                    if loss is not None:
                        existing['loss'] = loss
                    if map50 is not None:
                        existing['mAP50'] = map50
                    if map50_95 is not None:
                        existing['mAP50-95'] = map50_95
                else:
                    # 创建新记录
                    metrics.append({
                        'epoch': current_epoch,
                        'loss': loss,
                        'mAP50': map50,
                        'mAP50-95': map50_95,
                        'timestamp': datetime.now().isoformat()
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
