"""
本地文件路径工具模块
====================
OSS 功能已移除，所有文件存储在本地挂载目录中。

Docker 挂载示例：
    volumes:
      - /mnt/data/datasets:/app/datasets
      - /mnt/data/models:/app/models
"""

import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def is_oss_enabled() -> bool:
    """OSS 已禁用，始终返回 False"""
    return False


def normalize_key(rel_path: str) -> str:
    """将本地相对路径规范化为正斜杠格式"""
    return Path(rel_path).as_posix().lstrip("./").lstrip("/")


def is_url(path: str) -> bool:
    """判断路径是否为 http/https URL"""
    return path.startswith("http://") or path.startswith("https://")


def get_basename(path: str) -> str:
    """
    从本地路径中提取文件名。
    如果传入的是 URL 也能正常处理。
    """
    if is_url(path):
        from urllib.parse import urlparse
        return os.path.basename(urlparse(path).path)
    return os.path.basename(path)


def resolve_local_path(path: str) -> Optional[str]:
    """
    将 img.path（本地相对或绝对路径）解析为本地绝对路径。

    Returns:
        本地绝对路径字符串，文件不存在则返回 None
    """
    cwd = os.getcwd()

    if is_url(path):
        logger.warning(f"不支持从 URL 解析本地路径（OSS 已禁用）: {path}")
        return None

    local_path = path if os.path.isabs(path) else os.path.join(cwd, path)
    if os.path.isfile(local_path):
        return local_path

    logger.warning(f"本地文件不存在: {local_path}")
    return None
