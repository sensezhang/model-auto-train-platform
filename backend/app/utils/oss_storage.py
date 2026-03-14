"""
存储工具模块
============
根据环境变量自动选择 OSS（S3 兼容）或本地文件存储。

OSS 所需环境变量（全部设置后自动启用）：
    OSS_ENDPOINT     = "http://10.230.0.5:8009"
    OSS_ACCESS_KEY   = "your-access-key"
    OSS_SECRET_KEY   = "your-secret-key"
    OSS_BUCKET       = "yolo-datasets"
    OSS_PUBLIC_HOST  = "https://oss.ikingtec.com"  # 可选，生成访问 URL 用，默认同 ENDPOINT
    OSS_REGION       = "us-east-1"                 # 可选，默认 us-east-1
"""

import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── OSS 配置（从环境变量读取）────────────────────────────────
_OSS_ENDPOINT    = os.getenv("OSS_ENDPOINT", "")
_OSS_PUBLIC_HOST = os.getenv("OSS_PUBLIC_HOST", _OSS_ENDPOINT)
_OSS_ACCESS_KEY  = os.getenv("OSS_ACCESS_KEY", "")
_OSS_SECRET_KEY  = os.getenv("OSS_SECRET_KEY", "")
_OSS_BUCKET      = os.getenv("OSS_BUCKET", "")
_OSS_REGION      = os.getenv("OSS_REGION", "us-east-1")


def is_oss_enabled() -> bool:
    """当所有必要的 OSS 环境变量均已设置时返回 True"""
    return bool(_OSS_ENDPOINT and _OSS_ACCESS_KEY and _OSS_SECRET_KEY and _OSS_BUCKET)


def _get_s3_client():
    """创建 boto3 S3 客户端（禁用代理）"""
    import boto3
    from botocore.config import Config
    return boto3.client(
        "s3",
        endpoint_url=_OSS_ENDPOINT,
        aws_access_key_id=_OSS_ACCESS_KEY,
        aws_secret_access_key=_OSS_SECRET_KEY,
        region_name=_OSS_REGION,
        config=Config(proxies={}, connect_timeout=30, read_timeout=30),
    )


def upload_to_oss(local_path: str, key: str) -> Optional[str]:
    """
    上传本地文件到 OSS。

    Args:
        local_path: 本地绝对路径
        key:        OSS 对象键（正斜杠相对路径）

    Returns:
        成功返回公网访问 URL，失败返回 None
    """
    if not is_oss_enabled():
        return None
    try:
        client = _get_s3_client()
        client.upload_file(local_path, _OSS_BUCKET, key)
        public_host = _OSS_PUBLIC_HOST.rstrip("/")
        url = f"{public_host}/{_OSS_BUCKET}/{key}"
        logger.info(f"OSS 上传成功: {url}")
        return url
    except Exception as e:
        logger.error(f"OSS 上传失败 ({key}): {e}")
        return None


def normalize_key(rel_path: str) -> str:
    """将本地相对路径规范化为正斜杠格式（OSS 对象键）"""
    return Path(rel_path).as_posix().lstrip("./").lstrip("/")


def is_url(path: str) -> bool:
    """判断路径是否为 http/https URL"""
    return path.startswith("http://") or path.startswith("https://")


def get_basename(path: str) -> str:
    """从路径或 URL 中提取文件名"""
    if is_url(path):
        from urllib.parse import urlparse
        return os.path.basename(urlparse(path).path)
    return os.path.basename(path)


def resolve_local_path(path: str) -> Optional[str]:
    """
    将 img.path（本地相对/绝对路径）解析为本地绝对路径。
    OSS URL 无法解析为本地路径，返回 None。

    Returns:
        本地绝对路径字符串，文件不存在则返回 None
    """
    cwd = os.getcwd()

    if is_url(path):
        logger.warning(f"路径为 URL，无法解析为本地路径: {path}")
        return None

    local_path = path if os.path.isabs(path) else os.path.join(cwd, path)
    if os.path.isfile(local_path):
        return local_path

    logger.warning(f"本地文件不存在: {local_path}")
    return None
