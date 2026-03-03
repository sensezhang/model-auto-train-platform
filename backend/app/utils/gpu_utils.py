"""
GPU检测和管理工具
"""
from typing import List, Dict, Optional
import os
import sys


def get_available_gpus() -> List[Dict]:
    """
    获取所有可用的GPU信息

    Returns:
        GPU信息列表，每个GPU包含：id, name, memory_total, memory_free, utilization
    """
    try:
        import torch

        # 记录 CUDA 检测信息用于调试
        cuda_available = torch.cuda.is_available()
        print(f"[GPU Detection] PyTorch version: {torch.__version__}", file=sys.stderr)
        print(f"[GPU Detection] CUDA available: {cuda_available}", file=sys.stderr)

        if not cuda_available:
            # 提供更多诊断信息
            print(f"[GPU Detection] CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}", file=sys.stderr)
            print(f"[GPU Detection] torch.version.cuda: {torch.version.cuda}", file=sys.stderr)

            # 检查是否是因为 CUDA 编译问题
            if hasattr(torch, 'backends') and hasattr(torch.backends, 'cudnn'):
                print(f"[GPU Detection] cuDNN available: {torch.backends.cudnn.is_available()}", file=sys.stderr)

            return []

        gpus = []
        device_count = torch.cuda.device_count()
        print(f"[GPU Detection] Device count: {device_count}", file=sys.stderr)

        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            # 获取显存信息（单位：MB）
            total_memory = props.total_memory / 1024 / 1024

            # 尝试获取当前可用显存
            try:
                torch.cuda.set_device(i)
                free_memory = torch.cuda.mem_get_info()[0] / 1024 / 1024
                allocated_memory = torch.cuda.memory_allocated(i) / 1024 / 1024
            except Exception as mem_err:
                print(f"[GPU Detection] Error getting memory info for GPU {i}: {mem_err}", file=sys.stderr)
                free_memory = total_memory  # 假设全部可用
                allocated_memory = 0

            gpu_info = {
                'id': i,
                'name': props.name,
                'memory_total': int(total_memory),
                'memory_free': int(free_memory),
                'memory_used': int(allocated_memory),
                'compute_capability': f"{props.major}.{props.minor}",
            }
            gpus.append(gpu_info)
            print(f"[GPU Detection] Found GPU {i}: {props.name}", file=sys.stderr)

        return gpus
    except ImportError as e:
        print(f"[GPU Detection] PyTorch import error: {e}", file=sys.stderr)
        return []
    except Exception as e:
        print(f"[GPU Detection] Error getting GPU info: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return []


def check_gpu_availability(gpu_ids: List[int]) -> bool:
    """
    检查指定的GPU是否可用

    Args:
        gpu_ids: GPU ID列表

    Returns:
        是否所有指定的GPU都可用
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return False

        device_count = torch.cuda.device_count()
        return all(0 <= gpu_id < device_count for gpu_id in gpu_ids)
    except Exception:
        return False


def format_gpu_env(gpu_ids: List[int]) -> str:
    """
    格式化GPU环境变量

    Args:
        gpu_ids: GPU ID列表

    Returns:
        CUDA_VISIBLE_DEVICES环境变量值
    """
    return ','.join(str(i) for i in gpu_ids)


def get_optimal_batch_size(gpu_ids: List[int], base_batch_size: int = 16) -> int:
    """
    根据GPU数量计算最优batch size

    Args:
        gpu_ids: GPU ID列表
        base_batch_size: 单卡基础batch size

    Returns:
        推荐的batch size
    """
    num_gpus = len(gpu_ids) if gpu_ids else 1
    return base_batch_size * num_gpus


def setup_distributed_env(gpu_ids: List[int], rank: int = 0) -> Dict[str, str]:
    """
    设置分布式训练环境变量

    Args:
        gpu_ids: GPU ID列表
        rank: 当前进程的rank

    Returns:
        环境变量字典
    """
    env = os.environ.copy()

    # 设置可见GPU
    if gpu_ids:
        env['CUDA_VISIBLE_DEVICES'] = format_gpu_env(gpu_ids)

    # 设置分布式训练相关环境变量
    world_size = len(gpu_ids) if gpu_ids else 1

    env['WORLD_SIZE'] = str(world_size)
    env['RANK'] = str(rank)
    env['LOCAL_RANK'] = str(rank)

    # 设置后端
    env['NCCL_P2P_DISABLE'] = '0'  # 启用P2P通信
    env['NCCL_IB_DISABLE'] = '1'   # 禁用InfiniBand（一般服务器不需要）

    return env


def is_multi_gpu(gpu_ids: Optional[List[int]]) -> bool:
    """
    判断是否为多GPU训练

    Args:
        gpu_ids: GPU ID列表

    Returns:
        是否为多GPU
    """
    return gpu_ids is not None and len(gpu_ids) > 1
