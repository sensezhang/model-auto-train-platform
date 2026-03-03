"""系统信息API路由"""
import os
from fastapi import APIRouter

from ..utils.gpu_utils import get_available_gpus

router = APIRouter(tags=["system"])


@router.get("/system/gpus")
def list_gpus():
    """
    获取所有可用GPU信息

    Returns:
        GPU列表，每个GPU包含id, name, memory_total, memory_free等信息
    """
    return get_available_gpus()


@router.get("/system/info")
def get_system_info():
    """
    获取系统信息

    Returns:
        系统配置信息
    """
    import torch
    import platform

    gpus = get_available_gpus()

    return {
        "platform": platform.system(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_count": len(gpus),
        "gpus": gpus,
    }


@router.get("/system/gpu-diagnostics")
def get_gpu_diagnostics():
    """
    获取 GPU 诊断信息，用于排查 GPU 不可见问题

    Returns:
        详细的 GPU 和 CUDA 诊断信息
    """
    import platform

    diagnostics = {
        "platform": platform.system(),
        "python_version": platform.python_version(),
        "environment": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "not set"),
            "NVIDIA_VISIBLE_DEVICES": os.environ.get("NVIDIA_VISIBLE_DEVICES", "not set"),
            "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", "not set"),
        },
        "pytorch": {},
        "cuda": {},
        "gpus": [],
        "errors": []
    }

    # PyTorch 信息
    try:
        import torch
        diagnostics["pytorch"] = {
            "version": torch.__version__,
            "cuda_compiled": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        }

        # CUDA 信息
        diagnostics["cuda"] = {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "current_device": torch.cuda.current_device() if torch.cuda.is_available() and torch.cuda.device_count() > 0 else None,
        }

        # 尝试获取每个 GPU 的信息
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                try:
                    props = torch.cuda.get_device_properties(i)
                    gpu_info = {
                        "id": i,
                        "name": props.name,
                        "total_memory_mb": int(props.total_memory / 1024 / 1024),
                        "compute_capability": f"{props.major}.{props.minor}",
                        "multi_processor_count": props.multi_processor_count,
                    }

                    # 尝试获取显存信息
                    try:
                        torch.cuda.set_device(i)
                        free, total = torch.cuda.mem_get_info()
                        gpu_info["free_memory_mb"] = int(free / 1024 / 1024)
                    except Exception as e:
                        gpu_info["memory_error"] = str(e)

                    diagnostics["gpus"].append(gpu_info)
                except Exception as e:
                    diagnostics["errors"].append(f"Error getting info for GPU {i}: {str(e)}")
        else:
            # CUDA 不可用，尝试找出原因
            if torch.version.cuda is None:
                diagnostics["errors"].append("PyTorch was not compiled with CUDA support")
            else:
                diagnostics["errors"].append(
                    f"PyTorch compiled with CUDA {torch.version.cuda} but CUDA is not available. "
                    "Check if NVIDIA driver is installed and if running in Docker, ensure --gpus flag is used."
                )

    except ImportError as e:
        diagnostics["errors"].append(f"Failed to import PyTorch: {str(e)}")
    except Exception as e:
        diagnostics["errors"].append(f"Unexpected error: {str(e)}")

    # 检查 nvidia-smi
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            diagnostics["nvidia_smi"] = {
                "available": True,
                "output": result.stdout.strip().split("\n")
            }
        else:
            diagnostics["nvidia_smi"] = {
                "available": False,
                "error": result.stderr.strip()
            }
    except FileNotFoundError:
        diagnostics["nvidia_smi"] = {
            "available": False,
            "error": "nvidia-smi not found in PATH"
        }
    except subprocess.TimeoutExpired:
        diagnostics["nvidia_smi"] = {
            "available": False,
            "error": "nvidia-smi timed out"
        }
    except Exception as e:
        diagnostics["nvidia_smi"] = {
            "available": False,
            "error": str(e)
        }

    return diagnostics
