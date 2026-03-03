"""
模型导出服务 - 支持YOLO和RF-DETR导出为ONNX格式
"""
from __future__ import annotations

import os
import math
import warnings
from typing import Optional
from datetime import datetime

from ..db import get_session
from ..models import TrainingJob, ModelArtifact
from ..utils.oss_storage import resolve_local_path


def register_onnx_symbolic_for_sdpa():
    """
    注册 scaled_dot_product_attention 的 ONNX 符号函数
    这样在导出ONNX时，PyTorch会使用我们的实现
    """
    import torch
    from torch.onnx import register_custom_op_symbolic

    def scaled_dot_product_attention_symbolic(g, query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
        """
        ONNX symbolic function for scaled_dot_product_attention

        将 scaled_dot_product_attention 转换为 ONNX 支持的基础操作
        """
        import torch
        from torch.onnx import symbolic_helper

        # Q @ K^T
        key_transposed = g.op("Transpose", key, perm_i=[0, 1, 3, 2])
        attn_weight = g.op("MatMul", query, key_transposed)

        # 动态计算scale factor: 1/sqrt(head_dim)
        # query shape: [batch, num_heads, seq_len, head_dim]
        if scale is None:
            # 获取query的形状
            query_shape = g.op("Shape", query)

            # 提取head_dim (最后一个维度, index=3)
            head_dim_index = g.op("Constant", value_t=torch.tensor([3], dtype=torch.int64))
            head_dim = g.op("Gather", query_shape, head_dim_index, axis_i=0)

            # 转换为float
            head_dim_float = g.op("Cast", head_dim, to_i=1)  # 1 = FLOAT

            # 计算 sqrt(head_dim)
            sqrt_head_dim = g.op("Sqrt", head_dim_float)

            # 计算 1/sqrt(head_dim)
            one = g.op("Constant", value_t=torch.tensor([1.0], dtype=torch.float32))
            scale_factor = g.op("Div", one, sqrt_head_dim)
        else:
            # 使用提供的scale
            scale_factor = g.op("Constant", value_t=torch.tensor([scale], dtype=torch.float32))

        # Scale: attn_weight * scale_factor
        attn_weight = g.op("Mul", attn_weight, scale_factor)

        # Add attention mask if provided
        # 注意：RF-DETR在推理时通常不使用mask，所以我们跳过这一步
        # 这避免了处理None/空值的复杂性
        # if attn_mask is not None:
        #     attn_weight = g.op("Add", attn_weight, attn_mask)

        # Softmax
        attn_weight = g.op("Softmax", attn_weight, axis_i=-1)

        # Dropout (在推理时通常被忽略)
        # 在ONNX导出时，我们跳过dropout因为推理时不需要

        # attn_weight @ V
        output = g.op("MatMul", attn_weight, value)

        return output

    # 注册符号函数
    try:
        register_custom_op_symbolic(
            'aten::scaled_dot_product_attention',
            scaled_dot_product_attention_symbolic,
            opset_version=17
        )
        print("[OK] Registered ONNX symbolic for scaled_dot_product_attention")
        return True
    except Exception as e:
        print(f"[WARN] Failed to register ONNX symbolic: {e}")
        return False




def export_yolo_to_onnx(job_id: int, artifact_id: int) -> Optional[str]:
    """
    导出YOLO模型为ONNX格式

    Args:
        job_id: 训练任务ID
        artifact_id: PT模型文件的artifact ID

    Returns:
        导出的ONNX文件路径，失败返回None
    """
    with get_session() as session:
        job = session.get(TrainingJob, job_id)
        artifact = session.get(ModelArtifact, artifact_id)

        if not job or not artifact:
            return None

        if artifact.format != 'pt':
            return None

        pt_path = resolve_local_path(artifact.path)
        if not pt_path:
            return None

    try:
        from ultralytics import YOLO

        # 加载模型
        model = YOLO(pt_path)

        # 导出ONNX (opset=12, dynamic=True)
        onnx_path = model.export(format='onnx', opset=12, dynamic=True)

        if isinstance(onnx_path, str) and os.path.exists(onnx_path):
            # 保存到artifacts
            rel_path = os.path.relpath(onnx_path, os.getcwd())
            size = os.path.getsize(onnx_path)

            with get_session() as session:
                # 检查是否已存在ONNX artifact
                existing = (
                    session.query(ModelArtifact)
                    .filter(ModelArtifact.trainingJobId == job_id)
                    .filter(ModelArtifact.format == 'onnx')
                    .first()
                )

                if existing:
                    # 更新现有记录
                    existing.path = rel_path
                    existing.size = size
                    session.add(existing)
                else:
                    # 创建新记录
                    session.add(ModelArtifact(
                        trainingJobId=job_id,
                        format='onnx',
                        path=rel_path,
                        size=size
                    ))
                session.commit()

            return onnx_path
    except Exception as e:
        print(f"YOLO ONNX export failed: {e}")
        return None

    return None


def export_rfdetr_to_onnx(job_id: int, artifact_id: int, simplify: bool = False) -> Optional[str]:
    """
    导出RF-DETR模型为ONNX格式

    Args:
        job_id: 训练任务ID
        artifact_id: PTH模型文件的artifact ID
        simplify: 是否简化ONNX模型（需要onnx-simplifier）

    Returns:
        导出的ONNX文件路径，失败返回None
    """
    # 抑制已知的警告信息
    warnings.filterwarnings('ignore', category=UserWarning, module='pkg_resources')
    warnings.filterwarnings('ignore', category=UserWarning, message='.*torchvision.*')

    # 设置环境变量以避免ONNX运行时类型检查警告
    os.environ['TORCH_ONNX_EXPERIMENTAL_RUNTIME_TYPE_CHECK'] = '0'
    os.environ["TORCH_NNMHA_ENABLE_MATH"] = "1"

    with get_session() as session:
        job = session.get(TrainingJob, job_id)
        artifact = session.get(ModelArtifact, artifact_id)

        if not job or not artifact:
            return None

        if artifact.format != 'pth':
            return None

        pth_path = resolve_local_path(artifact.path)
        if not pth_path:
            return None

        # 获取模型变体
        model_variant = job.modelVariant.lower()

    try:
        from rfdetr import RFDETRSmall, RFDETRMedium, RFDETRLarge

        # 选择对应的模型
        if 'small' in model_variant:
            model = RFDETRSmall(pretrain_weights=pth_path)
        elif 'large' in model_variant:
            model = RFDETRLarge(pretrain_weights=pth_path)
        else:
            model = RFDETRMedium(pretrain_weights=pth_path)

        # 注册ONNX符号函数以支持scaled_dot_product_attention
        print(f"Registering ONNX symbolic for scaled_dot_product_attention...")
        register_onnx_symbolic_for_sdpa()

        # 将ONNX导出到模型权重所在目录，避免不同模型的ONNX文件互相覆盖
        model_dir = os.path.dirname(pth_path)

        # 导出ONNX
        # simplify=False 避免需要onnxsim包（需要cmake编译）
        print(f"Exporting RF-DETR model to ONNX in {model_dir} (simplify={simplify})...")
        result = model.export(output_dir=model_dir, simplify=simplify)

        # 查找生成的ONNX文件
        # RF-DETR会生成 inference_model.onnx 文件
        onnx_candidates = [
            os.path.join(model_dir, f)
            for f in os.listdir(model_dir)
            if f.endswith('.onnx')
        ]

        if onnx_candidates:
            # 使用最新生成的ONNX文件
            onnx_path = max(onnx_candidates, key=os.path.getmtime)

            # 保存到artifacts
            rel_path = os.path.relpath(onnx_path, os.getcwd())
            size = os.path.getsize(onnx_path)

            with get_session() as session:
                # 检查是否已存在ONNX artifact
                existing = (
                    session.query(ModelArtifact)
                    .filter(ModelArtifact.trainingJobId == job_id)
                    .filter(ModelArtifact.format == 'onnx')
                    .first()
                )

                if existing:
                    # 更新现有记录
                    existing.path = rel_path
                    existing.size = size
                    session.add(existing)
                else:
                    # 创建新记录
                    session.add(ModelArtifact(
                        trainingJobId=job_id,
                        format='onnx',
                        path=rel_path,
                        size=size
                    ))
                session.commit()

            return onnx_path
    except Exception as e:
        print(f"RF-DETR ONNX export failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    return None


def export_model_to_onnx(job_id: int, artifact_id: int, simplify: bool = False) -> dict:
    """
    根据训练框架自动选择导出方法

    Args:
        job_id: 训练任务ID
        artifact_id: 模型文件的artifact ID
        simplify: 是否简化ONNX模型（仅RF-DETR）

    Returns:
        导出结果字典
    """
    with get_session() as session:
        job = session.get(TrainingJob, job_id)
        if not job:
            return {"success": False, "error": "Training job not found"}

        framework = job.framework

    try:
        if framework == 'rfdetr':
            onnx_path = export_rfdetr_to_onnx(job_id, artifact_id, simplify)
        else:
            # 默认使用YOLO导出
            onnx_path = export_yolo_to_onnx(job_id, artifact_id)

        if onnx_path:
            return {
                "success": True,
                "onnx_path": onnx_path,
                "message": "Model exported to ONNX successfully"
            }
        else:
            return {
                "success": False,
                "error": "Export failed - check logs for details"
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
