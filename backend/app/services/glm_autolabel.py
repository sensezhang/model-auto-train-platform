"""
GLM-4.5V 自动标注服务占位：
- 实际实现需集成GLM官方API，限制并发=2，阈值过滤=0.5
- 输出标准化候选框并写入 ProposedAnnotation
"""

from typing import List, Dict


def run_autolabel_for_images(project_id: int, image_ids: List[int], threshold: float = 0.5) -> Dict:
    # TODO: Implement GLM-4.5V API calls and persistence
    return {"projectId": project_id, "processed": len(image_ids), "threshold": threshold}

