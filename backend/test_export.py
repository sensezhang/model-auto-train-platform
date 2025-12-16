"""
测试数据集导出功能的脚本

使用方法:
    python test_export.py <project_id>

示例:
    python test_export.py 2
"""

import sys
import os
import json

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.export_coco import export_dataset_to_coco


def test_export(project_id: int):
    """测试导出功能"""
    print(f"Testing export for project {project_id}...")

    # 创建输出目录
    output_dir = os.path.join(os.path.dirname(__file__), "..", "test_export_output")
    os.makedirs(output_dir, exist_ok=True)

    try:
        # 执行导出
        result = export_dataset_to_coco(
            project_id=project_id,
            output_dir=output_dir,
            seed=42
        )

        # 打印结果
        print("\n" + "=" * 50)
        print("Export completed successfully!")
        print("=" * 50)
        print(f"Project: {result['project_name']} (ID: {result['project_id']})")
        print(f"Output directory: {result['output_dir']}")
        print(f"Random seed: {result['seed']}")
        print(f"\nTotal images: {result['total_images']}")
        print(f"Total annotations: {result['total_annotations']}")
        print(f"Total classes: {result['total_classes']}")
        print("\nDataset splits:")
        for split_name, stats in result['splits'].items():
            print(f"  {split_name}:")
            print(f"    - Images: {stats['images']}")
            print(f"    - Annotations: {stats['annotations']}")

        # 验证文件是否生成
        print("\n" + "=" * 50)
        print("Verifying generated files...")
        print("=" * 50)
        for split in ['train', 'val', 'test']:
            split_dir = os.path.join(output_dir, split)
            ann_file = os.path.join(split_dir, '_annotations.json')

            if os.path.exists(ann_file):
                with open(ann_file, 'r', encoding='utf-8') as f:
                    coco_data = json.load(f)
                print(f"\n{split.upper()} split:")
                print(f"  - Annotation file: ✓")
                print(f"  - COCO images: {len(coco_data.get('images', []))}")
                print(f"  - COCO annotations: {len(coco_data.get('annotations', []))}")
                print(f"  - COCO categories: {len(coco_data.get('categories', []))}")

                # 统计实际图像文件
                image_files = [f for f in os.listdir(split_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                print(f"  - Image files: {len(image_files)}")
            else:
                print(f"\n{split.upper()} split: ✗ (annotation file not found)")

        print("\n" + "=" * 50)
        print(f"All files saved to: {output_dir}")
        print("=" * 50)

    except ValueError as e:
        print(f"\nError: {e}")
        return 1
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_export.py <project_id>")
        print("Example: python test_export.py 2")
        sys.exit(1)

    try:
        project_id = int(sys.argv[1])
    except ValueError:
        print("Error: project_id must be an integer")
        sys.exit(1)

    sys.exit(test_export(project_id))
