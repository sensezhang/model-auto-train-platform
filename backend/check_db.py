import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db import get_session
from app.models import Project, Image, Class, Annotation

with get_session() as session:
    # 检查项目
    proj = session.get(Project, 2)
    if proj:
        print(f"Project found: {proj.name}")
    else:
        print("Project 2 not found!")
        sys.exit(1)

    # 统计数据
    images = session.query(Image).filter(Image.projectId == 2).all()
    classes = session.query(Class).filter(Class.projectId == 2).all()

    print(f"Images: {len(images)}")
    print(f"Classes: {len(classes)}")

    if images:
        image_ids = [img.id for img in images]
        annotations = session.query(Annotation).filter(Annotation.imageId.in_(image_ids)).all()
        print(f"Annotations: {len(annotations)}")

        # 打印前几个图像路径
        print("\nSample images:")
        for img in images[:3]:
            print(f"  - {img.path}")
    else:
        print("No images found!")
