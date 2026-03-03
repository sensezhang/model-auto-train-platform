"""
测试 Python 3.11 环境和包兼容性
"""
import sys

# 设置 UTF-8 编码（Windows 兼容）
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

def test_imports():
    """测试所有关键包的导入"""
    print("=" * 60)
    print("测试包导入...")
    print("=" * 60)

    try:
        import torch
        print(f"[OK] PyTorch {torch.__version__}")
    except Exception as e:
        print(f"[FAIL] PyTorch 导入失败: {e}")
        return False

    try:
        import torchvision
        print(f"✓ TorchVision {torchvision.__version__}")
    except Exception as e:
        print(f"✗ TorchVision 导入失败: {e}")
        return False

    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
        if np.__version__.startswith('2.'):
            print("  ⚠ 警告: NumPy 版本为 2.x，建议降级到 1.x")
    except Exception as e:
        print(f"✗ NumPy 导入失败: {e}")
        return False

    try:
        import cv2
        print(f"✓ OpenCV {cv2.__version__}")
    except Exception as e:
        print(f"✗ OpenCV 导入失败: {e}")
        return False

    try:
        import onnxsim
        print(f"✓ onnxsim {onnxsim.__version__ if hasattr(onnxsim, '__version__') else 'unknown'}")
    except Exception as e:
        print(f"✗ onnxsim 导入失败: {e}")
        return False

    try:
        import transformers
        print(f"✓ Transformers {transformers.__version__}")
    except Exception as e:
        print(f"✗ Transformers 导入失败: {e}")
        return False

    try:
        import peft
        print(f"✓ PEFT {peft.__version__ if hasattr(peft, '__version__') else 'unknown'}")
    except Exception as e:
        print(f"✗ PEFT 导入失败: {e}")
        return False

    try:
        import ultralytics
        print(f"✓ Ultralytics {ultralytics.__version__ if hasattr(ultralytics, '__version__') else 'unknown'}")
    except Exception as e:
        print(f"✗ Ultralytics 导入失败: {e}")
        return False

    try:
        import rfdetr
        print(f"✓ RF-DETR {rfdetr.__version__ if hasattr(rfdetr, '__version__') else 'unknown'}")
    except Exception as e:
        print(f"✗ RF-DETR 导入失败: {e}")
        return False

    return True


def test_rfdetr_model():
    """测试 RF-DETR 模型创建"""
    print("\n" + "=" * 60)
    print("测试 RF-DETR 模型创建...")
    print("=" * 60)

    try:
        from rfdetr import RFDETRMedium
        print("✓ 导入 RFDETRMedium 成功")

        # 注意：这会下载预训练权重（约 386MB）
        # 如果不想下载，可以注释掉下面的代码
        print("  创建模型实例（可能会下载预训练权重...）")
        model = RFDETRMedium()
        print(f"✓ 模型创建成功: {model.__class__.__name__}")

        # 检查模型是否可以设置为训练模式
        if hasattr(model, 'model'):
            model.model.train()
            print("✓ 模型可以设置为训练模式")

        return True
    except Exception as e:
        print(f"✗ RF-DETR 模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_version_compatibility():
    """测试版本兼容性"""
    print("\n" + "=" * 60)
    print("检查版本兼容性...")
    print("=" * 60)

    import torch
    import numpy as np

    issues = []

    # 检查 PyTorch 版本
    if not torch.__version__.startswith('2.0'):
        issues.append(f"PyTorch 版本 {torch.__version__} 可能与 RF-DETR 不兼容，推荐 2.0.1")
    else:
        print(f"✓ PyTorch {torch.__version__} 版本正确")

    # 检查 NumPy 版本
    if np.__version__.startswith('2.'):
        issues.append(f"NumPy 版本 {np.__version__} 可能与 PyTorch 2.0.1 不兼容，推荐 <2.0")
    else:
        print(f"✓ NumPy {np.__version__} 版本正确")

    if issues:
        print("\n⚠ 发现以下兼容性问题:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✓ 所有版本兼容性检查通过")
        return True


def main():
    """主测试函数"""
    print("\n")
    print("=" * 60)
    print("Python 3.11 环境测试")
    print("=" * 60)

    import sys
    print(f"Python 版本: {sys.version}")
    print(f"Python 可执行文件: {sys.executable}")

    # 运行所有测试
    results = []

    results.append(("包导入测试", test_imports()))
    results.append(("版本兼容性测试", test_version_compatibility()))
    results.append(("RF-DETR 模型测试", test_rfdetr_model()))

    # 显示测试结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("✓ 所有测试通过！环境配置正确。")
    else:
        print("✗ 部分测试失败，请检查错误信息。")
    print("=" * 60 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
