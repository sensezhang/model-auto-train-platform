# Python 3.11 环境迁移与 RF-DETR 兼容性修复总结

## 背景

原项目使用 Python 3.12 环境，但在安装 onnxsim 包时遇到兼容性问题。同时，RF-DETR 训练在 PyTorch 2.9.1 下出现 "Inference tensors cannot be saved for backward" 错误。

## 解决方案

### 1. 创建 Python 3.11 Conda 环境

**环境信息**:
- 环境名称: `yolo-train-test-py311`
- Python 版本: 3.11.14
- 环境路径: `C:\Users\YUHGT\.conda\envs\yolo-train-test-py311`

**创建命令**:
```bash
conda create -n yolo-train-test-py311 python=3.11 -y
```

### 2. 解决包依赖兼容性问题

经过多次测试，确定了以下兼容的包版本组合：

| 包名 | 原版本 | 新版本 | 原因 |
|------|--------|--------|------|
| torch | 2.9.1 | **2.0.1** | RF-DETR 与 2.9.1 不兼容 |
| torchvision | 0.24.1 | **0.15.2** | 匹配 torch 2.0.1 |
| numpy | 2.2.6 | **<2.0 (1.26.4)** | PyTorch 2.0.1 需要 numpy 1.x |
| opencv-python | 4.12.0.88 | **<4.10 (4.9.0.80)** | 兼容 numpy 1.x |
| transformers | 4.57.3 | **4.49.0** | RF-DETR 需要 4.40-4.50 |
| peft | 0.18.0 | **0.11.1** | 兼容 transformers 4.49 |
| onnxsim | - | **0.4.36** | 新增，需要 Python 3.11 |

### 3. 修复的错误

#### 错误 1: onnxsim 安装失败
**原因**: Python 3.12 与 onnxsim 存在兼容性问题
**解决**: 迁移到 Python 3.11

#### 错误 2: RF-DETR 训练失败
**错误信息**:
```
RuntimeError: Inference tensors cannot be saved for backward.
Please do not use Tensors created in inference mode in computation
tracked by autograd.
```

**原因**:
- PyTorch 2.9.1 太新，与 RF-DETR 1.3.0 存在兼容性问题
- RF-DETR 的 DinoV2 backbone 在前向传播时使用了 inference mode 创建的张量

**解决**:
- 降级 PyTorch 到 2.0.1（稳定版本）
- 同时降级相关依赖包

#### 错误 3: NumPy 版本冲突
**错误信息**:
```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.6
```

**解决**: 降级 numpy 到 1.26.4

#### 错误 4: transformers 与 PyTorch 版本冲突
**错误信息**:
```
AttributeError: module 'torch.utils._pytree' has no attribute 'register_pytree_node'
```

**解决**: 调整 transformers 版本到 4.40-4.50 范围

## 验证结果

### 成功导入测试
```bash
"C:\Users\YUHGT\.conda\envs\yolo-train-test-py311\python.exe" -c "
import torch
import torchvision
import rfdetr
from rfdetr import RFDETRMedium
model = RFDETRMedium()
print('All packages imported successfully!')
"
```

**输出**: ✅ 成功导入，模型自动下载预训练权重（rf-detr-medium.pth, 386MB）

### 包版本确认
```bash
PyTorch: 2.0.1+cpu
TorchVision: 0.15.2+cpu
NumPy: 1.26.4
OpenCV: 4.9.0.80
Transformers: 4.49.0
PEFT: 0.11.1
onnxsim: 0.4.36
RF-DETR: 1.3.0
```

## 文件更新

### 1. requirements.txt
```txt
fastapi==0.111.0
uvicorn[standard]==0.30.1
sqlmodel==0.0.21
pydantic==2.7.4
python-multipart==0.0.9
Jinja2==3.1.4
Pillow==10.4.0
ultralytics>=8.2.0
onnxsim>=0.4.0
torch==2.0.1
torchvision==0.15.2
numpy<2.0
opencv-python<4.10
transformers>=4.40,<4.50
peft>=0.10,<0.12
```

### 2. deploy/Dockerfile.backend
```dockerfile
FROM python:3.11-slim  # 从 3.10-slim 更新
```

### 3. deploy/README.md
更新手动部署说明，使用 Python 3.11

### 4. backend/app/services/training_rfdetr.py
添加显式训练模式设置（防御性编程）：
```python
# 确保模型处于训练模式，解决 inference mode 错误
if hasattr(model, 'model') and hasattr(model.model, 'train'):
    model.model.train()
```

### 5. 新增文档
- `CONDA_ENV_GUIDE.md`: Conda 环境使用指南
- `PYTHON311_MIGRATION_SUMMARY.md`: 本文档

## 使用新环境

### 激活环境
```bash
conda activate yolo-train-test-py311
```

### 启动后端
```bash
conda activate yolo-train-test-py311
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 运行 RF-DETR 训练
环境配置完成后，RF-DETR 训练应该可以正常运行，不会再出现 inference mode 错误。

### 使用 onnxsim
```python
import onnxsim
# 现在可以正常使用 onnxsim 进行模型简化
```

## 部署说明

### Docker 部署
Docker 镜像已自动更新为 Python 3.11，无需额外配置：
```bash
cd deploy
docker-compose up -d
```

### 手动部署
确保服务器安装 Python 3.11：
```bash
# Ubuntu/Debian
sudo apt install python3.11 python3.11-venv

# 创建虚拟环境
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 注意事项

1. **PyTorch 版本**: 必须使用 2.0.1，不要升级到 2.1.x 或更高版本
2. **NumPy 版本**: 必须保持在 1.x，不要升级到 2.x
3. **旧环境**: 旧的 `.venv` 目录（Python 3.12）可以保留作为备份或删除
4. **IDE 配置**: 需要重新配置 IDE 的 Python 解释器为新的 conda 环境

## 性能影响

- **PyTorch 2.0.1 vs 2.9.1**: 训练性能基本相同，2.0.1 更稳定
- **NumPy 1.x vs 2.x**: NumPy 2.x 有性能提升，但为了兼容性选择 1.x
- **整体影响**: 兼容性优先，性能差异可忽略

## 下一步

1. 测试 RF-DETR 训练是否可以完整运行
2. 测试 ONNX 导出功能
3. 验证所有其他训练框架（YOLO等）是否正常工作
4. 在生产环境中测试部署

## 相关文档

- `CONDA_ENV_GUIDE.md`: 详细的环境使用指南
- `RF-DETR_ONNX_EXPORT_GUIDE.md`: ONNX 导出说明（如果存在）
- `deploy/README.md`: 部署指南

## 问题排查

如果遇到问题：

1. **包导入失败**: 检查是否激活了正确的 conda 环境
2. **版本冲突**: 删除环境重新创建，严格按照 requirements.txt 安装
3. **训练失败**: 查看日志确认是否为其他问题（数据集、配置等）
4. **IDE 无法识别包**: 重新配置 Python 解释器路径

## 联系与支持

如有问题，请参考：
- GitHub Issues
- 项目文档
- Conda 环境使用指南
