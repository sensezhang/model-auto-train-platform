# Conda 环境使用指南

本项目已迁移到 Python 3.11 的 Conda 环境，以支持 onnxsim 等包的完整功能。

## 环境信息

- **环境名称**: `yolo-train-test-py311`
- **Python 版本**: 3.11
- **环境路径**: `C:\Users\YUHGT\.conda\envs\yolo-train-test-py311`

## 激活环境

### 方式1：使用 conda activate（推荐）

```bash
conda activate yolo-train-test-py311
```

**注意**: 如果提示 `CondaError: Run 'conda init' before 'conda activate'`，需要先运行：

```bash
conda init
# 然后重启终端
conda activate yolo-train-test-py311
```

### 方式2：直接使用环境中的 Python

```bash
# Windows
"C:\Users\YUHGT\.conda\envs\yolo-train-test-py311\python.exe" your_script.py

# 或使用 pip 安装包
"C:\Users\YUHGT\.conda\envs\yolo-train-test-py311\python.exe" -m pip install package_name
```

## 已安装的主要依赖

- fastapi==0.111.0
- uvicorn[standard]==0.30.1
- sqlmodel==0.0.21
- pydantic==2.7.4
- ultralytics>=8.2.0
- **onnxsim>=0.4.0** (新增)
- **torch==2.0.1+cu118** (CUDA 11.8，兼容 RF-DETR)
- **torchvision==0.15.2+cu118**
- **numpy<2.0** (兼容 PyTorch 2.0.1)
- **opencv-python<4.10** (兼容 numpy 1.x)
- **transformers>=4.40,<4.50** (RF-DETR 依赖)
- **peft>=0.10,<0.12** (RF-DETR 依赖)
- 以及其他所有项目依赖

## 运行项目

### 启动后端

```bash
# 激活环境
conda activate yolo-train-test-py311

# 启动 FastAPI 服务
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

或者直接使用完整路径：

```bash
cd backend
"C:\Users\YUHGT\.conda\envs\yolo-train-test-py311\Scripts\uvicorn.exe" app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 运行训练脚本

```bash
# 激活环境后
conda activate yolo-train-test-py311
python backend/run_training.py <job_id>

# 或直接使用
"C:\Users\YUHGT\.conda\envs\yolo-train-test-py311\python.exe" backend/run_training.py <job_id>
```

## 安装新的依赖

如果需要安装新的 Python 包：

```bash
# 激活环境
conda activate yolo-train-test-py311

# 安装包
pip install package_name

# 或从 requirements.txt 安装
pip install -r backend/requirements.txt
```

## RF-DETR ONNX 导出

现在可以使用 onnxsim 进行模型优化了：

```bash
conda activate yolo-train-test-py311
python backend/app/services/export_model.py
```

onnxsim 将自动用于简化和优化导出的 ONNX 模型。

## 环境管理

### 查看所有 conda 环境

```bash
conda env list
```

### 删除环境（如需重建）

```bash
conda deactivate
conda env remove -n yolo-train-test-py311
```

### 重新创建环境

```bash
conda create -n yolo-train-test-py311 python=3.11 -y
conda activate yolo-train-test-py311
pip install -r backend/requirements.txt
```

## PyCharm / IDE 配置

### PyCharm 配置 Conda 环境

1. 打开 **Settings** → **Project** → **Python Interpreter**
2. 点击齿轮图标 → **Add Interpreter** → **Add Local Interpreter**
3. 选择 **Conda Environment** → **Existing environment**
4. 设置解释器路径为: `C:\Users\YUHGT\.conda\envs\yolo-train-test-py311\python.exe`
5. 确认并应用

### VS Code 配置

1. 按 `Ctrl+Shift+P` 打开命令面板
2. 输入 `Python: Select Interpreter`
3. 选择 `yolo-train-test-py311` 环境
4. 或手动输入路径: `C:\Users\YUHGT\.conda\envs\yolo-train-test-py311\python.exe`

## 部署说明

### Docker 部署

Docker 镜像已更新为使用 Python 3.11：
- `deploy/Dockerfile.backend` 现在使用 `python:3.11-slim`
- `requirements.txt` 已包含 `onnxsim>=0.4.0`

部署时无需额外配置，直接使用：

```bash
cd deploy
docker-compose up -d
```

### 手动部署

如果在服务器上手动部署，确保安装 Python 3.11：

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3.11 python3.11-venv

# 创建虚拟环境
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 常见问题

### Q: 为什么要使用 Python 3.11？

A: onnxsim 包在 Python 3.12 上可能存在兼容性问题，Python 3.11 是目前最稳定且兼容性最好的版本。

### Q: 为什么使用 PyTorch 2.0.1 而不是最新版本？

A: RF-DETR 库与 PyTorch 2.9.1 存在兼容性问题（inference mode 错误）。经过测试，PyTorch 2.0.1 是最稳定且与 RF-DETR 1.3.0 完全兼容的版本。

### Q: 如何确认 GPU 是否可用？

A: 运行以下命令：
```bash
conda activate yolo-train-test-py311
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

应该显示：
```
CUDA available: True
GPU: NVIDIA GeForce RTX 4070
```

### Q: 旧的 .venv 环境怎么办？

A: 旧的 `.venv` 目录（Python 3.12）可以保留作为备份，也可以删除以节省空间。项目现在使用 conda 环境。

### Q: 如何验证 onnxsim 是否正常工作？

A: 运行以下命令测试：

```bash
conda activate yolo-train-test-py311
python -c "import onnxsim; print(onnxsim.__version__)"
```

应该输出 onnxsim 的版本号（如 0.4.36）。

### Q: conda activate 命令不工作？

A: 需要先初始化 conda：

```bash
conda init
# 重启终端后再试
conda activate yolo-train-test-py311
```

或者使用备用方案直接指定 Python 路径。

## 更新日志

- **2025-12-17**: 迁移到 Python 3.11 Conda 环境
  - 创建新的 conda 环境 `yolo-train-test-py311`
  - 添加 onnxsim 支持
  - 更新 Docker 镜像为 Python 3.11
  - 修复 RF-DETR 兼容性问题：
    - 降级 PyTorch 到 2.0.1（从 2.9.1）
    - 降级 numpy 到 1.x（从 2.x）
    - 更新 transformers 和 peft 到兼容版本
    - 解决 "Inference tensors cannot be saved for backward" 错误
  - 更新部署文档
