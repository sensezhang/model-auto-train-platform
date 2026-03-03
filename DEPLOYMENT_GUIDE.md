# 多GPU训练和服务器部署指南

## 概述

本平台支持单卡和多卡GPU训练，可以灵活部署到本地或服务器环境。

## 多GPU训练功能

### 支持的训练模式

1. **单GPU训练** - 指定一张GPU卡
2. **多GPU训练** - 指定多张GPU卡（使用DataParallel或DDP）
3. **CPU训练** - 无GPU环境下的训练

### 框架支持

#### YOLO (Ultralytics)
- 单GPU: `device=0`
- 多GPU: `device=0,1,2` (自动使用DP或DDP)
- CPU: `device=cpu`

#### RF-DETR
- 单GPU: 设置 `CUDA_VISIBLE_DEVICES=0`
- 多GPU: 设置 `CUDA_VISIBLE_DEVICES=0,1,2` + DDP配置
- CPU: `device=cpu`

## 数据库迁移

首次部署需要运行迁移脚本：

```bash
# 添加framework字段
cd backend
python migrate_add_framework.py

# 添加gpuIds字段（多GPU支持）
python migrate_add_gpu_ids.py
```

## 本地开发部署

### 1. 环境准备

```bash
# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 如果有GPU，安装CUDA版本的PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2. 启动服务

```bash
# 后端
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 前端（新终端）
cd frontend
npm install
npm run dev
```

## 服务器部署

### 方案1：Docker部署（推荐）

#### 1. 构建镜像

```bash
# 构建后端镜像
cd deploy
docker build -f Dockerfile.backend -t yolo-train-backend:latest ..

# 构建前端镜像
docker build -f Dockerfile.frontend -t yolo-train-frontend:latest ..
```

#### 2. 使用Docker Compose部署

```bash
cd deploy
docker-compose up -d
```

#### 3. GPU支持

确保Docker支持NVIDIA GPU：

```bash
# 安装nvidia-docker2
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# 验证GPU可用
docker run --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### 方案2：系统服务部署

#### 1. 安装Python环境

```bash
# 安装Python 3.12
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3-pip

# 创建虚拟环境
cd /opt/yolo-train-test
python3.12 -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

#### 2. 配置Systemd服务

```bash
# 复制服务文件
sudo cp deploy/systemd/yolo-train-backend.service /etc/systemd/system/
sudo cp deploy/systemd/yolo-train-frontend.service /etc/systemd/system/

# 重新加载systemd
sudo systemctl daemon-reload

# 启动服务
sudo systemctl start yolo-train-backend
sudo systemctl start yolo-train-frontend

# 设置开机自启
sudo systemctl enable yolo-train-backend
sudo systemctl enable yolo-train-frontend

# 查看状态
sudo systemctl status yolo-train-backend
sudo systemctl status yolo-train-frontend
```

#### 3. 配置Nginx反向代理

```bash
# 复制Nginx配置
sudo cp deploy/nginx/yolo-train.conf /etc/nginx/sites-available/
sudo ln -s /etc/nginx/sites-available/yolo-train.conf /etc/nginx/sites-enabled/

# 测试配置
sudo nginx -t

# 重载Nginx
sudo systemctl reload nginx
```

## 多GPU训练配置

### Web界面配置

1. 进入训练中心
2. 创建训练任务时，展开"高级选项"
3. 选择要使用的GPU：
   - 单选：只使用一张GPU
   - 多选：使用多张GPU进行分布式训练
4. 系统会自动：
   - 单GPU：直接指定device
   - 多GPU：配置DDP环境变量

### API配置

```bash
POST /api/training/jobs
{
  "projectId": 1,
  "framework": "yolo",  # or "rfdetr"
  "modelVariant": "yolov11m",
  "epochs": 100,
  "gpuIds": "0,1,2,3"  # 使用GPU 0,1,2,3
}
```

### 环境变量配置

```bash
# 单GPU
CUDA_VISIBLE_DEVICES=0 python run_training.py

# 多GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_training.py
```

## 性能优化

### GPU选择建议

- **小模型（YOLOv11n/s）**: 1张GPU即可
- **中等模型（YOLOv11m）**: 1-2张GPU
- **大模型（YOLOv11l/x，RF-DETR）**: 2-4张GPU

### Batch Size调整

多GPU训练时，batch size会自动按GPU数量倍增：
- 单GPU: batch_size = 16
- 2 GPU: effective_batch_size = 16 * 2 = 32
- 4 GPU: effective_batch_size = 16 * 4 = 64

### 内存管理

- 显存不足时，减小batch size或图像尺寸
- 使用梯度累积：每N步更新一次参数
- 启用混合精度训练：AMP (Automatic Mixed Precision)

## 监控和日志

### 查看训练日志

```bash
# 实时日志（训练中心Web界面）
# 或通过文件
tail -f backend/models/{project_id}/{job_id}/logs.txt

# 训练进程日志
tail -f backend/models/training_logs/process_{job_id}.log
```

### GPU监控

```bash
# 实时监控GPU使用
watch -n 1 nvidia-smi

# 或使用gpustat
pip install gpustat
watch -n 1 gpustat -cpu
```

### TensorBoard监控

```bash
# YOLO训练
tensorboard --logdir=backend/models/{project_id}/{job_id}/train

# RF-DETR训练
tensorboard --logdir=backend/models/{project_id}/{job_id}
```

## 故障排除

### GPU不可用

```bash
# 检查CUDA
python -c "import torch; print(torch.cuda.is_available())"

# 检查GPU
nvidia-smi

# 重新安装PyTorch CUDA版本
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 多GPU训练失败

1. 检查GPU是否可见：`nvidia-smi`
2. 检查CUDA_VISIBLE_DEVICES环境变量
3. 确保所有GPU显存充足
4. 检查NCCL配置（仅Linux）

### 端口冲突

```bash
# 修改后端端口
uvicorn app.main:app --host 0.0.0.0 --port 8001

# 修改前端端口
# 编辑 frontend/vite.config.ts
```

## 安全建议

1. **生产环境**：
   - 使用HTTPS（SSL证书）
   - 配置防火墙规则
   - 添加用户认证
   - 限制API访问频率

2. **数据备份**：
   - 定期备份数据库
   - 备份训练模型文件
   - 备份项目数据

3. **资源限制**：
   - 限制单个用户的训练任务数
   - 限制GPU使用时长
   - 定期清理旧模型文件

## 扩展功能

### 分布式训练（多机多卡）

编辑训练脚本添加：

```python
# 主节点
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr="192.168.1.1" \
    --master_port=12355 \
    run_training.py

# 从节点
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr="192.168.1.1" \
    --master_port=12355 \
    run_training.py
```

### 云平台部署

- **AWS**: EC2 P3/P4 实例 + EBS存储
- **Google Cloud**: Compute Engine + GPU实例
- **Azure**: NC系列虚拟机
- **阿里云**: GPU计算型实例

## 维护命令

```bash
# 启动服务
./deploy/scripts/start.sh

# 停止服务
./deploy/scripts/stop.sh

# 重启服务
./deploy/scripts/restart.sh

# 查看状态
./deploy/scripts/status.sh

# 清理卡住的训练任务
curl -X POST http://localhost:8000/api/training/jobs/cleanup-stuck
```

## 更新部署

```bash
# 拉取最新代码
git pull

# 运行数据库迁移
cd backend
python migrate_*.py

# 重启服务
./deploy/scripts/restart.sh
```
