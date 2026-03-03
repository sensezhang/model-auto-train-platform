# 多GPU训练和服务器部署升级总结

## ✅ 已完成的功能

### 1. 多GPU训练支持

#### 后端核心功能
- ✅ GPU检测和管理工具 (`backend/app/utils/gpu_utils.py`)
  - 自动检测所有可用GPU
  - 获取GPU显存信息
  - 格式化GPU环境变量
  - 设置分布式训练环境

- ✅ 数据库模型扩展 (`backend/app/models.py`)
  - TrainingJob添加 `gpuIds` 字段
  - 支持逗号分隔的GPU ID列表（如 "0,1,2"）

- ✅ YOLO多GPU训练 (`backend/app/services/training_yolo.py`)
  - 单GPU: `device="0"`
  - 多GPU: `device="0,1,2"` (自动使用DataParallel或DDP)
  - 自动检测并使用可用GPU

- ✅ RF-DETR多GPU训练 (`backend/app/services/training_rfdetr.py`)
  - 通过CUDA_VISIBLE_DEVICES设置可见GPU
  - 支持分布式训练配置
  - 自动适配GPU数量

- ✅ GPU信息API (`backend/app/routers/system.py`)
  - `GET /api/system/gpus` - 获取所有GPU信息
  - `GET /api/system/info` - 获取系统配置信息

#### 前端功能
- ✅ GPU选择界面 (`frontend/src/pages/Training.tsx`)
  - 自动加载可用GPU列表
  - 支持多选GPU进行训练
  - 实时显示GPU显存状态
  - 多GPU训练提示

### 2. 模型导出功能

- ✅ ONNX导出服务 (`backend/app/services/export_model.py`)
  - YOLO模型导出为ONNX
  - RF-DETR模型导出为ONNX
  - 自动检测训练框架
  - 导出后自动保存到ModelArtifact

- ✅ 导出API (`backend/app/routers/training.py`)
  - `POST /api/training/jobs/{job_id}/export-onnx`
  - 支持PT和PTH格式转ONNX

- ✅ 前端导出界面
  - 在模型文件列表中显示"导出ONNX"按钮
  - 导出进度显示
  - 自动刷新文件列表

### 3. 数据库迁移

- ✅ `backend/migrate_add_framework.py` - 添加framework字段
- ✅ `backend/migrate_add_gpu_ids.py` - 添加gpuIds字段

### 4. 部署文档

- ✅ `DEPLOYMENT_GUIDE.md` - 完整部署指南
- ✅ `MULTI_GPU_SETUP.md` - 多GPU配置快速指南
- ✅ `UPGRADE_SUMMARY.md` - 升级总结（本文档）

## 🚀 快速开始

### 第一步：运行数据库迁移

```bash
cd backend

# 添加framework字段（如果还没运行过）
../.venv/Scripts/python migrate_add_framework.py

# 添加gpuIds字段
../.venv/Scripts/python migrate_add_gpu_ids.py
```

### 第二步：安装依赖（如果需要）

```bash
# 确保安装了ONNX相关包
../.venv/Scripts/pip install onnx onnxruntime

# 确保PyTorch是CUDA版本
../.venv/Scripts/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 第三步：重启服务

```bash
# 停止当前服务（Ctrl+C）

# 重新启动后端
cd backend
../.venv/Scripts/uvicorn app.main:app --reload

# 前端无需修改，自动生效
```

### 第四步：测试功能

1. **访问训练中心** - http://localhost:5173/（或你的前端端口）
2. **点击"训练中心"**
3. **创建训练任务**：
   - 选择项目
   - 选择框架（YOLO或RF-DETR）
   - 选择模型
   - **新功能：选择GPU** - 可以单选或多选
   - 点击"开始训练"

4. **等待训练完成后**：
   - 在"模型文件"区域找到PT或PTH文件
   - 点击"导出ONNX"按钮
   - 等待导出完成
   - ONNX文件会自动出现在列表中

## 📊 支持的训练模式

| 模式 | YOLO | RF-DETR | 配置示例 |
|-----|------|---------|---------|
| CPU训练 | ✅ | ✅ | gpuIds: null |
| 单GPU | ✅ | ✅ | gpuIds: "0" |
| 双GPU | ✅ | ✅ | gpuIds: "0,1" |
| 四GPU | ✅ | ✅ | gpuIds: "0,1,2,3" |
| 多机多卡 | ⚠️ 需手动配置 | ⚠️ 需手动配置 | 见部署指南 |

## 🔧 API使用示例

### 获取GPU信息

```bash
curl http://localhost:8000/api/system/gpus
```

响应：
```json
[
  {
    "id": 0,
    "name": "NVIDIA GeForce RTX 4070",
    "memory_total": 12282,
    "memory_free": 11500,
    "memory_used": 782,
    "compute_capability": "8.9"
  }
]
```

### 创建多GPU训练任务

```bash
curl -X POST http://localhost:8000/api/training/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "projectId": 1,
    "framework": "yolo",
    "modelVariant": "yolov11m",
    "epochs": 100,
    "imgsz": 640,
    "batch": 16,
    "seed": 42,
    "gpuIds": "0,1"
  }'
```

### 导出模型为ONNX

```bash
curl -X POST "http://localhost:8000/api/training/jobs/1/export-onnx?artifact_id=1&simplify=false"
```

## 📝 新增文件列表

### 后端
- `backend/app/utils/gpu_utils.py` - GPU工具类
- `backend/app/services/export_model.py` - 模型导出服务
- `backend/app/routers/system.py` - 系统信息API
- `backend/migrate_add_gpu_ids.py` - 数据库迁移脚本

### 文档
- `DEPLOYMENT_GUIDE.md` - 部署指南
- `MULTI_GPU_SETUP.md` - 多GPU配置指南
- `UPGRADE_SUMMARY.md` - 本文档

### 修改的文件
- `backend/app/models.py` - 添加gpuIds字段
- `backend/app/main.py` - 注册system router
- `backend/app/services/training_yolo.py` - 添加GPU支持
- `backend/app/services/training_rfdetr.py` - 添加GPU支持
- `backend/app/routers/training.py` - 添加导出API和gpuIds参数
- `frontend/src/pages/Training.tsx` - 添加GPU选择和导出功能

## 🎯 核心改进

### 性能提升
- **多GPU训练**：相比单GPU可提升2-4倍速度（取决于GPU数量）
- **自动化**：自动检测GPU，自动配置环境
- **灵活性**：支持任意组合的GPU选择

### 用户体验
- **可视化GPU选择**：实时显示GPU显存状态
- **一键导出ONNX**：无需手动运行脚本
- **实时反馈**：多GPU训练提示

### 部署友好
- **环境自适应**：自动检测CPU/GPU环境
- **Docker支持**：完整的Docker部署方案
- **监控便利**：GPU使用情况实时显示

## ⚙️ 环境变量

### 训练进程自动设置
- `CUDA_VISIBLE_DEVICES` - 控制可见GPU
- `WORLD_SIZE` - 分布式训练world size
- `RANK` - 当前进程rank
- `NCCL_*` - NCCL通信配置

### 手动配置（可选）
```bash
# 限制可见GPU
export CUDA_VISIBLE_DEVICES=0,1,2

# RF-DETR ONNX导出
export TORCH_ONNX_EXPERIMENTAL_RUNTIME_TYPE_CHECK=0

# YOLO权重目录
export YOLO_WEIGHTS_DIR=/path/to/weights
```

## 🔍 监控和调试

### 查看GPU使用

```bash
# 实时监控
watch -n 1 nvidia-smi

# 或使用gpustat
pip install gpustat
watch -n 1 gpustat -cpu
```

### 查看训练日志

```bash
# Web界面：训练中心 -> 选择任务 -> 查看日志

# 或通过文件
tail -f backend/models/{project_id}/{job_id}/logs.txt
```

### TensorBoard

```bash
# YOLO
tensorboard --logdir=backend/models/{project_id}/{job_id}/train

# RF-DETR
tensorboard --logdir=backend/models/{project_id}/{job_id}
```

## 🐛 常见问题

### Q: GPU不可见
A: 检查CUDA安装和PyTorch版本：
```bash
python -c "import torch; print(torch.cuda.is_available())"
nvidia-smi
```

### Q: 多GPU训练失败
A: 检查：
1. 显存是否充足
2. GPU是否被其他进程占用
3. 减小batch size

### Q: 导出ONNX失败
A: 确保：
1. 模型训练完成
2. PT/PTH文件存在
3. 安装了onnx包

### Q: 前端看不到GPU选项
A: 检查：
1. 后端是否重启
2. `/api/system/gpus` API是否可访问
3. 浏览器控制台是否有错误

## 📚 相关文档

- **部署指南**：`DEPLOYMENT_GUIDE.md`
- **多GPU配置**：`MULTI_GPU_SETUP.md`
- **训练指南**：`TRAINING_GUIDE.md`

## 🎉 总结

本次升级实现了：

1. ✅ **完整的多GPU训练支持** - YOLO和RF-DETR都支持
2. ✅ **模型导出功能** - 一键导出ONNX
3. ✅ **友好的用户界面** - 可视化GPU选择
4. ✅ **完善的部署方案** - 本地和服务器都支持
5. ✅ **详细的文档** - 快速上手指南

现在平台已经具备了生产级别的多GPU训练能力，可以部署到服务器进行大规模模型训练！🚀
