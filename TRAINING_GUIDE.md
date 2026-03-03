# 训练中心使用指南

## 概述

训练中心是一个独立的页面，支持使用YOLO和RF-DETR两种目标检测框架进行模型训练。

## 功能特性

### 支持的训练框架

1. **YOLO (YOLOv11)**
   - 模型变体：n, s, m, l, x（从小到大）
   - 数据格式：YOLO格式（自动生成）
   - 输出格式：PT（PyTorch）和ONNX

2. **RF-DETR**
   - 模型变体：Small, Medium, Large
   - 数据格式：COCO格式（自动生成）
   - 输出格式：PTH（PyTorch checkpoint）

## 快速开始

### 1. 安装依赖

#### YOLO训练
```bash
# 已经安装
pip install ultralytics
```

#### RF-DETR训练
```bash
# 需要安装RF-DETR库
pip install rfdetr
# 或者从源码安装
# git clone https://github.com/yourusername/rfdetr.git
# cd rfdetr && pip install -e .
```

### 2. 数据库迁移

首次使用需要运行数据库迁移脚本：

```bash
cd backend
python migrate_add_framework.py
```

### 3. 访问训练中心

1. 启动后端服务
   ```bash
   cd backend
   python -m uvicorn app.main:app --reload
   ```

2. 启动前端开发服务器
   ```bash
   cd frontend
   npm run dev
   ```

3. 在主页点击 **🚀 训练中心** 按钮

## 使用流程

### 创建训练任务

1. **选择项目**：从下拉列表中选择要训练的项目
2. **选择训练框架**：YOLO 或 RF-DETR
3. **选择模型变体**：根据框架显示不同的模型选项
4. **配置训练参数**：
   - **训练轮数 (Epochs)**：建议50-300
   - **图像尺寸 (Image Size)**：建议640（YOLO）
   - **Batch Size**：留空自动检测，或手动设置（取决于显存）
   - **随机种子**：用于数据集划分的可重复性
5. **开始训练**：点击"开始训练"按钮

### 监控训练进度

训练任务创建后，您可以：

1. **查看任务列表**：左侧显示所有训练任务及其状态
   - 🟢 succeeded：训练成功
   - 🔵 running：正在训练
   - 🟡 pending：等待开始
   - 🔴 failed：训练失败
   - ⚪ canceled：已取消

2. **查看任务详情**：点击任务查看详细信息
   - 训练配置参数
   - 训练指标（mAP50, mAP50-95, Precision, Recall）
   - 模型文件列表

3. **实时日志**：右侧显示训练的实时日志输出

4. **取消训练**：对于正在运行的任务，可以点击"取消训练"按钮

## 训练配置说明

### YOLO配置

- **数据集格式**：自动生成YOLO格式（80/20 train/val split）
- **输出目录**：`models/{project_id}/{job_id}/`
- **日志文件**：`models/{project_id}/{job_id}/logs.txt`
- **最佳权重**：`models/{project_id}/{job_id}/train/weights/best.pt`

### RF-DETR配置

基于您提供的训练脚本，RF-DETR使用以下配置：

- **数据集格式**：自动生成COCO格式（80/20 train/val split）
- **输出目录**：`models/{project_id}/{job_id}/`
- **日志文件**：`models/{project_id}/{job_id}/logs.txt`
- **最佳权重**：`models/{project_id}/{job_id}/checkpoint_best_ema.pth`
- **早停策略**：启用，patience=15
- **TensorBoard**：启用

可以在 `backend/app/services/training_rfdetr.py` 的 `_parse_config()` 函数中调整默认配置。

## 训练数据要求

- **最少图片数**：50张已标注图片
- **推荐图片数**：200+张已标注图片
- **类别数量**：至少1个类别
- **标注质量**：确保边界框准确

## 训练后的模型

训练完成后，可以在任务详情中找到：

### YOLO模型
- **PT格式**：用于继续训练或推理
- **ONNX格式**：用于部署（opset=12）

### RF-DETR模型
- **PTH格式**：PyTorch checkpoint文件

## 故障排除

### 常见问题

1. **训练失败：显存不足**
   - 减小batch size
   - 减小图像尺寸
   - 使用更小的模型变体

2. **训练失败：图片不足**
   - 确保项目至少有50张已标注图片

3. **RF-DETR导入错误**
   - 确保已安装rfdetr库：`pip install rfdetr`

4. **任务卡在pending或running状态**
   - 检查后端日志
   - 查看进程日志：`backend/models/training_logs/process_{job_id}.log`
   - 使用清理API：POST `/api/training/jobs/cleanup-stuck`

## API端点

- `POST /api/training/jobs` - 创建训练任务
- `GET /api/training/jobs` - 获取所有训练任务
- `GET /api/training/jobs/{job_id}` - 获取任务详情
- `GET /api/training/jobs/{job_id}/artifacts` - 获取模型文件
- `GET /api/training/jobs/{job_id}/logs/stream` - 流式获取日志
- `POST /api/training/jobs/{job_id}/cancel` - 取消训练
- `POST /api/training/jobs/cleanup-stuck` - 清理卡住的任务

## 扩展其他模型框架

要添加新的训练框架（例如DETR、Faster R-CNN等）：

1. 在 `backend/app/services/` 创建新的训练服务文件
2. 实现 `run_training_job(job_id)` 和 `start_training_async(job_id)` 函数
3. 在 `backend/app/routers/training.py` 的 `create_training_job` 中添加框架分支
4. 在前端 `Training.tsx` 中添加新的框架选项和模型变体

## 性能建议

- **使用GPU**：确保CUDA可用，训练速度可提升10-100倍
- **数据增强**：在导出数据集时启用数据增强以提高模型泛化能力
- **Early Stopping**：RF-DETR默认启用，避免过拟合
- **TensorBoard**：使用TensorBoard监控训练指标变化

## 下一步

- 集成模型评估功能
- 添加超参数搜索
- 支持分布式训练
- 添加模型部署功能
