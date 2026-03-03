# 故障排查指南

## 问题1: 看不到训练曲线

### 症状
- 选择训练任务后，没有显示Loss和mAP曲线
- 或者曲线显示但没有数据

### 可能原因和解决方法

#### 1. 训练刚开始，还没有完成第一个epoch
**解决：** 耐心等待第一个epoch完成（通常需要几分钟）

#### 2. 日志文件不存在或为空
**诊断：**
```bash
cd backend
# 找到你的训练任务日志文件
ls models/{project_id}/{job_id}/logs.txt
```

**解决：** 如果文件不存在，检查训练是否真的在运行

#### 3. 日志解析失败
**诊断：** 运行测试脚本
```bash
cd backend
../.venv/Scripts/python test_metrics.py models/7/8/logs.txt yolo
```

这会显示：
- 日志文件内容
- 解析出的指标
- 如果没有解析出数据，会提示可能的原因

**解决：**
- 如果日志格式不匹配，将日志内容发送给开发者
- 或者手动检查日志中是否有"epoch"、"loss"、"mAP"等关键词

#### 4. API调用失败
**诊断：** 打开浏览器开发者工具（F12），查看Console和Network标签

**检查：**
```
GET /api/training/jobs/{job_id}/metrics
```
是否返回200状态码

**解决：**
- 如果返回404：训练任务不存在
- 如果返回500：检查后端日志
- 如果返回空数组：日志还没有数据

### 快速测试

1. **创建一个快速测试任务**
   - 模型：yolov11n（最小）
   - Epochs：5
   - 项目：选择一个有标注的项目

2. **等待第一个epoch完成**（约2-5分钟）

3. **刷新页面或重新选择任务**

4. **应该能看到曲线了！**

---

## 问题2: ONNX导出失败

### 症状
```
ModuleNotFoundError: No module named 'onnxsim'
```

### 原因
RF-DETR的ONNX导出需要`onnxsim`包，但该包需要cmake编译

### 解决方法

#### 方法1：已修复（推荐）
我已经修改了代码，强制使用`simplify=False`，不需要onnxsim。

**重启后端服务即可：**
```bash
# 停止当前后端（Ctrl+C）
cd backend
../.venv/Scripts/uvicorn app.main:app --reload
```

#### 方法2：安装cmake和onnxsim（可选）

如果你想要simplify功能（可以让ONNX文件更小、更快），需要：

1. **安装cmake**
   - Windows: https://cmake.org/download/
   - 或使用scoop: `scoop install cmake`

2. **安装onnxsim**
   ```bash
   .venv/Scripts/pip install onnxsim
   ```

### YOLO ONNX导出

YOLO的ONNX导出**不需要**onnxsim，应该直接可以工作。

如果YOLO导出也失败，检查：
```bash
.venv/Scripts/pip list | findstr onnx
```

应该看到：
- onnx
- onnxruntime

---

## 完整测试流程

### 1. 测试后端健康
```bash
curl http://localhost:8000/api/health
```
应返回：`{"status":"ok"}`

### 2. 测试GPU信息
```bash
curl http://localhost:8000/api/system/gpus
```
应返回GPU列表

### 3. 测试训练任务列表
```bash
curl http://localhost:8000/api/training/jobs
```
应返回任务列表

### 4. 测试指标API
```bash
# 替换{job_id}为实际的任务ID
curl http://localhost:8000/api/training/jobs/8/metrics
```
应返回指标数组（如果训练已经开始）

### 5. 测试日志解析
```bash
cd backend
../.venv/Scripts/python test_metrics.py models/7/8/logs.txt yolo
```

---

## 常见错误信息

### 1. `no such column: trainingjob.gpuIds`
**原因：** 数据库迁移未运行

**解决：**
```bash
cd backend
../.venv/Scripts/python migrate_add_gpu_ids.py
```

### 2. `No module named 'recharts'`（前端）
**原因：** 前端依赖未安装

**解决：**
```bash
cd frontend
npm install recharts
```

### 3. `Connection refused`
**原因：** 后端服务未启动

**解决：**
```bash
cd backend
../.venv/Scripts/uvicorn app.main:app --reload
```

### 4. 图表显示但没有数据
**可能原因：**
- 训练刚开始
- 日志格式不匹配
- API调用失败

**解决：** 按照上面的"问题1"排查

---

## 获取帮助

如果上述方法都无法解决：

1. **检查日志文件内容**
   ```bash
   cd backend
   type models\{project_id}\{job_id}\logs.txt
   ```
   将前20行发送给开发者

2. **检查浏览器控制台错误**
   - 按F12打开开发者工具
   - 查看Console标签的红色错误
   - 截图发送给开发者

3. **检查后端错误**
   - 查看运行uvicorn的终端输出
   - 查找红色的ERROR或Traceback
   - 复制完整错误信息

4. **提供详细信息**
   - 操作系统
   - Python版本
   - 训练框架（YOLO/RF-DETR）
   - 模型变体
   - 完整的错误信息
