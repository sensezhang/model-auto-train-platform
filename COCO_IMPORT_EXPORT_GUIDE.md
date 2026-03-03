# COCO格式导入导出指南

## 概述

本系统现在支持**完整的COCO格式数据集导入导出循环**，确保导出的数据集可以无缝地重新导入。

## COCO导出格式

### 文件结构

```
dataset.zip
├── train/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── _annotations.coco.json
├── valid/
│   ├── image3.jpg
│   └── _annotations.coco.json
└── test/
    ├── image4.jpg
    └── _annotations.coco.json
```

### COCO JSON格式

每个`_annotations.coco.json`文件包含：

```json
{
  "info": {
    "description": "项目名称 train dataset",
    "version": "1.0",
    "year": 2025,
    "date_created": "2025-12-23T..."
  },
  "licenses": [],
  "images": [
    {
      "id": 0,
      "file_name": "image1.jpg",
      "width": 640,
      "height": 480
    }
  ],
  "annotations": [
    {
      "id": 0,
      "image_id": 0,
      "category_id": 0,
      "bbox": [x, y, width, height],
      "area": width * height,
      "iscrowd": 0,
      "segmentation": []
    }
  ],
  "categories": [
    {
      "id": 0,
      "name": "class_name",
      "supercategory": "object"
    }
  ]
}
```

### 关键特性

- **bbox格式**: `[x_min, y_min, width, height]` - 左上角坐标
- **类别ID**: 从0开始连续编号
- **图片ID**: 从0开始连续编号
- **标注ID**: 从0开始连续编号

## COCO导入功能

### 使用方法

1. **前端操作**：
   - 在项目列表页面点击"导入数据"按钮
   - 选择"COCO格式数据集"选项
   - 上传之前导出的ZIP文件

2. **后端处理**：
   - 解压ZIP文件到临时目录
   - 解析每个split的`_annotations.coco.json`文件
   - 自动创建缺失的类别
   - 导入图片和标注
   - 显示实时进度

### 类别处理

#### 类别匹配规则

- **按名称匹配**: COCO JSON中的`categories.name`与项目类别名称精确匹配
- **自动创建**: 如果项目中不存在某个类别，会自动创建
- **ID映射**: COCO的category_id会被映射到项目的class_id

#### 示例

**COCO文件中的类别**:
```json
{
  "categories": [
    {"id": 0, "name": "person"},
    {"id": 1, "name": "car"},
    {"id": 2, "name": "bicycle"}
  ]
}
```

**项目中已有类别**: person

**导入结果**:
- `person` → 匹配现有类别
- `car` → 自动创建新类别
- `bicycle` → 自动创建新类别

### 坐标转换

#### 导出时
```python
# 数据库: (x, y, w, h) - 左上角像素坐标
# COCO: [x, y, w, h] - 左上角像素坐标
# 直接使用，无需转换
```

#### 导入时
```python
# COCO: [x, y, w, h]
# 数据库: (x, y, w, h)
# 直接使用，无需转换
```

## API接口

### 导出COCO

**端点**: `POST /api/projects/{project_id}/export/coco`

**参数**:
```json
{
  "seed": 42,
  "train_ratio": 0.8,
  "val_ratio": 0.1,
  "test_ratio": 0.1,
  "augmentation": {
    "enabled": false,
    "resize_enabled": false
  }
}
```

**返回**:
```json
{
  "success": true,
  "file_path": "/path/to/export.zip",
  "stats": {
    "total_images": 100,
    "total_annotations": 500,
    "total_classes": 3,
    "splits": {
      "train": {"images": 80, "annotations": 400},
      "valid": {"images": 10, "annotations": 50},
      "test": {"images": 10, "annotations": 50}
    }
  }
}
```

### 导入COCO

**端点**: `POST /api/projects/{project_id}/import/coco`

**参数**: multipart/form-data
- `file`: ZIP文件

**返回**:
```json
{
  "job_id": 123,
  "status": "pending",
  "message": "导入任务已创建"
}
```

### 查询导入进度

**端点**: `GET /api/projects/{project_id}/import/jobs/{job_id}`

**返回**:
```json
{
  "id": 123,
  "status": "running",
  "total": 100,
  "current": 50,
  "progress": 50,
  "message": "正在导入图片...",
  "imported": 45,
  "duplicates": 3,
  "errors": 2,
  "annotations_imported": 200,
  "annotations_skipped": 5
}
```

## 完整流程示例

### 1. 导出数据集

```bash
# 前端操作
1. 选择项目 "我的标注项目"
2. 点击"导出数据集"
3. 选择"COCO格式"
4. 设置划分比例: 80/10/10
5. 下载 my_project_coco.zip
```

### 2. 查看导出结构

```bash
$ unzip -l my_project_coco.zip

Archive:  my_project_coco.zip
  Length      Date    Time    Name
---------  ---------- -----   ----
        0  12-23-2025 08:00   train/
   125678  12-23-2025 08:00   train/img001.jpg
   134567  12-23-2025 08:00   train/img002.jpg
    12345  12-23-2025 08:00   train/_annotations.coco.json
        0  12-23-2025 08:00   valid/
   123456  12-23-2025 08:00   valid/img003.jpg
     5678  12-23-2025 08:00   valid/_annotations.coco.json
        0  12-23-2025 08:00   test/
   127890  12-23-2025 08:00   test/img004.jpg
     5432  12-23-2025 08:00   test/_annotations.coco.json
```

### 3. 导入到新项目

```bash
# 前端操作
1. 创建新项目 "导入的项目" (类别可以为空)
2. 点击"导入数据"
3. 选择"COCO格式数据集"
4. 上传 my_project_coco.zip
5. 等待导入完成
```

### 4. 验证结果

导入完成后：
- ✅ 所有图片已导入
- ✅ 所有标注已恢复
- ✅ 类别自动创建（person, car, bicycle）
- ✅ bbox坐标完全一致
- ✅ 数据集划分保持不变（train/valid/test）

## 技术细节

### 后端实现

#### 导出服务
- **文件**: `backend/app/services/export_coco.py`
- **函数**: `export_dataset_to_coco()`
- **特性**:
  - 数据集划分
  - 数据增强支持
  - Resize支持
  - 坐标缩放

#### 导入服务
- **文件**: `backend/app/services/import_coco.py`
- **函数**: `import_coco_dataset()`
- **特性**:
  - ZIP解压
  - JSON解析
  - 类别自动创建
  - 图片去重（基于hash）
  - 进度回调

### 前端实现

#### UI组件
- **文件**: `frontend/src/pages/App.tsx`
- **组件**: `ImportDialog`
- **特性**:
  - 格式选择（COCO/YOLO/Images/Folder/Single）
  - 异步上传
  - 进度条显示
  - 结果统计

## 对比其他格式

| 特性 | COCO | YOLO | 纯图片 |
|------|------|------|--------|
| 包含标注 | ✅ | ✅ | ❌ |
| 包含类别定义 | ✅ | ✅ | ❌ |
| 数据集划分 | train/valid/test | train/valid/test | - |
| 标注格式 | JSON (绝对坐标) | TXT (归一化坐标) | - |
| 类别处理 | 自动创建/匹配 | 必须预先定义 | - |
| 适用场景 | 通用目标检测 | YOLO训练 | 手动标注 |

## 常见问题

### Q: 导出的COCO文件可以用于其他框架吗？

A: 可以！导出的COCO格式符合标准COCO规范，可以用于：
- PyTorch的torchvision
- TensorFlow的COCO API
- MMDetection
- Detectron2
- 其他支持COCO的框架

### Q: 导入时如果类别名称不匹配怎么办？

A: 系统会自动创建新类别。例如：
- 项目中有 `person`, `car`
- COCO中有 `person`, `car`, `bicycle`, `motorcycle`
- 导入后自动创建 `bicycle` 和 `motorcycle`

### Q: bbox坐标会丢失精度吗？

A: 不会。坐标以像素为单位存储，导出导入过程中保持完全一致。

### Q: 支持数据增强吗？

A: 支持！导出时可以选择：
- Resize: 调整图片尺寸
- 数据增强: 生成额外的增强样本（仅训练集）

### Q: 重复的图片会被跳过吗？

A: 是的。系统使用文件hash进行去重，重复的图片会被标记为`duplicate`。

## 总结

COCO导入导出功能实现了**完全自洽的数据循环**：

```
[项目数据]
    ↓ 导出
[COCO ZIP文件]
    ↓ 导入
[新项目数据]
    ↓ 验证
[完全一致] ✅
```

这使得数据集的分享、备份、迁移变得简单可靠！
