# ONNX导出完整指南

## 快速说明

### YOLO模型导出 ✅ 开箱即用
YOLO (PT格式) 可以直接导出为ONNX，无需额外依赖。

### RF-DETR模型导出 ⚠️ 需要额外步骤
RF-DETR (PTH格式) 导出需要 `onnxsim` 包，该包需要cmake编译。

---

## 方案1：只导出YOLO模型（推荐）

如果你主要使用YOLO训练，那么：

**✅ 无需任何额外配置**

1. 训练完成后，找到PT格式的模型文件
2. 点击"导出ONNX"
3. 完成！

---

## 方案2：为RF-DETR安装cmake和onnxsim

如果你需要导出RF-DETR模型，按以下步骤：

### Windows用户

#### 步骤1：安装cmake

**方法A：使用Scoop（推荐）**
```powershell
# 如果还没安装scoop
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
Invoke-RestMethod -Uri https://get.scoop.sh | Invoke-Expression

# 安装cmake
scoop install cmake

# 验证安装
cmake --version
```

**方法B：手动安装**
1. 访问 https://cmake.org/download/
2. 下载Windows x64 Installer
3. 运行安装程序，**勾选"Add CMake to PATH"**
4. 重启终端
5. 验证：`cmake --version`

#### 步骤2：安装onnxsim
```bash
cd E:\PycharmProjects\yolo-train-test
.venv\Scripts\pip install onnxsim
```

#### 步骤3：重启服务并测试
```bash
# 重启后端
cd backend
..\.venv\Scripts\uvicorn app.main:app --reload

# 测试RF-DETR导出
```

### Linux用户

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install cmake

# CentOS/RHEL
sudo yum install cmake

# 安装onnxsim
pip install onnxsim
```

### Mac用户

```bash
# 使用Homebrew
brew install cmake

# 安装onnxsim
pip install onnxsim
```

---

## 方案3：不导出ONNX（使用原始权重）

如果你不需要ONNX格式，可以直接使用：

- **YOLO**: PT格式文件
- **RF-DETR**: PTH格式文件

这些格式可以直接用于：
- PyTorch推理
- 继续训练
- 模型微调

---

## 为什么RF-DETR需要onnxsim？

根据RF-DETR官方文档，导出ONNX时会自动尝试简化模型（使用onnxsim），这样可以：
- ✅ 减小模型文件大小
- ✅ 提高推理速度
- ✅ 优化模型结构

但onnxsim需要cmake来编译C++扩展，这在Windows上比较麻烦。

---

## 当前状态说明

我已经修复了代码错误（`log_path未定义`），现在RF-DETR导出会：

1. 尝试使用 `simplify=False` 导出
2. 但RF-DETR在导入时就会检查onnxsim依赖
3. 如果没有onnxsim，会显示错误消息

**解决方案：**
- 如果需要RF-DETR ONNX导出 → 安装cmake和onnxsim（见上面步骤）
- 如果只用YOLO → 无需任何操作，直接导出即可
- 如果不需要ONNX → 直接使用PT/PTH文件

---

## 测试导出

### 测试YOLO导出
```bash
# 在Web界面：
1. 完成一个YOLO训练
2. 找到PT格式文件
3. 点击"导出ONNX"
4. 应该成功！
```

### 测试RF-DETR导出
```bash
# 前提：已安装cmake和onnxsim

# 在Web界面：
1. 完成一个RF-DETR训练
2. 找到PTH格式文件
3. 点击"导出ONNX"
4. 如果成功，ONNX文件会出现在列表中
```

---

## 常见问题

### Q1: 我必须导出ONNX吗？
**A:** 不是必须的。PT/PTH文件可以直接用于PyTorch推理。ONNX主要用于：
- 跨平台部署
- 使用其他推理引擎（TensorRT、OpenVINO等）
- 在非Python环境中使用

### Q2: 安装cmake很麻烦，有没有简单的方法？
**A:**
- **方案1**: 只使用YOLO训练（无需cmake）
- **方案2**: 使用Scoop安装cmake（一行命令）
- **方案3**: 在有cmake的Linux服务器上导出

### Q3: YOLO导出ONNX需要cmake吗？
**A:** 不需要！YOLO的ONNX导出开箱即用。

### Q4: 我已经有cmake，为什么还是失败？
**A:** 确保：
1. cmake在PATH中：`cmake --version` 能运行
2. 重启了所有终端窗口
3. 重新运行pip install：`.venv\Scripts\pip install onnxsim`

### Q5: 导出的ONNX文件在哪里？
**A:** 在模型权重的同一目录：
```
models/
  └── {project_id}/
      └── {job_id}/
          ├── best.pt (或 checkpoint_best.pth)
          └── model.onnx  ← 导出的ONNX文件
```

---

## 推荐方案

根据你的使用场景选择：

| 场景 | 推荐方案 |
|------|---------|
| 只用YOLO | 无需额外配置，直接导出 ✅ |
| 主要用RF-DETR | 安装cmake + onnxsim |
| 不确定 | 先用YOLO测试，需要时再装cmake |
| 服务器部署 | Linux服务器上安装cmake很简单 |
| 本地开发 | 使用PT/PTH文件即可 |

---

## 快速决策树

```
需要导出ONNX？
  ├─ 否 → 使用PT/PTH文件，无需操作
  └─ 是 → 什么模型？
         ├─ YOLO → 直接导出即可 ✅
         └─ RF-DETR → 有cmake吗？
                     ├─ 是 → 安装onnxsim，然后导出
                     └─ 否 → 安装cmake → 安装onnxsim → 导出
```

---

## 获取帮助

如果按照上述步骤仍然失败：

1. **重启后端服务** - 确保修复生效
2. **检查cmake版本** - `cmake --version`
3. **查看完整错误** - 在后端终端查看详细错误信息
4. **提供错误信息** - 包括cmake版本、pip输出等

现在代码错误已修复，YOLO导出应该可以直接工作了！🚀
