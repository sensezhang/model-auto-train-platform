# 图片标注与训练平台 — 脚手架

本仓库包含一个最小可运行的后端（FastAPI + SQLite）与前端（React Vite 风格）骨架，契合 `Prd.md` 中的 MVP 范畴。

## 快速开始

### 后端（FastAPI）
1. 在项目根目录安装依赖（推荐使用项目根目录的虚拟环境）：
   ```bash
   # Windows
   .venv\Scripts\python -m pip install fastapi uvicorn[standard] sqlmodel pydantic python-multipart Jinja2 Pillow ultralytics

   # Linux/Mac
   .venv/bin/python -m pip install fastapi uvicorn[standard] sqlmodel pydantic python-multipart Jinja2 Pillow ultralytics
   ```

2. 启动后端服务：
   ```bash
   # Windows
   cd backend
   ..\.venv\Scripts\uvicorn app.main:app --reload --port 8000

   # Linux/Mac
   cd backend
   ../venv/bin/uvicorn app.main:app --reload --port 8000
   ```

3. 健康检查：访问 http://localhost:8000/api/health

> **注意事项：**
> - 数据库路径：后端在当前工作目录创建 SQLite 数据库（`backend/app.db`）
> - 可通过环境变量 `APP_DB_PATH` 指定绝对路径，例如 `APP_DB_PATH=E:/data/my.db`
> - 首次训练会自动下载模型权重（如 yolo11n.pt），需网络可用
> - 离线环境：请将预训练权重放置到 `backend/models/weights/<variant>.pt`

### 前端（React）
1. 安装依赖并启动：
   ```bash
   cd frontend
   npm install
   npm run dev
   ```
2. 访问 http://localhost:5173 ，主页会调用后端 `/api/health` 与 `/api/projects`。

> 如需本地开发代理，请在 `vite.config.ts` 中配置 `server.proxy['/api'] = 'http://localhost:8000'`。

## 目录结构（节选）
```
backend/
  app/
    main.py
    db.py
    models.py
    routers/
      projects.py images.py annotations.py autolabel.py training.py
    services/
      glm_autolabel.py training_yolo.py
    workers/queue.py
  requirements.txt
frontend/
  index.html
  package.json
  src/
    main.tsx
    pages/App.tsx
Prd.md
```

## 下一步（实现建议）
- 路由填充：按 `Prd.md` 接口草案完善导入zip、AI自动标注任务、训练作业与下载产物。
- 服务集成：在 `services/` 中实现 GLM-4.5V 调用（并发=2，阈值=0.5）与 YOLOv11 训练&ONNX导出。
- 前端：添加项目创建、导入、标注工作台（Canvas/Konva）、AI待审、训练面板与下载入口。
- 任务：用内存队列先跑通，V2 切换到 RQ/Celery。

## 训练说明
- 训练入口位于标注工作台右上角“开始训练”，需要≥50张已标注图片。
- 训练状态会轮询显示；完成后可直接下载 `.pt` 与 `.onnx`。
