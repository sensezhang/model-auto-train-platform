import os

# ══════════════════════════════════════════════════════════════
# 最先加载 .env 文件！
# 必须在所有本地模块 import 之前执行，否则 db.py 的模块级变量
# 在 import 时已经读完环境变量，.env 中的值永远不会生效。
# ══════════════════════════════════════════════════════════════
try:
    from dotenv import load_dotenv as _load_dotenv
    _env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
    if os.path.isfile(_env_file):
        _load_dotenv(_env_file, override=True)
        print(f"[startup] .env 加载成功: {os.path.abspath(_env_file)}")
    else:
        print(f"[startup] 未找到 .env 文件: {os.path.abspath(_env_file)}，使用系统环境变量")
except ImportError:
    print("[startup] python-dotenv 未安装，跳过 .env 加载（pip install python-dotenv）")

# ── 以下 import 才能正确读到 .env 中的环境变量 ─────────────────
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .routers import projects, images, annotations, autolabel, training, export, system, inference, videos


def create_app() -> FastAPI:
    app = FastAPI(title="Image Labeling & Training Platform", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(projects.router,    prefix="/api")
    app.include_router(images.router,      prefix="/api")
    app.include_router(annotations.router, prefix="/api")
    app.include_router(autolabel.router,   prefix="/api")
    app.include_router(training.router,    prefix="/api")
    app.include_router(export.router,      prefix="/api")
    app.include_router(system.router,      prefix="/api")
    app.include_router(inference.router,   prefix="/api")
    app.include_router(videos.router,      prefix="/api")

    @app.get("/api/health")
    def health():
        from .db import DB_TYPE, DB_PATH
        return {
            "status": "ok",
            "db_type": DB_TYPE,
            "db_path": DB_PATH,
        }

    # 本地文件服务：挂载当前工作目录，供前端访问图片
    app.mount("/files", StaticFiles(directory="."), name="files")

    return app


app = create_app()
