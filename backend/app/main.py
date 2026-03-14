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
        from .db import DB_TYPE, DB_PATH, APP_DB_URL
        from .utils.oss_storage import is_oss_enabled, _OSS_ENDPOINT, _OSS_BUCKET
        return {
            "status": "ok",
            "db_type": DB_TYPE,
            "db_path": DB_PATH if DB_TYPE == "sqlite" else None,
            "db_host": APP_DB_URL.split("@")[-1] if APP_DB_URL else None,
            "oss_enabled": is_oss_enabled(),
            "oss_endpoint": _OSS_ENDPOINT if is_oss_enabled() else None,
            "oss_bucket": _OSS_BUCKET if is_oss_enabled() else None,
        }

    # ── 本地文件服务 ──────────────────────────────────────────────
    # 当 path 已经是 OSS/HTTP URL 时（数据库存的是公网地址），302 重定向到原 URL；
    # 否则按相对路径在当前工作目录查找本地文件。
    @app.get("/files/{file_path:path}")
    async def serve_file(file_path: str):
        import urllib.parse
        from fastapi.responses import RedirectResponse, FileResponse
        from fastapi import HTTPException as _HTTPException

        decoded = urllib.parse.unquote(file_path)

        # OSS / 外部 URL → 直接重定向，避免 Windows 路径拼接错误
        if decoded.startswith("http://") or decoded.startswith("https://"):
            return RedirectResponse(url=decoded, status_code=302)

        # 本地相对路径 → 拼接 CWD 后返回文件
        local_path = os.path.join(os.getcwd(), decoded.replace("/", os.sep))
        if not os.path.isfile(local_path):
            raise _HTTPException(status_code=404, detail=f"File not found: {decoded}")
        return FileResponse(local_path)

    return app


app = create_app()
