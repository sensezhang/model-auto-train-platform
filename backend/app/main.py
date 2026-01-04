from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .routers import projects, images, annotations, autolabel, training, export, system


def create_app() -> FastAPI:
    app = FastAPI(title="Image Labeling & Training Platform", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(projects.router, prefix="/api")
    app.include_router(images.router, prefix="/api")
    app.include_router(annotations.router, prefix="/api")
    app.include_router(autolabel.router, prefix="/api")
    app.include_router(training.router, prefix="/api")
    app.include_router(export.router, prefix="/api")
    app.include_router(system.router, prefix="/api")

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    # 静态文件：暴露数据集与模型等（最小化：仓库根目录，仅用于开发）
    app.mount("/files", StaticFiles(directory="."), name="files")

    return app


app = create_app()
