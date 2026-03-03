from contextlib import contextmanager
from sqlmodel import SQLModel, create_engine, Session
from sqlalchemy import text
import os


# ──────────────────────────────────────────────────────────────
# 数据库配置（始终使用 SQLite，路径通过环境变量配置）
#
# Docker 挂载示例：
#   volumes:
#     - /mnt/data/db:/app/db
#   environment:
#     - APP_DB_PATH=/app/db/app.db
# ──────────────────────────────────────────────────────────────

DB_PATH = os.getenv("APP_DB_PATH", os.path.join(os.getcwd(), "app.db"))

# 确保数据库目录存在
os.makedirs(os.path.dirname(os.path.abspath(DB_PATH)), exist_ok=True)

engine = create_engine(
    f"sqlite:///{DB_PATH}",
    connect_args={"check_same_thread": False},
)
DB_TYPE = "sqlite"


def _run_sqlite_migrations():
    """自动为已有表补充新列（向后兼容）"""
    with engine.connect() as conn:
        # image 表：labeled 列
        result = conn.execute(text("PRAGMA table_info(image)"))
        img_cols = [row[1] for row in result]
        if "labeled" not in img_cols:
            conn.execute(text("ALTER TABLE image ADD COLUMN labeled BOOLEAN DEFAULT 0"))
            conn.commit()

        # autolabeljob 表：classId / prompt / processedCount 列
        result = conn.execute(text("PRAGMA table_info(autolabeljob)"))
        alj_cols = [row[1] for row in result]
        for col, ddl in [
            ("classId",       "ALTER TABLE autolabeljob ADD COLUMN classId INTEGER DEFAULT NULL"),
            ("prompt",        "ALTER TABLE autolabeljob ADD COLUMN prompt TEXT DEFAULT NULL"),
            ("processedCount","ALTER TABLE autolabeljob ADD COLUMN processedCount INTEGER DEFAULT 0"),
        ]:
            if col not in alj_cols:
                conn.execute(text(ddl))
                conn.commit()


def init_db():
    """初始化数据库：建表 + 迁移补丁"""
    SQLModel.metadata.create_all(engine)
    _run_sqlite_migrations()


@contextmanager
def get_session():
    with Session(engine) as session:
        yield session
