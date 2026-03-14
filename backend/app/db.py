from contextlib import contextmanager
from sqlmodel import SQLModel, create_engine, Session
from sqlalchemy import text
import os


# ──────────────────────────────────────────────────────────────
# 数据库配置
#
# MySQL（本地开发）：
#   APP_DB_URL=mysql+pymysql://user:pass@host:3306/dbname
#
# SQLite（服务器 Docker，不设置 APP_DB_URL 时自动使用）：
#   APP_DB_PATH=/app/db/app.db   （不设置则默认 app.db）
# ──────────────────────────────────────────────────────────────

APP_DB_URL = os.getenv("APP_DB_URL")  # MySQL 连接串，设置后优先使用 MySQL

if APP_DB_URL:
    engine = create_engine(APP_DB_URL, pool_pre_ping=True)
    DB_TYPE = "mysql"
    DB_PATH = None
else:
    DB_PATH = os.getenv("APP_DB_PATH", os.path.join(os.getcwd(), "app.db"))
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
    if DB_TYPE == "sqlite":
        _run_sqlite_migrations()


@contextmanager
def get_session():
    with Session(engine) as session:
        yield session
