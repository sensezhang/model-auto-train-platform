from contextlib import contextmanager
from sqlmodel import SQLModel, create_engine, Session
import os


DB_PATH = os.getenv("APP_DB_PATH", os.path.join(os.getcwd(), "app.db"))
engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})


def init_db():
    SQLModel.metadata.create_all(engine)


@contextmanager
def get_session():
    with Session(engine) as session:
        yield session

