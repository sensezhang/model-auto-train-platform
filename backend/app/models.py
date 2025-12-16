from typing import Optional, List
from enum import Enum
from datetime import datetime
from sqlmodel import SQLModel, Field, Relationship


class Project(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    description: Optional[str] = None
    status: Optional[str] = Field(default="active")
    createdAt: datetime = Field(default_factory=datetime.utcnow)

    classes: List["Class"] = Relationship(back_populates="project")


class Class(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    projectId: int = Field(foreign_key="project.id")
    name: str
    color: Optional[str] = None
    hotkey: Optional[str] = None

    project: Optional[Project] = Relationship(back_populates="classes")


class ImageStatus(str, Enum):
    unannotated = "unannotated"
    ai_pending = "ai_pending"
    annotated = "annotated"


class Image(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    projectId: int = Field(foreign_key="project.id")
    path: str
    width: Optional[int] = None
    height: Optional[int] = None
    checksum: Optional[str] = None
    status: ImageStatus = Field(default="unannotated")
    createdAt: datetime = Field(default_factory=datetime.utcnow)


class Annotation(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    imageId: int = Field(foreign_key="image.id")
    classId: int = Field(foreign_key="class.id")
    type: str = Field(default="bbox")
    x: float
    y: float
    w: float
    h: float
    confidence: Optional[float] = None
    source: str = Field(default="manual")  # manual | ai
    createdAt: datetime = Field(default_factory=datetime.utcnow)


class ProposedAnnotation(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    imageId: int = Field(foreign_key="image.id")
    classId: Optional[int] = Field(default=None)
    x: float
    y: float
    w: float
    h: float
    confidence: float
    provider: str = Field(default="glm4.5v")
    payload: Optional[str] = None  # raw JSON from provider
    createdAt: datetime = Field(default_factory=datetime.utcnow)


class JobStatus(str, Enum):
    pending = "pending"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    canceled = "canceled"


class AutoLabelJob(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    projectId: int = Field(foreign_key="project.id")
    status: JobStatus = Field(default="pending")
    concurrency: int = Field(default=2)
    threshold: float = Field(default=0.5)
    imagesCount: int = Field(default=0)
    boxesCount: int = Field(default=0)
    logsRef: Optional[str] = None
    startedAt: Optional[datetime] = None
    finishedAt: Optional[datetime] = None


class DatasetVersion(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    projectId: int = Field(foreign_key="project.id")
    split: str  # train | val
    images: int
    annotations: int
    seed: int
    createdAt: datetime = Field(default_factory=datetime.utcnow)


class TrainingJob(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    projectId: int = Field(foreign_key="project.id")
    datasetVersionId: Optional[int] = Field(default=None, foreign_key="datasetversion.id")
    status: JobStatus = Field(default="pending")
    modelVariant: str = Field(default="yolov11n")
    epochs: int = Field(default=50)
    imgsz: int = Field(default=640)
    batch: Optional[int] = None
    seed: int = Field(default=42)
    logsRef: Optional[str] = None
    map50: Optional[float] = None
    map50_95: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    startedAt: Optional[datetime] = None
    finishedAt: Optional[datetime] = None


class ModelArtifact(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    trainingJobId: int = Field(foreign_key="trainingjob.id")
    format: str  # pt | onnx
    path: str
    size: Optional[int] = None
    checksum: Optional[str] = None
    createdAt: datetime = Field(default_factory=datetime.utcnow)


class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    username: str
    passwordHash: str
    role: str = Field(default="admin")


class ActivityLog(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    userId: Optional[int] = Field(default=None, foreign_key="user.id")
    projectId: Optional[int] = Field(default=None, foreign_key="project.id")
    type: str
    payload: Optional[str] = None
    createdAt: datetime = Field(default_factory=datetime.utcnow)


class ImportJob(SQLModel, table=True):
    """导入任务，用于追踪导入进度"""
    id: Optional[int] = Field(default=None, primary_key=True)
    projectId: int = Field(foreign_key="project.id")
    status: JobStatus = Field(default="pending")
    format: str = Field(default="yolo")  # yolo | images
    total: int = Field(default=0)
    current: int = Field(default=0)
    imported: int = Field(default=0)
    duplicates: int = Field(default=0)
    errors: int = Field(default=0)
    annotations_imported: int = Field(default=0)
    annotations_skipped: int = Field(default=0)
    message: Optional[str] = None
    startedAt: Optional[datetime] = None
    finishedAt: Optional[datetime] = None
    createdAt: datetime = Field(default_factory=datetime.utcnow)
