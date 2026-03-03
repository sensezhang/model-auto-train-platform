"""
YOLOv11 训练服务�?- 生成YOLO数据集快照（80/20划分�?- 调用Ultralytics进行训练
- 解析核心metrics并导出ONNX(opset=12)
注意：需要本地已安装 `ultralytics` 与合适的 `torch` (GPU推荐)�?"""

from __future__ import annotations

import os
import random
import shutil
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
from datetime import datetime

from sqlmodel import select

from ..db import get_session
from ..models import Image as DBImage, Annotation as DBAnn, Class as DBClass, TrainingJob, ModelArtifact
from ..utils.oss_storage import resolve_local_path, get_basename


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _yolo_name_for_variant(variant: str) -> str:
    # 兼容传入 yolov11n / yolo11n / yolo11s ...
    v = variant.lower().replace("yolov11", "yolo11")
    if not v.endswith('.pt'):
        v = f"{v}.pt"
    return v


def _resolve_weights(variant: str) -> tuple[str, bool, str]:
    """Return (path_or_name, used_local, weights_root). If local exists, use it; else return model name to allow auto-download.
    Local dir can be overridden by env YOLO_WEIGHTS_DIR.
    """
    name = _yolo_name_for_variant(variant)
    weights_root = os.getenv("YOLO_WEIGHTS_DIR", os.path.join(os.getcwd(), "models", "weights"))
    os.makedirs(weights_root, exist_ok=True)
    local_path = os.path.join(weights_root, name)
    if os.path.isfile(local_path):
        return local_path, True, weights_root
    return name, False, weights_root


def _collect_annotated_images(project_id: int) -> Tuple[List[DBImage], Dict[int, List[DBAnn]]]:
    with get_session() as session:
        images = session.exec(select(DBImage).where(DBImage.projectId == project_id)).all()
        anns_map: Dict[int, List[DBAnn]] = {}
        for img in images:
            anns = session.exec(select(DBAnn).where(DBAnn.imageId == img.id)).all()
            if anns:
                anns_map[img.id] = anns
        imgs = [img for img in images if img.id in anns_map]
        return imgs, anns_map


def _class_index(project_id: int) -> Dict[int, int]:
    with get_session() as session:
        classes = session.exec(select(DBClass).where(DBClass.projectId == project_id).order_by(DBClass.id)).all()
        return {c.id: i for i, c in enumerate(classes)}


def _make_dataset(project_id: int, seed: int = 42) -> Tuple[str, str, Dict[str, int]]:
    imgs, anns_map = _collect_annotated_images(project_id)
    base_cwd = os.getcwd()

    # 解析每张图片的本地路径（支持 OSS URL 和本地相对路径）
    # resolve_local_path 在本地无缓存时会自动从 OSS 下载
    imgs_resolved: List[Tuple[DBImage, str]] = []
    for img in imgs:
        local = resolve_local_path(img.path)
        if local:
            imgs_resolved.append((img, local))
        else:
            print(f"[training] 跳过无法定位的图片: {img.path}")

    counts = {"images": len(imgs_resolved), "annotations": sum(len(v) for v in anns_map.values())}
    base_dir = os.path.join(base_cwd, "datasets", str(project_id))
    images_dir = os.path.join(base_dir, "images")
    labels_dir = os.path.join(base_dir, "labels")
    for split in ("train", "val"):
        _ensure_dir(os.path.join(images_dir, split))
        _ensure_dir(os.path.join(labels_dir, split))

    if counts["images"] == 0:
        raise RuntimeError("No annotated images found")

    # Split 80/20 with safeguards to keep both splits non-empty
    random.Random(seed).shuffle(imgs_resolved)
    n = len(imgs_resolved)
    if n < 2:
        raise RuntimeError("Need at least 2 annotated images with files present for train/val split")
    k = max(1, min(n - 1, int(0.8 * n)))
    train_set = set(img.id for img, _ in imgs_resolved[:k])

    cls_map = _class_index(project_id)

    def yolo_line(a: DBAnn, img_w: int, img_h: int) -> str:
        # convert (x,y,w,h) pixel -> (cx,cy,w,h) normalized
        cx = (a.x + a.w / 2.0) / img_w
        cy = (a.y + a.h / 2.0) / img_h
        nw = a.w / img_w
        nh = a.h / img_h
        cid = cls_map.get(a.classId, 0)
        return f"{cid} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"

    train_written = 0
    val_written = 0
    for img, src_path in imgs_resolved:
        split = "train" if img.id in train_set else "val"
        dst_img = os.path.join(images_dir, split, os.path.basename(src_path))
        if os.path.exists(dst_img):
            os.remove(dst_img)
        try:
            # Prefer hardlink to save space; fallback to copy
            os.link(src_path, dst_img)
        except Exception:
            shutil.copy2(src_path, dst_img)

        # Write label
        lbl_name = os.path.splitext(os.path.basename(src_path))[0] + ".txt"
        dst_lbl = os.path.join(labels_dir, split, lbl_name)
        with open(dst_lbl, "w", encoding="utf-8") as f:
            for a in anns_map[img.id]:
                f.write(yolo_line(a, img.width or 1, img.height or 1) + "\n")
        if split == "train":
            train_written += 1
        else:
            val_written += 1

    # dataset yaml
    with get_session() as session:
        classes = session.exec(
            select(DBClass).where(DBClass.projectId == project_id).order_by(DBClass.id)
        ).all()
        if not classes:
            raise RuntimeError("No classes defined for this project. Please add at least one class before training.")

        names_list = [c.name if getattr(c, "name", None) else f"class_{i}" for i, c in enumerate(classes)]

    yaml_path = os.path.join(base_dir, "dataset.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        lines = [
            f"path: {base_dir}",
            "train: images/train",
            "val: images/val",
            f"names: {names_list}",
        ]
        f.write("\n".join(lines))

    counts.update({"train": train_written, "val": val_written, "classes": len(names_list)})
    if counts["train"] == 0 or counts["val"] == 0:
        raise RuntimeError(f"Invalid split: train={counts['train']} val={counts['val']}")
    return base_dir, yaml_path, counts


def _log(log_path: str, msg: str):
    ts = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    try:
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f"[{ts}] {msg}\n")
            f.flush()  # Force write to disk immediately
    except Exception as e:
        # If logging fails, try to write to a backup log
        try:
            backup_log = log_path + '.error'
            with open(backup_log, 'a', encoding='utf-8') as f:
                f.write(f"[{ts}] ERROR logging to {log_path}: {e}\n")
                f.write(f"[{ts}] Original message: {msg}\n")
        except Exception:
            pass


def _train_with_ultralytics(yolo_model: str, data_yaml: str, out_dir: str, epochs: int, imgsz: int, batch: Optional[int], log_path: Optional[str] = None, job_id: Optional[int] = None, device: Optional[str] = None) -> Dict:
    if log_path:
        _log(log_path, f"Entering _train_with_ultralytics function")

    try:
        # Ensure Ultralytics config dir is writable (avoid ~/.config permissions)
        ultralytics_dir = os.path.join(os.getcwd(), ".ultralytics")
        os.makedirs(ultralytics_dir, exist_ok=True)
        os.environ.setdefault("YOLO_CONFIG_DIR", ultralytics_dir)
        # Skip Polars CPU feature check warning
        os.environ.setdefault("POLARS_SKIP_CPU_CHECK", "1")

        if log_path:
            _log(log_path, f"Importing YOLO from ultralytics...")

        from ultralytics import YOLO

        if log_path:
            _log(log_path, f"YOLO imported successfully")

    except Exception as e:
        if log_path:
            _log(log_path, f"Failed to import Ultralytics: {e}")
        raise RuntimeError("Ultralytics is not installed. Please install 'ultralytics' and compatible torch.") from e

    model = YOLO(yolo_model)
    if log_path:
        _log(log_path, f"YOLO model loaded: {yolo_model}")

        try:
            def on_pretrain_routine_start(trainer):
                try:
                    _log(log_path, "Pre-training routine started")
                except Exception as e:
                    _log(log_path, f"Error in on_pretrain_routine_start: {e}")

            def on_train_start(trainer):
                try:
                    _log(log_path, "Training loop started")
                    _log(log_path, f"Total epochs: {getattr(trainer, 'epochs', 'unknown')}")
                except Exception as e:
                    _log(log_path, f"Error in on_train_start: {e}")

            def on_train_epoch_start(trainer):
                try:
                    ep = getattr(trainer, 'epoch', None)
                    total = getattr(trainer, 'epochs', None)
                    _log(log_path, f"Epoch {(ep+1) if isinstance(ep, int) else '?'}/{total if total is not None else '?'} started")
                except Exception as e:
                    _log(log_path, f"Error in on_train_epoch_start: {e}")

            def on_train_epoch_end(trainer):
                try:
                    ep = getattr(trainer, 'epoch', None)
                    total = getattr(trainer, 'epochs', None)
                    # Try to get loss from different sources
                    loss_val = None
                    if hasattr(trainer, 'loss') and trainer.loss is not None:
                        try:
                            loss_val = float(trainer.loss.item() if hasattr(trainer.loss, 'item') else trainer.loss)
                        except:
                            pass
                    if loss_val is None and hasattr(trainer, 'tloss') and trainer.tloss is not None:
                        try:
                            loss_val = float(trainer.tloss.item() if hasattr(trainer.tloss, 'item') else trainer.tloss)
                        except:
                            pass

                    loss_str = f"{loss_val:.4f}" if loss_val is not None else "N/A"
                    _log(log_path, f"Epoch {(ep+1) if isinstance(ep,int) else '?'}/{total if total is not None else '?'} completed, loss={loss_str}")

                    # cancellation check
                    if job_id is not None:
                        try:
                            with get_session() as session:
                                jb = session.get(TrainingJob, job_id)
                                if jb and jb.status == 'canceled':
                                    _log(log_path, "Cancellation requested. Stopping training...")
                                    trainer.stop = True
                        except Exception as e2:
                            _log(log_path, f"Error checking cancellation: {e2}")
                except Exception as e:
                    _log(log_path, f"Error in on_train_epoch_end: {e}")

            def on_fit_epoch_end(trainer):
                try:
                    # Log validation metrics if available
                    metrics = getattr(trainer, 'metrics', None)
                    if metrics:
                        map50 = metrics.get('metrics/mAP50', None)
                        map50_95 = metrics.get('metrics/mAP50-95', None)
                        if map50 is not None or map50_95 is not None:
                            _log(log_path, f"  Validation: mAP50={map50:.4f if map50 else 'N/A'}, mAP50-95={map50_95:.4f if map50_95 else 'N/A'}")
                except Exception as e:
                    pass  # Silent fail for metrics logging

            model.add_callback('on_pretrain_routine_start', on_pretrain_routine_start)
            model.add_callback('on_train_start', on_train_start)
            model.add_callback('on_train_epoch_start', on_train_epoch_start)
            model.add_callback('on_train_epoch_end', on_train_epoch_end)
            model.add_callback('on_fit_epoch_end', on_fit_epoch_end)
            _log(log_path, "Training callbacks registered")
        except Exception as e:
            _log(log_path, f"Error setting up callbacks: {e}")
    # ``batch='auto'`` is no longer accepted by recent Ultralytics releases, expect numeric values only.
    # Convert ``None`` or text like ``'auto'`` to the supported sentinel ``0`` (auto-batch mode).
    if isinstance(batch, str):
        normalized = batch.strip().lower()
        if normalized in {"", "auto"}:
            batch = 0
        else:
            try:
                batch = float(normalized)
            except ValueError:
                batch = 0

    # Use a safe default batch size instead of auto (0) to avoid OOM during auto-batch detection
    # Auto-batch detection can consume too much VRAM and cause OOM errors
    DEFAULT_BATCH = 16

    if batch is None:
        train_batch = DEFAULT_BATCH
    elif isinstance(batch, (int, float)):
        if batch <= 0:
            train_batch = DEFAULT_BATCH
        else:
            if isinstance(batch, float) and batch.is_integer():
                train_batch = int(batch)
            else:
                train_batch = batch
    else:
        train_batch = DEFAULT_BATCH

    if log_path:
        _log(log_path, f"Starting model.train() with batch={train_batch}, device={device or 'auto'}")

    # 准备训练参数
    # Windows平台必须设置workers=0，否则多进程数据加载器会死锁
    import sys
    num_workers = 0 if sys.platform == 'win32' else 8

    train_kwargs = {
        'data': data_yaml,
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': train_batch,
        'project': out_dir,
        'name': "train",
        'exist_ok': True,
        'verbose': True,
        'workers': num_workers,
        'deterministic': False,  # Windows下deterministic模式可能导致阻塞
        'amp': False,  # 禁用混合精度，避免Windows兼容性问题
    }

    # 添加device参数（支持多GPU）
    if device:
        train_kwargs['device'] = device

    if log_path:
        _log(log_path, f"Training kwargs: {train_kwargs}")

    try:
        results = model.train(**train_kwargs)
    except Exception as train_error:
        if log_path:
            import traceback
            _log(log_path, f"Training error: {type(train_error).__name__}: {train_error}")
            _log(log_path, f"Traceback:\n{traceback.format_exc()}")
        raise

    if log_path:
        _log(log_path, "model.train() completed, extracting metrics...")
    # metrics extraction
    metrics = {}
    try:
        m = results.results_dict if hasattr(results, 'results_dict') else getattr(results, 'metrics', {})
        if isinstance(m, dict):
            metrics = {
                'map50': float(m.get('metrics/mAP50', m.get('map50', 0.0))),
                'map50_95': float(m.get('metrics/mAP50-95', m.get('map', 0.0))),
                'precision': float(m.get('metrics/precision', m.get('precision', 0.0))),
                'recall': float(m.get('metrics/recall', m.get('recall', 0.0))),
            }
    except Exception:
        pass
    # best weight path convention: runs/detect/train/weights/best.pt or out_dir/train/weights/best.pt
    best_pt = None
    for cand in [os.path.join(out_dir, "train", "weights", "best.pt"), os.path.join(out_dir, "weights", "best.pt")]:
        if os.path.isfile(cand):
            best_pt = cand
            break
    return {"metrics": metrics, "best_pt": best_pt, "model": model}


def _export_onnx(model_or_path, out_dir: str, opset: int = 12, dynamic: bool = True) -> Optional[str]:
    try:
        from ultralytics import YOLO
        # Accept either a YOLO model instance or a weights path/name
        model_obj = None
        try:
            if hasattr(model_or_path, 'export') and not isinstance(model_or_path, str):
                model_obj = model_or_path
        except Exception:
            model_obj = None
        if model_obj is None:
            model_obj = YOLO(model_or_path)

        onnx_path = model_obj.export(format='onnx', opset=opset, dynamic=dynamic)
        if isinstance(onnx_path, str) and onnx_path.endswith('.onnx'):
            return onnx_path
        # fallback: search in out_dir
        for root, _, files in os.walk(out_dir):
            for fn in files:
                if fn.endswith('.onnx'):
                    return os.path.join(root, fn)
    except Exception:
        return None
    return None


def run_training_job(job_id: int):
    # Load job and capture fields within session to avoid detached instances
    with get_session() as session:
        job = session.get(TrainingJob, job_id)
        if not job:
            return
        project_id = job.projectId
        model_variant = job.modelVariant
        epochs = job.epochs
        imgsz = job.imgsz
        batch = job.batch
        seed = job.seed
        gpu_ids_str = job.gpuIds
        job.status = 'running'
        job.startedAt = job.startedAt or __import__('datetime').datetime.utcnow()
        session.add(job)
        session.commit()

    # 解析GPU配置
    device = None
    if gpu_ids_str:
        gpu_ids = [int(x.strip()) for x in gpu_ids_str.split(',') if x.strip()]
        if len(gpu_ids) == 1:
            device = str(gpu_ids[0])
        elif len(gpu_ids) > 1:
            # 多GPU训练：YOLO支持逗号分隔的设备列表
            device = ','.join(str(i) for i in gpu_ids)

    # Prepare dataset
    out_dir = os.path.join(os.getcwd(), 'models', str(project_id), str(job_id))
    _ensure_dir(out_dir)
    log_path = os.path.join(out_dir, 'logs.txt')
    rel_logs = os.path.relpath(log_path, os.getcwd())
    with get_session() as session:
        job = session.get(TrainingJob, job_id)
        job.logsRef = rel_logs
        session.add(job)
        session.commit()
    _log(log_path, f"Job {job_id} started")
    try:
        base_dir, yaml_path, _counts = _make_dataset(project_id, seed=seed)
        _log(
            log_path,
            f"Dataset prepared at {base_dir} (train={_counts.get('train')} val={_counts.get('val')} classes={_counts.get('classes')})",
        )
    except Exception as e:
        with get_session() as session:
            job = session.get(TrainingJob, job_id)
            job.status = 'failed'
            session.add(job)
            session.commit()
        _log(log_path, f"Dataset preparation failed: {e}")
        return

    # weights resolve (offline-friendly)
    weight_path, used_local, weights_root = _resolve_weights(model_variant)

    # Train
    try:
        if not os.path.isfile(weight_path):
            _log(log_path, (
                f"Weights '{weight_path}' not found locally. Ultralytics will attempt to download. "
                f"If your environment cannot access GitHub, please manually place the weight file as '{os.path.join(weights_root, _yolo_name_for_variant(model_variant))}'."
            ))
        _log(log_path, f"Training start weights={weight_path} epochs={epochs} imgsz={imgsz} batch={batch or 'auto'} device={device or 'auto'}")
        res = _train_with_ultralytics(weight_path, yaml_path, out_dir, epochs, imgsz, batch, log_path, job_id, device)
        _log(log_path, f"Training function returned: {list(res.keys()) if res else 'None'}")
        metrics = res.get('metrics', {})
        best_pt = res.get('best_pt')
        mdl = res.get('model')
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        with get_session() as session:
            job = session.get(TrainingJob, job_id)
            job.status = 'failed'
            job.finishedAt = __import__('datetime').datetime.utcnow()
            session.add(job)
            session.commit()
        _log(log_path, f"Training failed with exception: {type(e).__name__}: {e}")
        _log(log_path, f"Traceback:\n{tb}")
        return

    # Save metrics and artifact PT
    with get_session() as session:
        job = session.get(TrainingJob, job_id)
        job.map50 = metrics.get('map50')
        job.map50_95 = metrics.get('map50_95')
        job.precision = metrics.get('precision')
        job.recall = metrics.get('recall')
        session.add(job)
        session.commit()

    if best_pt and os.path.isfile(best_pt):
        try:
            size = os.path.getsize(best_pt)
        except Exception:
            size = None
        rel = os.path.relpath(best_pt, os.getcwd())
        artifact_path = rel
        with get_session() as session:
            session.add(ModelArtifact(trainingJobId=job_id, format='pt', path=artifact_path, size=size))
            session.commit()
        _log(log_path, f"Saved best weights {artifact_path}")

    # Export ONNX (prefer best weights path when available)
    onnx_path = _export_onnx(best_pt or mdl, out_dir, 12, True)
    if onnx_path and os.path.isfile(onnx_path):
        try:
            size = os.path.getsize(onnx_path)
        except Exception:
            size = None
        rel = os.path.relpath(onnx_path, os.getcwd())
        artifact_path = rel
        with get_session() as session:
            session.add(ModelArtifact(trainingJobId=job_id, format='onnx', path=artifact_path, size=size))
            session.commit()
        _log(log_path, f"Exported ONNX {artifact_path}")

    with get_session() as session:
        job = session.get(TrainingJob, job_id)
        if job and job.status == 'canceled':
            job.finishedAt = __import__('datetime').datetime.utcnow()
            session.add(job)
            session.commit()
            _log(log_path, "Job canceled")
            return
        if job:
            job.status = 'succeeded'
            job.finishedAt = __import__('datetime').datetime.utcnow()
            session.add(job)
            session.commit()
    _log(log_path, "Job succeeded")


def start_training_async(job_id: int):
    """Start training in a completely separate process to avoid blocking the main server"""
    import subprocess
    import sys

    from ..db import DB_PATH

    # Get the Python executable from current environment
    python_exe = sys.executable

    # Path to the standalone training script
    backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    script_path = os.path.join(backend_dir, "run_training.py")

    # Redirect stdout/stderr to log file for debugging
    log_dir = os.path.join(backend_dir, "models", "training_logs")
    os.makedirs(log_dir, exist_ok=True)
    process_log = os.path.join(log_dir, f"process_{job_id}.log")

    # Start as a completely independent process
    creationflags = 0
    if sys.platform == 'win32':
        # CREATE_NO_WINDOW prevents console window from appearing
        creationflags = subprocess.CREATE_NO_WINDOW

    # Set environment variable to ensure subprocess uses same database
    env = os.environ.copy()
    env['APP_DB_PATH'] = str(DB_PATH)
    # 跳过Polars CPU检查，避免兼容性问题
    env['POLARS_SKIP_CPU_CHECK'] = '1'
    # 禁用Python输出缓冲，避免I/O阻塞
    env['PYTHONUNBUFFERED'] = '1'

    # Open log file for the process (keep it open for subprocess)
    log_f = open(process_log, 'w', encoding='utf-8', buffering=1)  # Line buffering
    log_f.write(f"Starting training job {job_id} at {__import__('datetime').datetime.now()}\n")
    log_f.write(f"Database: sqlite:///{DB_PATH}\n")
    log_f.write(f"Working directory: {backend_dir}\n")
    log_f.flush()

    process = subprocess.Popen(
        [python_exe, script_path, str(job_id)],
        stdout=log_f,
        stderr=subprocess.STDOUT,  # Redirect stderr to stdout
        creationflags=creationflags,
        cwd=backend_dir,
        env=env,
        close_fds=False  # Keep file descriptor open
    )

    return process



