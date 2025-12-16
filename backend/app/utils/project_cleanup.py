import os
import shutil


def remove_project_files(project_id: int):
    base = os.getcwd()
    ds_dir = os.path.join(base, "datasets", str(project_id))
    if os.path.isdir(ds_dir):
        shutil.rmtree(ds_dir, ignore_errors=True)
    models_dir = os.path.join(base, "models", str(project_id))
    if os.path.isdir(models_dir):
        shutil.rmtree(models_dir, ignore_errors=True)

