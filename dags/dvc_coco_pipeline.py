from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator


REPO_DIR = os.environ.get("PIPELINE_REPO_DIR", "/opt/airflow/repo")

DEFAULT_ARGS = {
    "owner": "mlops",
    "depends_on_past": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="dvc_coco_pipeline",
    description="COCO2017 data pipeline orchestrated by Airflow running DVC stages",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2026, 1, 1),
    schedule_interval=None,  # manual trigger
    catchup=False,
    tags=["dvc", "coco", "pipeline"],
    max_active_runs=1,
    max_active_tasks=1,
) as dag:

    # Helper: run inside repo dir
    def dvc_stage(stage_name: str) -> BashOperator:
        return BashOperator(
            task_id=f"dvc_{stage_name}",
            bash_command=f"cd {REPO_DIR} && dvc repro {stage_name}",
        )

    download_val_and_ann = dvc_stage("download_val_and_ann")
    extract_val_and_ann = dvc_stage("extract_val_and_ann")

    download_train = dvc_stage("download_train")
    extract_train = dvc_stage("extract_train")

    coco_to_yolo = dvc_stage("coco_to_yolo")
    preprocess_images_link = dvc_stage("preprocess_images_link")
    splits = dvc_stage("splits")
    reports = dvc_stage("reports")

    # Dependencies
    download_val_and_ann >> extract_val_and_ann
    download_train >> extract_train

    [extract_val_and_ann, extract_train] >> coco_to_yolo
    coco_to_yolo >> preprocess_images_link >> splits >> reports
