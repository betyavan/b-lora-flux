from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

import boto3
from airflow import DAG
from airflow.decorators import task
from airflow.models.param import Param

from airflow_provider_mlcore.operators.mlc import CreateJobParams, CreateJobRequest  # type: ignore[import]

from dags.plugins.job_runner_wrapper import JobSubmitOperatorEnv

JOBS_DIR = Path(__file__).parent / "jobs"

_S3_BASE_PATH_DEFAULT = os.environ.get("CORP_S3_BASE_PATH", "")
_S3_ENDPOINT = os.environ.get("CORP_S3_ENDPOINT", "")
_S3_BUCKET_DATA = os.environ.get("CORP_S3_BUCKET_DATA", "")
_AIRFLOW_CONN_ID = os.environ.get("CORP_AIRFLOW_CONN_ID", "dp_conn")

with DAG(
    "blora_flux_pipeline",
    default_args={
        "depends_on_past": False,
        "email_on_failure": False,
        "email_on_retry": False,
        "retries": 1,
        "retry_delay": timedelta(hours=1),
    },
    description="B-LoRA FLUX pipeline: train → generate → metrics → update plan",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    tags=["blora", "flux"],
    template_searchpath="dags_folder/dags",
    params={
        "EXPERIMENT_NAME": Param(
            default="e01_blora_flux_van_gogh_img1",
            type="string",
            description="Experiment config name (must match configs/experiments/*.yaml)",
        ),
        "S3_BASE_PATH": Param(
            default=_S3_BASE_PATH_DEFAULT,
            type="string",
            description="Base S3 path for all experiment data",
        ),
    },
) as dag:

    @task
    def prepare_paths(**context) -> dict[str, str]:
        """Build all S3 paths used by downstream jobs."""
        params = context["params"]
        base = params["S3_BASE_PATH"].rstrip("/")
        exp = params["EXPERIMENT_NAME"]
        return {
            "EXPERIMENT_NAME": exp,
            "TRAIN_OUTPUT_S3_PATH": f"{base}/exp_logs/{exp}/loras",
            "GENERATED_OUTPUT_S3_PATH": f"{base}/exp_logs/{exp}/generated",
            "METRICS_OUTPUT_S3_PATH": f"{base}/exp_logs/{exp}/metrics",
        }

    @task
    def prepare_train_env(paths, **context) -> dict[str, str]:
        return {
            "EXPERIMENT_NAME": paths["EXPERIMENT_NAME"],
            "TRAIN_OUTPUT_S3_PATH": paths["TRAIN_OUTPUT_S3_PATH"],
        }

    @task
    def prepare_generate_env(paths, **context) -> dict[str, str]:
        return {
            "EXPERIMENT_NAME": paths["EXPERIMENT_NAME"],
            "TRAIN_OUTPUT_S3_PATH": paths["TRAIN_OUTPUT_S3_PATH"],
            "GENERATED_OUTPUT_S3_PATH": paths["GENERATED_OUTPUT_S3_PATH"],
        }

    @task
    def prepare_metrics_env(paths, **context) -> dict[str, str]:
        return {
            "EXPERIMENT_NAME": paths["EXPERIMENT_NAME"],
            "GENERATED_OUTPUT_S3_PATH": paths["GENERATED_OUTPUT_S3_PATH"],
            "METRICS_OUTPUT_S3_PATH": paths["METRICS_OUTPUT_S3_PATH"],
        }

    @task
    def update_plan(**context):
        """Download metrics.json from S3 and update experiments/plan.md."""
        import subprocess

        params = context["params"]
        paths = context["ti"].xcom_pull(task_ids="prepare_paths")
        exp = params["EXPERIMENT_NAME"]

        bucket = _S3_BUCKET_DATA
        from urllib.parse import urlparse
        parsed = urlparse(paths["METRICS_OUTPUT_S3_PATH"])
        key = parsed.path.lstrip("/") + "/metrics.json"
        local_path = Path(f"output/results/{exp}.json")
        local_path.parent.mkdir(parents=True, exist_ok=True)

        s3 = boto3.client("s3", endpoint_url=_S3_ENDPOINT)
        s3.download_file(bucket, key, str(local_path))

        result = subprocess.run(
            ["poetry", "run", "python", "scripts/update_exp_plan.py"],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            import logging as _log
            _log.getLogger(__name__).error("update_exp_plan.py failed:\n%s", result.stderr)
            raise RuntimeError(f"update_exp_plan.py exited with code {result.returncode}")

    paths_dict = prepare_paths()
    train_env = prepare_train_env(paths=paths_dict)
    generate_env = prepare_generate_env(paths=paths_dict)
    metrics_env = prepare_metrics_env(paths=paths_dict)

    train_task = JobSubmitOperatorEnv(
        task_id="remote-train-blora",
        dp_conn_id=_AIRFLOW_CONN_ID,
        autosensor=True,
        create_job_params=CreateJobParams(
            job_path=JOBS_DIR / "train_blora",
            print_func=lambda x: print(x, end=""),
            args=CreateJobRequest(
                preset_file="train_blora_preset.yml",
                env=train_env,
            ),
        ),
        executor_config={},
    )

    generate_task = JobSubmitOperatorEnv(
        task_id="remote-generate-blora",
        dp_conn_id=_AIRFLOW_CONN_ID,
        autosensor=True,
        create_job_params=CreateJobParams(
            job_path=JOBS_DIR / "generate_blora",
            print_func=lambda x: print(x, end=""),
            args=CreateJobRequest(
                preset_file="generate_blora_preset.yml",
                env=generate_env,
            ),
        ),
        executor_config={},
    )

    metrics_task = JobSubmitOperatorEnv(
        task_id="remote-metrics-blora",
        dp_conn_id=_AIRFLOW_CONN_ID,
        autosensor=True,
        create_job_params=CreateJobParams(
            job_path=JOBS_DIR / "metrics_blora",
            print_func=lambda x: print(x, end=""),
            args=CreateJobRequest(
                preset_file="metrics_blora_preset.yml",
                env=metrics_env,
            ),
        ),
        executor_config={},
    )

    update_plan_task = update_plan()

    (
        paths_dict
        >> train_env >> train_task
        >> generate_env >> generate_task
        >> metrics_env >> metrics_task
        >> update_plan_task
    )
