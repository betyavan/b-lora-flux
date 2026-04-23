from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.decorators import task
from airflow.models.param import Param

from dags.plugins.job_runner_wrapper import JobSubmitDictOperator

JOBS_DIR = Path(__file__).parent / "jobs"

_S3_BASE_PATH_DEFAULT = os.environ.get("CORP_S3_BASE_PATH", "")
_AIRFLOW_CONN_ID = os.environ.get("CORP_AIRFLOW_CONN_ID", "dp_conn")

GROUP_EXPERIMENTS = {
    "ablation_a": [
        "a01_blocks_34_37",
        "a02_blocks_30_37",
        "a03_blocks_24_37",
        "a04_blocks_19_37",
    ],
    "ablation_b": [
        "b01_rank_4",
        "b02_rank_8",
        "b03_rank_16",
        "b04_rank_32",
    ],
    "ablation_c": [
        "c01_steps_100",
        "c02_steps_250",
        "c03_steps_500",
        "c04_steps_1000",
    ],
    "compare_e": [
        "e01_blora_flux_van_gogh_img1",
        "e01_blora_flux_van_gogh_img2",
        "e01_blora_flux_van_gogh_img3",
        "e01_blora_flux_van_gogh_img4",
        "e02_full_lora_flux_van_gogh_img1",
        "e02_full_lora_flux_van_gogh_img2",
        "e02_full_lora_flux_van_gogh_img3",
        "e02_full_lora_flux_van_gogh_img4",
        "e01_blora_flux_monet_img1",
        "e01_blora_flux_monet_img2",
        "e01_blora_flux_monet_img3",
        "e01_blora_flux_monet_img4",
        "e02_full_lora_flux_monet_img1",
        "e02_full_lora_flux_monet_img2",
        "e02_full_lora_flux_monet_img3",
        "e02_full_lora_flux_monet_img4",
    ],
    "compare_f": [
        "e04_blora_sdxl_van_gogh_img1",
        "e04_blora_sdxl_van_gogh_img2",
        "e04_blora_sdxl_van_gogh_img3",
        "e04_blora_sdxl_van_gogh_img4",
    ],
    "diag_d": [
        "e00_no_lora_baseline",
        "d01_double_stream_1000steps",
        "d02_double_stream_2000steps",
        "d03_double_stream_rank32",
    ],
}


with DAG(
    "blora_flux_group_pipeline",
    default_args={
        "depends_on_past": False,
        "email_on_failure": False,
        "email_on_retry": False,
        "retries": 1,
        "retry_delay": timedelta(hours=1),
    },
    description="B-LoRA FLUX group pipeline: parallel train → parallel generate → parallel metrics",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    tags=["blora", "flux"],
    template_searchpath="dags_folder/dags",
    params={
        "GROUP": Param(
            default="ablation_a",
            enum=["ablation_a", "ablation_b", "ablation_c", "compare_e", "compare_f", "diag_d"],
            description="Experiment group to run",
        ),
        "S3_BASE_PATH": Param(
            default=_S3_BASE_PATH_DEFAULT,
            type="string",
            description="Base S3 path for all experiment data",
        ),
    },
) as dag:

    @task
    def prepare_train_env_dicts(**context):
        params = context["params"]
        base = params["S3_BASE_PATH"].rstrip("/")
        run_ts = context["logical_date"].strftime("%Y%m%dT%H%M%S")
        return [
            {"env": {
                "EXPERIMENT_NAME": exp,
                "TRAIN_OUTPUT_S3_PATH": f"{base}/exp_logs/{run_ts}/{exp}/loras",
            }}
            for exp in GROUP_EXPERIMENTS[params["GROUP"]]
        ]

    @task
    def prepare_generate_env_dicts(**context):
        params = context["params"]
        base = params["S3_BASE_PATH"].rstrip("/")
        run_ts = context["logical_date"].strftime("%Y%m%dT%H%M%S")
        return [
            {"env": {
                "EXPERIMENT_NAME": exp,
                "TRAIN_OUTPUT_S3_PATH": f"{base}/exp_logs/{run_ts}/{exp}/loras",
                "GENERATED_OUTPUT_S3_PATH": f"{base}/exp_logs/{run_ts}/{exp}/generated",
            }}
            for exp in GROUP_EXPERIMENTS[params["GROUP"]]
        ]

    @task
    def prepare_metrics_env_dicts(**context):
        params = context["params"]
        base = params["S3_BASE_PATH"].rstrip("/")
        run_ts = context["logical_date"].strftime("%Y%m%dT%H%M%S")
        return [
            {"env": {
                "EXPERIMENT_NAME": exp,
                "GENERATED_OUTPUT_S3_PATH": f"{base}/exp_logs/{run_ts}/{exp}/generated",
                "METRICS_OUTPUT_S3_PATH": f"{base}/exp_logs/{run_ts}/{exp}/metrics",
            }}
            for exp in GROUP_EXPERIMENTS[params["GROUP"]]
        ]

    train_env_dicts = prepare_train_env_dicts()
    generate_env_dicts = prepare_generate_env_dicts()
    metrics_env_dicts = prepare_metrics_env_dicts()

    train_all = JobSubmitDictOperator.partial(
        task_id="remote-train-blora-group",
        dp_conn_id=_AIRFLOW_CONN_ID,
        autosensor=True,
        runner_job_path=str(JOBS_DIR / "train_blora"),
        runner_preset_file="train_blora_preset.yml",
        executor_config={},
        base_env={},
    )

    generate_all = JobSubmitDictOperator.partial(
        task_id="remote-generate-blora-group",
        dp_conn_id=_AIRFLOW_CONN_ID,
        autosensor=False,
        runner_job_path=str(JOBS_DIR / "generate_blora"),
        runner_preset_file="generate_blora_preset.yml",
        executor_config={},
        base_env={},
    )

    metrics_all = JobSubmitDictOperator.partial(
        task_id="remote-metrics-blora-group",
        dp_conn_id=_AIRFLOW_CONN_ID,
        autosensor=False,
        runner_job_path=str(JOBS_DIR / "metrics_blora"),
        runner_preset_file="metrics_blora_preset.yml",
        executor_config={},
        base_env={},
    )

    (
        train_env_dicts
        >> train_all.expand(job_env_dict=train_env_dicts)
        >> generate_env_dicts
        >> generate_all.expand(job_env_dict=generate_env_dicts)
        >> metrics_env_dicts
        >> metrics_all.expand(job_env_dict=metrics_env_dicts)
    )
