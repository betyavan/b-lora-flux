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
    # ---- Legacy single-stream ablations (early prototype, pre-plan) ----
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

    # ---- Phase 1 — double-stream diagnostic ----
    "diag_d": [
        "e00_no_lora_baseline",
        "d01_double_stream_1000steps",
        "d02_double_stream_2000steps",
        "d03_double_stream_rank32",
    ],

    # ---- Phase 1b — Θ_content search ----
    "phase_1b": [
        "dc_content_early_ds",
        "dc_content_late_ds",
        "dc_content_from_p0",
    ],

    # ---- Phase 2 — ablations (plan-aligned naming) ----
    "ablation_da": [
        "da01_ds_blocks_0_6",
        "da02_ds_blocks_0_12",
        "da03_ds_blocks_0_18",
        "da04_ds_blocks_6_18",
    ],
    "ablation_db": [
        "db01_rank_4",
        "db02_rank_16",
        "db03_rank_32",
        "db04_rank_64",
    ],
    "ablation_dc": [
        "dc01_steps_500",
        "dc02_steps_1000",
        "dc03_steps_2000",
        "dc04_steps_4000",
    ],
    "ablation_dp": [
        "dp01_prompt_sks",
        "dp02_prompt_sks_class",
        "dp03_prompt_v",
        "dp04_prompt_v_class",
    ],

    # ---- Phase 3 — alpha ablation (inference-only, process: []) ----
    # generate step uses LORA_SCALE from GROUP_LORA_SCALES below; no training.
    "phase3_alpha": [
        "g01_alpha_0.3",
        "g02_alpha_0.5",
        "g03_alpha_0.7",
        "g04_alpha_1.0",
        "g05_alpha_1.5",
        "g06_alpha_2.0",
    ],

    # ---- Phase 4 — method comparison ----
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
    "compare_e03": [
        "e03_splitflux_van_gogh_img1",
        "e03_splitflux_van_gogh_img2",
        "e03_splitflux_van_gogh_img3",
        "e03_splitflux_van_gogh_img4",
        "e03_splitflux_monet_img1",
        "e03_splitflux_monet_img2",
        "e03_splitflux_monet_img3",
        "e03_splitflux_monet_img4",
    ],

    # ---- Phase 5 — SDXL cross-architecture ----
    "compare_f": [
        "e04_blora_sdxl_van_gogh_img1",
        "e04_blora_sdxl_van_gogh_img2",
        "e04_blora_sdxl_van_gogh_img3",
        "e04_blora_sdxl_van_gogh_img4",
    ],
}

# Per-experiment LORA_SCALE overrides for inference-only groups (Phase 3).
# The generate step reads LORA_SCALE env var (default 1.0); entries here override it.
GROUP_LORA_SCALES: dict[str, float] = {
    "g01_alpha_0.3": 0.3,
    "g02_alpha_0.5": 0.5,
    "g03_alpha_0.7": 0.7,
    "g04_alpha_1.0": 1.0,
    "g05_alpha_1.5": 1.5,
    "g06_alpha_2.0": 2.0,
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
            default="diag_d",
            enum=[
                # Phase 1
                "diag_d",
                # Phase 1b
                "phase_1b",
                # Phase 2 (plan-aligned)
                "ablation_da", "ablation_db", "ablation_dc", "ablation_dp",
                # Phase 3
                "phase3_alpha",
                # Phase 4
                "compare_e", "compare_e03",
                # Phase 5
                "compare_f",
                # Legacy proto ablations
                "ablation_a", "ablation_b", "ablation_c",
            ],
            description="Experiment group to run",
        ),
        "PHASE3_BASE_EXP": Param(
            default="",
            type="string",
            description="(Phase 3 only) Experiment name whose trained LoRA is reused for all alpha variants",
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
        group = params["GROUP"]
        phase3_base = params.get("PHASE3_BASE_EXP", "").strip()

        env_dicts = []
        for exp in GROUP_EXPERIMENTS[group]:
            env: dict[str, str] = {
                "EXPERIMENT_NAME": exp,
                "GENERATED_OUTPUT_S3_PATH": f"{base}/exp_logs/{run_ts}/{exp}/generated",
            }

            # Phase 3: reuse trained LoRA from a base experiment; vary LORA_SCALE per variant.
            if exp in GROUP_LORA_SCALES:
                env["LORA_SCALE"] = str(GROUP_LORA_SCALES[exp])
                if phase3_base:
                    env["TRAIN_OUTPUT_S3_PATH"] = f"{base}/exp_logs/{run_ts}/{phase3_base}/loras"
                else:
                    env["TRAIN_OUTPUT_S3_PATH"] = f"{base}/exp_logs/{run_ts}/{exp}/loras"
            else:
                env["TRAIN_OUTPUT_S3_PATH"] = f"{base}/exp_logs/{run_ts}/{exp}/loras"

            env_dicts.append({"env": env})
        return env_dicts

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
