from __future__ import annotations

from pathlib import Path
from typing import Any
from airflow.utils.context import Context
from airflow_provider_mlcore.operators.mlc import (  # type: ignore[import]
    MLCoreJobSubmitOperator as _JobSubmitBase,
    CreateJobParams,
    CreateJobRequest,
)
from typing_extensions import override


class JobSubmitDictOperator(_JobSubmitBase):
    """
    Thin wrapper around the corporate job submit operator that accepts job environment
    variables as a dictionary, along with separate job_path and preset_file arguments,
    then constructs CreateJobParams internally.
    """

    template_fields = (
        "dp_conn_id",
        "preset_file",
        "_create_job_params",
        "job_env_dict",
    )

    def __init__(
        self,
        *args,
        runner_job_path: str | Path,
        runner_preset_file: str,
        job_env_dict: dict[str, Any],
        base_env: dict[str, Any] = None,
        **kwargs,
    ) -> None:
        env = job_env_dict.get("env", {})
        if base_env:
            env.update(base_env)

        self.job_env_dict = env

        create_job_request = CreateJobRequest(
            preset_file=runner_preset_file,
            env=env,
        )

        job_path = Path(runner_job_path)

        create_job_params_obj = CreateJobParams(
            args=create_job_request,
            job_path=job_path,
            print_func=lambda x: print(x, end=""),
        )

        super().__init__(*args, create_job_params=create_job_params_obj, **kwargs)

    @override
    def execute(self, context: Context) -> Any:
        self.render_template_fields(context)
        self._create_job_params.args.env.update(self.job_env_dict)
        return super().execute(context=context)


class JobSubmitOperatorEnv(_JobSubmitBase):
    template_fields = (
        "dp_conn_id",
        "preset_file",
        "_create_job_params",
        "template_env",
    )

    def __init__(
        self,
        *args,
        create_job_params,
        **kwargs,
    ) -> None:
        self.template_env = create_job_params.args.env
        create_job_params.args.env = {}
        super().__init__(*args, create_job_params=create_job_params, **kwargs)

    @override
    def execute(self, context: Context) -> Any:
        self.render_template_fields(context)
        self._create_job_params.args.env.update(self.template_env)
        return super().execute(context=context)
