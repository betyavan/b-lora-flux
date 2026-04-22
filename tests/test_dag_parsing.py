"""
Tests: Airflow DAGs are structurally correct (no Airflow install required).

Covers:
- Python syntax of all DAG files
- GROUP_EXPERIMENTS has expected keys and all configs exist on disk
- Correct task IDs and operator names in source
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

ROOT = Path(__file__).parent.parent
CONFIGS_DIR = ROOT / "configs" / "experiments"
DAGS_DIR = ROOT / "dags"

pytestmark = pytest.mark.fast

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_airflow() -> None:
    """Inject mock modules so DAG files can be imported without Airflow."""
    mocks = [
        "airflow",
        "airflow.decorators",
        "airflow.models",
        "airflow.models.param",
        "airflow.utils",
        "airflow.utils.context",
        "airflow_provider_mlcore",
        "airflow_provider_mlcore.operators",
        "airflow_provider_mlcore.operators.mlc",
        "typing_extensions",
        "boto3",
    ]
    for mod in mocks:
        if mod not in sys.modules:
            sys.modules[mod] = MagicMock()


def _load_group_experiments() -> dict:
    """Extract GROUP_EXPERIMENTS from AST — no Airflow or mock needed."""
    src = (DAGS_DIR / "blora_flux_group_pipeline.py").read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "GROUP_EXPERIMENTS":
                    return ast.literal_eval(node.value)
    raise RuntimeError("GROUP_EXPERIMENTS not found in source")


# ---------------------------------------------------------------------------
# Syntax tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rel_path", [
    "dags/blora_flux_pipeline.py",
    "dags/blora_flux_group_pipeline.py",
    "dags/plugins/job_runner_wrapper.py",
])
def test_dag_files_valid_syntax(rel_path: str) -> None:
    path = ROOT / rel_path
    ast.parse(path.read_text(), filename=rel_path)


# ---------------------------------------------------------------------------
# GROUP_EXPERIMENTS structure
# ---------------------------------------------------------------------------

def test_group_experiments_has_expected_keys() -> None:
    ge = _load_group_experiments()
    expected = {"ablation_a", "ablation_b", "ablation_c", "compare_e", "compare_f"}
    assert expected == set(ge.keys())


def test_group_experiments_correct_counts() -> None:
    ge = _load_group_experiments()
    assert len(ge["ablation_a"]) == 4
    assert len(ge["ablation_b"]) == 4
    assert len(ge["ablation_c"]) == 4
    assert len(ge["compare_e"]) == 16  # e01+e02 × van_gogh+monet × img1-4
    assert len(ge["compare_f"]) == 4


def test_all_group_experiment_configs_exist() -> None:
    ge = _load_group_experiments()
    missing = []
    for group, exps in ge.items():
        for exp in exps:
            cfg = CONFIGS_DIR / f"{exp}.yaml"
            if not cfg.exists():
                missing.append(f"{group}: {exp}")
    assert not missing, f"Missing config files:\n" + "\n".join(missing)


# ---------------------------------------------------------------------------
# Config filename ↔ config.name consistency
# ---------------------------------------------------------------------------

def test_experiment_config_names_match_filenames() -> None:
    mismatches = []
    for path in CONFIGS_DIR.glob("*.yaml"):
        if path.stem == "base_flux_lora":
            continue
        cfg = yaml.safe_load(path.read_text())
        name_in_file = cfg.get("config", {}).get("name", "")
        if name_in_file != path.stem:
            mismatches.append(f"{path.name}: config.name='{name_in_file}'")
    assert not mismatches, "config.name ≠ filename:\n" + "\n".join(mismatches)


# ---------------------------------------------------------------------------
# Task IDs and operator names in source text
# ---------------------------------------------------------------------------

def test_single_pipeline_task_ids_and_operators() -> None:
    src = (DAGS_DIR / "blora_flux_pipeline.py").read_text()
    assert '"remote-train-blora"' in src
    assert '"remote-generate-blora"' in src
    assert '"remote-metrics-blora"' in src
    assert "JobSubmitOperatorEnv" in src
    assert "job_runner_wrapper" in src


def test_group_pipeline_task_ids_and_operators() -> None:
    src = (DAGS_DIR / "blora_flux_group_pipeline.py").read_text()
    assert '"remote-train-blora-group"' in src
    assert '"remote-generate-blora-group"' in src
    assert '"remote-metrics-blora-group"' in src
    assert "JobSubmitDictOperator" in src
    assert "runner_job_path" in src
    assert "runner_preset_file" in src


def test_pipeline_has_no_old_mlc_task_ids() -> None:
    for fname in ("blora_flux_pipeline.py", "blora_flux_group_pipeline.py"):
        src = (DAGS_DIR / fname).read_text()
        assert '"mlc-train' not in src, f"{fname} still has old mlc- task ID"
        assert '"mlc-generate' not in src
        assert '"mlc-metrics' not in src
        assert "ml_core_wrapper" not in src


def test_dag_has_autosensor() -> None:
    for fname in ("blora_flux_pipeline.py", "blora_flux_group_pipeline.py"):
        src = (DAGS_DIR / fname).read_text()
        assert "autosensor=True" in src, f"{fname}: autosensor=True missing"
