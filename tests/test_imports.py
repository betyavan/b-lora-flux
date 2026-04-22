"""
Tests: Python syntax, non-ML imports, YAML configs, and infra templates.

Covers:
- All .py files parse without syntax errors
- Core non-ML modules import cleanly
- All 32 experiment configs are valid YAML with required fields
- infra.env.template has all required keys
- Preset templates use CORP_RUNNER_* (not CORP_MLC_*)
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parent.parent
CONFIGS_DIR = ROOT / "configs" / "experiments"

pytestmark = pytest.mark.fast

# ---------------------------------------------------------------------------
# Syntax: all .py files
# ---------------------------------------------------------------------------

def _collect_py_files() -> list[Path]:
    skip = {".venv", "src", "__pycache__", ".git", ".mypy_cache", ".pytest_cache"}
    files = []
    for p in ROOT.rglob("*.py"):
        if any(part in skip for part in p.parts):
            continue
        files.append(p)
    return files


@pytest.mark.parametrize("py_file", _collect_py_files(), ids=lambda p: str(p.relative_to(ROOT)))
def test_python_file_valid_syntax(py_file: Path) -> None:
    ast.parse(py_file.read_text(), filename=str(py_file))


# ---------------------------------------------------------------------------
# Non-ML imports
# ---------------------------------------------------------------------------

def test_update_exp_plan_imports() -> None:
    from scripts.update_exp_plan import (  # noqa: F401
        ExpResult,
        _update_table_row,
        _update_plan,
        PLAN_PATH,
        EXPERIMENT_IDS,
    )


def test_experiment_ids_count() -> None:
    from scripts.update_exp_plan import EXPERIMENT_IDS
    assert len(EXPERIMENT_IDS) == 32, f"Expected 32 experiment IDs, got {len(EXPERIMENT_IDS)}"


# ---------------------------------------------------------------------------
# Experiment YAML configs
# ---------------------------------------------------------------------------

def _experiment_configs() -> list[Path]:
    return [p for p in CONFIGS_DIR.glob("*.yaml") if p.stem != "base_flux_lora"]


@pytest.mark.parametrize("cfg_path", _experiment_configs(), ids=lambda p: p.stem)
def test_experiment_config_required_fields(cfg_path: Path) -> None:
    cfg = yaml.safe_load(cfg_path.read_text())
    assert cfg.get("job") is not None, "missing 'job' key"
    assert cfg["config"]["name"], "config.name is empty"
    process = cfg["config"]["process"]
    assert isinstance(process, list) and len(process) >= 1
    datasets = process[0].get("datasets", [])
    assert datasets, "no datasets defined"
    assert datasets[0].get("folder_path"), "folder_path is empty"


def test_experiment_configs_total_count() -> None:
    configs = _experiment_configs()
    assert len(configs) == 32, f"Expected 32 configs, got {len(configs)}"


def test_ablation_configs_have_block_constraints() -> None:
    """A/B/C ablations are all B-LoRA variants — must restrict to blocks 30-37."""
    for pattern in ("a0*.yaml", "b0*.yaml", "c0*.yaml"):
        for path in CONFIGS_DIR.glob(pattern):
            cfg = yaml.safe_load(path.read_text())
            kwargs = cfg["config"]["process"][0]["network"].get("network_kwargs", {})
            assert "only_if_contains" in kwargs, \
                f"{path.name}: B-LoRA ablation must have only_if_contains"


def test_e01_configs_have_block_selection() -> None:
    """B-LoRA configs (e01) must select single_transformer_blocks 30-37."""
    for path in CONFIGS_DIR.glob("e01_*.yaml"):
        cfg = yaml.safe_load(path.read_text())
        kwargs = cfg["config"]["process"][0]["network"].get("network_kwargs", {})
        blocks = kwargs.get("only_if_contains", [])
        assert any("single_transformer_blocks.30" in b for b in blocks), \
            f"{path.name}: B-LoRA block selection missing"


def test_e02_configs_have_no_block_selection() -> None:
    """Full-LoRA configs (e02) must NOT have only_if_contains (train all blocks)."""
    for path in CONFIGS_DIR.glob("e02_*.yaml"):
        cfg = yaml.safe_load(path.read_text())
        kwargs = cfg["config"]["process"][0]["network"].get("network_kwargs", {})
        assert "only_if_contains" not in kwargs, \
            f"{path.name}: Full-LoRA must not restrict blocks"


def test_monet_configs_reference_monet_data() -> None:
    for path in CONFIGS_DIR.glob("*monet*.yaml"):
        cfg = yaml.safe_load(path.read_text())
        folder = cfg["config"]["process"][0]["datasets"][0]["folder_path"]
        assert "monet" in folder, f"{path.name}: folder_path should reference monet data"


def test_van_gogh_configs_reference_van_gogh_data() -> None:
    for path in CONFIGS_DIR.glob("*van_gogh*.yaml"):
        cfg = yaml.safe_load(path.read_text())
        folder = cfg["config"]["process"][0]["datasets"][0]["folder_path"]
        assert "van_gogh" in folder, f"{path.name}: folder_path should reference van_gogh data"


# ---------------------------------------------------------------------------
# infra.env.template
# ---------------------------------------------------------------------------

REQUIRED_INFRA_KEYS = [
    "CORP_S3_BUCKET_DATA",
    "CORP_S3_BUCKET_MODELS",
    "CORP_S3_DATA_PATH",
    "CORP_S3_BASE_PATH",
    "CORP_VAULT_PATH",
    "CORP_DOCKER_IMAGE",
    "CORP_PIP_INDEX_URL",
    "CORP_S3_ENDPOINT",
    "CORP_RUNNER_PROJECT",
    "CORP_RUNNER_REGION",
    "CORP_GITHUB_REPO",
    "CORP_AIRFLOW_CONN_ID",
    "CORP_SDXL_MODEL_SRC",
    "CORP_SDXL_MODEL_VERSION",
]


def test_infra_env_template_has_required_keys() -> None:
    text = (ROOT / "infra.env.template").read_text()
    missing = [k for k in REQUIRED_INFRA_KEYS if k not in text]
    assert not missing, f"infra.env.template missing keys: {missing}"


def test_infra_env_template_no_real_values() -> None:
    """Template must have empty values (no real secrets committed)."""
    for line in (ROOT / "infra.env.template").read_text().splitlines():
        if line.startswith("#") or not line.strip() or "=" not in line:
            continue
        key, _, value = line.partition("=")
        # Allow a few known safe defaults
        safe_defaults = {"dp_conn", "stabilityai/stable-diffusion-xl-base-1.0"}
        if value.strip() and value.strip() not in safe_defaults:
            pytest.fail(f"infra.env.template has non-empty value for {key.strip()}: '{value.strip()[:30]}...'")


# ---------------------------------------------------------------------------
# Preset templates use CORP_RUNNER_* (not CORP_MLC_*)
# ---------------------------------------------------------------------------

PRESET_TEMPLATES = [
    "dags/jobs/train_blora/train_blora_preset.yml.template",
    "dags/jobs/generate_blora/generate_blora_preset.yml.template",
    "dags/jobs/metrics_blora/metrics_blora_preset.yml.template",
]


@pytest.mark.parametrize("rel_path", PRESET_TEMPLATES)
def test_preset_template_uses_runner_vars(rel_path: str) -> None:
    text = (ROOT / rel_path).read_text()
    assert "${CORP_RUNNER_REGION}" in text, f"{rel_path}: missing CORP_RUNNER_REGION"
    assert "${CORP_MLC_REGION}" not in text, f"{rel_path}: still uses old CORP_MLC_REGION"
    assert "${CORP_DOCKER_IMAGE}" in text
    assert "${CORP_VAULT_PATH}" in text


@pytest.mark.parametrize("rel_path", PRESET_TEMPLATES)
def test_preset_template_has_s3_mounts(rel_path: str) -> None:
    text = (ROOT / rel_path).read_text()
    assert "s3msk" in text, f"{rel_path}: missing S3 mount"
    assert "${CORP_S3_BUCKET_DATA}" in text
