"""
Tests: results upload/download logic and metrics processing.

Covers:
- ExpResult formatting and status chars
- _update_table_row rewrites Markdown correctly
- _update_plan modifies plan.md in place
- _load_prompts reads files correctly
- S3 path construction in prepare_paths task
- boto3 download pattern present in DAG source
- metrics_job_script.sh style detection logic
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytestmark = pytest.mark.fast

# ---------------------------------------------------------------------------
# ExpResult: status chars and formatting
# ---------------------------------------------------------------------------

from scripts.update_exp_plan import ExpResult, _update_table_row, _update_plan  # noqa: E402


def test_status_char_pending() -> None:
    assert ExpResult(exp_id="A01", status="pending").status_char() == "[ ]"


def test_status_char_running() -> None:
    assert ExpResult(exp_id="A01", status="running").status_char() == "[~]"


def test_status_char_done() -> None:
    assert ExpResult(exp_id="A01", status="done").status_char() == "[x]"


def test_status_char_failed() -> None:
    assert ExpResult(exp_id="A01", status="failed").status_char() == "[!]"


def test_fmt_float() -> None:
    r = ExpResult(exp_id="A01")
    assert r.fmt(0.8523, 4) == "0.8523"


def test_fmt_none() -> None:
    assert ExpResult(exp_id="A01").fmt(None) == "—"


def test_fmt_one_decimal() -> None:
    assert ExpResult(exp_id="A01").fmt(12.3, 1) == "12.3"


# ---------------------------------------------------------------------------
# _update_table_row
# ---------------------------------------------------------------------------

_SAMPLE_ROW = "| A01  | a01_blocks_34_37.yaml | [34–37] | — | — | — | — | [ ] |"


def test_update_table_row_pending_unchanged() -> None:
    result = _update_table_row(_SAMPLE_ROW, ExpResult(exp_id="A01", status="pending"))
    assert "[ ]" in result
    assert result.count("—") >= 4


def test_update_table_row_done_fills_metrics() -> None:
    r = ExpResult(exp_id="A01", status="done",
                  clip_style=0.8523, clip_content=0.7912, fid=12.3, lpips=0.2341)
    result = _update_table_row(_SAMPLE_ROW, r)
    assert "0.8523" in result
    assert "0.7912" in result
    assert "12.3" in result
    assert "0.2341" in result
    assert "[x]" in result


def test_update_table_row_wrong_id_unchanged() -> None:
    r = ExpResult(exp_id="A02", status="done",
                  clip_style=0.9, clip_content=0.9, fid=5.0, lpips=0.1)
    result = _update_table_row(_SAMPLE_ROW, r)
    assert result == _SAMPLE_ROW


def test_update_table_row_failed_status() -> None:
    r = ExpResult(exp_id="A01", status="failed")
    result = _update_table_row(_SAMPLE_ROW, r)
    assert "[!]" in result


# ---------------------------------------------------------------------------
# _update_plan: writes to file correctly
# ---------------------------------------------------------------------------

_MINIMAL_PLAN = """\
# Experiment Plan

Auto-generated. Last updated: —

- Завершено: 0 / 32
- Запущено: 0
- Ожидают: 32
- Ошибки: 0

| A01  | a01_blocks_34_37.yaml | [34–37] | — | — | — | — | [ ] |
| A02  | a02_blocks_30_37.yaml | [30–37] | — | — | — | — | [ ] |
"""


def test_update_plan_writes_metrics(tmp_path: Path, monkeypatch) -> None:
    plan_file = tmp_path / "plan.md"
    plan_file.write_text(_MINIMAL_PLAN)

    import scripts.update_exp_plan as uep
    monkeypatch.setattr(uep, "PLAN_PATH", plan_file)

    results = {
        "A01": ExpResult(exp_id="A01", status="done",
                         clip_style=0.85, clip_content=0.79, fid=10.0, lpips=0.23),
    }
    _update_plan(results)

    updated = plan_file.read_text()
    assert "[x]" in updated
    assert "0.8500" in updated
    assert "0.7900" in updated


def test_update_plan_updates_progress_counters(tmp_path: Path, monkeypatch) -> None:
    plan_file = tmp_path / "plan.md"
    plan_file.write_text(_MINIMAL_PLAN)

    import scripts.update_exp_plan as uep
    monkeypatch.setattr(uep, "PLAN_PATH", plan_file)
    monkeypatch.setattr(uep, "EXPERIMENT_IDS", ["A01", "A02"])

    results = {
        "A01": ExpResult(exp_id="A01", status="done"),
        "A02": ExpResult(exp_id="A02", status="pending"),
    }
    _update_plan(results)

    updated = plan_file.read_text()
    assert "Завершено: 1 / 2" in updated
    assert "Ожидают: 1" in updated


def test_update_plan_dry_run_no_write(tmp_path: Path, monkeypatch, capsys) -> None:
    plan_file = tmp_path / "plan.md"
    plan_file.write_text(_MINIMAL_PLAN)
    original = _MINIMAL_PLAN

    import scripts.update_exp_plan as uep
    monkeypatch.setattr(uep, "PLAN_PATH", plan_file)

    _update_plan({"A01": ExpResult(exp_id="A01", status="done")}, dry_run=True)

    assert plan_file.read_text() == original  # file unchanged
    captured = capsys.readouterr()
    assert "[x]" in captured.out  # printed to stdout


# ---------------------------------------------------------------------------
# _load_prompts
# ---------------------------------------------------------------------------

def _load_prompts(prompt_file: str) -> list[str]:
    """Mirror of _load_prompts from generate_images.py / compute_metrics.py."""
    path = Path(prompt_file)
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def test_load_prompts_reads_lines(tmp_path: Path) -> None:
    prompt_file = tmp_path / "prompts.txt"
    prompt_file.write_text("a cat\na dog\na bird\n")
    assert _load_prompts(str(prompt_file)) == ["a cat", "a dog", "a bird"]


def test_load_prompts_skips_blank_lines(tmp_path: Path) -> None:
    prompt_file = tmp_path / "prompts.txt"
    prompt_file.write_text("a cat\n\n\na dog\n")
    assert len(_load_prompts(str(prompt_file))) == 2


def test_load_prompts_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        _load_prompts("/nonexistent/path/prompts.txt")


# ---------------------------------------------------------------------------
# S3 path construction (prepare_paths logic, no Airflow needed)
# ---------------------------------------------------------------------------

def _prepare_paths(base: str, exp: str) -> dict[str, str]:
    """Mirror of the prepare_paths task logic from blora_flux_pipeline.py."""
    base = base.rstrip("/")
    return {
        "EXPERIMENT_NAME": exp,
        "TRAIN_OUTPUT_S3_PATH": f"{base}/exp_logs/{exp}/loras",
        "GENERATED_OUTPUT_S3_PATH": f"{base}/exp_logs/{exp}/generated",
        "METRICS_OUTPUT_S3_PATH": f"{base}/exp_logs/{exp}/metrics",
    }


def test_prepare_paths_all_keys_present() -> None:
    result = _prepare_paths("s3://bucket/base", "e01_blora_flux_van_gogh_img1")
    assert set(result.keys()) == {
        "EXPERIMENT_NAME", "TRAIN_OUTPUT_S3_PATH",
        "GENERATED_OUTPUT_S3_PATH", "METRICS_OUTPUT_S3_PATH",
    }


def test_prepare_paths_trailing_slash_stripped() -> None:
    r = _prepare_paths("s3://bucket/base/", "exp1")
    assert not r["TRAIN_OUTPUT_S3_PATH"].startswith("s3://bucket/base//")


def test_prepare_paths_correct_subdirs() -> None:
    r = _prepare_paths("s3://bucket/base", "my_exp")
    assert r["TRAIN_OUTPUT_S3_PATH"] == "s3://bucket/base/exp_logs/my_exp/loras"
    assert r["GENERATED_OUTPUT_S3_PATH"] == "s3://bucket/base/exp_logs/my_exp/generated"
    assert r["METRICS_OUTPUT_S3_PATH"] == "s3://bucket/base/exp_logs/my_exp/metrics"


def test_prepare_paths_experiment_name_in_result() -> None:
    r = _prepare_paths("s3://bucket/base", "e01_blora_flux_van_gogh_img1")
    assert r["EXPERIMENT_NAME"] == "e01_blora_flux_van_gogh_img1"


# ---------------------------------------------------------------------------
# DAG source: boto3 download and metrics.json pattern
# ---------------------------------------------------------------------------

def test_dag_pipeline_has_boto3_download() -> None:
    src = (ROOT / "dags" / "blora_flux_pipeline.py").read_text()
    assert 'boto3.client("s3"' in src
    assert "endpoint_url" in src
    assert "download_file" in src
    assert "metrics.json" in src


# ---------------------------------------------------------------------------
# Job script: style detection from experiment name
# ---------------------------------------------------------------------------

def test_metrics_script_detects_van_gogh() -> None:
    src = (ROOT / "dags/jobs/metrics_blora/files/metrics_job_script.sh").read_text()
    assert "van_gogh" in src
    assert "STYLE_REFS_DIR" in src


def test_metrics_script_detects_monet() -> None:
    src = (ROOT / "dags/jobs/metrics_blora/files/metrics_job_script.sh").read_text()
    assert "monet" in src


def test_train_script_uploads_safetensors() -> None:
    src = (ROOT / "dags/jobs/train_blora/files/train_job_script.sh").read_text()
    assert "safetensors" in src
    assert "s3cmd sync" in src
    assert "TRAIN_OUTPUT_S3_PATH" in src


def test_generate_script_downloads_lora_and_uploads_images() -> None:
    src = (ROOT / "dags/jobs/generate_blora/files/generate_job_script.sh").read_text()
    assert "s3cmd sync" in src
    assert "TRAIN_OUTPUT_S3_PATH" in src
    assert "GENERATED_OUTPUT_S3_PATH" in src
    assert "safetensors" in src


def test_metrics_script_uploads_metrics_json() -> None:
    src = (ROOT / "dags/jobs/metrics_blora/files/metrics_job_script.sh").read_text()
    assert "metrics.json" in src
    assert "METRICS_OUTPUT_S3_PATH" in src
    assert "s3cmd put" in src


# ---------------------------------------------------------------------------
# metrics.json schema (output of compute_metrics.py)
# ---------------------------------------------------------------------------

def test_metrics_json_schema(tmp_path: Path) -> None:
    """Verify that the expected metrics.json keys align with what update_exp_plan reads."""
    from scripts.update_exp_plan import _METRIC_KEYS  # type: ignore[attr-defined]

    # _METRIC_KEYS maps attr → "group/series" — check the keys are as expected
    expected_attrs = {"clip_style", "clip_content", "fid", "lpips"}
    assert set(_METRIC_KEYS.keys()) == expected_attrs
