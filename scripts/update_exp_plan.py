"""Sync experiment results from ClearML into experiments/plan.md.

Usage:
    python scripts/update_exp_plan.py                  # update all
    python scripts/update_exp_plan.py --dry-run        # print diff only
    python scripts/update_exp_plan.py --project blora-flux-eval

The script queries ClearML for tasks whose names match the experiment IDs
defined in EXPERIMENT_REGISTRY, pulls scalar metrics (clip_style, clip_content,
fid, lpips) from the last reported value, and rewrites the status cells in
experiments/plan.md in place.

Metric keys expected in ClearML (logged as scalars):
    eval/clip_style, eval/clip_content, eval/fid, eval/lpips
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

PLAN_PATH = Path(__file__).resolve().parents[1] / "experiments" / "plan.md"
CLEARML_PROJECT = "blora-flux-eval"

# Map: experiment ID prefix → ClearML task name prefix
# task name format: "generate/<exp_name>" or "<exp_name>"
EXPERIMENT_IDS = [
    # Ablation A
    "A01", "A02", "A03", "A04",
    # Ablation B
    "B01", "B02", "B03", "B04",
    # Ablation C
    "C01", "C02", "C03", "C04",
    # Group E (Van Gogh ×4 images, 4 methods)
    "E01-1", "E01-2", "E01-3", "E01-4",
    "E02-1", "E02-2", "E02-3", "E02-4",
    "E03-1", "E03-2", "E03-3", "E03-4",
    "E04-1", "E04-2", "E04-3", "E04-4",
    # Group F (Monet ×4)
    "F01-1", "F01-2", "F01-3", "F01-4",
]

# ClearML task name → experiment ID mapping (lower-cased prefix match)
_ID_TO_TASK_PREFIX: dict[str, str] = {
    "A01": "a01", "A02": "a02", "A03": "a03", "A04": "a04",
    "B01": "b01", "B02": "b02", "B03": "b03", "B04": "b04",
    "C01": "c01", "C02": "c02", "C03": "c03", "C04": "c04",
    "E01-1": "e01_blora_flux_van_gogh_img1",
    "E01-2": "e01_blora_flux_van_gogh_img2",
    "E01-3": "e01_blora_flux_van_gogh_img3",
    "E01-4": "e01_blora_flux_van_gogh_img4",
    "E02-1": "e02_full_lora_flux_van_gogh_img1",
    "E02-2": "e02_full_lora_flux_van_gogh_img2",
    "E02-3": "e02_full_lora_flux_van_gogh_img3",
    "E02-4": "e02_full_lora_flux_van_gogh_img4",
    "E03-1": "e03_ip_adapter_flux_van_gogh_img1",
    "E03-2": "e03_ip_adapter_flux_van_gogh_img2",
    "E03-3": "e03_ip_adapter_flux_van_gogh_img3",
    "E03-4": "e03_ip_adapter_flux_van_gogh_img4",
    "E04-1": "e04_blora_sdxl_van_gogh_img1",
    "E04-2": "e04_blora_sdxl_van_gogh_img2",
    "E04-3": "e04_blora_sdxl_van_gogh_img3",
    "E04-4": "e04_blora_sdxl_van_gogh_img4",
    "F01-1": "e01_blora_flux_monet_img1",
    "F01-2": "e01_blora_flux_monet_img2",
    "F01-3": "e01_blora_flux_monet_img3",
    "F01-4": "e01_blora_flux_monet_img4",
}

_METRIC_KEYS = {
    "clip_style": "eval/clip_style",
    "clip_content": "eval/clip_content",
    "fid": "eval/fid",
    "lpips": "eval/lpips",
}


@dataclass
class ExpResult:
    exp_id: str
    status: str = "pending"       # pending | running | done | failed
    clip_style: float | None = None
    clip_content: float | None = None
    fid: float | None = None
    lpips: float | None = None

    def status_char(self) -> str:
        return {"pending": "[ ]", "running": "[~]", "done": "[x]", "failed": "[!]"}.get(
            self.status, "[ ]"
        )

    def fmt(self, v: float | None, decimals: int = 4) -> str:
        return f"{v:.{decimals}f}" if v is not None else "—"


def _query_clearml(project: str) -> dict[str, ExpResult]:
    try:
        from clearml import Task  # type: ignore[import]
    except ImportError:
        log.error("clearml not installed. Run: pip install clearml")
        return {}

    results: dict[str, ExpResult] = {}

    for exp_id in EXPERIMENT_IDS:
        task_prefix = _ID_TO_TASK_PREFIX.get(exp_id, exp_id.lower())
        tasks = Task.get_tasks(project_name=project, task_name=task_prefix)
        if not tasks:
            results[exp_id] = ExpResult(exp_id=exp_id, status="pending")
            continue

        # take the most recently created matching task
        task = sorted(tasks, key=lambda t: t.data.created, reverse=True)[0]
        status_str = str(task.get_status())

        if status_str in ("completed", "published"):
            status = "done"
        elif status_str in ("in_progress", "queued"):
            status = "running"
        elif status_str == "failed":
            status = "failed"
        else:
            status = "pending"

        scalars = task.get_reported_scalars() or {}
        r = ExpResult(exp_id=exp_id, status=status)
        for attr, key in _METRIC_KEYS.items():
            group, series = key.split("/", 1)
            try:
                series_data = scalars[group][series]["y"]
                setattr(r, attr, float(series_data[-1]))
            except (KeyError, IndexError, TypeError):
                pass
        results[exp_id] = r

    return results


def _update_table_row(line: str, result: ExpResult) -> str:
    """Replace metric cells and status cell in a Markdown table row."""
    # Rows look like: | A01  | a01_blocks_34_37.yaml | [34–37] | — | — | — | — | [ ] |
    parts = [p.strip() for p in line.split("|")]
    if len(parts) < 8:
        return line

    # find ID column (parts[0] is empty due to leading |)
    row_id = parts[1].strip()
    if row_id != result.exp_id:
        return line

    # columns: id, config/desc, param, clip_style, clip_content, fid, lpips, status
    offset = 3  # first metric column index in parts[]
    parts[offset] = f" {result.fmt(result.clip_style)} "
    parts[offset + 1] = f" {result.fmt(result.clip_content)} "
    parts[offset + 2] = f" {result.fmt(result.fid, 1)} "
    parts[offset + 3] = f" {result.fmt(result.lpips)} "
    parts[offset + 4] = f" {result.status_char()} "

    return "|".join(parts)


def _update_plan(results: dict[str, ExpResult], dry_run: bool = False) -> None:
    text = PLAN_PATH.read_text()
    lines = text.splitlines(keepends=True)

    for i, line in enumerate(lines):
        for exp_id, result in results.items():
            if f"| {exp_id} " in line or f"| {exp_id}-" in line:
                updated = _update_table_row(line.rstrip("\n"), result)
                if updated != line.rstrip("\n"):
                    lines[i] = updated + "\n"

    # Update "Последнее обновление" timestamp
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        re.sub(r"Last updated: .*", f"Last updated: {now}", l) for l in lines
    ]

    # Update progress counters
    done = sum(1 for r in results.values() if r.status == "done")
    running = sum(1 for r in results.values() if r.status == "running")
    failed = sum(1 for r in results.values() if r.status == "failed")
    total = len(EXPERIMENT_IDS)
    pending = total - done - running - failed
    for i, line in enumerate(lines):
        if "- Завершено:" in line:
            lines[i] = f"- Завершено: {done} / {total}\n"
        elif "- Запущено:" in line:
            lines[i] = f"- Запущено: {running}\n"
        elif "- Ожидают:" in line:
            lines[i] = f"- Ожидают: {pending}\n"
        elif "- Ошибки:" in line:
            lines[i] = f"- Ошибки: {failed}\n"

    new_text = "".join(lines)
    if dry_run:
        print(new_text)
        return

    PLAN_PATH.write_text(new_text)
    log.info("Updated %s", PLAN_PATH)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync ClearML results → experiments/plan.md")
    parser.add_argument("--project", default=CLEARML_PROJECT)
    parser.add_argument("--dry-run", action="store_true", help="Print updated plan without writing")
    args = parser.parse_args()

    results = _query_clearml(args.project)
    if not results:
        log.warning("No ClearML results found; plan unchanged.")
        return

    _update_plan(results, dry_run=args.dry_run)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
