"""Fast tests that validate smoke_test.yaml without any infrastructure."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

CONFIGS_DIR = Path(__file__).parent.parent / "configs" / "experiments"
SMOKE_CONFIG_PATH = CONFIGS_DIR / "smoke_test.yaml"


@pytest.fixture(scope="module")
def smoke_cfg() -> dict:
    with SMOKE_CONFIG_PATH.open() as fh:
        return yaml.safe_load(fh)


@pytest.fixture(scope="module")
def process(smoke_cfg: dict) -> dict:
    """Return the single process entry (sd_trainer block)."""
    return smoke_cfg["config"]["process"][0]


@pytest.mark.fast
def test_config_name(smoke_cfg: dict) -> None:
    assert smoke_cfg["config"]["name"] == "smoke_test"


@pytest.mark.fast
def test_train_steps(process: dict) -> None:
    assert process["train"]["steps"] == 10


@pytest.mark.fast
def test_only_if_contains_present(process: dict) -> None:
    blocks = process["network"]["network_kwargs"]["only_if_contains"]
    assert isinstance(blocks, list), "only_if_contains must be a list"
    assert len(blocks) > 0, "only_if_contains must not be empty"
    # Verify B-LoRA single-stream blocks 30-37 are all present
    expected = {f"single_transformer_blocks.{i}" for i in range(30, 38)}
    actual = set(blocks)
    assert expected == actual, f"Expected blocks {expected}, got {actual}"


@pytest.mark.fast
def test_sample_prompts_exactly_one(process: dict) -> None:
    prompts = process["sample"]["prompts"]
    assert isinstance(prompts, list), "sample.prompts must be a list"
    assert len(prompts) == 1, f"Expected exactly 1 prompt, got {len(prompts)}"
