"""Generate a 3×3 style-content mixing grid (Phase 6, M01).

For each (style_i, content_j) pair:
  - Take transformer_blocks.0–8  weights from style_lora[i]
  - Take transformer_blocks.9–18 weights from content_lora[j]
  - Merge into a single LoRA, load into FLUX, generate one image

Outputs:
  {output_dir}/{exp_name}/style_{i}_content_{j}.png  — individual cells
  {output_dir}/{exp_name}/grid.png                   — composite 3×3 figure

Usage:
    python scripts/eval/generate_mixing.py \\
        "mixing.style_loras=[/path/vg_img1.safetensors,/path/vg_img4.safetensors,/path/monet_img1.safetensors]" \\
        "mixing.content_loras=[/path/m01_cat.safetensors,/path/m01_dog.safetensors,/path/m01_backpack.safetensors]" \\
        "mixing.style_names=[van_gogh_img1,van_gogh_img4,monet_img1]" \\
        "mixing.content_names=[cat,dog,backpack]" \\
        "mixing.prompts=[a cat sitting on a chair,a dog running in a park,a backpack on a table]"
"""

from __future__ import annotations

import io
import logging
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from safetensors.torch import load_file, save_file

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LoRA key helpers
# ---------------------------------------------------------------------------

def _block_index(key: str) -> int | None:
    """Return the transformer block index for a LoRA key, or None."""
    m = re.search(r"transformer_blocks\.(\d+)\.", key)
    return int(m.group(1)) if m else None


def _filter_keys(state: dict[str, Any], blocks: list[int]) -> dict[str, Any]:
    """Keep only keys belonging to the given block indices."""
    block_set = set(blocks)
    return {k: v for k, v in state.items() if _block_index(k) in block_set}


def _merge_loras(
    style_path: str,
    content_path: str,
    style_blocks: list[int],
    content_blocks: list[int],
) -> Path:
    """Merge two LoRA safetensors into a temp file and return its path.

    Keys from style_path for style_blocks are combined with keys from
    content_path for content_blocks.  Overlapping keys (same block in both
    ranges) are taken from content_path so that content always wins.
    """
    style_state = load_file(style_path)
    content_state = load_file(content_path)

    style_part = _filter_keys(style_state, style_blocks)
    content_part = _filter_keys(content_state, content_blocks)

    if not style_part:
        raise ValueError(
            f"No keys found for style_blocks={style_blocks} in {style_path}. "
            "Check that the LoRA was trained on transformer_blocks.0–8."
        )
    if not content_part:
        raise ValueError(
            f"No keys found for content_blocks={content_blocks} in {content_path}. "
            "Check that the LoRA was trained on transformer_blocks.9–18."
        )

    merged = {**style_part, **content_part}
    log.debug(
        "Merged LoRA: %d style keys + %d content keys = %d total",
        len(style_part),
        len(content_part),
        len(merged),
    )

    tmp = tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False)
    save_file(merged, tmp.name)
    return Path(tmp.name)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _load_pipeline(cfg: DictConfig):
    from diffusers import FluxPipeline  # type: ignore[import]

    log.info("Loading FLUX pipeline from %s", cfg.model.name_or_path)
    pipe = FluxPipeline.from_pretrained(
        cfg.model.name_or_path,
        torch_dtype=torch.bfloat16,
    )
    pipe = pipe.to("cuda")
    return pipe


def _generate_cell(
    pipe,
    merged_lora_path: Path,
    prompt: str,
    lora_scale: float,
    cfg: DictConfig,
    seed: int,
) -> Image.Image:
    """Load merged LoRA, generate one image, then unload the LoRA."""
    pipe.load_lora_weights(str(merged_lora_path), adapter_name="merged")
    pipe.set_adapters(["merged"], adapter_weights=[lora_scale])

    generator = torch.Generator("cpu").manual_seed(seed)
    result = pipe(
        prompt=prompt,
        num_inference_steps=int(cfg.sampling.steps),
        guidance_scale=float(cfg.sampling.guidance_scale),
        width=int(cfg.sampling.width),
        height=int(cfg.sampling.height),
        generator=generator,
    )
    image: Image.Image = result.images[0]

    pipe.unload_lora_weights()
    return image


# ---------------------------------------------------------------------------
# Grid assembly
# ---------------------------------------------------------------------------

def _make_grid(
    cells: list[list[Image.Image]],
    style_names: list[str],
    content_names: list[str],
    cell_size: int = 512,
    label_height: int = 40,
) -> Image.Image:
    """Assemble cell images into a labelled 3×3 PNG grid."""
    from PIL import ImageDraw, ImageFont  # type: ignore[import]

    n_styles = len(style_names)
    n_contents = len(content_names)
    pad = label_height

    total_w = pad + n_contents * cell_size
    total_h = pad + n_styles * cell_size
    grid = Image.new("RGB", (total_w, total_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(grid)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    except OSError:
        font = ImageFont.load_default()

    # Column labels (content names)
    for j, name in enumerate(content_names):
        x = pad + j * cell_size + cell_size // 2
        draw.text((x, label_height // 4), name, fill=(0, 0, 0), font=font, anchor="mm")

    # Row labels (style names) + cells
    for i, s_name in enumerate(style_names):
        y_center = pad + i * cell_size + cell_size // 2
        draw.text((pad // 2, y_center), s_name, fill=(0, 0, 0), font=font, anchor="mm")
        for j in range(n_contents):
            cell = cells[i][j].resize((cell_size, cell_size))
            grid.paste(cell, (pad + j * cell_size, pad + i * cell_size))

    return grid


# ---------------------------------------------------------------------------
# ClearML
# ---------------------------------------------------------------------------

def _setup_clearml(cfg: DictConfig, task_name: str):
    if not cfg.clearml.enabled:
        return None
    try:
        from clearml import Task  # type: ignore[import]

        task = Task.init(
            project_name=cfg.clearml.project,
            task_name=f"{cfg.clearml.task_prefix}/{task_name}",
            reuse_last_task_id=False,
        )
        task.connect(OmegaConf.to_container(cfg, resolve=True))
        return task
    except Exception as exc:
        log.warning("ClearML init failed (%s), continuing without tracking.", exc)
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cfg: DictConfig) -> None:
    m = cfg.mixing

    # Validate input lists
    assert len(m.style_loras) == len(m.style_names) == 3, \
        "style_loras and style_names must each have exactly 3 entries"
    assert len(m.content_loras) == len(m.content_names) == len(m.prompts) == 3, \
        "content_loras, content_names, and prompts must each have exactly 3 entries"

    for p in list(m.style_loras) + list(m.content_loras):
        if not Path(p).exists():
            raise FileNotFoundError(f"LoRA file not found: {p}")

    out_dir = Path(m.output_dir) / m.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output dir: %s", out_dir)

    task = _setup_clearml(cfg, m.exp_name)
    pipe = _load_pipeline(cfg)

    cells: list[list[Image.Image]] = []

    for i, (s_path, s_name) in enumerate(zip(m.style_loras, m.style_names)):
        row: list[Image.Image] = []
        for j, (c_path, c_name, prompt) in enumerate(
            zip(m.content_loras, m.content_names, m.prompts)
        ):
            log.info("Cell (%d,%d): style=%s  content=%s  prompt=%r", i, j, s_name, c_name, prompt)

            merged_path = _merge_loras(
                style_path=s_path,
                content_path=c_path,
                style_blocks=list(m.style_only_blocks),
                content_blocks=list(m.content_blocks),
            )
            try:
                seed = int(cfg.sampling.seed) + i * 10 + j
                img = _generate_cell(pipe, merged_path, prompt, float(m.lora_scale), cfg, seed)
            finally:
                merged_path.unlink(missing_ok=True)

            fname = out_dir / f"style_{i}_{s_name}__content_{j}_{c_name}.png"
            img.save(fname)
            log.info("  saved: %s", fname)

            if task is not None:
                task.get_logger().report_image(
                    "mixing_grid",
                    f"{s_name} × {c_name}",
                    iteration=0,
                    image=img,
                )
            row.append(img)
        cells.append(row)

    if m.save_grid:
        grid = _make_grid(cells, list(m.style_names), list(m.content_names))
        grid_path = out_dir / "grid.png"
        grid.save(grid_path)
        log.info("Grid saved: %s", grid_path)
        if task is not None:
            task.get_logger().report_image("mixing_grid", "3x3 grid", iteration=0, image=grid)

    log.info("Done. %d images generated.", len(m.style_loras) * len(m.content_loras))
    if task is not None:
        task.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    config_dir = str(
        (Path(__file__).resolve().parents[2] / "configs" / "eval").resolve()
    )
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(config_name="mixing", overrides=sys.argv[1:])
    main(cfg)
