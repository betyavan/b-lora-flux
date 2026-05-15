"""Generate SDXL B-LoRA style-transfer images for the F02 pair-driver protocol.

For each pair (content_subject_id, style_id) in `experiments/data/b_lora_eval_pairs.json`:
  1. Resolve style-LoRA path via cfg.mixing_sdxl.style_id_to_path[style_id]
  2. Resolve content-LoRA path as  <content_lora_dir>/m02_content_sdxl_<subject>.safetensors
  3. Load both safetensors state-dicts and merge them by union of keys.
     (SDXL B-LoRA blocks DO NOT overlap — style uses up_blocks.0.attentions.1,
      content uses up_blocks.0.attentions.0 — so a plain {**style, **content} is
      semantically equivalent to a per-block filter. We assert no key overlap
      and emit a warning if any is detected.)
  4. Save merged state-dict to a temp .safetensors file.
  5. Load merged adapter into a single SDXL pipeline (reused across the whole loop).
  6. Generate one image with seed = sampling.seed_base + pair_id.
  7. Save as pair_{pair_id:03d}_{style_id}__{content_subject_id}.png.

The prompt is `f"a {content_subject_id}{style_suffix}"`, where `style_suffix` is
filled from `style_suffix_template.format(artist=...)` using the artist mapping
keyed on the style_id prefix (wikivg_* -> "Van Gogh", wikimo_* -> "Claude Monet").

Usage:
    python scripts/eval/generate_mixing_sdxl.py \\
        mixing_sdxl.manifest_path=experiments/data/b_lora_eval_pairs.json \\
        mixing_sdxl.style_lora_dir=/models/sdxl_loras/style \\
        mixing_sdxl.content_lora_dir=/models/sdxl_loras/content \\
        mixing_sdxl.exp_name=f02_blora_sdxl_pairs

    # User-study subset only (30 pairs):
    python scripts/eval/generate_mixing_sdxl.py \\
        mixing_sdxl.manifest_path=experiments/data/b_lora_eval_pairs.json \\
        mixing_sdxl.pair_subset=user_study \\
        mixing_sdxl.exp_name=f02_blora_sdxl_user_study
"""

from __future__ import annotations

import json
import logging
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
# Manifest + path resolution
# ---------------------------------------------------------------------------

def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with path.open() as f:
        manifest: dict[str, Any] = json.load(f)
    if "pairs" not in manifest:
        raise ValueError(f"Manifest {path} missing 'pairs' field")
    return manifest


def _select_pairs(manifest: dict[str, Any], subset: str | None) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = list(manifest["pairs"])
    if subset is not None:
        s = str(subset).strip().strip('"').strip("'")
        subset = s if s else None
    if subset is None:
        return pairs
    if subset == "user_study":
        ids = manifest.get("user_study_pair_ids")
        if not ids:
            raise ValueError("pair_subset=user_study requested but manifest has no user_study_pair_ids")
        id_set = set(int(i) for i in ids)
        return [p for p in pairs if int(p["pair_id"]) in id_set]
    raise ValueError(f"Unknown pair_subset: {subset!r} (expected null or 'user_study')")


def _artist_for_style_id(style_id: str, mapping: dict[str, str]) -> str:
    """Resolve style_id (e.g. 'wikivg_03') to artist name via prefix mapping.

    `mapping` keys may be wildcards like 'wikivg_*' / 'wikimo_*' or exact ids.
    """
    if style_id in mapping:
        return str(mapping[style_id])
    prefix = style_id.split("_", 1)[0]
    wildcard = f"{prefix}_*"
    if wildcard in mapping:
        return str(mapping[wildcard])
    raise KeyError(f"No artist mapping for style_id={style_id!r}; mapping keys: {list(mapping.keys())}")


def _resolve_style_lora_path(style_id: str, style_lora_dir: Path, id_to_path: dict[str, str]) -> Path:
    if style_id not in id_to_path:
        raise KeyError(
            f"style_id={style_id!r} missing from mixing_sdxl.style_id_to_path; "
            f"available: {list(id_to_path.keys())}"
        )
    fname = id_to_path[style_id]
    p = (style_lora_dir / fname).resolve() if not Path(fname).is_absolute() else Path(fname)
    if not p.exists():
        raise FileNotFoundError(f"Style LoRA not found for {style_id}: {p}")
    return p


def _resolve_content_lora_path(subject_id: str, content_lora_dir: Path, template: str) -> Path:
    fname = template.format(subject=subject_id)
    p = (content_lora_dir / fname).resolve() if not Path(fname).is_absolute() else Path(fname)
    if not p.exists():
        raise FileNotFoundError(f"Content LoRA not found for subject={subject_id!r}: {p}")
    return p


# ---------------------------------------------------------------------------
# LoRA merge (SDXL: non-overlapping block ranges, plain union)
# ---------------------------------------------------------------------------

def _merge_state_dicts(style_path: Path, content_path: Path) -> Path:
    style_state = load_file(str(style_path))
    content_state = load_file(str(content_path))

    if not style_state:
        raise ValueError(f"Style LoRA state-dict is empty: {style_path}")
    if not content_state:
        raise ValueError(f"Content LoRA state-dict is empty: {content_path}")

    overlap = set(style_state).intersection(content_state)
    if overlap:
        log.warning(
            "Unexpected key overlap between style and content SDXL LoRAs (%d keys); "
            "content will win for overlapping keys. Example: %s",
            len(overlap),
            next(iter(overlap)),
        )

    merged: dict[str, Any] = {**style_state, **content_state}
    log.debug(
        "Merged SDXL LoRA: %d style + %d content = %d total (overlap=%d)",
        len(style_state),
        len(content_state),
        len(merged),
        len(overlap),
    )

    tmp = tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False)
    tmp.close()
    save_file(merged, tmp.name)
    return Path(tmp.name)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _load_pipeline(cfg: DictConfig) -> Any:
    from diffusers import StableDiffusionXLPipeline  # type: ignore[import]

    model_path = str(cfg.model.name_or_path)
    log.info("Loading SDXL pipeline from %s", model_path)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    )
    return pipe.to("cuda")


def _generate_one(
    pipe: Any,
    merged_lora_path: Path,
    prompt: str,
    lora_scale: float,
    cfg: DictConfig,
    seed: int,
) -> Image.Image:
    """Load the merged LoRA, generate one image, then unload."""
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
# ClearML
# ---------------------------------------------------------------------------

def _setup_clearml(cfg: DictConfig, task_name: str) -> Any | None:
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
    m = cfg.mixing_sdxl

    manifest_path = Path(str(m.manifest_path))
    manifest = _load_manifest(manifest_path)
    subset = m.get("pair_subset", None)
    pairs = _select_pairs(manifest, subset)
    if not pairs:
        raise ValueError(f"No pairs to process (subset={subset!r})")

    style_lora_dir = Path(str(m.style_lora_dir))
    content_lora_dir = Path(str(m.content_lora_dir))
    style_id_to_path = dict(OmegaConf.to_container(m.style_id_to_path, resolve=True))  # type: ignore[arg-type]
    artist_map = dict(OmegaConf.to_container(m.artist_from_style_id, resolve=True))  # type: ignore[arg-type]
    content_lora_template = str(m.content_lora_template)
    style_suffix_template = str(m.style_suffix_template)
    lora_scale = float(m.lora_scale)
    seed_base = int(cfg.sampling.seed_base)

    out_dir = Path(str(m.output_dir)) / str(m.exp_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output dir: %s", out_dir)
    log.info("Pairs to process: %d (subset=%s)", len(pairs), subset)

    task = _setup_clearml(cfg, str(m.exp_name))
    pipe = _load_pipeline(cfg)

    log_every = int(m.get("log_every", 10))
    n_generated = 0

    for idx, pair in enumerate(pairs):
        pair_id = int(pair["pair_id"])
        subject_id = str(pair["content_subject_id"])
        style_id = str(pair["style_id"])

        style_lora = _resolve_style_lora_path(style_id, style_lora_dir, style_id_to_path)
        content_lora = _resolve_content_lora_path(subject_id, content_lora_dir, content_lora_template)
        artist = _artist_for_style_id(style_id, artist_map)
        style_suffix = style_suffix_template.format(artist=artist)
        prompt = f"a {subject_id}{style_suffix}"

        log.info(
            "Pair %d/%d (id=%d): style=%s subject=%s prompt=%r",
            idx + 1, len(pairs), pair_id, style_id, subject_id, prompt,
        )

        merged_path = _merge_state_dicts(style_lora, content_lora)
        try:
            seed = seed_base + pair_id
            img = _generate_one(pipe, merged_path, prompt, lora_scale, cfg, seed)
        finally:
            merged_path.unlink(missing_ok=True)

        fname = out_dir / f"pair_{pair_id:03d}_{style_id}__{subject_id}.png"
        img.save(fname)
        n_generated += 1

        if task is not None:
            task.get_logger().report_image(
                "f02_pairs",
                f"{style_id}__{subject_id}",
                iteration=pair_id,
                image=img,
            )

        if (idx + 1) % log_every == 0:
            log.info("  progress: %d / %d pairs done", idx + 1, len(pairs))

    log.info("Done. %d images generated.", n_generated)
    if task is not None:
        task.get_logger().report_scalar("f02", "n_generated", value=n_generated, iteration=0)
        task.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    config_dir = str((Path(__file__).resolve().parents[2] / "configs" / "eval").resolve())
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(config_name="mixing_sdxl", overrides=sys.argv[1:])
    main(cfg)
