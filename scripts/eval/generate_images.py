"""Generate style-transferred images using a trained LoRA adapter on FLUX.1.

Usage:
    python scripts/eval/generate_images.py \
        generate.lora_path=output/e01_blora_flux_van_gogh_img1/e01_blora_flux_van_gogh_img1.safetensors \
        generate.exp_name=e01_blora_flux_van_gogh_img1
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
from PIL import Image

log = logging.getLogger(__name__)


def _load_prompts(prompt_file: str, suffix: str = "") -> list[str]:
    path = Path(prompt_file)
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    prompts = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if suffix:
        prompts = [f"{p}, {suffix}" for p in prompts]
    log.info("Loaded %d prompts from %s (suffix=%r)", len(prompts), path, suffix)
    return prompts


def _build_pipeline(cfg: DictConfig):
    pipeline_type = cfg.model.get("pipeline_type", "flux")

    if pipeline_type == "sdxl":
        from diffusers import StableDiffusionXLPipeline  # type: ignore[import]

        log.info("Loading SDXL pipeline: %s", cfg.model.name_or_path)
        pipe = StableDiffusionXLPipeline.from_pretrained(
            cfg.model.name_or_path,
            torch_dtype=torch.float16,
        )
    else:
        from diffusers import FluxPipeline  # type: ignore[import]

        log.info("Loading FLUX pipeline: %s", cfg.model.name_or_path)
        pipe = FluxPipeline.from_pretrained(
            cfg.model.name_or_path,
            torch_dtype=torch.bfloat16,
        )

    pipe = pipe.to("cuda")

    lora_path = cfg.generate.lora_path
    if lora_path is not None:
        log.info("Loading LoRA: %s  (scale=%.2f)", lora_path, cfg.model.lora_scale)
        pipe.load_lora_weights(str(lora_path))
        pipe.fuse_lora(lora_scale=float(cfg.model.lora_scale))
    else:
        log.info("No LoRA specified — running baseline (no LoRA)")

    return pipe


def _setup_clearml(cfg: DictConfig, task_name: str) -> object | None:
    if not cfg.clearml.enabled:
        return None
    try:
        from clearml import Task  # type: ignore[import]

        task = Task.init(
            project_name=cfg.clearml.project,
            task_name=f"{cfg.clearml.task_prefix}/{task_name}",
            reuse_last_task_id=False,
        )
        task.connect(cfg)
        return task
    except Exception as exc:
        log.warning("ClearML init failed (%s), continuing without tracking.", exc)
        return None


def main(cfg: DictConfig) -> None:
    task_name = cfg.generate.exp_name
    task = _setup_clearml(cfg, task_name)

    out_dir = Path(cfg.generate.output_dir) / task_name
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = _load_prompts(cfg.generate.prompt_file, cfg.generate.get("prompt_suffix", ""))
    pipe = _build_pipeline(cfg)

    base_seed = int(cfg.sampling.seed)
    pipeline_type = cfg.model.get("pipeline_type", "flux")

    # SDXL VAE has fp16 numerical instability; upcast_vae() forces the decoder to run in float32.
    # autocast alone is insufficient because it doesn't promote the entire VAE to float32.
    if pipeline_type == "sdxl":
        pipe.upcast_vae()

    log.info("Generating %d images -> %s", len(prompts), out_dir)
    for idx, prompt in enumerate(prompts):
        # Fresh generator per image: seed+idx gives deterministic, non-coupled outputs.
        generator = torch.Generator("cpu").manual_seed(base_seed + idx)
        result = pipe(
            prompt=prompt,
            num_inference_steps=int(cfg.sampling.steps),
            guidance_scale=float(cfg.sampling.guidance_scale),
            width=int(cfg.sampling.width),
            height=int(cfg.sampling.height),
            generator=generator,
        )
        image: Image.Image = result.images[0]
        out_path = out_dir / f"{idx:04d}.png"
        image.save(out_path)

        if (idx + 1) % 10 == 0:
            log.info("  %d / %d", idx + 1, len(prompts))

    log.info("Done. Images saved to %s", out_dir)

    if task is not None:
        task.get_logger().report_scalar("generate", "total_images", value=len(prompts), iteration=0)
        task.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    config_dir = str((Path(__file__).resolve().parents[2] / "configs" / "eval").resolve())
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(config_name="generate", overrides=sys.argv[1:])
    main(cfg)
