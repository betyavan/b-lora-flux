"""Quantitative evaluation of B-LoRA-FLUX limitations (Phase 4b).

Three limitation cases:
  L01 — Color leakage: style colors bleed into objects with characteristic colors.
         Metric: ΔE CIE76 between dominant color of generated region and expected object color.
  L02 — Background leakage: style backgrounds bleed into generated subjects.
         Metric: DINO ViT-B/8 cosine similarity between generated image and a clean reference crop.
  L03 — Complex scenes: style transfer degrades fidelity in multi-object scenes.
         Metric: CLIP text-image cosine similarity (CLIP-content) for complex COCO prompts.

Usage:
    # L01 — color leakage
    python scripts/eval/limitations_eval.py \\
        lim.case=L01 \\
        lim.lora_path=output/PHASE2_BEST/PHASE2_BEST.safetensors \\
        lim.exp_name=l01_color_leakage \\
        lim.alpha_style=1.0

    # L02 — background leakage (full-frame vs center-cropped style training)
    python scripts/eval/limitations_eval.py \\
        lim.case=L02 \\
        lim.lora_path=output/PHASE2_BEST/PHASE2_BEST.safetensors \\
        lim.lora_path_crop=output/PHASE2_BEST_CROP/PHASE2_BEST_CROP.safetensors \\
        lim.exp_name=l02_background_leakage \\
        lim.style_ref_dir=data/styles/van_gogh

    # L03 — complex scenes
    python scripts/eval/limitations_eval.py \\
        lim.case=L03 \\
        lim.lora_path=output/PHASE2_BEST/PHASE2_BEST.safetensors \\
        lim.exp_name=l03_complex_scenes
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from PIL import Image

log = logging.getLogger(__name__)

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

# Expected dominant colors for L01 prompts (approximate sRGB values).
_L01_EXPECTED_COLORS: dict[str, tuple[int, int, int]] = {
    "a red apple on a wooden table": (200, 30, 30),
    "a yellow school bus parked on a street": (255, 200, 0),
    "blue jeans folded on a chair": (60, 100, 170),
    "a green parrot sitting on a branch": (50, 160, 50),
    "an orange pumpkin in a field": (210, 100, 20),
}


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def _build_flux_pipeline(model_path: str, lora_path: str | None, lora_scale: float):
    from diffusers import FluxPipeline  # type: ignore[import]

    pipe = FluxPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16).to("cuda")
    if lora_path:
        pipe.load_lora_weights(lora_path)
        pipe.fuse_lora(lora_scale=lora_scale)
    return pipe


def _generate_images(
    pipe,
    prompts: list[str],
    out_dir: Path,
    seed: int,
    steps: int,
    guidance: float,
    width: int,
    height: int,
    n_per_prompt: int = 5,
) -> dict[str, list[Path]]:
    """Generate n_per_prompt images per prompt; return mapping prompt → [paths]."""
    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, list[Path]] = {}

    for pidx, prompt in enumerate(prompts):
        paths: list[Path] = []
        for k in range(n_per_prompt):
            gen = torch.Generator("cuda").manual_seed(seed + pidx * 100 + k)
            img = pipe(
                prompt=prompt,
                num_inference_steps=steps,
                guidance_scale=guidance,
                width=width,
                height=height,
                generator=gen,
            ).images[0]
            p = out_dir / f"{pidx:02d}_{k:02d}.png"
            img.save(p)
            paths.append(p)
        result[prompt] = paths
        log.info("Generated %d images for prompt: %s", n_per_prompt, prompt[:60])

    return result


# ---------------------------------------------------------------------------
# L01 — Color leakage (ΔE CIE76)
# ---------------------------------------------------------------------------

def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert (H, W, 3) uint8 RGB to CIE Lab via naive sRGB→XYZ→Lab."""
    from colormath.color_objects import sRGBColor, LabColor  # type: ignore[import]
    from colormath.color_conversions import convert_color  # type: ignore[import]

    r, g, b = rgb[:, :, 0].mean(), rgb[:, :, 1].mean(), rgb[:, :, 2].mean()
    src = sRGBColor(r / 255.0, g / 255.0, b / 255.0)
    lab = convert_color(src, LabColor)
    return np.array([lab.lab_l, lab.lab_a, lab.lab_b])


def _delta_e(lab1: np.ndarray, lab2: np.ndarray) -> float:
    """CIE76 ΔE between two Lab triplets."""
    return float(np.linalg.norm(lab1 - lab2))


def evaluate_l01(
    generated: dict[str, list[Path]],
    expected_colors: dict[str, tuple[int, int, int]],
) -> dict[str, float]:
    """Return mean ΔE per prompt."""
    scores: dict[str, float] = {}
    for prompt, paths in generated.items():
        if prompt not in expected_colors:
            log.warning("No expected color for prompt: %s", prompt)
            continue
        ref_rgb = np.array([[list(expected_colors[prompt])]], dtype=np.uint8)
        ref_lab = _rgb_to_lab(ref_rgb)

        delta_es: list[float] = []
        for p in paths:
            img = np.array(Image.open(p).convert("RGB"))
            gen_lab = _rgb_to_lab(img)
            delta_es.append(_delta_e(ref_lab, gen_lab))

        scores[prompt] = float(np.mean(delta_es))
        log.info("L01 ΔE prompt '%s...': %.2f", prompt[:40], scores[prompt])

    return scores


# ---------------------------------------------------------------------------
# L02 — Background leakage (DINO-object cosine similarity)
# ---------------------------------------------------------------------------

def _load_dino(model_name: str, device: torch.device):
    from transformers import AutoImageProcessor, AutoModel  # type: ignore[import]

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    return model, processor


@torch.no_grad()
def _dino_embed(paths: list[Path], model, processor, device: torch.device) -> torch.Tensor:
    embeddings: list[torch.Tensor] = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        inputs = processor(images=[img], return_tensors="pt").to(device)
        out = model(**inputs)
        cls = out.last_hidden_state[:, 0, :]
        embeddings.append(F.normalize(cls, dim=-1).cpu())
    return torch.cat(embeddings, dim=0)


def evaluate_l02(
    generated_full: dict[str, list[Path]],
    generated_crop: dict[str, list[Path]],
    style_ref_paths: list[Path],
    dino_model_name: str,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    """DINO cosine similarity for full-frame vs center-crop style training.

    Returns {'full': {prompt: score}, 'crop': {prompt: score}}.
    """
    dino_model, dino_proc = _load_dino(dino_model_name, device)
    ref_embs = _dino_embed(style_ref_paths, dino_model, dino_proc, device)

    results: dict[str, dict[str, float]] = {"full": {}, "crop": {}}

    for label, gen_dict in [("full", generated_full), ("crop", generated_crop)]:
        for prompt, paths in gen_dict.items():
            gen_embs = _dino_embed(paths, dino_model, dino_proc, device)
            sim = float((gen_embs @ ref_embs.T).mean())
            results[label][prompt] = sim
            log.info("L02 DINO-object [%s] '%s...': %.4f", label, prompt[:40], sim)

    return results


# ---------------------------------------------------------------------------
# L03 — Complex scenes (CLIP-content)
# ---------------------------------------------------------------------------

def _load_clip(model_name: str, device: torch.device):
    from transformers import CLIPModel, CLIPProcessor  # type: ignore[import]

    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    return model, processor


@torch.no_grad()
def _clip_image_embed(paths: list[Path], model, processor, device: torch.device) -> torch.Tensor:
    images = [Image.open(p).convert("RGB") for p in paths]
    inputs = processor(images=images, return_tensors="pt").to(device)
    feats = model.get_image_features(**inputs)
    return F.normalize(feats, dim=-1).cpu()


@torch.no_grad()
def _clip_text_embed(texts: list[str], model, processor, device: torch.device) -> torch.Tensor:
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
    feats = model.get_text_features(**inputs)
    return F.normalize(feats, dim=-1).cpu()


def evaluate_l03(
    generated: dict[str, list[Path]],
    clip_model_name: str,
    device: torch.device,
) -> dict[str, float]:
    """Mean CLIP image-text cosine similarity per prompt."""
    clip_model, clip_proc = _load_clip(clip_model_name, device)
    scores: dict[str, float] = {}

    for prompt, paths in generated.items():
        img_embs = _clip_image_embed(paths, clip_model, clip_proc, device)
        txt_emb = _clip_text_embed([prompt], clip_model, clip_proc, device)
        sim = float((img_embs @ txt_emb.T).mean())
        scores[prompt] = sim
        log.info("L03 CLIP-content '%s...': %.4f", prompt[:50], sim)

    return scores


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
            task_name=f"limitations/{task_name}",
            reuse_last_task_id=False,
        )
        task.connect(cfg)
        return task
    except Exception as exc:
        log.warning("ClearML init failed (%s), continuing without tracking.", exc)
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_DEFAULT_CFG = {
    "lim": {
        "case": "L01",
        "lora_path": None,
        "lora_path_crop": None,
        "exp_name": "limitations_eval",
        "alpha_style": 1.0,
        "prompts_file": "experiments/data/limitations_prompts.json",
        "style_ref_dir": None,
        "n_images": 5,
    },
    "model": {
        "name_or_path": "/models/flux-dev",
        "clip": "openai/clip-vit-large-patch14",
        "dino": "facebook/dino-vitb8",
    },
    "sampling": {
        "seed": 42,
        "steps": 28,
        "guidance_scale": 3.5,
        "width": 1024,
        "height": 1024,
    },
    "clearml": {
        "enabled": True,
        "project": "blora-flux-limitations",
    },
}


def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Case: %s  Device: %s", cfg.lim.case, device)

    task = _setup_clearml(cfg, f"{cfg.lim.case}_{cfg.lim.exp_name}")

    with open(cfg.lim.prompts_file) as f:
        all_prompts: dict[str, list[str]] = json.load(f)

    out_root = Path("output") / "limitations" / cfg.lim.exp_name
    results: dict = {}

    if cfg.lim.case == "L01":
        prompts = all_prompts["L01_color_leakage"]
        pipe = _build_flux_pipeline(
            cfg.model.name_or_path,
            cfg.lim.lora_path,
            float(cfg.lim.alpha_style),
        )
        gen = _generate_images(
            pipe, prompts, out_root / "images",
            seed=int(cfg.sampling.seed),
            steps=int(cfg.sampling.steps),
            guidance=float(cfg.sampling.guidance_scale),
            width=int(cfg.sampling.width),
            height=int(cfg.sampling.height),
            n_per_prompt=int(cfg.lim.n_images),
        )
        scores = evaluate_l01(gen, _L01_EXPECTED_COLORS)
        results = {"case": "L01", "metric": "delta_e_cie76", "per_prompt": scores,
                   "mean": float(np.mean(list(scores.values())))}

    elif cfg.lim.case == "L02":
        style_refs = []
        if cfg.lim.style_ref_dir:
            d = Path(cfg.lim.style_ref_dir)
            style_refs = sorted(p for p in d.rglob("*") if p.suffix.lower() in _IMAGE_EXTS)

        # Generate with full-frame style LoRA
        prompts_l01 = all_prompts.get("L01_color_leakage", [])
        pipe_full = _build_flux_pipeline(cfg.model.name_or_path, cfg.lim.lora_path, 1.0)
        gen_full = _generate_images(
            pipe_full, prompts_l01, out_root / "full_frame",
            seed=int(cfg.sampling.seed), steps=int(cfg.sampling.steps),
            guidance=float(cfg.sampling.guidance_scale),
            width=int(cfg.sampling.width), height=int(cfg.sampling.height),
            n_per_prompt=int(cfg.lim.n_images),
        )
        del pipe_full

        # Generate with center-crop style LoRA
        gen_crop: dict[str, list[Path]] = {}
        if cfg.lim.lora_path_crop:
            pipe_crop = _build_flux_pipeline(cfg.model.name_or_path, cfg.lim.lora_path_crop, 1.0)
            gen_crop = _generate_images(
                pipe_crop, prompts_l01, out_root / "center_crop",
                seed=int(cfg.sampling.seed), steps=int(cfg.sampling.steps),
                guidance=float(cfg.sampling.guidance_scale),
                width=int(cfg.sampling.width), height=int(cfg.sampling.height),
                n_per_prompt=int(cfg.lim.n_images),
            )
            del pipe_crop

        dino_scores = evaluate_l02(
            gen_full, gen_crop, style_refs,
            str(cfg.model.dino), device,
        )
        results = {"case": "L02", "metric": "dino_cosine", "scores": dino_scores}

    elif cfg.lim.case == "L03":
        prompts = all_prompts["L03_complex_scenes"]
        pipe = _build_flux_pipeline(cfg.model.name_or_path, cfg.lim.lora_path, 1.0)
        gen = _generate_images(
            pipe, prompts, out_root / "images",
            seed=int(cfg.sampling.seed), steps=int(cfg.sampling.steps),
            guidance=float(cfg.sampling.guidance_scale),
            width=int(cfg.sampling.width), height=int(cfg.sampling.height),
            n_per_prompt=int(cfg.lim.n_images),
        )
        scores = evaluate_l03(gen, str(cfg.model.clip), device)
        results = {"case": "L03", "metric": "clip_content", "per_prompt": scores,
                   "mean": float(np.mean(list(scores.values())))}

    else:
        raise ValueError(f"Unknown limitation case: {cfg.lim.case!r}. Choose L01, L02, or L03.")

    # Save results
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / "results.json"
    out_path.write_text(json.dumps(results, indent=2))
    log.info("Results saved to %s", out_path)

    # Print summary
    print(f"\n{'='*50}")
    print(f"  Case: {cfg.lim.case}  ({cfg.lim.exp_name})")
    print(f"{'='*50}")
    if "mean" in results:
        print(f"  Mean {results['metric']}: {results['mean']:.4f}")
    if "per_prompt" in results:
        for prompt, val in results["per_prompt"].items():
            print(f"  {prompt[:45]:<45} {val:.4f}")
    print(f"{'='*50}\n")

    if task is not None:
        logger = task.get_logger()
        if "mean" in results:
            logger.report_scalar("limitations", results["metric"], value=results["mean"], iteration=0)
        if "per_prompt" in results:
            for i, (prompt, val) in enumerate(results["per_prompt"].items()):
                logger.report_scalar(f"limitations/{cfg.lim.case}", f"prompt_{i}", value=val, iteration=0)
        task.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    overrides = sys.argv[1:]
    cfg = OmegaConf.create(_DEFAULT_CFG)
    if overrides:
        cli_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, cli_cfg)

    main(cfg)
