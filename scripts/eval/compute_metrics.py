"""Compute CLIP-style, CLIP-content, DINO-style, FID and LPIPS for a folder of generated images.

Usage (single style reference):
    python scripts/eval/compute_metrics.py \\
        metrics.generated_dir=output/generated/e01_blora_flux_van_gogh_img1 \\
        metrics.style_ref=data/styles/van_gogh/img1/reference.jpg \\
        metrics.prompt_file=data/coco_prompts.txt \\
        metrics.artbench_dir=data/artbench10 \\
        metrics.exp_name=e01_blora_flux_van_gogh_img1

Usage (multiple style references — averaged):
    python scripts/eval/compute_metrics.py \\
        metrics.generated_dir=output/generated/e01_blora_flux_van_gogh_img1 \\
        metrics.style_refs_dir=data/styles/van_gogh \\
        metrics.prompt_file=data/coco_prompts.txt \\
        metrics.exp_name=e01_blora_flux_van_gogh_img1

ClearML scalar keys logged (match update_exp_plan.py _METRIC_KEYS):
    eval / clip_style, eval / clip_content, eval / dino_style, eval / fid, eval / lpips

DINO ViT-B/8 is the primary style metric matching the original B-LoRA paper.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from PIL import Image

log = logging.getLogger(__name__)

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_style_refs(cfg: DictConfig) -> list[Path]:
    """Return sorted list of style reference image paths."""
    if cfg.metrics.style_refs_dir is not None:
        d = Path(str(cfg.metrics.style_refs_dir))
        refs = sorted(p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in _IMAGE_EXTS)
        if not refs:
            raise FileNotFoundError(f"No images found in style_refs_dir: {d}")
        log.info("Style refs (dir): %d images from %s", len(refs), d)
        return refs

    if cfg.metrics.style_ref is not None:
        p = Path(str(cfg.metrics.style_ref))
        if not p.exists():
            raise FileNotFoundError(f"style_ref not found: {p}")
        log.info("Style ref (single): %s", p)
        return [p]

    return []


def _load_generated(generated_dir: Path) -> list[Path]:
    paths = sorted(p for p in generated_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTS)
    if not paths:
        raise FileNotFoundError(f"No generated images found in {generated_dir}")
    log.info("Generated images: %d from %s", len(paths), generated_dir)
    return paths


def _load_prompts(prompt_file: str) -> list[str]:
    path = Path(prompt_file)
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    prompts = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    log.info("Prompts: %d from %s", len(prompts), path)
    return prompts


# ---------------------------------------------------------------------------
# CLIP helpers
# ---------------------------------------------------------------------------

def _load_clip(model_name: str, device: torch.device):
    from transformers import CLIPModel, CLIPProcessor  # type: ignore[import]

    log.info("Loading CLIP: %s", model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    return model, processor


@torch.no_grad()
def _encode_images_clip(
    paths: list[Path],
    model,
    processor,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """Return L2-normalised image embeddings, shape (N, D)."""
    embeddings: list[torch.Tensor] = []
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i : i + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = processor(images=images, return_tensors="pt").to(device)
        feats = model.get_image_features(**inputs)
        embeddings.append(F.normalize(feats, dim=-1).cpu())
    return torch.cat(embeddings, dim=0)  # (N, D)


@torch.no_grad()
def _encode_texts_clip(
    texts: list[str],
    model,
    processor,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """Return L2-normalised text embeddings, shape (N, D)."""
    embeddings: list[torch.Tensor] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = processor(text=batch, return_tensors="pt", padding=True, truncation=True).to(device)
        feats = model.get_text_features(**inputs)
        embeddings.append(F.normalize(feats, dim=-1).cpu())
    return torch.cat(embeddings, dim=0)  # (N, D)


# ---------------------------------------------------------------------------
# DINO helpers
# ---------------------------------------------------------------------------

def _load_dino(model_name: str, device: torch.device):
    from transformers import AutoImageProcessor, AutoModel  # type: ignore[import]

    log.info("Loading DINO: %s", model_name)
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    return model, processor


@torch.no_grad()
def _encode_images_dino(
    paths: list[Path],
    model,
    processor,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """Return L2-normalised [CLS] token embeddings from DINO, shape (N, D)."""
    embeddings: list[torch.Tensor] = []
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i : i + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = processor(images=images, return_tensors="pt").to(device)
        outputs = model(**inputs)
        cls_tokens = outputs.last_hidden_state[:, 0, :]  # (B, D) — [CLS] token
        embeddings.append(F.normalize(cls_tokens, dim=-1).cpu())
    return torch.cat(embeddings, dim=0)  # (N, D)


def compute_dino_style(
    generated_paths: list[Path],
    style_refs: list[Path],
    dino_model,
    dino_processor,
    device: torch.device,
    batch_size: int,
) -> float:
    """Mean DINO ViT-B/8 cosine similarity between generated images and style references.

    This is the primary metric used in the original B-LoRA paper (Table 1).
    For M references and N generated images, returns mean of all M×N similarity values.
    """
    ref_embs = _encode_images_dino(style_refs, dino_model, dino_processor, device, batch_size)
    gen_embs = _encode_images_dino(generated_paths, dino_model, dino_processor, device, batch_size)
    sim_matrix = gen_embs @ ref_embs.T  # (N, M)
    return float(sim_matrix.mean())


# ---------------------------------------------------------------------------
# Metric computations
# ---------------------------------------------------------------------------

def compute_clip_style(
    generated_paths: list[Path],
    style_refs: list[Path],
    clip_model,
    clip_processor,
    device: torch.device,
    batch_size: int,
) -> float:
    """Mean cosine similarity between each generated image and each style reference.

    For M references and N generated images, computes the mean of M×N similarity values.
    This is mathematically correct: mean(cos_sim(gen_i, ref_j) for all i,j).
    Note: "similarity to mean ref embedding" ≠ "mean similarity" when M > 1.
    """
    ref_embs = _encode_images_clip(style_refs, clip_model, clip_processor, device, batch_size)
    gen_embs = _encode_images_clip(generated_paths, clip_model, clip_processor, device, batch_size)

    # ref_embs: (M, D),  gen_embs: (N, D)  — already L2-normalised
    sim_matrix = gen_embs @ ref_embs.T  # (N, M)
    return float(sim_matrix.mean())


def compute_clip_content(
    generated_paths: list[Path],
    prompts: list[str],
    clip_model,
    clip_processor,
    device: torch.device,
    batch_size: int,
) -> float:
    """Mean cosine similarity between each generated image and its corresponding prompt."""
    n = min(len(generated_paths), len(prompts))
    if n < len(generated_paths):
        log.warning(
            "Fewer prompts (%d) than images (%d); truncating to %d pairs.",
            len(prompts), len(generated_paths), n,
        )

    gen_embs = _encode_images_clip(generated_paths[:n], clip_model, clip_processor, device, batch_size)
    txt_embs = _encode_texts_clip(prompts[:n], clip_model, clip_processor, device, batch_size)

    # Element-wise dot product of already-normalised vectors = cosine similarity
    similarities = (gen_embs * txt_embs).sum(dim=-1)  # (N,)
    return float(similarities.mean())


def compute_lpips_score(
    generated_paths: list[Path],
    style_refs: list[Path],
    device: torch.device,
    net: str = "vgg",
    target_size: int = 512,
) -> float:
    """Mean LPIPS between generated images and each style reference, averaged over refs.

    normalize=True tells lpips to scale [0, 1] → [-1, 1] internally;
    without it the library expects [-1, 1] and [0, 1] inputs give wrong values.
    """
    import lpips  # type: ignore[import]
    import torchvision.transforms.functional as TF  # type: ignore[import]

    loss_fn = lpips.LPIPS(net=net).to(device)
    loss_fn.eval()

    def _to_tensor(path: Path) -> torch.Tensor:
        img = Image.open(path).convert("RGB").resize((target_size, target_size))
        t = TF.to_tensor(img).unsqueeze(0)  # (1, 3, H, W) in [0, 1]
        return t * 2 - 1  # LPIPS expects [-1, 1]

    scores_per_ref: list[float] = []
    for ref_path in style_refs:
        ref_t = _to_tensor(ref_path).to(device)
        scores: list[float] = []
        for gen_path in generated_paths:
            gen_t = _to_tensor(gen_path).to(device)
            with torch.no_grad():
                d = loss_fn(gen_t, ref_t)
            scores.append(float(d))
        scores_per_ref.append(sum(scores) / len(scores))
        log.debug("LPIPS vs %s: %.4f", ref_path.name, scores_per_ref[-1])

    return sum(scores_per_ref) / len(scores_per_ref)


def compute_fid(generated_dir: Path, artbench_dir: Path) -> float:
    """Fréchet Inception Distance via clean-fid (mode='clean')."""
    from cleanfid import fid  # type: ignore[import]

    for label, d in [("generated_dir", generated_dir), ("artbench_dir", artbench_dir)]:
        if not d.is_dir():
            raise FileNotFoundError(f"{label} not found or not a directory: {d}")
        imgs = [p for p in d.iterdir() if p.suffix.lower() in _IMAGE_EXTS]
        if not imgs:
            raise FileNotFoundError(f"No images found in {label}: {d}")
        log.info("%s: %d images", label, len(imgs))

    log.info("Computing FID: %s vs %s", generated_dir, artbench_dir)
    return float(fid.compute_fid(str(generated_dir), str(artbench_dir), mode="clean"))


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
        task.connect(cfg)
        return task
    except Exception as exc:
        log.warning("ClearML init failed (%s), continuing without tracking.", exc)
        return None


def _log_metrics(task, results: dict[str, float | None]) -> None:
    if task is None:
        return
    logger = task.get_logger()
    for series, value in results.items():
        if value is not None:
            logger.report_scalar("eval", series, value=value, iteration=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    task = _setup_clearml(cfg, cfg.metrics.exp_name)

    generated_dir = Path(str(cfg.metrics.generated_dir))
    generated_paths = _load_generated(generated_dir)
    style_refs = _load_style_refs(cfg)
    prompts = _load_prompts(str(cfg.metrics.prompt_file))

    results: dict[str, float | None] = {
        "clip_style": None,
        "clip_content": None,
        "dino_style": None,
        "fid": None,
        "lpips": None,
    }

    # --- CLIP (shared model for style + content) ---
    clip_model, clip_processor = _load_clip(str(cfg.model.clip), device)
    batch_size = int(cfg.model.batch_size)

    if style_refs:
        log.info("Computing CLIP-style (%d refs)...", len(style_refs))
        results["clip_style"] = compute_clip_style(
            generated_paths, style_refs, clip_model, clip_processor, device, batch_size
        )
    else:
        log.warning("No style_ref / style_refs_dir provided — skipping CLIP-style.")

    log.info("Computing CLIP-content...")
    results["clip_content"] = compute_clip_content(
        generated_paths, prompts, clip_model, clip_processor, device, batch_size
    )

    del clip_model, clip_processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- DINO ViT-B/8 (primary metric from B-LoRA paper) ---
    if style_refs:
        log.info("Computing DINO-style (%d refs)...", len(style_refs))
        dino_model, dino_processor = _load_dino(str(cfg.model.dino), device)
        results["dino_style"] = compute_dino_style(
            generated_paths, style_refs, dino_model, dino_processor, device, batch_size
        )
        del dino_model, dino_processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        log.warning("No style_ref / style_refs_dir provided — skipping DINO-style.")

    # --- LPIPS ---
    if style_refs:
        log.info("Computing LPIPS (%d refs)...", len(style_refs))
        results["lpips"] = compute_lpips_score(
            generated_paths, style_refs, device, net=str(cfg.model.lpips_net)
        )
    else:
        log.warning("No style_ref / style_refs_dir provided — skipping LPIPS.")

    # --- FID ---
    if cfg.metrics.artbench_dir is not None:
        artbench_dir = Path(str(cfg.metrics.artbench_dir))
        results["fid"] = compute_fid(generated_dir, artbench_dir)
    else:
        log.warning("artbench_dir not set — skipping FID.")

    # --- Log + print ---
    _log_metrics(task, results)

    print("\n" + "=" * 50)
    print(f"  Experiment: {cfg.metrics.exp_name}")
    print("=" * 50)
    for name, value in results.items():
        display = f"{value:.4f}" if value is not None else "—"
        print(f"  {name:<16} {display}")
    print("=" * 50 + "\n")

    out_path = Path("output") / "results" / f"{cfg.metrics.exp_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"exp_name": cfg.metrics.exp_name, **{k: v for k, v in results.items()}}
    out_path.write_text(json.dumps(payload, indent=2))
    log.info("Metrics saved to %s", out_path)


if __name__ == "__main__":
    import sys
    from hydra import compose, initialize_config_dir  # type: ignore[import]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    config_dir = str((Path(__file__).resolve().parents[2] / "configs" / "eval").resolve())
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(config_name="metrics", overrides=sys.argv[1:])
    main(cfg)
