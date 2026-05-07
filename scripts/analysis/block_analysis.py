"""Block-level prompt-injection analysis for FLUX.1-dev (Phase 0, B-LoRA thesis).

Reproduces Section 4.1 of the B-LoRA paper (arXiv:2403.14572):
for each transformer block, T5/CLIP embeddings from P_inject are substituted
via forward pre-hooks and CLIP text-image similarity of the resulting image is
measured. The procedure is run in both directions (style_inject, content_inject)
to separate style-sensitive blocks from content-sensitive ones.

Usage:
    python scripts/analysis/block_analysis.py
    python scripts/analysis/block_analysis.py analysis.n_prompts=20 model.name_or_path=...
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tqdm import tqdm

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Built-in prompt corpus (used when prompts_file does not exist)
# ---------------------------------------------------------------------------

_OBJECTS = [
    "dog", "cat", "bird", "horse", "car", "boat", "chair", "table", "tree",
    "flower", "house", "mountain", "river", "cloud", "city", "beach", "forest",
    "castle", "tower", "bridge", "lighthouse", "windmill", "barn", "cottage",
    "temple", "church", "statue", "fountain", "garden", "waterfall", "volcano",
    "glacier", "desert", "savanna", "tundra", "canyon", "cliff", "island",
    "valley", "meadow",
]

_COLORS = [
    "red", "blue", "green", "yellow", "purple", "orange", "pink", "white",
    "black", "golden", "silver", "emerald", "crimson", "violet", "turquoise",
    "amber", "azure", "scarlet", "indigo", "teal",
]


def _build_prompt_corpus() -> list[dict[str, str]]:
    """Generate 200 (content, style) prompt pairs from built-in object/colour lists."""
    pairs: list[dict[str, str]] = []
    rng = random.Random(42)
    seen: set[tuple[str, str]] = set()
    while len(pairs) < 200:
        obj = rng.choice(_OBJECTS)
        color = rng.choice(_COLORS)
        key = (obj, color)
        if key in seen:
            continue
        seen.add(key)
        pairs.append({"content": f"A photo of a {obj}", "style": f"A photo of a {color} {obj}"})
    return pairs


# ---------------------------------------------------------------------------
# Seed helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set all RNG seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Prompt file helpers
# ---------------------------------------------------------------------------

def _load_or_create_prompts(prompts_file: str, n_prompts: int, seed: int) -> list[dict[str, str]]:
    """Load prompt pairs from JSON, creating the file with a built-in corpus if absent."""
    path = Path(prompts_file)
    if not path.exists():
        log.info("Prompts file not found — generating built-in corpus: %s", path)
        path.parent.mkdir(parents=True, exist_ok=True)
        corpus = _build_prompt_corpus()
        path.write_text(json.dumps(corpus, indent=2, ensure_ascii=False))
        log.info("Wrote %d prompt pairs to %s", len(corpus), path)
    else:
        corpus = json.loads(path.read_text())
        log.info("Loaded %d prompt pairs from %s", len(corpus), path)

    if len(corpus) > n_prompts:
        rng = random.Random(seed)
        corpus = rng.sample(corpus, n_prompts)
        log.info("Subsampled to %d pairs (seed=%d)", n_prompts, seed)

    return corpus


# ---------------------------------------------------------------------------
# FLUX pipeline + embedding extraction
# ---------------------------------------------------------------------------

def _resolve_dtype(dtype_str: str) -> torch.dtype:
    mapping: dict[str, torch.dtype] = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype_str not in mapping:
        raise ValueError(f"Unknown dtype '{dtype_str}'. Choose from {list(mapping)}")
    return mapping[dtype_str]


def _build_pipeline(model_path: str, dtype: torch.dtype) -> Any:
    """Load FLUX pipeline onto CUDA."""
    from diffusers import FluxPipeline  # type: ignore[import]

    log.info("Loading FLUX pipeline: %s  (dtype=%s)", model_path, dtype)
    pipe = FluxPipeline.from_pretrained(model_path, torch_dtype=dtype)
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe


@torch.no_grad()
def _encode_prompt(pipe: Any, prompt: str, device: torch.device) -> dict[str, torch.Tensor]:
    """Encode a prompt through FLUX's dual text encoders (T5 + CLIP).

    Returns:
        Dict with keys:
          - prompt_embeds: T5 hidden states (1, seq_len, 4096)
          - pooled_prompt_embeds: CLIP pooled output (1, 768)
          - text_ids: T5 token positional ids (1, seq_len, 3)
    """
    prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=None,
        device=device,
        num_images_per_prompt=1,
        max_sequence_length=512,
    )
    return {
        "prompt_embeds": prompt_embeds,
        "pooled_prompt_embeds": pooled_prompt_embeds,
        "text_ids": text_ids,
    }


# ---------------------------------------------------------------------------
# Single image generation with block-level injection
# ---------------------------------------------------------------------------

@torch.no_grad()
def _generate_with_injection(
    pipe: Any,
    base_embeds: dict[str, torch.Tensor],
    inject_embeds: dict[str, torch.Tensor],
    block_type: str,
    block_idx: int,
    cfg_sampling: DictConfig,
    generator: torch.Generator,
) -> Image.Image:
    """Run one denoising pass with T5 embedding injection at a specific block.

    For double-stream blocks: replaces encoder_hidden_states kwarg at the target
    block only. pooled_projections is intentionally left unchanged — it feeds the
    timestep modulation (same for all blocks) and replacing it globally would
    confound the per-block sensitivity measurement.

    For single-stream blocks: fused hidden_states = cat([img_tokens, txt_tokens]).
    The T5 embeddings are projected from 4096→3072 via the transformer's own
    context_embedder (Linear(4096, 3072)) before being concatenated with image
    tokens. We apply the same projection to inject_embeds before replacing the
    text slice of the fused tensor.

    All hooks are removed in a finally block to prevent state leakage across calls.
    """
    transformer = pipe.transformer
    device = next(transformer.parameters()).device

    inject_enc_hs = inject_embeds["prompt_embeds"].to(device)
    handles: list[Any] = []

    if block_type == "double":
        block = transformer.transformer_blocks[block_idx]

        def _ds_pre_hook(module: torch.nn.Module, args: tuple, kwargs: dict) -> tuple[tuple, dict]:  # type: ignore[type-arg]
            if "encoder_hidden_states" in kwargs:
                kwargs = {**kwargs, "encoder_hidden_states": inject_enc_hs}
            elif len(args) >= 2:
                args = (args[0], inject_enc_hs) + args[2:]
            return args, kwargs

        # with_kwargs=True: receive and return both args and kwargs from pre-hook
        h = block.register_forward_pre_hook(_ds_pre_hook, with_kwargs=True)
        handles.append(h)

    else:
        # Single-stream: project T5 embeddings (4096) → FLUX hidden dim (3072)
        # using the transformer's own context_embedder weights (already trained).
        ctx_embedder = transformer.context_embedder  # Linear(4096, 3072)
        with torch.no_grad():
            inject_enc_proj = ctx_embedder(inject_enc_hs.to(ctx_embedder.weight.dtype))
        # inject_enc_proj: (1, seq_len, 3072)

        _state: dict[str, int | None] = {"img_seq_len": None}

        def _transformer_img_ids_hook(
            module: torch.nn.Module, args: tuple, kwargs: dict
        ) -> tuple[tuple, dict]:  # type: ignore[type-arg]
            """Capture img_seq_len from img_ids before the transformer runs."""
            if "img_ids" in kwargs:
                _state["img_seq_len"] = kwargs["img_ids"].shape[1]
            return args, kwargs

        h_t = transformer.register_forward_pre_hook(_transformer_img_ids_hook, with_kwargs=True)
        handles.append(h_t)

        target_ss_block = transformer.single_transformer_blocks[block_idx]

        def _ss_pre_hook(
            module: torch.nn.Module, args: tuple, kwargs: dict
        ) -> tuple[tuple, dict]:  # type: ignore[type-arg]
            """Replace the text slice of fused hidden_states with inject embeddings."""
            img_seq_len = _state["img_seq_len"]
            if img_seq_len is None or not args:
                return args, kwargs

            hidden = args[0]  # (B, img_seq + txt_seq, 3072)
            txt_seq_len = inject_enc_proj.shape[1]
            total_seq = hidden.shape[1]

            if img_seq_len + txt_seq_len != total_seq:
                log.debug(
                    "SS block %d: seq mismatch img=%d txt=%d total=%d — skipping",
                    block_idx, img_seq_len, txt_seq_len, total_seq,
                )
                return args, kwargs

            img_part = hidden[:, :img_seq_len, :]
            fused = torch.cat([img_part, inject_enc_proj.to(hidden.dtype)], dim=1)
            args = (fused,) + args[1:]
            return args, kwargs

        h_ss = target_ss_block.register_forward_pre_hook(_ss_pre_hook, with_kwargs=True)
        handles.append(h_ss)

    try:
        result = pipe(
            prompt_embeds=base_embeds["prompt_embeds"].to(device),
            pooled_prompt_embeds=base_embeds["pooled_prompt_embeds"].to(device),
            text_ids=base_embeds["text_ids"].to(device),
            num_inference_steps=int(cfg_sampling.steps),
            guidance_scale=float(cfg_sampling.guidance_scale),
            width=int(cfg_sampling.width),
            height=int(cfg_sampling.height),
            generator=generator,
            output_type="pil",
        )
        image: Image.Image = result.images[0]
    finally:
        for h in handles:
            h.remove()

    return image


# ---------------------------------------------------------------------------
# CLIP similarity
# ---------------------------------------------------------------------------

def _load_clip(model_name: str, device: torch.device) -> tuple[Any, Any]:
    from transformers import CLIPModel, CLIPProcessor  # type: ignore[import]

    log.info("Loading CLIP: %s", model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    return model, processor


@torch.no_grad()
def _clip_text_image_sim(
    image: Image.Image,
    text: str,
    clip_model: Any,
    clip_processor: Any,
    device: torch.device,
) -> float:
    """L2-normalised cosine similarity between one image and one text string."""
    img_inputs = clip_processor(images=image, return_tensors="pt").to(device)
    txt_inputs = clip_processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(device)

    img_feat = F.normalize(clip_model.get_image_features(**img_inputs), dim=-1)
    txt_feat = F.normalize(clip_model.get_text_features(**txt_inputs), dim=-1)

    return float((img_feat * txt_feat).sum(dim=-1).item())


# ---------------------------------------------------------------------------
# ClearML helpers
# ---------------------------------------------------------------------------

def _setup_clearml(cfg: DictConfig) -> Any:
    if not cfg.clearml.enabled:
        return None
    try:
        from clearml import Task  # type: ignore[import]

        task = Task.init(
            project_name=cfg.clearml.project,
            task_name=f"{cfg.clearml.task_prefix}/block_analysis",
            reuse_last_task_id=False,
        )
        task.connect(cfg)
        return task
    except Exception as exc:
        log.warning("ClearML init failed (%s), continuing without tracking.", exc)
        return None


def _log_clearml_scalar(task: Any, series: str, value: float, iteration: int) -> None:
    if task is None:
        return
    task.get_logger().report_scalar("block_analysis", series, value=value, iteration=iteration)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _write_report(results: dict[str, dict[str, Any]], output_dir: Path) -> None:
    """Write a ranked Markdown report to output_dir/block_analysis_report.md."""
    lines: list[str] = [
        "# Block Analysis Report — FLUX.1-dev",
        "",
        "Scores are mean CLIP text-image cosine similarity over all prompt pairs.",
        "Higher score = stronger sensitivity to the injected embedding.",
        "",
        "## Style Injection (base=P_content, inject=P_style)",
        "Blocks where score is high are **style-sensitive**.",
        "",
        "| Block ID | Type | CLIP-style score |",
        "|----------|------|-----------------|",
    ]
    style_rows = sorted(
        ((bid, info) for bid, info in results.items() if info.get("direction") == "style_inject"),
        key=lambda x: x[1]["clip_sim"],
        reverse=True,
    )
    for block_id, info in style_rows:
        lines.append(f"| {block_id} | {info['block_type']} | {info['clip_sim']:.4f} |")

    lines += [
        "",
        "## Content Injection (base=P_style, inject=P_content)",
        "Blocks where score is high are **content-sensitive**.",
        "",
        "| Block ID | Type | CLIP-content score |",
        "|----------|------|-------------------|",
    ]
    content_rows = sorted(
        ((bid, info) for bid, info in results.items() if info.get("direction") == "content_inject"),
        key=lambda x: x[1]["clip_sim"],
        reverse=True,
    )
    for block_id, info in content_rows:
        lines.append(f"| {block_id} | {info['block_type']} | {info['clip_sim']:.4f} |")

    lines += ["", "---", "_Generated by scripts/analysis/block_analysis.py_"]
    report_path = output_dir / "block_analysis_report.md"
    report_path.write_text("\n".join(lines))
    log.info("Report written to %s", report_path)


# ---------------------------------------------------------------------------
# Main analysis loop
# ---------------------------------------------------------------------------

def _run_analysis(cfg: DictConfig) -> None:
    set_seed(int(cfg.analysis.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    output_dir = Path(str(cfg.analysis.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    config_snapshot = output_dir / "block_analysis_config.yaml"
    config_snapshot.write_text(OmegaConf.to_yaml(cfg))
    log.info("Config saved to %s", config_snapshot)

    task = _setup_clearml(cfg)

    pairs = _load_or_create_prompts(
        str(cfg.analysis.prompts_file),
        int(cfg.analysis.n_prompts),
        int(cfg.analysis.seed),
    )
    log.info("Using %d prompt pairs", len(pairs))

    target_blocks: list[tuple[str, int, str]] = []
    for idx in cfg.analysis.target_ds_blocks:
        target_blocks.append(("double", int(idx), f"ds_{int(idx):02d}"))
    for idx in cfg.analysis.target_ss_blocks:
        target_blocks.append(("single", int(idx), f"ss_{int(idx):02d}"))

    log.info(
        "Target blocks: %d double-stream, %d single-stream",
        len(cfg.analysis.target_ds_blocks),
        len(cfg.analysis.target_ss_blocks),
    )

    dtype = _resolve_dtype(str(cfg.model.dtype))
    pipe = _build_pipeline(str(cfg.model.name_or_path), dtype)
    clip_model, clip_processor = _load_clip(str(cfg.model.clip), device)

    # Cache embeddings for all pairs once — avoids 48x redundant T5 encode calls.
    log.info("Pre-encoding %d prompt pairs...", len(pairs))
    all_embeds: list[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]] = []
    for pair in tqdm(pairs, desc="Encoding prompts"):
        content_emb = _encode_prompt(pipe, pair["content"], device)
        style_emb = _encode_prompt(pipe, pair["style"], device)
        all_embeds.append((content_emb, style_emb))

    base_seed = int(cfg.sampling.seed)
    accumulator: dict[str, list[float]] = {}
    structured: dict[str, dict[str, Any]] = {}
    results_path = output_dir / "block_analysis_results.json"

    for block_type, block_idx, block_id in tqdm(target_blocks, desc="Blocks"):
        key_style = f"{block_id}_style"
        key_content = f"{block_id}_content"
        accumulator[key_style] = []
        accumulator[key_content] = []

        log.info("Processing block %s (%s idx=%d)", block_id, block_type, block_idx)

        for pair_idx, pair in enumerate(tqdm(pairs, desc=block_id, leave=False)):
            content_embeds, style_embeds = all_embeds[pair_idx]

            # Direction 1: style_inject — base=P_content, inject=P_style at block N.
            # High CLIP-sim(output, P_style) → block is style-sensitive.
            gen_style = torch.Generator("cpu").manual_seed(base_seed + pair_idx)
            img_style_inject = _generate_with_injection(
                pipe=pipe,
                base_embeds=content_embeds,
                inject_embeds=style_embeds,
                block_type=block_type,
                block_idx=block_idx,
                cfg_sampling=cfg.sampling,
                generator=gen_style,
            )
            sim_style = _clip_text_image_sim(img_style_inject, pair["style"], clip_model, clip_processor, device)
            accumulator[key_style].append(sim_style)

            # Direction 2: content_inject — base=P_style, inject=P_content at block N.
            # High CLIP-sim(output, P_content) → block is content-sensitive.
            # Seed offset * 2 avoids any overlap with style seeds for any n_prompts.
            gen_content = torch.Generator("cpu").manual_seed(base_seed * 2 + pair_idx)
            img_content_inject = _generate_with_injection(
                pipe=pipe,
                base_embeds=style_embeds,
                inject_embeds=content_embeds,
                block_type=block_type,
                block_idx=block_idx,
                cfg_sampling=cfg.sampling,
                generator=gen_content,
            )
            sim_content = _clip_text_image_sim(
                img_content_inject, pair["content"], clip_model, clip_processor, device
            )
            accumulator[key_content].append(sim_content)

        mean_style = float(np.mean(accumulator[key_style]))
        mean_content = float(np.mean(accumulator[key_content]))
        log.info("Block %s — CLIP-style=%.4f  CLIP-content=%.4f", block_id, mean_style, mean_content)

        _log_clearml_scalar(task, f"clip_style_{block_id}", mean_style, iteration=0)
        _log_clearml_scalar(task, f"clip_content_{block_id}", mean_content, iteration=0)

        structured[key_style] = {
            "direction": "style_inject",
            "block_type": block_type,
            "block_idx": block_idx,
            "clip_sim": mean_style,
        }
        structured[key_content] = {
            "direction": "content_inject",
            "block_type": block_type,
            "block_idx": block_idx,
            "clip_sim": mean_content,
        }

        # Incremental save — preserves progress if the run crashes after many hours.
        payload: dict[str, Any] = {
            "config": OmegaConf.to_container(cfg, resolve=True),
            "n_pairs": len(pairs),
            "results": structured,
            "raw_per_block": {k: accumulator[k] for k in accumulator},
        }
        results_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

        torch.cuda.empty_cache()

    log.info("Final results saved to %s", results_path)
    _write_report(structured, output_dir)

    mean_scores: dict[str, float] = {k: float(np.mean(v)) for k, v in accumulator.items()}

    print("\n" + "=" * 60)
    print("  Block Analysis Summary")
    print("=" * 60)
    print(f"  {'Block':<12} {'Type':<8} {'CLIP-style':>12} {'CLIP-content':>14}")
    print("  " + "-" * 50)
    for block_type, block_idx, block_id in target_blocks:
        s = mean_scores.get(f"{block_id}_style", 0.0)
        c = mean_scores.get(f"{block_id}_content", 0.0)
        print(f"  {block_id:<12} {block_type:<8} {s:>12.4f} {c:>14.4f}")
    print("=" * 60 + "\n")

    if task is not None:
        task.close()


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------

def main(cfg: DictConfig) -> None:
    """Hydra entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    _run_analysis(cfg)


if __name__ == "__main__":
    import hydra  # type: ignore[import]

    @hydra.main(version_base="1.3", config_path="../../configs/analysis", config_name="block_analysis")
    def _hydra_main(cfg: DictConfig) -> None:
        main(cfg)

    _hydra_main()
