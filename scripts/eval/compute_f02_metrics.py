"""Compute per-pair DINO-style and DINO-content for the F02 pair-driver protocol.

Inputs:
  - manifest_path: experiments/data/b_lora_eval_pairs.json
  - generated_dir: folder produced by generate_mixing_sdxl.py with files named
                   pair_{pair_id:03d}_{style_id}__{subject_id}.png

For each pair:
  DINO-style[pair_id]   = cos(emb(generated_pair_image), emb(single_style_ref_jpg))
  DINO-content[pair_id] = cos(emb(generated_pair_image), mean_emb(all_imgs_in_content_subject_dir))

Aggregates: grand_mean DINO-style, grand_mean DINO-content,
            STI = DINO-content_mean - DINO-style_mean,
            per-artist means (Van Gogh / Claude Monet).

Output: output/results/<exp_name>_f02_metrics.json
"""

from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

# Reuse DINO helpers from the project-level metrics module
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_metrics import _IMAGE_EXTS, _encode_images_dino, _load_dino  # noqa: E402

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def _resolve_style_ref(style_id: str, base_dir: Path) -> Path:
    """style_id like 'wikimo_04' -> data/styles/monet/img4/<first .jpg>."""
    prefix, num = style_id.split("_", 1)
    artist_dir = {"wikimo": "monet", "wikivg": "van_gogh"}.get(prefix)
    if artist_dir is None:
        raise ValueError(f"Unknown style_id prefix in {style_id!r} (expected wikimo_/wikivg_)")
    img_dir = base_dir / artist_dir / f"img{int(num)}"
    if not img_dir.is_dir():
        raise FileNotFoundError(f"Style ref dir not found for {style_id}: {img_dir}")
    imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTS)
    if not imgs:
        raise FileNotFoundError(f"No images under {img_dir}")
    if len(imgs) > 1:
        log.debug("Multiple style refs in %s (%d); using first: %s", img_dir, len(imgs), imgs[0].name)
    return imgs[0]


def _resolve_content_refs(subject_id: str, base_dir: Path) -> list[Path]:
    d = base_dir / subject_id
    if not d.is_dir():
        raise FileNotFoundError(f"Content refs dir not found for subject={subject_id!r}: {d}")
    imgs = sorted(p for p in d.iterdir() if p.suffix.lower() in _IMAGE_EXTS)
    if not imgs:
        raise FileNotFoundError(f"No images under {d}")
    return imgs


def _resolve_generated(generated_dir: Path, pair_id: int, style_id: str, subject_id: str) -> Path:
    p = generated_dir / f"pair_{pair_id:03d}_{style_id}__{subject_id}.png"
    if not p.exists():
        raise FileNotFoundError(f"Generated image not found: {p}")
    return p


def _artist_for_style_id(style_id: str, mapping: dict[str, str]) -> str:
    if style_id in mapping:
        return str(mapping[style_id])
    prefix = style_id.split("_", 1)[0]
    wildcard = f"{prefix}_*"
    if wildcard in mapping:
        return str(mapping[wildcard])
    raise KeyError(f"No artist mapping for style_id={style_id!r}")


# ---------------------------------------------------------------------------
# Manifest
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
    if subset is None:
        return pairs
    if subset == "user_study":
        ids = manifest.get("user_study_pair_ids")
        if not ids:
            raise ValueError("pair_subset=user_study requested but manifest has no user_study_pair_ids")
        id_set = set(int(i) for i in ids)
        return [p for p in pairs if int(p["pair_id"]) in id_set]
    raise ValueError(f"Unknown pair_subset: {subset!r}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cfg: DictConfig) -> None:
    m = cfg.f02_metrics
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    manifest = _load_manifest(Path(str(m.manifest_path)))
    subset = m.get("pair_subset", None)
    pairs = _select_pairs(manifest, subset)
    if not pairs:
        raise ValueError(f"No pairs selected (subset={subset!r})")
    log.info("Pairs to evaluate: %d (subset=%s)", len(pairs), subset)

    generated_dir = Path(str(m.generated_dir))
    style_refs_base = Path(str(m.style_refs_dir_base))
    content_refs_base = Path(str(m.content_refs_dir_base))
    artist_map = dict(OmegaConf.to_container(m.artist_from_style_id, resolve=True))  # type: ignore[arg-type]

    dino_model, dino_proc = _load_dino(str(m.dino_model), device)
    batch_size = int(m.batch_size)

    # Cache content-subject mean embeddings: each subject's mean is computed once.
    content_mean_cache: dict[str, torch.Tensor] = {}
    # Cache style ref embeddings: one image per style_id, computed once.
    style_emb_cache: dict[str, torch.Tensor] = {}

    per_pair: list[dict[str, Any]] = []
    log_every = int(m.get("log_every", 10))

    for idx, pair in enumerate(pairs):
        pair_id = int(pair["pair_id"])
        subject_id = str(pair["content_subject_id"])
        style_id = str(pair["style_id"])

        gen_path = _resolve_generated(generated_dir, pair_id, style_id, subject_id)
        gen_emb = _encode_images_dino([gen_path], dino_model, dino_proc, device, batch_size)[0]

        if style_id not in style_emb_cache:
            ref_path = _resolve_style_ref(style_id, style_refs_base)
            style_emb_cache[style_id] = _encode_images_dino(
                [ref_path], dino_model, dino_proc, device, batch_size
            )[0]
        style_emb = style_emb_cache[style_id]

        if subject_id not in content_mean_cache:
            ref_paths = _resolve_content_refs(subject_id, content_refs_base)
            ref_embs = _encode_images_dino(ref_paths, dino_model, dino_proc, device, batch_size)
            # Mean of unit-norm vectors, then re-normalize so cosine remains well-defined.
            mean_vec = ref_embs.mean(dim=0)
            mean_vec = mean_vec / mean_vec.norm().clamp_min(1e-12)
            content_mean_cache[subject_id] = mean_vec
        content_emb = content_mean_cache[subject_id]

        d_style = float((gen_emb @ style_emb).item())
        d_content = float((gen_emb @ content_emb).item())

        per_pair.append({
            "pair_id": pair_id,
            "style_id": style_id,
            "content_subject_id": subject_id,
            "dino_style": d_style,
            "dino_content": d_content,
        })

        if (idx + 1) % log_every == 0:
            log.info("  scored %d / %d pairs", idx + 1, len(pairs))

    # Aggregates ---------------------------------------------------------
    n = len(per_pair)
    dino_style_mean = sum(p["dino_style"] for p in per_pair) / n
    dino_content_mean = sum(p["dino_content"] for p in per_pair) / n
    sti = dino_content_mean - dino_style_mean

    by_artist_acc: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for p in per_pair:
        artist = _artist_for_style_id(p["style_id"], artist_map)
        by_artist_acc[artist].append((p["dino_style"], p["dino_content"]))
    by_artist: dict[str, dict[str, float]] = {}
    for artist, vals in by_artist_acc.items():
        s_mean = sum(v[0] for v in vals) / len(vals)
        c_mean = sum(v[1] for v in vals) / len(vals)
        by_artist[artist] = {
            "n_pairs": len(vals),
            "dino_style_mean": s_mean,
            "dino_content_mean": c_mean,
        }

    payload: dict[str, Any] = {
        "exp_name": str(m.exp_name),
        "protocol": "b_lora_tab1_style_transfer",
        "n_pairs_evaluated": n,
        "pair_subset": subset,
        "per_pair": per_pair,
        "dino_style_grand_mean": dino_style_mean,
        "dino_content_grand_mean": dino_content_mean,
        "sti": sti,
        "by_artist": by_artist,
    }

    out_path = Path("output/results") / f"{m.exp_name}_f02_metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    log.info("Saved %s", out_path)

    # Console summary ----------------------------------------------------
    print("\n" + "=" * 60)
    print(f"  {m.exp_name} — F02 pair-driver DINO metrics")
    print("=" * 60)
    print(f"  pairs evaluated:       {n}")
    print(f"  DINO-style mean:       {dino_style_mean:.4f}")
    print(f"  DINO-content mean:     {dino_content_mean:.4f}")
    print(f"  STI (content - style): {sti:+.4f}")
    print("-" * 60)
    for artist, stats in by_artist.items():
        print(
            f"  {artist:<14} n={stats['n_pairs']:<3}  "
            f"style={stats['dino_style_mean']:.4f}  "
            f"content={stats['dino_content_mean']:.4f}"
        )
    print("=" * 60 + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config_dir = str((Path(__file__).resolve().parents[2] / "configs" / "eval").resolve())
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(config_name="f02_metrics", overrides=sys.argv[1:])
    main(cfg)
