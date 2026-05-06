"""Build DS8: B-LoRA paired eval manifest (§5.1, reduced protocol).

Protocol (fixed 2026-05-06):
  Nc = 8 content subjects (DreamBooth DS5)
  Ns = 8 style images     (WikiArt DS1)
  N  = min(50, 8×8) = 50 random pairs, seed=42, no duplicate (subject, style)

Outputs:
  experiments/data/b_lora_eval_pairs.json

Usage:
  python scripts/data/build_eval_pairs.py
  python scripts/data/build_eval_pairs.py --validate-only
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

SEED = 42
N_PAIRS = 50
N_USER_STUDY = 30

REPO_ROOT = Path(__file__).resolve().parents[2]

CONTENT_SUBJECTS = {
    "dog":      "data/dreambooth_dog",
    "cat":      "data/dreambooth_subjects/cat",
    "backpack": "data/dreambooth_subjects/backpack",
    "bowl":     "data/dreambooth_subjects/bowl",
    "can":      "data/dreambooth_subjects/can",
    "clock":    "data/dreambooth_subjects/clock",
    "vase":     "data/dreambooth_subjects/vase",
    "bear":     "data/dreambooth_subjects/bear",
}

STYLE_IMAGES = {
    "wikivg_01": "data/styles/van_gogh/img1",
    "wikivg_02": "data/styles/van_gogh/img2",
    "wikivg_03": "data/styles/van_gogh/img3",
    "wikivg_04": "data/styles/van_gogh/img4",
    "wikimo_01": "data/styles/monet/img1",
    "wikimo_02": "data/styles/monet/img2",
    "wikimo_03": "data/styles/monet/img3",
    "wikimo_04": "data/styles/monet/img4",
}


def _find_repr_image(directory: Path) -> Path | None:
    """Return first .jpg/.jpeg/.png in directory, sorted for determinism."""
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        imgs = sorted(directory.glob(ext))
        if imgs:
            return imgs[0]
    return None


def _resolve_style_path(style_dir: Path) -> Path | None:
    return _find_repr_image(style_dir)


def build_manifest(repo_root: Path) -> dict:
    rng = random.Random(SEED)

    # Resolve representative paths
    content_items: list[tuple[str, str]] = []
    for subj, rel in CONTENT_SUBJECTS.items():
        d = repo_root / rel
        img = _find_repr_image(d)
        if img is None:
            print(f"[WARN] No image found for subject '{subj}' at {d}", file=sys.stderr)
            continue
        content_items.append((subj, str(img.relative_to(repo_root))))

    style_items: list[tuple[str, str]] = []
    for sid, rel in STYLE_IMAGES.items():
        d = repo_root / rel
        img = _resolve_style_path(d)
        if img is None:
            print(f"[WARN] No image found for style '{sid}' at {d}", file=sys.stderr)
            continue
        style_items.append((sid, str(img.relative_to(repo_root))))

    # All unique (content, style) combinations
    all_combos = [(c, s) for c in content_items for s in style_items]
    rng.shuffle(all_combos)
    selected = all_combos[:N_PAIRS]

    pairs = []
    for i, ((subj, c_path), (sid, s_path)) in enumerate(selected):
        pairs.append({
            "pair_id": i,
            "content_subject_id": subj,
            "content_repr_path": c_path,
            "style_id": sid,
            "style_repr_path": s_path,
        })

    # User-study subset: 30 random pair_ids, derived seed
    all_ids = list(range(len(pairs)))
    rng2 = random.Random(SEED + 1)
    user_study_ids = sorted(rng2.sample(all_ids, min(N_USER_STUDY, len(all_ids))))

    return {
        "protocol": "b_lora_tab1_style_transfer",
        "seed": SEED,
        "n_content": len(content_items),
        "n_styles": len(style_items),
        "n_pairs": len(pairs),
        "pairs": pairs,
        "user_study_pair_ids": user_study_ids,
    }


def validate_manifest(manifest: dict, repo_root: Path) -> bool:
    ok = True
    for pair in manifest["pairs"]:
        for key in ("content_repr_path", "style_repr_path"):
            p = repo_root / pair[key]
            if not p.exists():
                print(f"[MISSING] {pair[key]}", file=sys.stderr)
                ok = False
    if ok:
        print(f"All {len(manifest['pairs'])} paths exist.")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    out_path = REPO_ROOT / "experiments" / "data" / "b_lora_eval_pairs.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.validate_only:
        if not out_path.exists():
            print(f"Manifest not found: {out_path}", file=sys.stderr)
            sys.exit(1)
        manifest = json.loads(out_path.read_text())
        ok = validate_manifest(manifest, REPO_ROOT)
        sys.exit(0 if ok else 1)

    manifest = build_manifest(REPO_ROOT)
    out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"Wrote {len(manifest['pairs'])} pairs → {out_path}")
    print(f"User study subset: {len(manifest['user_study_pair_ids'])} pairs")
    validate_manifest(manifest, REPO_ROOT)


if __name__ == "__main__":
    main()
