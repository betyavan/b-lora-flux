"""Compute per-cell DINO-style and DINO-content for a 3×3 mixing grid (Phase 6, M02).

For each cell (i, j):
  - DINO-style[i, j]   = cos(DINO_emb(cell), DINO_emb(style_refs[i]))
  - DINO-content[i, j] = cos(DINO_emb(cell), mean DINO_emb over content_refs_dirs[j])

Output JSON shape:
  {
    "exp_name": ...,
    "style_names":   [...], "content_names": [...],
    "dino_style":   {"per_cell": [[..]x3]x3, "row_means":[..], "col_means":[..], "grand_mean": ..},
    "dino_content": {"per_cell": [[..]x3]x3, "row_means":[..], "col_means":[..], "grand_mean": ..},
    "disentanglement_score": ..  # mean(diag DINO-style) - mean(off-diag DINO-style)
  }

Usage:
    python scripts/eval/compute_mixing_metrics.py \\
        mixing_metrics.grid_dir=output/mixing/m01_mixing_grid_v3_with_suffix \\
        mixing_metrics.style_refs=[data/styles/van_gogh/img1/red-vineyards-at-arles-1888.jpg,data/styles/van_gogh/img4/the-starry-night-1889.jpg,data/styles/monet/img1/haystack-at-giverny-1886.jpg] \\
        mixing_metrics.content_refs_dirs=[data/eval_content/cat,data/eval_content/dog,data/eval_content/backpack] \\
        mixing_metrics.style_names=[van_gogh_starry_road,van_gogh_starry_night,monet_garden] \\
        mixing_metrics.content_names=[cat,dog,backpack] \\
        mixing_metrics.exp_name=m01_mixing_grid_v3_with_suffix
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

# Reuse loader + encoder from the project-level metrics module
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_metrics import _IMAGE_EXTS, _encode_images_dino, _load_dino  # noqa: E402

log = logging.getLogger(__name__)


def _list_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in _IMAGE_EXTS)


def _cell_path(grid_dir: Path, i: int, sname: str, j: int, cname: str) -> Path:
    return grid_dir / f"style_{i}_{sname}__content_{j}_{cname}.png"


def main(cfg: DictConfig) -> None:
    m = cfg.mixing_metrics
    assert len(m.style_refs) == len(m.style_names) == 3, "need 3 style refs + names"
    assert len(m.content_refs_dirs) == len(m.content_names) == 3, "need 3 content dirs + names"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    grid_dir = Path(m.grid_dir)
    cells: list[Path] = []
    for i, sname in enumerate(m.style_names):
        for j, cname in enumerate(m.content_names):
            p = _cell_path(grid_dir, i, sname, j, cname)
            if not p.exists():
                raise FileNotFoundError(f"Missing cell: {p}")
            cells.append(p)

    style_ref_paths = [Path(p) for p in m.style_refs]
    for p in style_ref_paths:
        if not p.exists():
            raise FileNotFoundError(f"Style ref not found: {p}")
    content_ref_imgs: list[list[Path]] = []
    for d in m.content_refs_dirs:
        imgs = _list_images(Path(d))
        if not imgs:
            raise FileNotFoundError(f"No images found under content_refs_dir: {d}")
        content_ref_imgs.append(imgs)

    log.info("Cells: %d, style refs: %d (one per row), content refs: %s",
             len(cells), len(style_ref_paths), [len(x) for x in content_ref_imgs])

    dino, proc = _load_dino(str(m.dino_model), device)
    bs = int(m.batch_size)

    cell_emb = _encode_images_dino(cells, dino, proc, device, bs)  # (9, D)
    style_emb = _encode_images_dino(style_ref_paths, dino, proc, device, bs)  # (3, D)
    content_emb_per_col = [
        _encode_images_dino(imgs, dino, proc, device, bs).mean(dim=0, keepdim=True)
        for imgs in content_ref_imgs
    ]
    content_emb = torch.cat(content_emb_per_col, dim=0)  # (3, D), unit-normed average

    dino_style = [[0.0] * 3 for _ in range(3)]
    dino_content = [[0.0] * 3 for _ in range(3)]
    idx = 0
    for i in range(3):
        for j in range(3):
            v = cell_emb[idx]
            dino_style[i][j] = float((v @ style_emb[i]).item())
            dino_content[i][j] = float((v @ content_emb[j]).item())
            idx += 1

    def _stats(mat: list[list[float]]) -> dict:
        rows = [sum(r) / 3 for r in mat]
        cols = [sum(mat[i][j] for i in range(3)) / 3 for j in range(3)]
        grand = sum(rows) / 3
        return {"per_cell": mat, "row_means": rows, "col_means": cols, "grand_mean": grand}

    # disentanglement proxy: for the style axis, style[i, j] should be high for the matching style
    # row (it's the same ref) and may be lower for off-diagonal — but here all cells in row i share
    # the same style ref, so all 3 share the SAME row index. We define a per-row dispersion proxy
    # as std along the content axis: low std → style invariant to content (good disentanglement).
    style_row_std = [
        (sum((x - sum(dino_style[i]) / 3) ** 2 for x in dino_style[i]) / 3) ** 0.5
        for i in range(3)
    ]
    content_col_std = [
        (sum((dino_content[i][j] - sum(dino_content[k][j] for k in range(3)) / 3) ** 2 for i in range(3)) / 3) ** 0.5
        for j in range(3)
    ]

    payload = {
        "exp_name": m.exp_name,
        "style_names": list(m.style_names),
        "content_names": list(m.content_names),
        "dino_style": _stats(dino_style),
        "dino_content": _stats(dino_content),
        "style_row_std": style_row_std,
        "content_col_std": content_col_std,
    }

    out = Path("output/results") / f"{m.exp_name}_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    log.info("Saved %s", out)

    print("\n" + "=" * 60)
    print(f"  {m.exp_name} — DINO style / content")
    print("=" * 60)
    print("DINO-style (rows=styles, cols=contents):")
    for i, sn in enumerate(m.style_names):
        print(f"  {sn:<22}  " + "  ".join(f"{dino_style[i][j]:.4f}" for j in range(3))
              + f"   | row_mean={payload['dino_style']['row_means'][i]:.4f}")
    print(f"  grand_mean = {payload['dino_style']['grand_mean']:.4f}")
    print("\nDINO-content (rows=styles, cols=contents):")
    for i, sn in enumerate(m.style_names):
        print(f"  {sn:<22}  " + "  ".join(f"{dino_content[i][j]:.4f}" for j in range(3)))
    print(f"  col_means = {[round(x, 4) for x in payload['dino_content']['col_means']]}")
    print(f"  grand_mean = {payload['dino_content']['grand_mean']:.4f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config_dir = str((Path(__file__).resolve().parents[2] / "configs" / "eval").resolve())
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(config_name="mixing_metrics", overrides=sys.argv[1:])
    main(cfg)
