"""Download and prepare datasets for B-LoRA style transfer experiments."""

from __future__ import annotations

import logging
import urllib.request
import zipfile
from pathlib import Path

import hydra
from omegaconf import DictConfig

log = logging.getLogger(__name__)


def _hf_download(hf_id: str, out_dir: Path, split: str, image_column: str) -> int:
    """Download a HuggingFace dataset and save images to out_dir."""
    try:
        from datasets import load_dataset  # type: ignore[import]
    except ImportError:
        raise ImportError("Run: pip install datasets Pillow")

    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Downloading %s (split=%s) -> %s", hf_id, split, out_dir)

    ds = load_dataset(hf_id, split=split, streaming=False)
    count = 0
    for idx, sample in enumerate(ds):  # type: ignore[var-annotated]
        img = sample[image_column]
        img.save(out_dir / f"{idx:06d}.jpg")
        count += 1
        if count % 1000 == 0:
            log.info("  saved %d images...", count)

    log.info("Done: %d images -> %s", count, out_dir)
    return count


def _http_download_zip(url: str, out_dir: Path) -> None:
    """Download a zip archive and extract it to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / Path(url).name
    log.info("Downloading %s -> %s", url, zip_path)

    def _reporthook(count: int, block_size: int, total_size: int) -> None:
        done = count * block_size
        pct = done / total_size * 100 if total_size > 0 else 0.0
        if count % 500 == 0:
            log.info("  %.1f%% (%d MB / %d MB)", pct, done // 1_000_000, total_size // 1_000_000)

    urllib.request.urlretrieve(url, zip_path, reporthook=_reporthook)  # noqa: S310
    log.info("Extracting %s...", zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    zip_path.unlink()
    log.info("Extracted to %s", out_dir)


@hydra.main(config_path="../configs", config_name="download", version_base="1.3")
def main(cfg: DictConfig) -> None:
    d = cfg.data
    dl = cfg.download

    if dl.wikiart:
        wc = d.wikiart
        _hf_download(str(wc.hf_id), Path(str(wc.dir)), str(wc.split), str(wc.image_column))

    if dl.best_artworks:
        log.info(
            "Best Artworks: download manually from Kaggle -> "
            "kaggle.com/datasets/ikarus777/best-artworks-of-all-time"
        )
        log.info("Then unzip to: %s", d.best_artworks.dir)

    if dl.coco_val2017:
        coco = d.coco_val2017
        coco_path = Path(str(coco.dir))
        _http_download_zip(str(coco.url), coco_path)
        _http_download_zip(str(coco.annotations_url), coco_path)

    if dl.artbench10:
        ab = d.artbench10
        _hf_download(str(ab.hf_id), Path(str(ab.dir)), str(ab.split), str(ab.image_column))

    log.info("All downloads complete. Run: dvc add data/ && dvc push")


if __name__ == "__main__":
    main()
