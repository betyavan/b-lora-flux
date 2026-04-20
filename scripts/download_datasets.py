"""Download and prepare datasets for B-LoRA style transfer experiments."""

from __future__ import annotations

import logging
import os
import ssl
import sys
import urllib.request
import zipfile
from pathlib import Path

import httpx
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

log = logging.getLogger(__name__)

_CHUNK = 8192


def _tls_verify() -> bool:
    return os.environ.get("DIPLOMA_TLS_VERIFY", "").strip().lower() in ("1", "true", "yes")


def _configure_tls() -> None:
    """Disable TLS verification by default (corporate proxy). Set DIPLOMA_TLS_VERIFY=1 to enable."""
    if _tls_verify():
        log.info("TLS verification enabled.")
        return

    log.warning("TLS verification disabled. Set DIPLOMA_TLS_VERIFY=1 to enable.")

    from huggingface_hub import set_client_factory
    from huggingface_hub.utils._http import hf_request_event_hook  # type: ignore[import]

    set_client_factory(
        lambda: httpx.Client(
            verify=False,
            event_hooks={"request": [hf_request_event_hook]},
            follow_redirects=True,
            timeout=None,
        )
    )


def _ssl_context() -> ssl.SSLContext | None:
    if _tls_verify():
        return None
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _hf_download(hf_id: str, out_dir: Path, split: str, image_column: str) -> int:
    try:
        from datasets import load_dataset  # type: ignore[import]
    except ImportError:
        raise ImportError("Run: pip install datasets Pillow")

    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("HF download %s[%s] -> %s", hf_id, split, out_dir)

    ds = load_dataset(hf_id, split=split, streaming=False)
    count = 0
    for idx, sample in enumerate(ds):  # type: ignore[var-annotated]
        sample[image_column].save(out_dir / f"{idx:06d}.jpg")
        count += 1
        if count % 1000 == 0:
            log.info("  %d images saved...", count)

    log.info("Done: %d images -> %s", count, out_dir)
    return count


def _download_zip(url: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / Path(url).name
    log.info("Downloading %s", url)

    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=_ssl_context()))
    with opener.open(urllib.request.Request(url)) as resp:  # noqa: S310
        total = int(resp.headers.get("Content-Length") or 0) or None
        done, last_logged_mb = 0, -1
        with zip_path.open("wb") as f:
            while chunk := resp.read(_CHUNK):
                f.write(chunk)
                done += len(chunk)
                mb = done // 1_000_000
                if total and mb % 50 == 0 and mb != last_logged_mb:
                    log.info("  %d / %d MB", mb, total // 1_000_000)
                    last_logged_mb = mb

    log.info("Extracting -> %s", out_dir)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
    zip_path.unlink()


def main(cfg: DictConfig) -> None:
    d = cfg.data
    dl = cfg.download

    styles_dir = Path(str(d.styles.dir))
    if not styles_dir.exists() or not any(styles_dir.iterdir()):
        log.warning(
            "data/styles/ is empty — add style images manually "
            "(10-20 per style, one subdir per style, e.g. data/styles/van_gogh/)."
        )

    if dl.coco_val2017:
        coco = d.coco_val2017
        coco_path = Path(str(coco.dir))
        _download_zip(str(coco.url), coco_path)
        _download_zip(str(coco.annotations_url), coco_path)

    if dl.artbench10:
        ab = d.artbench10
        _hf_download(str(ab.hf_id), Path(str(ab.dir)), str(ab.split), str(ab.image_column))

    log.info("Done. Run: dvc add data/ && dvc push")


if __name__ == "__main__":
    # compose API avoids Hydra/argparse conflict with Python 3.14 (LazyCompletionHelp)
    logging.basicConfig(level=logging.INFO)
    _configure_tls()
    config_dir = str((Path(__file__).resolve().parent.parent / "configs").resolve())
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(config_name="download", overrides=sys.argv[1:])
    main(cfg)
