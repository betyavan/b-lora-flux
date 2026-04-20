"""Download and prepare datasets for B-LoRA style transfer experiments."""

from __future__ import annotations

import io
import logging
import os
import ssl
import sys
import time
import typing as t
import urllib.request
import zipfile
from pathlib import Path

import httpx
import huggingface_hub.utils._http as hf_http_mod
from huggingface_hub.utils._typing import HTTP_METHOD_T
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

log = logging.getLogger(__name__)


def _apply_hf_hub_backoff_client_refresh_patch() -> None:
    """Re-bind httpx client each retry (upstream caches client before the while-loop)."""
    m = hf_http_mod

    if getattr(m._http_backoff_base, "_diploma_client_refresh_patch", False):
        return

    get_session = m.get_session
    close_session = m.close_session
    logger = m.logger
    SliceFileObj = m.SliceFileObj
    hf_raise_for_status = m.hf_raise_for_status
    parse_ratelimit_headers = m.parse_ratelimit_headers
    _exc = m._DEFAULT_RETRY_ON_EXCEPTIONS
    _codes = m._DEFAULT_RETRY_ON_STATUS_CODES

    def _http_backoff_base_fixed(
        method: HTTP_METHOD_T,
        url: str,
        *,
        max_retries: int = 5,
        base_wait_time: float = 1,
        max_wait_time: float = 8,
        retry_on_exceptions: type[Exception] | tuple[type[Exception], ...] = _exc,
        retry_on_status_codes: int | tuple[int, ...] = _codes,
        stream: bool = False,
        **kwargs: t.Any,
    ) -> t.Generator[httpx.Response, None, None]:
        if isinstance(retry_on_exceptions, type):
            retry_on_exceptions = (retry_on_exceptions,)

        if isinstance(retry_on_status_codes, int):
            retry_on_status_codes = (retry_on_status_codes,)

        nb_tries = 0
        sleep_time = base_wait_time
        ratelimit_reset: int | None = None

        io_obj_initial_pos = None
        if "data" in kwargs and isinstance(kwargs["data"], (io.IOBase, SliceFileObj)):
            io_obj_initial_pos = kwargs["data"].tell()

        while True:
            nb_tries += 1
            client = get_session()
            ratelimit_reset = None
            try:
                if io_obj_initial_pos is not None:
                    kwargs["data"].seek(io_obj_initial_pos)

                def _should_retry(response: httpx.Response) -> bool:
                    nonlocal ratelimit_reset

                    if response.status_code not in retry_on_status_codes:
                        return False

                    logger.warning(f"HTTP Error {response.status_code} thrown while requesting {method} {url}")
                    if nb_tries > max_retries:
                        hf_raise_for_status(response)
                        return False

                    if response.status_code == 429:
                        ratelimit_info = parse_ratelimit_headers(response.headers)
                        if ratelimit_info is not None:
                            ratelimit_reset = ratelimit_info.reset_in_seconds

                    return True

                if stream:
                    with client.stream(method=method, url=url, **kwargs) as response:
                        if not _should_retry(response):
                            yield response
                            return
                else:
                    response = client.request(method=method, url=url, **kwargs)
                    if not _should_retry(response):
                        yield response
                        return

            except retry_on_exceptions as err:
                logger.warning(f"'{err}' thrown while requesting {method} {url}")

                if isinstance(err, httpx.ConnectError):
                    close_session()

                if nb_tries > max_retries:
                    raise err

            if ratelimit_reset is not None:
                actual_sleep = float(ratelimit_reset) + 1
                logger.warning(
                    f"Rate limited. Waiting {actual_sleep}s before retry [Retry {nb_tries}/{max_retries}]."
                )
            else:
                actual_sleep = sleep_time
                logger.warning(f"Retrying in {actual_sleep}s [Retry {nb_tries}/{max_retries}].")

            time.sleep(actual_sleep)

            sleep_time = min(max_wait_time, sleep_time * 2)

    setattr(_http_backoff_base_fixed, "_diploma_client_refresh_patch", True)
    m._http_backoff_base = _http_backoff_base_fixed


def _tls_verify_enabled() -> bool:
    """If true, verify TLS certificates (default for this script is verify off)."""
    return os.environ.get("DIPLOMA_TLS_VERIFY", "").strip().lower() in ("1", "true", "yes")


def _configure_tls() -> None:
    """TLS for Hugging Face Hub (httpx) and urllib HTTPS downloads.

    By default **certificate verification is disabled** for this script (common behind
    corporate proxies). Set ``DIPLOMA_TLS_VERIFY=1`` to enforce verification.

    Optionally ``DIPLOMA_USE_CERTIFI=1`` sets ``SSL_CERT_FILE`` from certifi (only useful
    when verification is on or for other libs reading those env vars).
    """
    if os.environ.get("DIPLOMA_USE_CERTIFI", "").strip().lower() in ("1", "true", "yes"):
        try:
            import certifi  # type: ignore[import]

            bundle = certifi.where()
            os.environ.setdefault("SSL_CERT_FILE", bundle)
            os.environ.setdefault("REQUESTS_CA_BUNDLE", bundle)
            log.info("DIPLOMA_USE_CERTIFI=1: using certifi bundle at %s", bundle)
        except ImportError:
            log.warning("DIPLOMA_USE_CERTIFI=1 but certifi is not installed; skipping")

    if _tls_verify_enabled():
        log.info("DIPLOMA_TLS_VERIFY=1: TLS certificate verification enabled for HF Hub / downloads.")
        return

    log.warning("TLS certificate verification is disabled for dataset downloads (default). Set DIPLOMA_TLS_VERIFY=1 to enable.")
    from huggingface_hub import set_client_factory
    from huggingface_hub.utils._http import hf_request_event_hook

    def _insecure_factory() -> httpx.Client:
        return httpx.Client(
            verify=False,
            event_hooks={"request": [hf_request_event_hook]},
            follow_redirects=True,
            timeout=None,
        )

    set_client_factory(_insecure_factory)


def _unverified_ssl_context() -> ssl.SSLContext | None:
    if _tls_verify_enabled():
        return None
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


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

    ssl_ctx = _unverified_ssl_context()
    if ssl_ctx is not None:
        opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_ctx))
        req = urllib.request.Request(url)
        block_size = 8192
        count = 0
        with opener.open(req) as resp:  # noqa: S310
            cl = resp.headers.get("Content-Length")
            total_size = int(cl) if cl and cl.isdigit() else -1
            with zip_path.open("wb") as out:
                while True:
                    chunk = resp.read(block_size)
                    if not chunk:
                        break
                    out.write(chunk)
                    count += 1
                    _reporthook(count, block_size, total_size)
    else:
        urllib.request.urlretrieve(url, zip_path, reporthook=_reporthook)  # noqa: S310
    log.info("Extracting %s...", zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    zip_path.unlink()
    log.info("Extracted to %s", out_dir)


def main(cfg: DictConfig) -> None:
    d = cfg.data
    dl = cfg.download

    styles_dir = Path(str(d.styles.dir))
    if not styles_dir.exists() or not any(styles_dir.iterdir() if styles_dir.exists() else []):
        log.warning(
            "data/styles/ is empty. Add style reference images manually "
            "(10-20 images per style, one subdirectory per style, e.g. data/styles/van_gogh/)."
        )

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
    # Use compose API instead of @hydra.main so argparse is not used. Hydra 1.3.x is
    # incompatible with Python 3.14's stricter help validation (LazyCompletionHelp).
    logging.basicConfig(level=logging.INFO)
    _configure_tls()
    _apply_hf_hub_backoff_client_refresh_patch()
    _config_dir = (Path(__file__).resolve().parent.parent / "configs").resolve()
    with initialize_config_dir(version_base="1.3", config_dir=str(_config_dir)):
        _cfg = compose(config_name="download", overrides=sys.argv[1:])
    main(_cfg)
