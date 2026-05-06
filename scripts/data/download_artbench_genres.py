"""Download impressionism and post_impressionism subsets from ArtBench-10 (DS3).

Source: https://artbench.eecs.berkeley.edu/files/artbench-10-imagefolder.tar
Extracts exactly N_PER_GENRE images per genre into:
  data/artbench10/impressionism/
  data/artbench10/post_impressionism/

Streams the tar — never writes the full 2 GB to disk.
SSL verification disabled for corporate proxy (same as download_datasets.py).
"""

from __future__ import annotations

import io
import logging
import shutil
import ssl
import sys
import tarfile
import urllib.request
from pathlib import Path

log = logging.getLogger(__name__)

TAR_URL = "https://artbench.eecs.berkeley.edu/files/artbench-10-imagefolder.tar"
TARGET_GENRES = {"impressionism", "post_impressionism"}
N_PER_GENRE = 500
CHUNK = 1 << 20  # 1 MB read chunks


def _ssl_ctx() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def download_genres(out_root: Path, n: int = N_PER_GENRE, resume: bool = True) -> None:
    out_root.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    for genre in TARGET_GENRES:
        genre_dir = out_root / genre
        if resume and genre_dir.exists():
            existing = len(list(genre_dir.glob("*.jpg")))
            counts[genre] = existing
            log.info("Resume: %s has %d / %d images already", genre, existing, n)
        else:
            genre_dir.mkdir(parents=True, exist_ok=True)
            counts[genre] = 0

    if all(counts[g] >= n for g in TARGET_GENRES):
        log.info("Both genres already complete (%d each). Nothing to do.", n)
        return

    log.info("Streaming tar from %s ...", TAR_URL)
    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=_ssl_ctx()))
    resp = opener.open(urllib.request.Request(TAR_URL))  # noqa: S310

    total_bytes = int(resp.headers.get("Content-Length") or 0)
    log.info("File size: %.1f GB", total_bytes / 1e9)

    class _StreamWrapper(io.RawIOBase):
        """Wraps the HTTP response so tarfile can read it as a seekable-free stream."""

        def __init__(self, response: object) -> None:
            self._resp = response
            self._buf = b""
            self._pos = 0

        def read(self, n: int = -1) -> bytes:  # type: ignore[override]
            if n < 0:
                return self._resp.read()
            data = self._resp.read(n)
            self._pos += len(data)
            return data

        def readinto(self, b: bytearray | memoryview) -> int:  # type: ignore[override]
            data = self.read(len(b))
            n = len(data)
            b[:n] = data
            return n

        def readable(self) -> bool:
            return True

    stream = io.BufferedReader(_StreamWrapper(resp), buffer_size=CHUNK)

    with tarfile.open(fileobj=stream, mode="r|") as tf:
        logged_mb = -1
        for member in tf:
            if not member.isfile():
                continue

            parts = Path(member.name).parts
            if len(parts) < 2:
                continue
            # Skip __MACOSX resource-fork entries and AppleDouble metadata files
            if "__MACOSX" in parts or parts[-1].startswith("._"):
                continue
            genre = parts[-2]

            if genre not in TARGET_GENRES:
                continue
            if counts[genre] >= n:
                continue

            idx = counts[genre]
            dest = out_root / genre / f"{idx:06d}.jpg"
            if resume and dest.exists():
                counts[genre] += 1
                continue

            fobj = tf.extractfile(member)
            if fobj is None:
                continue
            dest.write_bytes(fobj.read())
            counts[genre] += 1

            if counts[genre] % 50 == 0:
                log.info("  %s: %d / %d", genre, counts[genre], n)

            if all(counts[g] >= n for g in TARGET_GENRES):
                log.info("All genres complete — stopping stream early.")
                break

            # Progress log
            if total_bytes > 0:
                try:
                    mb = stream.tell() // 1_000_000
                except OSError:
                    mb = 0
                if mb // 100 != logged_mb // 100:
                    logged_mb = mb
                    log.info("  streamed ~%d MB / %d MB", mb, total_bytes // 1_000_000)

    for genre in TARGET_GENRES:
        final = len(list((out_root / genre).glob("*.jpg")))
        log.info("Final count %s: %d images", genre, final)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    repo_root = Path(__file__).resolve().parents[2]
    out_root = repo_root / "data" / "artbench10"

    # Remove existing flat dump if any (numbered files without genre subdir)
    flat_files = list(out_root.glob("*.jpg"))
    if flat_files:
        ans = input(
            f"Found {len(flat_files)} flat .jpg files in {out_root}.\n"
            "These are unlabelled mixed-genre images (old download).\n"
            "Remove them before proceeding? [y/N] "
        ).strip().lower()
        if ans == "y":
            for f in flat_files:
                f.unlink()
            log.info("Removed %d flat files.", len(flat_files))
        else:
            log.info("Keeping flat files — genre subdirs will be created alongside them.")

    download_genres(out_root)
    log.info("Done. Run: dvc add data/artbench10 && dvc push")


if __name__ == "__main__":
    sys.exit(main())
