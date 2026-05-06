"""Generate 512×512 center-cropped style images for DS7 (Phase 4b, L02).

Reads  : data/styles/{van_gogh,monet}/img{1..4}/<filename>.jpg
Writes : data/styles_cropped/{van_gogh,monet}/img{1..4}.jpg
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

CROP_SIZE = 512
STYLES_DIR = Path("data/styles")
OUT_DIR = Path("data/styles_cropped")


def center_crop(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    left = (w - size) // 2
    top = (h - size) // 2
    return img.crop((left, top, left + size, top + size))


def main() -> None:
    jpgs = sorted(STYLES_DIR.rglob("*.jpg"))
    if not jpgs:
        print(f"No .jpg files found under {STYLES_DIR}", file=sys.stderr)
        sys.exit(1)

    for src in jpgs:
        # src = data/styles/{artist}/img{N}/{name}.jpg
        # dst = data/styles_cropped/{artist}/img{N}.jpg
        artist = src.parent.parent.name
        slot = src.parent.name          # img1, img2, …
        dst = OUT_DIR / artist / f"{slot}.jpg"
        dst.parent.mkdir(parents=True, exist_ok=True)

        img = Image.open(src).convert("RGB")
        w, h = img.size
        s = min(w, h)
        cropped = center_crop(img, s).resize((CROP_SIZE, CROP_SIZE), Image.LANCZOS)
        cropped.save(dst, "JPEG", quality=95)
        print(f"  {src.relative_to('.')} → {dst}  ({w}×{h} → {CROP_SIZE}×{CROP_SIZE})")

    print(f"\nDone. {len(jpgs)} images written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
