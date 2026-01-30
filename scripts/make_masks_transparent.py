#!/usr/bin/env python3
"""Make black background transparent and crop to object bbox."""

from __future__ import annotations

import argparse
from pathlib import Path
from PIL import Image


def is_bg(pixel: tuple[int, int, int, int], threshold: int) -> bool:
    r, g, b, _a = pixel
    return r <= threshold and g <= threshold and b <= threshold


def process_image(src: Path, dst: Path, threshold: int) -> tuple[int, int, int, int] | None:
    im = Image.open(src).convert("RGBA")
    w, h = im.size
    data = list(im.getdata())

    min_x, min_y = w, h
    max_x, max_y = -1, -1

    new_data: list[tuple[int, int, int, int]] = []
    for i, px in enumerate(data):
        if is_bg(px, threshold):
            r, g, b, _a = px
            new_data.append((r, g, b, 0))
            continue
        r, g, b, _a = px
        new_data.append((r, g, b, 255))
        x = i % w
        y = i // w
        if x < min_x:
            min_x = x
        if y < min_y:
            min_y = y
        if x > max_x:
            max_x = x
        if y > max_y:
            max_y = y

    im.putdata(new_data)

    if max_x < 0:
        # No foreground found; keep full image.
        bbox = None
    else:
        bbox = (min_x, min_y, max_x + 1, max_y + 1)
        im = im.crop(bbox)

    dst.parent.mkdir(parents=True, exist_ok=True)
    im.save(dst)
    return bbox


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Make black background transparent and crop to object bbox.")
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Input directory containing images.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <input_dir>/transparent).",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=5,
        help="Background threshold for RGB (default: 5).",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite input images in place.")

    args = parser.parse_args()
    input_dir: Path = args.input_dir
    if not input_dir.is_dir():
        raise SystemExit(f"Input dir not found: {input_dir}")

    if args.inplace:
        output_dir = input_dir
    else:
        output_dir = args.output_dir or (input_dir / "transparent")

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    files = [p for p in sorted(input_dir.iterdir()) if p.suffix.lower() in exts]
    if not files:
        raise SystemExit(f"No images found in: {input_dir}")

    for src in files:
        dst = output_dir / src.name
        bbox = process_image(src, dst, args.threshold)
        if bbox is None:
            print(f"{src.name}: no foreground detected, saved full image")
        else:
            print(f"{src.name}: cropped to {bbox}, saved -> {dst}")


if __name__ == "__main__":
    main()
