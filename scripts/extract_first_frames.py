#!/usr/bin/env python3
"""
Extract the first image from each immediate sub-dataset and save it
as <subdir_name>.<ext> in the output directory.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def find_first_image(root: Path, exclude_dir: Path | None = None) -> Path | None:
    """
    Find the lexicographically first image under root (recursive).
    Does not load images; only scans paths.
    """
    best_path = None
    best_key = None

    for dirpath, dirnames, filenames in os.walk(root):
        dirpath_p = Path(dirpath)

        # Skip hidden and excluded directories
        filtered = []
        for d in dirnames:
            if d.startswith("."):
                continue
            candidate = dirpath_p / d
            if exclude_dir and _is_within(candidate, exclude_dir):
                continue
            filtered.append(d)
        dirnames[:] = sorted(filtered)

        for name in sorted(filenames):
            if name.startswith("."):
                continue
            path = dirpath_p / name
            if not _is_image(path):
                continue

            rel_key = path.relative_to(root).as_posix()
            if best_key is None or rel_key < best_key:
                best_key = rel_key
                best_path = path

    return best_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract the first image from each sub-dataset.",
    )
    parser.add_argument(
        "--input-root",
        required=True,
        help="Root directory containing sub-datasets (e.g., /path/to/traj_group0).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save extracted images.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files if they already exist.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)

    if not input_root.exists() or not input_root.is_dir():
        print(f"Error: input root not found or not a directory: {input_root}")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    # Iterate immediate subdirectories only
    subdirs = [p for p in input_root.iterdir() if p.is_dir() and not p.name.startswith(".")]
    if not subdirs:
        print(f"No subdirectories found under: {input_root}")
        return 1

    for subdir in sorted(subdirs):
        first_image = find_first_image(subdir, exclude_dir=output_dir if _is_within(output_dir, subdir) else None)
        if first_image is None:
            print(f"[Skip] No image found under: {subdir}")
            continue

        out_path = output_dir / f"{subdir.name}{first_image.suffix.lower()}"
        if out_path.exists() and not args.overwrite:
            print(f"[Skip] Exists: {out_path}")
            continue

        shutil.copy2(first_image, out_path)
        print(f"[OK] {subdir.name} -> {out_path.name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
