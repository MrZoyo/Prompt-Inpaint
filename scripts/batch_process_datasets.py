#!/usr/bin/env python3
"""
Batch process Bridge V2 datasets.

For each sub-dataset, find the first image and run the segmentation pipeline.
Results are saved to output_dir/<subdir_name>/.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import SegmentConfig, parse_output_size
from src.pipeline import SegmentInpaintPipeline

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
        description="Batch process Bridge V2 datasets with segmentation pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python scripts/batch_process_datasets.py \\
      --input-root /path/to/traj_group0 \\
      --output-dir ./batch_outputs \\
      --config configs/items.yml

  # With resize and individual masks
  python scripts/batch_process_datasets.py \\
      --input-root /path/to/traj_group0 \\
      --output-dir ./batch_outputs \\
      --config configs/items.yml \\
      --resize-output 448x448 \\
      --save-individual-masks

  # Skip inpainting for faster processing
  python scripts/batch_process_datasets.py \\
      --input-root /path/to/traj_group0 \\
      --output-dir ./batch_outputs \\
      --config configs/items.yml \\
      --no-inpaint
        """,
    )
    parser.add_argument(
        "--input-root",
        required=True,
        help="Root directory containing sub-datasets (e.g., /path/to/traj_group0)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save processing results",
    )
    parser.add_argument(
        "--config", "-c",
        default=None,
        help="Path to YAML config file (default: configs/items.yml)",
    )
    parser.add_argument(
        "--resize-output",
        nargs="?",
        const="448x448",
        default=None,
        help="Resize output images to WxH (default: 448x448 when flag is set)",
    )
    parser.add_argument(
        "--save-individual-masks",
        action="store_true",
        help="Save all object RGB masks to a separate 'masks' folder",
    )
    parser.add_argument(
        "--no-inpaint",
        action="store_true",
        help="Skip background inpainting",
    )
    parser.add_argument(
        "--save-debug",
        action="store_true",
        help="Save debug artifacts",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing results",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to run on (default: cuda)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)

    if not input_root.exists() or not input_root.is_dir():
        print(f"Error: input root not found or not a directory: {input_root}")
        return 1

    # Load config
    if args.config:
        config_path = Path(args.config)
    else:
        config_path = Path(__file__).parent.parent / "configs" / "items.yml"

    if not config_path.exists():
        print(f"Error: config file not found: {config_path}")
        return 1

    config = SegmentConfig.from_yaml(str(config_path))
    print(f"Loaded config from: {config_path}")

    # Apply CLI overrides
    if args.resize_output is not None:
        try:
            config.output_size = parse_output_size(args.resize_output)
        except ValueError as exc:
            print(f"Error: invalid --resize-output value '{args.resize_output}': {exc}")
            return 1
    if args.save_individual_masks:
        config.save_individual_masks = True
    if args.no_inpaint:
        config.inpaint_backend = "none"
    if args.save_debug:
        config.save_debug = True
    if args.device:
        config.device = args.device

    output_dir.mkdir(parents=True, exist_ok=True)

    # Iterate immediate subdirectories only
    subdirs = [p for p in input_root.iterdir() if p.is_dir() and not p.name.startswith(".")]
    if not subdirs:
        print(f"No subdirectories found under: {input_root}")
        return 1

    subdirs = sorted(subdirs)
    total = len(subdirs)
    print(f"Found {total} sub-datasets to process")
    print("=" * 60)

    # Print config summary
    print("Configuration:")
    print(f"  Config: {config_path}")
    print(f"  Prompts: {len(config.prompts)} items")
    print(f"  Inpaint: {config.inpaint_backend}")
    print(f"  Save Individual Masks: {config.save_individual_masks}")
    if config.output_size:
        print(f"  Output Resize: {config.output_size[0]}x{config.output_size[1]}")
    print("=" * 60)

    # Initialize pipeline once (models will be loaded lazily on first use)
    pipeline = SegmentInpaintPipeline(config)

    processed = 0
    skipped = 0
    failed = 0

    for idx, subdir in enumerate(subdirs, 1):
        subdir_output = output_dir / subdir.name

        # Check if already processed
        if subdir_output.exists() and not args.overwrite:
            report_file = subdir_output / "report.json"
            if report_file.exists():
                print(f"[{idx}/{total}] [Skip] Already exists: {subdir.name}")
                skipped += 1
                continue

        # Find first image
        first_image = find_first_image(
            subdir,
            exclude_dir=output_dir if _is_within(output_dir, subdir) else None
        )
        if first_image is None:
            print(f"[{idx}/{total}] [Skip] No image found: {subdir.name}")
            skipped += 1
            continue

        print(f"[{idx}/{total}] Processing: {subdir.name}")
        print(f"           Image: {first_image.relative_to(subdir)}")

        try:
            report = pipeline.process(
                image_path=str(first_image),
                output_dir=str(subdir_output),
            )
            print(f"           -> {report['num_objects']} objects detected")
            processed += 1
        except Exception as e:
            print(f"           [Error] {e}")
            failed += 1

    print("=" * 60)
    print("Summary:")
    print(f"  Total: {total}")
    print(f"  Processed: {processed}")
    print(f"  Skipped: {skipped}")
    print(f"  Failed: {failed}")
    print(f"  Output: {output_dir}")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
