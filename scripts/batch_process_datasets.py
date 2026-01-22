#!/usr/bin/env python3
"""
Batch process traj datasets.

For each traj* directory under the input root, find the first image and run main.py.
Results mirror the input structure under the output root.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
TRAJ_DIR_RE = re.compile(r"^traj.*$")
IMAGE_DIR_RE = re.compile(r"^images(\d+)$")


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def _pick_image_in_dir(images_dir: Path) -> Path | None:
    if not images_dir.exists() or not images_dir.is_dir():
        return None
    candidates = sorted(
        [p for p in images_dir.iterdir() if _is_image(p)],
        key=lambda p: p.name,
    )
    if not candidates:
        return None
    for p in candidates:
        if p.stem == "im_0":
            return p
    return candidates[0]


def find_first_image(root: Path, exclude_dir: Path | None = None) -> Path | None:
    """
    Prefer imagesN/im_0.* (images0 if available), otherwise fallback to
    the lexicographically first image under root (recursive).
    Does not load images; only scans paths.
    """
    image_dirs = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        match = IMAGE_DIR_RE.match(child.name)
        if match:
            idx = int(match.group(1))
            image_dirs.append((idx, child))
    if image_dirs:
        image_dirs.sort(key=lambda x: x[0])
        for _, img_dir in image_dirs:
            picked = _pick_image_in_dir(img_dir)
            if picked is not None:
                return picked

    best_path = None
    best_key = None

    for dirpath, dirnames, filenames in os.walk(root):
        dirpath_p = Path(dirpath)

        # Skip hidden and excluded directories
        filtered = []
        for d in dirnames:
            if d.startswith("."):
                continue
            if d.startswith("depth") or d.startswith("mask"):
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


def find_traj_dirs(root: Path, exclude_dir: Path | None = None) -> list[Path]:
    """
    Find all trajX_Y directories under root (recursive), excluding output dirs.
    """
    traj_dirs: list[Path] = []

    for dirpath, dirnames, _ in os.walk(root):
        dirpath_p = Path(dirpath)

        filtered = []
        for d in dirnames:
            if d.startswith("."):
                continue
            candidate = dirpath_p / d
            if exclude_dir and _is_within(candidate, exclude_dir):
                continue
            if TRAJ_DIR_RE.match(d):
                traj_dirs.append(candidate)
                continue
            filtered.append(d)
        dirnames[:] = sorted(filtered)

    return sorted(traj_dirs, key=lambda p: p.as_posix())


def _is_task_root(root: Path) -> bool:
    seen_traj = False
    seen_other = False
    for child in root.iterdir():
        if not child.is_dir() or child.name.startswith("."):
            continue
        if TRAJ_DIR_RE.match(child.name):
            seen_traj = True
        else:
            seen_other = True
    return seen_traj and not seen_other


def _compute_output_base(input_root: Path, output_root: Path) -> tuple[Path, str, str | None]:
    """
    Decide output base directory based on whether input_root is a scene or task dir
    (task dir = contains only traj* subdirs).
    Returns (output_base, scene_name, task_name).
    """
    if _is_task_root(input_root):
        scene_name = input_root.parent.name
        task_name = input_root.name
        base = output_root
        if output_root.name != scene_name:
            base = output_root / scene_name
        return base / task_name, scene_name, task_name

    scene_name = input_root.name
    base = output_root
    if output_root.name != scene_name:
        base = output_root / scene_name
    return base, scene_name, None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch process traj datasets by looping main.py.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python scripts/batch_process_datasets.py \\
      --input-root /path/to/datacol1_toykitchen1 \\
      --output-dir ./batch_outputs \\
      --config configs/items.yml

  # With resize and individual masks
  python scripts/batch_process_datasets.py \\
      --input-root /path/to/datacol1_toykitchen1 \\
      --output-dir ./batch_outputs \\
      --config configs/items.yml \\
      --resize-output 448x448 \\
      --save-individual-masks

  # Skip inpainting for faster processing
  python scripts/batch_process_datasets.py \\
      --input-root /path/to/datacol1_toykitchen1 \\
      --output-dir ./batch_outputs \\
      --config configs/items.yml \\
      --no-inpaint

  # Task directory input (traj* directly under many_skills)
  python scripts/batch_process_datasets.py \\
      --input-root /path/to/datacol1_toykitchen1/many_skills \\
      --output-dir ./batch_outputs \\
      --config configs/items.yml

Notes:
  - Any extra args not recognized by this script are forwarded to main.py.
  - Do not pass --image/-i or --output-dir/-o; they are set per sub-dataset.
        """,
    )
    parser.add_argument(
        "--input-root",
        required=True,
        help="Scene directory or task directory (only traj* subdirs)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help=(
            "Output root directory. If input-root is a task dir (contains traj*), "
            "the output will include <scene>/<task>. Otherwise it includes <scene>."
        ),
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
        nargs="?",
        const=0,
        default=None,
        type=int,
        choices=[0, 1],
        help=(
            "Save object RGB masks to a separate 'masks' folder. "
            "Optionally set to 1 to also save robot arm / gripper masks."
        ),
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
    args, extra_args = parser.parse_known_args()
    args._extra_args = extra_args
    return args


def _filter_conflicting_args(extra_args: list[str]) -> tuple[list[str], list[str]]:
    removed = []
    filtered = []
    skip_next = False
    for arg in extra_args:
        if skip_next:
            removed.append(arg)
            skip_next = False
            continue
        if arg in {"--image", "-i", "--output-dir", "-o"}:
            removed.append(arg)
            skip_next = True
            continue
        if arg.startswith("--image=") or arg.startswith("--output-dir="):
            removed.append(arg)
            continue
        filtered.append(arg)
    return filtered, removed


def _build_forward_args(args: argparse.Namespace) -> list[str]:
    forward_args = list(args._extra_args)

    if args.config:
        forward_args += ["--config", args.config]
    if args.resize_output is not None:
        forward_args += ["--resize-output", args.resize_output]
    if args.save_individual_masks is not None:
        forward_args += ["--save-individual-masks", str(args.save_individual_masks)]
    if args.no_inpaint:
        forward_args.append("--no-inpaint")
    if args.save_debug:
        forward_args.append("--save-debug")
    if args.device:
        forward_args += ["--device", args.device]

    forward_args, removed = _filter_conflicting_args(forward_args)
    if removed:
        print(f"Warning: ignored conflicting args: {' '.join(removed)}")

    return forward_args


def _run_main(
    image_path: Path,
    output_dir: Path,
    forward_args: list[str],
    main_path: Path,
) -> int:
    cmd = [
        sys.executable,
        str(main_path),
        "--image",
        str(image_path),
        "--output-dir",
        str(output_dir),
        *forward_args,
    ]
    result = subprocess.run(cmd)
    return result.returncode


def main() -> int:
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_dir)

    if not input_root.exists() or not input_root.is_dir():
        print(f"Error: input root not found or not a directory: {input_root}")
        return 1

    main_path = Path(__file__).parent.parent / "main.py"
    if not main_path.exists():
        print(f"Error: main.py not found: {main_path}")
        return 1

    forward_args = _build_forward_args(args)

    output_base, scene_name, task_name = _compute_output_base(input_root, output_root)
    output_base.mkdir(parents=True, exist_ok=True)

    exclude_dir = output_base if _is_within(output_base, input_root) else None
    traj_dirs = find_traj_dirs(input_root, exclude_dir=exclude_dir)
    if not traj_dirs:
        print(f"No trajX_Y directories found under: {input_root}")
        return 1

    total = len(traj_dirs)
    print(f"Found {total} traj datasets to process")
    print("=" * 60)

    # Print config summary (as forwarded args)
    if forward_args:
    print("Forwarded args to main.py:")
    print(f"  {' '.join(forward_args)}")
    print("=" * 60)

    processed = 0
    skipped = 0
    failed = 0

    for idx, traj_dir in enumerate(traj_dirs, 1):
        rel_path = traj_dir.relative_to(input_root)
        subdir_output = output_base / rel_path

        # Check if already processed
        if subdir_output.exists() and not args.overwrite:
            report_file = subdir_output / "report.json"
            if report_file.exists():
                print(f"[{idx}/{total}] [Skip] Already exists: {rel_path.as_posix()}")
                skipped += 1
                continue

        # Find first image
        first_image = find_first_image(
            traj_dir,
            exclude_dir=output_base if _is_within(output_base, traj_dir) else None
        )
        if first_image is None:
            print(f"[{idx}/{total}] [Skip] No image found: {rel_path.as_posix()}")
            skipped += 1
            continue

        print(f"[{idx}/{total}] Processing: {rel_path.as_posix()}")
        print(f"           Image: {first_image.relative_to(traj_dir)}")

        try:
            code = _run_main(
                image_path=first_image,
                output_dir=subdir_output,
                forward_args=forward_args,
                main_path=main_path,
            )
            if code != 0:
                print(f"           [Error] main.py exited with code {code}")
                failed += 1
                continue

            report_file = subdir_output / "report.json"
            if report_file.exists():
                with open(report_file, "r", encoding="utf-8") as f:
                    report = json.load(f)
                print(f"           -> {report.get('num_objects', 0)} objects detected")
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
    output_label = f"{scene_name}/{task_name}" if task_name else scene_name
    print(f"  Output: {output_base} ({output_label})")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
