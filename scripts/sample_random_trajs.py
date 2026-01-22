#!/usr/bin/env python3
"""
Randomly sample up to N trajectories per scene and run main.py.

Input root should be the raw_data directory (containing scene folders) or a single scene folder.
Outputs are flattened per scene: <output_root>/<scene>/traj0_0, traj0_1, ...
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
TRAJ_DIR_RE = re.compile(r"^traj.*$")

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import SegmentConfig, parse_output_size
from src.pipeline import SegmentInpaintPipeline

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


def find_traj_dirs(root: Path, exclude_dir: Path | None = None) -> list[Path]:
    """
    Find all traj* directories under root (recursive), excluding output dirs.
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


def _detect_scene_dirs(input_root: Path) -> list[Path]:
    if (input_root / "path_map.json").exists():
        return [input_root]

    scenes = []
    for child in input_root.iterdir():
        if child.is_dir() and not child.name.startswith("."):
            scenes.append(child)
    return sorted(scenes, key=lambda p: p.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample random trajectories per scene and run main.py.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sample 5 trajs per scene from raw_data
  python scripts/sample_random_trajs.py \\
      --input-root /home/discover/sam3d_gs/raw_data \\
      --output-dir /home/discover/sam3d_gs/data \\
      --max-per-scene 5 \\
      --config configs/items.yml

  # Reuse model in-process (default)
  python scripts/sample_random_trajs.py \\
      --input-root /home/discover/sam3d_gs/raw_data \\
      --output-dir /home/discover/sam3d_gs/data \\
      --max-per-scene 5 \\
      --sam-model sam2_hiera_large \\
      --dino-model grounding-dino-base

Notes:
  - Any extra args not recognized by this script are forwarded to main.py,
    unless reuse is enabled (default), in which case they are ignored.
  - Do not pass --image/-i or --output-dir/-o; they are set per sample.
        """,
    )
    parser.add_argument(
        "--input-root",
        required=True,
        help="Raw data root (contains scene dirs) or a single scene dir",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output root directory; per-scene folders will be created under it",
    )
    parser.add_argument(
        "--max-per-scene",
        type=int,
        required=True,
        help="Maximum number of trajectories sampled per scene",
    )
    parser.add_argument(
        "--no-reuse-model",
        action="store_true",
        help="Disable in-process reuse; call main.py for each sample",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible sampling",
    )
    parser.add_argument(
        "--config", "-c",
        default=None,
        help="Path to YAML config file (default: configs/items.yml)",
    )
    parser.add_argument(
        "--prompts", "-p",
        nargs="+",
        default=None,
        help="Text prompts for object detection (override config)",
    )
    parser.add_argument(
        "--sam-model",
        default=None,
        help="SAM model to use (default: sam2_hiera_small)",
    )
    parser.add_argument(
        "--dino-model",
        default=None,
        choices=["grounding-dino-tiny", "grounding-dino-base"],
        help="Grounding DINO model (default: grounding-dino-tiny)",
    )
    parser.add_argument(
        "--box-threshold",
        type=float,
        default=None,
        help="Grounding DINO box confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--text-threshold",
        type=float,
        default=None,
        help="Grounding DINO text confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=None,
        help="IoU threshold for mask deduplication (default: 0.5)",
    )
    parser.add_argument(
        "--mask-dilate-pixels",
        type=int,
        default=None,
        help="Mask dilation pixels for inpainting (default: 12)",
    )
    parser.add_argument(
        "--inpaint-backend",
        choices=["iopaint", "opencv", "none"],
        default=None,
        help="Inpainting backend (default: iopaint)",
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


def _load_config(args: argparse.Namespace) -> SegmentConfig:
    if args.config:
        config_path = Path(args.config)
    else:
        config_path = Path(__file__).parent.parent / "configs" / "items.yml"

    if config_path.exists():
        config = SegmentConfig.from_yaml(str(config_path))
        print(f"Loaded config from: {config_path}")
    else:
        if args.config:
            print(f"Error: Config file not found: {config_path}")
            raise FileNotFoundError(str(config_path))
        config = SegmentConfig()

    if args.prompts:
        config.prompts = args.prompts
    if args.sam_model is not None:
        config.sam_model = args.sam_model
    if args.dino_model is not None:
        config.grounding_dino_model = args.dino_model
    if args.device is not None:
        config.device = args.device
    if args.box_threshold is not None:
        config.box_threshold = args.box_threshold
    if args.text_threshold is not None:
        config.text_threshold = args.text_threshold
    if args.iou_threshold is not None:
        config.iou_threshold = args.iou_threshold
    if args.mask_dilate_pixels is not None:
        config.mask_dilate_pixels = args.mask_dilate_pixels
    if args.no_inpaint:
        config.inpaint_backend = "none"
    elif args.inpaint_backend is not None:
        config.inpaint_backend = args.inpaint_backend
    if args.save_debug:
        config.save_debug = True
    if args.save_individual_masks is not None:
        config.save_individual_masks = True
        config.save_individual_masks_include_robot_gripper = bool(args.save_individual_masks)
    if args.resize_output is not None:
        config.output_size = parse_output_size(args.resize_output)

    if not config.prompts:
        raise ValueError("No prompts provided. Use --prompts or --config.")

    return config


def main() -> int:
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_dir)

    if not input_root.exists() or not input_root.is_dir():
        print(f"Error: input root not found or not a directory: {input_root}")
        return 1
    if args.max_per_scene <= 0:
        print("Error: --max-per-scene must be > 0")
        return 1

    main_path = Path(__file__).parent.parent / "main.py"
    if not main_path.exists():
        print(f"Error: main.py not found: {main_path}")
        return 1

    pipeline = None
    if not args.no_reuse_model:
        if args._extra_args:
            print(f"Warning: ignored extra args in reuse mode: {' '.join(args._extra_args)}")
        try:
            config = _load_config(args)
        except Exception as exc:
            print(f"Error: {exc}")
            return 1
        pipeline = SegmentInpaintPipeline(config)
        forward_args = []
    else:
        forward_args = _build_forward_args(args)
        if forward_args:
            print("Forwarded args to main.py:")
            print(f"  {' '.join(forward_args)}")
            print("=" * 60)

    rng = random.Random(args.seed) if args.seed is not None else random

    scene_dirs = _detect_scene_dirs(input_root)
    if not scene_dirs:
        print(f"No scene directories found under: {input_root}")
        return 1

    total_scenes = 0
    processed = 0
    skipped = 0
    failed = 0

    for scene_dir in scene_dirs:
        if not scene_dir.exists():
            continue

        output_scene_dir = output_root / scene_dir.name
        output_scene_dir.mkdir(parents=True, exist_ok=True)

        exclude_dir = output_scene_dir if _is_within(output_scene_dir, scene_dir) else None
        traj_dirs = find_traj_dirs(scene_dir, exclude_dir=exclude_dir)
        if not traj_dirs:
            print(f"[Skip] No traj* directories in scene: {scene_dir.name}")
            continue

        total_scenes += 1
        rng.shuffle(traj_dirs)
        selected = traj_dirs[: args.max_per_scene]

        print("=" * 60)
        print(f"Scene: {scene_dir.name}")
        print(f"  Total trajs: {len(traj_dirs)}")
        print(f"  Sampled: {len(selected)}")

        map_path = output_scene_dir / "path_map.json"
        mapping: dict[str, str] = {}
        if map_path.exists():
            try:
                with open(map_path, "r", encoding="utf-8") as f:
                    mapping = json.load(f)
            except Exception:
                mapping = {}

        for i, traj_dir in enumerate(selected):
            output_name = f"traj0_{i}"
            output_dir = output_scene_dir / output_name

            if output_dir.exists() and not args.overwrite:
                report_file = output_dir / "report.json"
                if report_file.exists():
                    print(f"  [Skip] Exists: {output_name}")
                    skipped += 1
                    continue

            first_image = find_first_image(
                traj_dir,
                exclude_dir=output_scene_dir if _is_within(output_scene_dir, traj_dir) else None,
            )
            if first_image is None:
                print(f"  [Skip] No image found: {traj_dir}")
                skipped += 1
                continue

            print(f"  Processing: {output_name}")
            print(f"    Source: {traj_dir}")
            print(f"    Image: {first_image.relative_to(traj_dir)}")

            if pipeline is not None:
                try:
                    report = pipeline.process(
                        image_path=str(first_image),
                        output_dir=str(output_dir),
                    )
                    print(f"    -> {report.get('num_objects', 0)} objects detected")
                except Exception as exc:
                    print(f"    [Error] {exc}")
                    failed += 1
                    continue
            else:
                try:
                    code = _run_main(
                        image_path=first_image,
                        output_dir=output_dir,
                        forward_args=forward_args,
                        main_path=main_path,
                    )
                except Exception as exc:
                    print(f"    [Error] {exc}")
                    failed += 1
                    continue

                if code != 0:
                    print(f"    [Error] main.py exited with code {code}")
                    failed += 1
                    continue

            mapping[output_name] = str(traj_dir)
            processed += 1

        with open(map_path, "w", encoding="utf-8") as f:
            json.dump(mapping, f, indent=2, ensure_ascii=False)

    print("=" * 60)
    print("Summary:")
    print(f"  Scenes processed: {total_scenes}")
    print(f"  Processed: {processed}")
    print(f"  Skipped: {skipped}")
    print(f"  Failed: {failed}")
    print(f"  Output root: {output_root}")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
