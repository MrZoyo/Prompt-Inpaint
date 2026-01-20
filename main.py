#!/usr/bin/env python3
"""
Grounded Segment Inpaint - CLI Entry Point

Detect objects using text prompts, segment with SAM, and inpaint background.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from src.config import SegmentConfig, parse_output_size
from src.pipeline import SegmentInpaintPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect, segment, and inpaint objects in images using Grounding DINO + SAM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with config file
  python main.py --image photo.jpg --config configs/items.yml

  # Quick test with inline prompts
  python main.py --image photo.jpg --prompts "cup" "keyboard" "book"

  # Use SAM1 instead of SAM2
  python main.py --image photo.jpg --prompts "cup" --sam-model vit_h

  # Skip inpainting
  python main.py --image photo.jpg --config configs/items.yml --no-inpaint

SAM Models:
  SAM2: sam2_hiera_tiny, sam2_hiera_small, sam2_hiera_base_plus, sam2_hiera_large
  SAM1: vit_b, vit_l, vit_h
        """,
    )

    # Input/Output
    parser.add_argument(
        "--image", "-i",
        required=True,
        help="Path to input image",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Output directory (default: outputs/<timestamp>)",
    )
    parser.add_argument(
        "--config", "-c",
        default=None,
        help="Path to YAML config file (e.g., configs/items.yml)",
    )

    # Prompts (alternative to config)
    parser.add_argument(
        "--prompts", "-p",
        nargs="+",
        default=None,
        help="Text prompts for object detection (alternative to config file)",
    )

    # Model selection
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
        "--device",
        default=None,
        help="Device to run on (default: cuda)",
    )

    # Detection thresholds
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

    # Inpainting
    parser.add_argument(
        "--inpaint-backend",
        choices=["iopaint", "opencv", "none"],
        default=None,
        help="Inpainting backend (default: iopaint)",
    )
    parser.add_argument(
        "--no-inpaint",
        action="store_true",
        help="Skip background inpainting",
    )
    parser.add_argument(
        "--save-debug",
        action="store_true",
        help="Save debug artifacts under outputs/<timestamp>/debug",
    )
    parser.add_argument(
        "--save-individual-masks",
        action="store_true",
        help="Save all object RGB masks to a separate 'masks' folder with object names",
    )
    parser.add_argument(
        "--resize-output",
        nargs="?",
        const="448x448",
        default=None,
        help="Force resize all saved images to WxH (default: 448x448 when flag is set)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Validate input
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)

    # Load config (default: configs/items.yml)
    if args.config:
        config_path = Path(args.config)
    else:
        config_path = Path(__file__).parent / "configs" / "items.yml"

    if config_path.exists():
        config = SegmentConfig.from_yaml(str(config_path))
        print(f"Loaded config from: {config_path}")
    else:
        if args.config:
            # User specified a config file but it doesn't exist
            print(f"Error: Config file not found: {config_path}")
            sys.exit(1)
        # Default config file not found, use defaults (prompts must be provided via CLI)
        config = SegmentConfig()

    # Override config with CLI arguments
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
    if args.save_individual_masks:
        config.save_individual_masks = True
    if args.resize_output is not None:
        try:
            config.output_size = parse_output_size(args.resize_output)
        except ValueError as exc:
            print(f"Error: invalid --resize-output value '{args.resize_output}': {exc}")
            sys.exit(1)

    # Validate prompts
    if not config.prompts:
        print("Error: No prompts provided. Use --prompts or --config.")
        sys.exit(1)

    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"outputs/{timestamp}"

    # Print config summary
    print("=" * 50)
    print("Configuration:")
    print(f"  Image: {image_path}")
    print(f"  Output: {output_dir}")
    print(f"  DINO Model: {config.grounding_dino_model}")
    print(f"  SAM Model: {config.sam_model}")
    print(f"  Prompts: {config.prompts}")
    print(f"  Box Threshold: {config.box_threshold}")
    print(f"  Text Threshold: {config.text_threshold}")
    print(f"  IoU Threshold: {config.iou_threshold}")
    print(f"  Inpaint: {config.inpaint_backend}")
    print(f"  Save Debug: {config.save_debug}")
    print(f"  Save Individual Masks: {config.save_individual_masks}")
    if config.output_size:
        print(f"  Output Resize: {config.output_size[0]}x{config.output_size[1]}")
    print("=" * 50)

    # Run pipeline
    pipeline = SegmentInpaintPipeline(config)
    report = pipeline.process(
        image_path=str(image_path),
        output_dir=output_dir,
    )

    # Print summary
    print("\n" + "=" * 50)
    print("Summary:")
    print(f"  Detections: {report['num_detections']}")
    print(f"  Unique Objects: {report['num_objects']}")
    if report.get("objects"):
        for obj in report["objects"]:
            print(f"    - {obj['labels']} (area: {obj['area']}, score: {obj['score']:.3f})")
    print(f"  Output Directory: {output_dir}")
    print("=" * 50)

    return 0


if __name__ == "__main__":
    sys.exit(main())
