"""Configuration loading and management."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import yaml


@dataclass
class SegmentConfig:
    """Configuration for the segmentation pipeline."""

    # Prompts for object detection
    prompts: List[str] = field(default_factory=list)

    # Grounding DINO settings
    box_threshold: float = 0.25
    text_threshold: float = 0.25

    # Mask processing
    iou_threshold: float = 0.5  # For deduplication
    mask_dilate_pixels: int = 12  # Mask dilation for inpainting
    containment_overlap_ratio: float = 0.9  # Containment merge threshold
    contour_overlap_ratio: float = 0.3  # Contour overlap merge threshold

    # SAM model selection
    # Options: "sam2_hiera_tiny", "sam2_hiera_small", "sam2_hiera_base_plus", "sam2_hiera_large"
    #          "vit_b", "vit_l", "vit_h" (SAM1)
    sam_model: str = "sam2_hiera_small"

    # Grounding DINO model selection
    # Options: "grounding-dino-tiny", "grounding-dino-base"
    grounding_dino_model: str = "grounding-dino-tiny"

    # Inpainting
    inpaint_backend: str = "iopaint"  # "iopaint", "opencv", "none"
    save_debug: bool = False
    output_size: Optional[Tuple[int, int]] = None  # (width, height)

    # Output options
    save_individual_masks: bool = False  # Save all RGB masks to a separate folder

    # Device
    device: str = "cuda"

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "SegmentConfig":
        """Load configuration from a YAML file."""
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        prompts = data.get("prompts", [])
        settings = data.get("settings", {})

        return cls(
            prompts=prompts,
            box_threshold=settings.get("box_threshold", 0.25),
            text_threshold=settings.get("text_threshold", 0.25),
            iou_threshold=settings.get("iou_threshold", 0.5),
            mask_dilate_pixels=settings.get("mask_dilate_pixels", 12),
            containment_overlap_ratio=settings.get("containment_overlap_ratio", 0.9),
            contour_overlap_ratio=settings.get("contour_overlap_ratio", 0.3),
            sam_model=settings.get("sam_model", "sam2_hiera_small"),
            grounding_dino_model=settings.get("grounding_dino_model", "grounding-dino-tiny"),
            inpaint_backend=settings.get("inpaint_backend", "iopaint"),
            save_debug=settings.get("save_debug", False),
            output_size=parse_output_size(settings.get("output_size")),
            save_individual_masks=settings.get("save_individual_masks", False),
            device=settings.get("device", "cuda"),
        )

    def to_yaml(self, yaml_path: str) -> None:
        """Save configuration to a YAML file."""
        data = {
            "prompts": self.prompts,
            "settings": {
                "box_threshold": self.box_threshold,
                "text_threshold": self.text_threshold,
                "iou_threshold": self.iou_threshold,
                "mask_dilate_pixels": self.mask_dilate_pixels,
                "containment_overlap_ratio": self.containment_overlap_ratio,
                "contour_overlap_ratio": self.contour_overlap_ratio,
                "sam_model": self.sam_model,
                "inpaint_backend": self.inpaint_backend,
                "save_debug": self.save_debug,
                "save_individual_masks": self.save_individual_masks,
                "device": self.device,
            },
        }
        if self.output_size:
            data["settings"]["output_size"] = list(self.output_size)
        Path(yaml_path).parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False)


def parse_output_size(value: Optional[object]) -> Optional[Tuple[int, int]]:
    """Parse output size from config/CLI input."""
    if value is None:
        return None

    if isinstance(value, int):
        if value <= 0:
            raise ValueError("size must be positive")
        return (value, value)

    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError("size must have two values")
        width, height = value
        width = int(width)
        height = int(height)
        if width <= 0 or height <= 0:
            raise ValueError("size must be positive")
        return (width, height)

    if isinstance(value, str):
        cleaned = value.lower().replace(" ", "")
        if "x" in cleaned:
            parts = cleaned.split("x")
        elif "," in cleaned:
            parts = cleaned.split(",")
        else:
            parts = [cleaned, cleaned]
        if len(parts) != 2:
            raise ValueError("size must be WIDTHxHEIGHT")
        width = int(parts[0])
        height = int(parts[1])
        if width <= 0 or height <= 0:
            raise ValueError("size must be positive")
        return (width, height)

    raise ValueError("size must be int, list/tuple, or string")


# SAM2 model configurations
SAM2_MODELS = {
    "sam2_hiera_tiny": {
        "config": "sam2_hiera_t.yaml",
        "checkpoint": "sam2_hiera_tiny.pt",
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt",
    },
    "sam2_hiera_small": {
        "config": "sam2_hiera_s.yaml",
        "checkpoint": "sam2_hiera_small.pt",
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt",
    },
    "sam2_hiera_base_plus": {
        "config": "sam2_hiera_b+.yaml",
        "checkpoint": "sam2_hiera_base_plus.pt",
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt",
    },
    "sam2_hiera_large": {
        "config": "sam2_hiera_l.yaml",
        "checkpoint": "sam2_hiera_large.pt",
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt",
    },
}

# SAM1 model configurations
SAM1_MODELS = {
    "vit_b": {
        "checkpoint": "sam_vit_b_01ec64.pth",
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    },
    "vit_l": {
        "checkpoint": "sam_vit_l_0b3195.pth",
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    },
    "vit_h": {
        "checkpoint": "sam_vit_h_4b8939.pth",
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    },
}
