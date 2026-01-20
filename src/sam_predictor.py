"""SAM/SAM2 segmentation predictor."""

import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import requests
import torch
from PIL import Image

from .config import SAM1_MODELS, SAM2_MODELS

CHECKPOINT_FOLDER = Path(__file__).parent.parent / "checkpoints"
# Also check Track-Anything checkpoints
TRACK_ANYTHING_CHECKPOINTS = Path(__file__).parent.parent.parent / "Track-Anything" / "checkpoints"


def _download_checkpoint(url: str, path: Path) -> None:
    """Download a checkpoint file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading checkpoint to {path}...")
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    with open(path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print("Download completed.")


def _find_checkpoint(filename: str, url: Optional[str] = None) -> Path:
    """Find or download a checkpoint file."""
    # Check local checkpoints folder first
    local_path = CHECKPOINT_FOLDER / filename
    if local_path.exists():
        return local_path

    # Check Track-Anything checkpoints
    ta_path = TRACK_ANYTHING_CHECKPOINTS / filename
    if ta_path.exists():
        return ta_path

    # Download if URL provided
    if url:
        _download_checkpoint(url, local_path)
        return local_path

    raise FileNotFoundError(f"Checkpoint not found: {filename}")


class SAMPredictor:
    """Unified SAM/SAM2 predictor for mask generation from bounding boxes."""

    def __init__(self, model_name: str = "sam2_hiera_small", device: str = "cuda"):
        """
        Initialize SAM predictor.

        Args:
            model_name: Model name. Options:
                SAM2: "sam2_hiera_tiny", "sam2_hiera_small", "sam2_hiera_base_plus", "sam2_hiera_large"
                SAM1: "vit_b", "vit_l", "vit_h"
            device: Device to run on
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.using_sam2 = model_name.startswith("sam2")
        self.predictor = None
        self._image_set = False

        self._load_model()

    def _load_model(self) -> None:
        """Load the SAM model."""
        if self.using_sam2:
            self._load_sam2()
        else:
            self._load_sam1()

    def _load_sam2(self) -> None:
        """Load SAM2 model."""
        if self.model_name not in SAM2_MODELS:
            raise ValueError(f"Unknown SAM2 model: {self.model_name}")

        model_info = SAM2_MODELS[self.model_name]
        checkpoint_path = _find_checkpoint(model_info["checkpoint"], model_info["url"])

        print(f"Loading SAM2 model: {self.model_name}...")
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        sam2 = build_sam2(model_info["config"], str(checkpoint_path), device=self.device)
        self.predictor = SAM2ImagePredictor(sam2)
        print("SAM2 loaded.")

    def _load_sam1(self) -> None:
        """Load SAM1 model."""
        if self.model_name not in SAM1_MODELS:
            raise ValueError(f"Unknown SAM1 model: {self.model_name}")

        model_info = SAM1_MODELS[self.model_name]
        checkpoint_path = _find_checkpoint(model_info["checkpoint"], model_info["url"])

        print(f"Loading SAM1 model: {self.model_name}...")
        from segment_anything import SamPredictor, sam_model_registry

        sam = sam_model_registry[self.model_name](checkpoint=str(checkpoint_path))
        sam.to(device=self.device)
        self.predictor = SamPredictor(sam)
        print("SAM1 loaded.")

    def set_image(self, image: np.ndarray) -> None:
        """
        Set the image for prediction.

        Args:
            image: RGB image as numpy array (H, W, 3)
        """
        self.predictor.set_image(image)
        self._image_set = True

    def predict_box(
        self,
        box: Tuple[int, int, int, int],
        multimask_output: bool = False,
    ) -> Tuple[np.ndarray, float]:
        """
        Predict mask from a bounding box.

        Args:
            box: Bounding box (x1, y1, x2, y2)
            multimask_output: Whether to return multiple masks

        Returns:
            Tuple of (mask, score) where mask is a boolean numpy array
        """
        if not self._image_set:
            raise RuntimeError("Must call set_image() before predict_box()")

        box_np = np.array(box)

        if self.using_sam2:
            masks, scores, _ = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box_np,
                multimask_output=multimask_output,
            )
        else:
            masks, scores, _ = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box_np,
                multimask_output=multimask_output,
            )

        # Return the best mask
        best_idx = np.argmax(scores)
        return masks[best_idx], float(scores[best_idx])

    def predict_boxes(
        self,
        boxes: List[Tuple[int, int, int, int]],
    ) -> List[Tuple[np.ndarray, float]]:
        """
        Predict masks from multiple bounding boxes.

        Args:
            boxes: List of bounding boxes (x1, y1, x2, y2)

        Returns:
            List of (mask, score) tuples
        """
        results = []
        for box in boxes:
            mask, score = self.predict_box(box)
            results.append((mask, score))
        return results

    def reset_image(self) -> None:
        """Reset the image state."""
        self._image_set = False
