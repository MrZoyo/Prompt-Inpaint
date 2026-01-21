"""Grounding DINO object detection using HuggingFace transformers."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


@dataclass
class Detection:
    """A single object detection result."""

    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    score: float
    label: str


class GroundingDINODetector:
    """Grounding DINO detector using HuggingFace transformers."""

    def __init__(self, device: str = "cuda", model_id: str = "IDEA-Research/grounding-dino-tiny"):
        """
        Initialize the Grounding DINO detector.

        Args:
            device: Device to run the model on ("cuda" or "cpu")
            model_id: HuggingFace model ID. Options:
                - "IDEA-Research/grounding-dino-tiny" (default, smaller)
                - "IDEA-Research/grounding-dino-base"
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_id = model_id
        resolved_model_id = self._resolve_model_id(model_id)

        print(f"Loading Grounding DINO from {resolved_model_id}...")
        local_only = Path(resolved_model_id).exists()
        self.processor = AutoProcessor.from_pretrained(
            resolved_model_id,
            local_files_only=local_only,
            trust_remote_code=True,
        )
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            resolved_model_id,
            local_files_only=local_only,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()
        print("Grounding DINO loaded.")

    @staticmethod
    def _resolve_model_id(model_id: str) -> str:
        """Resolve model_id to a local checkpoints path if available."""
        path = Path(model_id)
        if path.exists():
            return str(path)

        checkpoints_dir = Path(__file__).parent.parent / "checkpoints"
        if model_id in {"grounding-dino-tiny", "grounding-dino-base"}:
            local_dir = checkpoints_dir / model_id
        elif model_id.startswith("IDEA-Research/grounding-dino-"):
            local_dir = checkpoints_dir / model_id.split("/", 1)[1]
        else:
            local_dir = checkpoints_dir / model_id

        if local_dir.exists():
            return str(local_dir)

        return model_id

    def detect(
        self,
        image: Image.Image,
        prompts: List[str],
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
    ) -> List[Detection]:
        """
        Detect objects in an image based on text prompts.

        Args:
            image: PIL Image to process
            prompts: List of text prompts describing objects to detect
            box_threshold: Confidence threshold for bounding boxes
            text_threshold: Confidence threshold for text matching

        Returns:
            List of Detection objects
        """
        if not prompts:
            return []

        # Combine prompts into a single text query (Grounding DINO format)
        # Format: "prompt1. prompt2. prompt3."
        text = ". ".join(prompts) + "."

        # Process inputs
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process results
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]],  # (height, width)
        )[0]

        detections = []
        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        labels = results["labels"]

        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box.astype(int)
            detections.append(
                Detection(
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    score=float(score),
                    label=label,
                )
            )

        return detections

    def detect_single_prompt(
        self,
        image: Image.Image,
        prompt: str,
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
    ) -> List[Detection]:
        """Detect objects matching a single prompt."""
        return self.detect(image, [prompt], box_threshold, text_threshold)
