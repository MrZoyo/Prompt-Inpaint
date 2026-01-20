"""Main segmentation pipeline."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

from .config import SegmentConfig
from .grounding_dino import Detection, GroundingDINODetector
from .inpainter import Inpainter
from .mask_processor import (
    SegmentedObject,
    combine_all_masks,
    deduplicate_objects,
    dilate_mask,
    mask_to_image,
    mask_to_rgb,
)
from .sam_predictor import SAMPredictor


class SegmentInpaintPipeline:
    """Pipeline for grounded segmentation and background inpainting."""

    def __init__(self, config: SegmentConfig):
        """
        Initialize the pipeline.

        Args:
            config: Pipeline configuration
        """
        self.config = config

        # Initialize models lazily
        self._detector: Optional[GroundingDINODetector] = None
        self._sam: Optional[SAMPredictor] = None
        self._inpainter: Optional[Inpainter] = None

    @property
    def detector(self) -> GroundingDINODetector:
        if self._detector is None:
            # Map short names to full HuggingFace model IDs
            model_map = {
                "grounding-dino-tiny": "IDEA-Research/grounding-dino-tiny",
                "grounding-dino-base": "IDEA-Research/grounding-dino-base",
            }
            model_id = model_map.get(
                self.config.grounding_dino_model,
                f"IDEA-Research/{self.config.grounding_dino_model}"
            )
            self._detector = GroundingDINODetector(device=self.config.device, model_id=model_id)
        return self._detector

    @property
    def sam(self) -> SAMPredictor:
        if self._sam is None:
            self._sam = SAMPredictor(
                model_name=self.config.sam_model,
                device=self.config.device,
            )
        return self._sam

    @property
    def inpainter(self) -> Inpainter:
        if self._inpainter is None:
            self._inpainter = Inpainter(
                backend=self.config.inpaint_backend,
                device=self.config.device,
            )
        return self._inpainter

    def _save_image(self, image: Image.Image, path: Path, is_mask: bool = False) -> None:
        """Save image, optionally resizing for output."""
        if self.config.output_size:
            resample = Image.NEAREST if is_mask else Image.LANCZOS
            image = image.resize(self.config.output_size, resample=resample)
        image.save(path)

    def detect_objects(
        self,
        image: Image.Image,
        prompts: Optional[List[str]] = None,
    ) -> List[Detection]:
        """Detect objects in image using prompts (defaults to config)."""
        all_detections = []

        active_prompts = prompts if prompts is not None else self.config.prompts
        for prompt in active_prompts:
            detections = self.detector.detect_single_prompt(
                image,
                prompt,
                box_threshold=self.config.box_threshold,
                text_threshold=self.config.text_threshold,
            )
            all_detections.extend(detections)
            if detections:
                print(f"  '{prompt}': found {len(detections)} objects")
            else:
                print(f"  '{prompt}': no objects found")

        return all_detections

    def segment_detections(
        self,
        image_np: np.ndarray,
        detections: List[Detection],
    ) -> List[SegmentedObject]:
        """Generate masks for all detections using SAM."""
        self.sam.set_image(image_np)

        objects = []
        for i, det in enumerate(detections):
            mask, score = self.sam.predict_box(det.bbox)
            obj = SegmentedObject(
                id=i,
                mask=mask,
                bbox=det.bbox,
                score=det.score * score,  # Combined score
                labels=[det.label],
            )
            objects.append(obj)

        self.sam.reset_image()
        return objects

    def process(
        self,
        image_path: str,
        output_dir: str,
        prompts: Optional[List[str]] = None,
    ) -> Dict:
        """
        Process an image: detect, segment, deduplicate, and inpaint.

        Args:
            image_path: Path to input image
            output_dir: Output directory
            prompts: Override prompts (uses config prompts if None)

        Returns:
            Report dictionary
        """
        # Use provided prompts or config prompts
        if prompts:
            self.config.prompts = prompts

        if not self.config.prompts:
            raise ValueError("No prompts provided. Set prompts in config or pass them directly.")

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        objects_dir = output_path / "objects"
        objects_dir.mkdir(exist_ok=True)

        # Load image
        print(f"Loading image: {image_path}")
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        image_area = image_np.shape[0] * image_np.shape[1]
        max_mask_area = int(image_area * 0.2)

        # Save input image
        input_copy = output_path / "input_image.png"
        self._save_image(image, input_copy)

        # Step 1: Detect objects
        print("Detecting objects...")
        detections = self.detect_objects(image)
        print(f"Total detections: {len(detections)}")

        # Save detections
        detections_data = [
            {"bbox": list(d.bbox), "score": d.score, "label": d.label}
            for d in detections
        ]
        with open(output_path / "detections.json", "w") as f:
            json.dump(detections_data, f, indent=2)

        if not detections:
            print("No objects detected. Skipping segmentation.")
            report = {
                "input_image": str(input_copy),
                "num_detections": 0,
                "num_objects": 0,
                "objects_dir": str(objects_dir),
                "combined_mask": None,
                "clean_background": None,
            }
            with open(output_path / "report.json", "w") as f:
                json.dump(report, f, indent=2)
            return report

        # Step 2: Segment detections
        print("Segmenting objects with SAM...")
        objects = self.segment_detections(image_np, detections)
        print(f"Segmented {len(objects)} objects")

        # Filter oversized masks before deduplication
        if objects:
            before = len(objects)
            objects = [obj for obj in objects if obj.area <= max_mask_area]
            dropped = before - len(objects)
            if dropped:
                print(f"Filtered {dropped} oversized masks (>20% image area)")

        # Step 3: Deduplicate
        print(f"Deduplicating with IoU threshold {self.config.iou_threshold}...")
        objects = deduplicate_objects(
            objects,
            self.config.iou_threshold,
            self.config.containment_overlap_ratio,
            self.config.contour_overlap_ratio,
        )
        print(f"After deduplication: {len(objects)} unique objects")

        # Filter oversized masks after deduplication
        if objects:
            before = len(objects)
            objects = [obj for obj in objects if obj.area <= max_mask_area]
            dropped = before - len(objects)
            if dropped:
                print(f"Filtered {dropped} oversized masks after dedup (>20% image area)")
            for i, obj in enumerate(objects):
                obj.id = i

        # Step 4: Save individual objects
        print("Saving individual objects...")
        for obj in objects:
            # Create object directory
            label_slug = obj.labels[0].replace(" ", "_").replace("/", "-")[:30]
            obj_dir = objects_dir / f"{obj.id:03d}_{label_slug}"
            obj_dir.mkdir(exist_ok=True)

            # Save binary mask (white on black)
            mask_img = mask_to_image(obj.mask)
            self._save_image(mask_img, obj_dir / "mask.png", is_mask=True)

            # Save RGB mask (original colors on black background, full image size)
            rgb_mask = mask_to_rgb(image_np, obj.mask)
            self._save_image(rgb_mask, obj_dir / "mask_rgb.png")

            # Save info
            info = {
                "id": obj.id,
                "labels": obj.labels,
                "bbox": list(obj.bbox),
                "area": obj.area,
                "score": obj.score,
            }
            with open(obj_dir / "info.json", "w") as f:
                json.dump(info, f, indent=2)

        # Step 4.5: Save individual RGB masks to a separate folder (if enabled)
        masks_dir = None
        if self.config.save_individual_masks and objects:
            print("Saving individual RGB masks...")
            masks_dir = output_path / "masks"
            masks_dir.mkdir(exist_ok=True)

            # Track used names to handle duplicates
            name_counts = {}
            for obj in objects:
                # Use first label as filename
                base_name = obj.labels[0].replace(" ", "_").replace("/", "-")

                # Handle duplicate names by adding suffix
                if base_name in name_counts:
                    name_counts[base_name] += 1
                    filename = f"{base_name}_{name_counts[base_name]}.png"
                else:
                    name_counts[base_name] = 0
                    filename = f"{base_name}.png"

                # Save RGB mask
                rgb_mask = mask_to_rgb(image_np, obj.mask)
                self._save_image(rgb_mask, masks_dir / filename)

            print(f"  Saved {len(objects)} RGB masks to: {masks_dir}")

        # Step 5: Combine masks (original precise masks for reference)
        combined_mask = combine_all_masks(objects)
        combined_path = output_path / "combined_mask.png"
        if combined_mask is not None:
            self._save_image(mask_to_image(combined_mask), combined_path, is_mask=True)

        # Step 6: Iterative inpaint with re-detection for mask expansion
        # 1. Initial detection already done - we have base masks for all objects
        # 2. For each object: inpaint → re-detect FIRST-PASS LABELS → expand if adjacent
        # 3. Expansion rules: same label, adjacent to original, area <= 3x original
        clean_path = output_path / "clean_background.png"
        if objects and self.config.inpaint_backend != "none":
            print(f"Inpainting background with {self.config.inpaint_backend}...")
            print(f"  Iterative inpaint + re-detection for {len(objects)} objects...")
            mask_dilate_pixels = self.config.mask_dilate_pixels
            save_debug = self.config.save_debug

            # Create debug directory
            debug_dir = None
            if save_debug:
                debug_dir = output_path / "debug"
                debug_dir.mkdir(exist_ok=True)

            # Only re-detect labels seen in the first pass
            detected_labels = sorted({label for obj in objects for label in obj.labels})

            # Build initial object registry: label -> (mask, area)
            # Track expanded masks for each original object
            object_masks = {}  # obj_id -> current mask (may be expanded)
            for obj in objects:
                object_masks[obj.id] = {
                    "mask": obj.mask.astype(bool),
                    "labels": set(obj.labels),
                    "original_area": obj.area,
                    "current_area": obj.area,
                    "name": obj.labels[0].replace(" ", "_"),
                }

            # Current working image for sequential inpainting
            current_image = image_np.copy()

            # Process each object: inpaint from ORIGINAL then check for expansion of OTHER objects
            for i, obj in enumerate(objects):
                obj_info = object_masks[obj.id]
                current_mask = obj_info["mask"]
                obj_name = obj_info["name"]

                # Create step debug directory
                step_dir = None
                if save_debug:
                    step_dir = debug_dir / f"step_{i+1:02d}_remove_{obj_name}"
                    step_dir.mkdir(exist_ok=True)

                # Dilate mask for inpainting
                dilated = dilate_mask(current_mask, pixels=mask_dilate_pixels)
                print(f"    [{i+1}/{len(objects)}] Checking expansion after removing '{obj.labels[0]}' (area: {obj_info['current_area']})...")

                # Save the mask being removed
                if save_debug:
                    self._save_image(mask_to_image(current_mask), step_dir / "removed_mask.png", is_mask=True)
                    self._save_image(mask_to_rgb(image_np, current_mask), step_dir / "removed_mask_rgb.png")

                # Inpaint this object from the current sequential image
                temp_inpainted = self.inpainter.inpaint(current_image, dilated)

                # Save inpainted result
                if save_debug:
                    self._save_image(Image.fromarray(temp_inpainted), step_dir / "inpainted.png")

                # Re-detect to find expansions of OTHER objects (not the one we just removed)
                print(f"        Re-detecting for mask expansion...")
                temp_pil = Image.fromarray(temp_inpainted)
                new_detections = self.detect_objects(temp_pil, prompts=detected_labels)

                if new_detections:
                    new_objects = self.segment_detections(temp_inpainted, new_detections)

                    # Save all re-detected masks
                    if save_debug:
                        redetect_dir = step_dir / "redetected"
                        redetect_dir.mkdir(exist_ok=True)
                        for j, new_obj in enumerate(new_objects):
                            label_slug = new_obj.labels[0].replace(" ", "_")[:20]
                            self._save_image(
                                mask_to_image(new_obj.mask),
                                redetect_dir / f"{j:02d}_{label_slug}_mask.png",
                                is_mask=True,
                            )
                            self._save_image(
                                mask_to_rgb(temp_inpainted, new_obj.mask),
                                redetect_dir / f"{j:02d}_{label_slug}_rgb.png",
                            )

                    # Check each new detection for valid expansion
                    for new_obj in new_objects:
                        new_mask = new_obj.mask.astype(bool)
                        new_labels = set(new_obj.labels)
                        new_area = new_mask.sum()

                        # Try to match with existing objects (not the one we just inpainted)
                        for other_id, other_info in object_masks.items():
                            if other_id == obj.id:
                                continue  # Skip the object we just removed

                            # Check if labels match (same type)
                            if not (new_labels & other_info["labels"]):
                                continue

                            # Check if adjacent (masks touch or overlap)
                            # Use larger dilation to catch nearby masks
                            other_dilated = dilate_mask(other_info["mask"], pixels=15)
                            overlap_pixels = np.logical_and(new_mask, other_dilated).sum()
                            touches = overlap_pixels > 0

                            if not touches:
                                print(f"        [Skip] '{new_obj.labels[0]}' not adjacent to '{list(other_info['labels'])[0]}'")
                                continue

                            # Check area constraint: expanded area <= 3x original
                            combined = np.logical_or(other_info["mask"], new_mask)
                            combined_area = combined.sum()

                            if combined_area > other_info["original_area"] * 3:
                                print(f"        [Skip] '{new_obj.labels[0]}' would exceed 3x area ({combined_area} > {other_info['original_area'] * 3})")
                                continue

                            # Valid expansion! Merge masks
                            expansion_area = combined_area - other_info["current_area"]
                            if expansion_area > 0:
                                print(f"        Expanding '{list(other_info['labels'])[0]}': +{expansion_area} pixels (overlap: {overlap_pixels})")
                                other_info["mask"] = combined
                                other_info["current_area"] = combined_area

                # Update current image for the next iteration
                current_image = temp_inpainted

            # Save final expanded masks to debug directory
            if save_debug:
                final_masks_dir = debug_dir / "final_expanded_masks"
                final_masks_dir.mkdir(exist_ok=True)
                for obj_id, obj_info in object_masks.items():
                    name = obj_info["name"]
                    self._save_image(
                        mask_to_image(obj_info["mask"]),
                        final_masks_dir / f"{obj_id:02d}_{name}_mask.png",
                        is_mask=True,
                    )
                    self._save_image(
                        mask_to_rgb(image_np, obj_info["mask"]),
                        final_masks_dir / f"{obj_id:02d}_{name}_rgb.png",
                    )

            # Final inpainting: sequentially apply expanded masks on current image
            print("  Final inpainting with expanded masks (sequential)...")
            final_image = current_image
            for obj in objects:
                obj_info = object_masks[obj.id]
                labels = list(obj_info["labels"])
                orig_area = obj_info["original_area"]
                final_area = obj_info["current_area"]
                print(f"    '{labels[0]}': {orig_area} -> {final_area} (+{final_area - orig_area})")

                dilated = dilate_mask(obj_info["mask"], pixels=mask_dilate_pixels)
                final_image = self.inpainter.inpaint(final_image, dilated)

            self._save_image(Image.fromarray(final_image), clean_path)
            print(f"Background inpainting complete.")
        else:
            clean_path = None

        # Generate report
        report = {
            "input_image": str(input_copy),
            "num_detections": len(detections),
            "num_objects": len(objects),
            "objects_dir": str(objects_dir),
            "masks_dir": str(masks_dir) if masks_dir else None,
            "objects": [
                {
                    "id": obj.id,
                    "labels": obj.labels,
                    "bbox": list(obj.bbox),
                    "area": obj.area,
                    "score": obj.score,
                }
                for obj in objects
            ],
            "combined_mask": str(combined_path) if combined_mask is not None else None,
            "clean_background": str(clean_path) if clean_path else None,
            "config": {
                "prompts": self.config.prompts,
                "sam_model": self.config.sam_model,
                "box_threshold": self.config.box_threshold,
                "text_threshold": self.config.text_threshold,
                "iou_threshold": self.config.iou_threshold,
                "inpaint_backend": self.config.inpaint_backend,
                "save_debug": self.config.save_debug,
                "save_individual_masks": self.config.save_individual_masks,
            },
        }
        with open(output_path / "report.json", "w") as f:
            json.dump(report, f, indent=2)

        print(f"Results saved to: {output_path}")
        return report
