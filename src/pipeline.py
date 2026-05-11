"""Main segmentation pipeline."""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

from .config import SegmentConfig
from .inpainter import Inpainter
from .mask_processor import (
    SegmentedObject,
    combine_all_masks,
    deduplicate_objects,
    dilate_mask,
    extract_object_crop,
    mask_to_image,
    mask_to_rgb,
)
from .sam3_predictor import Detection, SAM3Predictor


class SegmentInpaintPipeline:
    """Pipeline for SAM3-based concept segmentation and background inpainting."""

    def __init__(self, config: SegmentConfig):
        self.config = config
        self._sam3: Optional[SAM3Predictor] = None
        self._inpainter: Optional[Inpainter] = None

    @property
    def sam3(self) -> SAM3Predictor:
        if self._sam3 is None:
            self._sam3 = SAM3Predictor(
                model_id=self.config.sam3_model,
                device=self.config.device,
                threshold=self.config.sam3_threshold,
                mask_threshold=self.config.sam3_mask_threshold,
            )
        return self._sam3

    @property
    def inpainter(self) -> Inpainter:
        if self._inpainter is None:
            self._inpainter = Inpainter(
                backend=self.config.inpaint_backend,
                device=self.config.device,
            )
        return self._inpainter

    def _save_image(self, image: Image.Image, path: Path, is_mask: bool = False) -> None:
        if self.config.output_size:
            resample = Image.NEAREST if is_mask else Image.LANCZOS
            image = image.resize(self.config.output_size, resample=resample)
        image.save(path)

    def detect_and_segment(
        self,
        image_np: np.ndarray,
        prompts: Optional[List[str]] = None,
    ) -> List[Detection]:
        """Run SAM3 over each prompt on the given image, sharing vision features."""
        active_prompts = prompts if prompts is not None else self.config.prompts
        if not active_prompts:
            return []

        self.sam3.set_image(image_np)
        try:
            all_detections: List[Detection] = []
            for prompt in active_prompts:
                detections = self.sam3.detect(prompt)
                if detections:
                    print(f"  '{prompt}': found {len(detections)} objects")
                else:
                    print(f"  '{prompt}': no objects found")
                all_detections.extend(detections)
            return all_detections
        finally:
            self.sam3.reset_image()

    @staticmethod
    def _detections_to_objects(detections: List[Detection]) -> List[SegmentedObject]:
        objects: List[SegmentedObject] = []
        for i, det in enumerate(detections):
            if det.mask is None:
                continue
            objects.append(
                SegmentedObject(
                    id=i,
                    mask=det.mask,
                    bbox=det.bbox,
                    score=det.score,
                    labels=[det.label],
                )
            )
        return objects

    def process(
        self,
        image_path: str,
        output_dir: str,
        prompts: Optional[List[str]] = None,
    ) -> Dict:
        """Process an image: detect+segment, deduplicate, and inpaint."""
        if prompts:
            self.config.prompts = prompts

        if not self.config.prompts:
            raise ValueError("No prompts provided. Set prompts in config or pass them directly.")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        objects_dir = output_path / "objects"
        objects_dir.mkdir(exist_ok=True)

        print(f"Loading image: {image_path}")
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        image_area = image_np.shape[0] * image_np.shape[1]
        max_mask_area = int(image_area * 0.2)

        input_copy = output_path / "input_image.png"
        self._save_image(image, input_copy)

        print("Detecting and segmenting objects with SAM3...")
        detections = self.detect_and_segment(image_np)
        print(f"Total detections: {len(detections)}")

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

        objects = self._detections_to_objects(detections)
        print(f"Segmented {len(objects)} objects")

        if objects:
            before = len(objects)
            objects = [obj for obj in objects if obj.area <= max_mask_area]
            dropped = before - len(objects)
            if dropped:
                print(f"Filtered {dropped} oversized masks (>20% image area)")

        print(f"Deduplicating with IoU threshold {self.config.iou_threshold}...")
        objects = deduplicate_objects(
            objects,
            self.config.iou_threshold,
            self.config.containment_overlap_ratio,
            self.config.contour_overlap_ratio,
        )
        print(f"After deduplication: {len(objects)} unique objects")

        if objects:
            before = len(objects)
            objects = [obj for obj in objects if obj.area <= max_mask_area]
            dropped = before - len(objects)
            if dropped:
                print(f"Filtered {dropped} oversized masks after dedup (>20% image area)")
            for i, obj in enumerate(objects):
                obj.id = i

        print("Saving individual objects...")
        for obj in objects:
            label_slug = obj.labels[0].replace(" ", "_").replace("/", "-")[:30]
            obj_dir = objects_dir / f"{obj.id:03d}_{label_slug}"
            obj_dir.mkdir(exist_ok=True)

            mask_img = mask_to_image(obj.mask)
            self._save_image(mask_img, obj_dir / "mask.png", is_mask=True)

            rgb_mask = mask_to_rgb(image_np, obj.mask)
            self._save_image(rgb_mask, obj_dir / "mask_rgb.png")

            info = {
                "id": obj.id,
                "labels": obj.labels,
                "bbox": list(obj.bbox),
                "area": obj.area,
                "score": obj.score,
            }
            with open(obj_dir / "info.json", "w") as f:
                json.dump(info, f, indent=2)

        masks_dir = None
        if self.config.save_individual_masks and objects:
            print("Saving individual RGB masks...")
            masks_dir = output_path / "masks"
            masks_dir.mkdir(exist_ok=True)

            skip_labels = {"robot arm", "gripper"}
            include_robot_gripper = self.config.save_individual_masks_include_robot_gripper

            name_counts = {}
            saved_count = 0
            for obj in objects:
                if not include_robot_gripper:
                    primary_label = obj.labels[0].strip().lower() if obj.labels else ""
                    if primary_label in skip_labels:
                        continue

                base_name = obj.labels[0].replace(" ", "_").replace("/", "-")

                if base_name in name_counts:
                    name_counts[base_name] += 1
                    filename = f"{base_name}_{name_counts[base_name]}.png"
                else:
                    name_counts[base_name] = 0
                    filename = f"{base_name}.png"

                rgb_mask = mask_to_rgb(image_np, obj.mask)
                self._save_image(rgb_mask, masks_dir / filename)

                saved_count += 1

            print(f"  Saved {saved_count} RGB masks to: {masks_dir}")

        transparent_masks_dir = None
        if self.config.save_individual_transparent_masks and objects:
            print("Saving individual transparent cutouts...")
            transparent_masks_dir = output_path / "masks_transparent"
            transparent_masks_dir.mkdir(exist_ok=True)

            skip_labels = {"robot arm", "gripper"}
            include_robot_gripper = (
                self.config.save_individual_transparent_masks_include_robot_gripper
            )

            name_counts = {}
            saved_count = 0
            for obj in objects:
                if not include_robot_gripper:
                    primary_label = obj.labels[0].strip().lower() if obj.labels else ""
                    if primary_label in skip_labels:
                        continue

                base_name = obj.labels[0].replace(" ", "_").replace("/", "-")

                if base_name in name_counts:
                    name_counts[base_name] += 1
                    filename = f"{base_name}_{name_counts[base_name]}.png"
                else:
                    name_counts[base_name] = 0
                    filename = f"{base_name}.png"

                rgba_cutout = extract_object_crop(image_np, obj.mask)
                self._save_image(rgba_cutout, transparent_masks_dir / filename)
                saved_count += 1

            print(f"  Saved {saved_count} transparent cutouts to: {transparent_masks_dir}")

        combined_mask = combine_all_masks(objects)
        combined_path = output_path / "combined_mask.png"
        if combined_mask is not None:
            self._save_image(mask_to_image(combined_mask), combined_path, is_mask=True)

        clean_path = output_path / "clean_background.png"
        final_image_np: Optional[np.ndarray] = None
        if objects and self.config.inpaint_backend != "none":
            print(f"Inpainting background with {self.config.inpaint_backend}...")
            print(f"  Iterative inpaint + re-detection for {len(objects)} objects...")
            mask_dilate_pixels = self.config.mask_dilate_pixels
            save_debug = self.config.save_debug

            debug_dir = None
            if save_debug:
                debug_dir = output_path / "debug"
                debug_dir.mkdir(exist_ok=True)

            detected_labels = sorted({label for obj in objects for label in obj.labels})

            def _match_prompts_for_redetect(prompts, labels):
                if not prompts or not labels:
                    return []
                cleaned_labels = []
                for label in labels:
                    if not label:
                        continue
                    cleaned = label.lstrip("#").strip().lower()
                    if cleaned:
                        cleaned_labels.append(cleaned)
                if not cleaned_labels:
                    return []
                matched = []
                for prompt in prompts:
                    prompt_text = str(prompt).strip()
                    if not prompt_text:
                        continue
                    prompt_lower = prompt_text.lower()
                    if any(token in prompt_lower for token in cleaned_labels):
                        matched.append(prompt_text)
                seen = set()
                unique = []
                for prompt in matched:
                    if prompt not in seen:
                        seen.add(prompt)
                        unique.append(prompt)
                return unique

            redetect_prompts = _match_prompts_for_redetect(self.config.prompts, detected_labels)

            object_masks = {}
            for obj in objects:
                object_masks[obj.id] = {
                    "mask": obj.mask.astype(bool),
                    "labels": set(obj.labels),
                    "original_area": obj.area,
                    "current_area": obj.area,
                    "name": obj.labels[0].replace(" ", "_"),
                }

            current_image = image_np.copy()

            for i, obj in enumerate(objects):
                obj_info = object_masks[obj.id]
                current_mask = obj_info["mask"]
                obj_name = obj_info["name"]

                step_dir = None
                if save_debug:
                    step_dir = debug_dir / f"step_{i+1:02d}_remove_{obj_name}"
                    step_dir.mkdir(exist_ok=True)

                dilated = dilate_mask(current_mask, pixels=mask_dilate_pixels)
                print(f"    [{i+1}/{len(objects)}] Checking expansion after removing '{obj.labels[0]}' (area: {obj_info['current_area']})...")

                if save_debug:
                    self._save_image(mask_to_image(current_mask), step_dir / "removed_mask.png", is_mask=True)
                    self._save_image(mask_to_rgb(image_np, current_mask), step_dir / "removed_mask_rgb.png")

                temp_inpainted = self.inpainter.inpaint(current_image, dilated)

                if save_debug:
                    self._save_image(Image.fromarray(temp_inpainted), step_dir / "inpainted.png")

                print(f"        Re-detecting for mask expansion...")
                new_detections = (
                    self.detect_and_segment(temp_inpainted, prompts=redetect_prompts)
                    if redetect_prompts
                    else []
                )

                if new_detections:
                    new_objects = self._detections_to_objects(new_detections)

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

                    for new_obj in new_objects:
                        new_mask = new_obj.mask.astype(bool)
                        new_labels = set(new_obj.labels)

                        for other_id, other_info in object_masks.items():
                            if other_id == obj.id:
                                continue

                            if not (new_labels & other_info["labels"]):
                                continue

                            other_dilated = dilate_mask(other_info["mask"], pixels=15)
                            overlap_pixels = np.logical_and(new_mask, other_dilated).sum()
                            touches = overlap_pixels > 0

                            if not touches:
                                print(f"        [Skip] '{new_obj.labels[0]}' not adjacent to '{list(other_info['labels'])[0]}'")
                                continue

                            combined = np.logical_or(other_info["mask"], new_mask)
                            combined_area = combined.sum()

                            if combined_area > other_info["original_area"] * 3:
                                print(f"        [Skip] '{new_obj.labels[0]}' would exceed 3x area ({combined_area} > {other_info['original_area'] * 3})")
                                continue

                            expansion_area = combined_area - other_info["current_area"]
                            if expansion_area > 0:
                                print(f"        Expanding '{list(other_info['labels'])[0]}': +{expansion_area} pixels (overlap: {overlap_pixels})")
                                other_info["mask"] = combined
                                other_info["current_area"] = combined_area

                current_image = temp_inpainted

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
            final_image_np = final_image
            print(f"Background inpainting complete.")
        else:
            clean_path = None

        desktop_mask = None
        desktop_mask_path = output_path / "bg_mask.png"
        if final_image_np is None:
            print("Skipping desktop mask: clean background not available.")
        else:
            print("Detecting desktop surface...")
            desktop_prompts = ["tabletop", "desk surface", "desktop surface"]
            desktop_detections = self.detect_and_segment(final_image_np, prompts=desktop_prompts)

            if desktop_detections:
                desktop_objects = self._detections_to_objects(desktop_detections)
                desktop_objects = deduplicate_objects(
                    desktop_objects,
                    self.config.iou_threshold,
                    self.config.containment_overlap_ratio,
                    self.config.contour_overlap_ratio,
                )
                desktop_mask = combine_all_masks(desktop_objects)
                if desktop_mask is not None:
                    self._save_image(mask_to_image(desktop_mask), desktop_mask_path, is_mask=True)
                    print("Background surface mask saved.")
                else:
                    print("No desktop mask produced after segmentation.")
            else:
                print("No desktop surface detected.")

        report = {
            "input_image": str(input_copy),
            "num_detections": len(detections),
            "num_objects": len(objects),
            "objects_dir": str(objects_dir),
            "masks_dir": str(masks_dir) if masks_dir else None,
            "masks_transparent_dir": str(transparent_masks_dir) if transparent_masks_dir else None,
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
            "bg_mask": str(desktop_mask_path) if desktop_mask is not None else None,
            "config": {
                "prompts": self.config.prompts,
                "sam3_model": self.config.sam3_model,
                "sam3_threshold": self.config.sam3_threshold,
                "sam3_mask_threshold": self.config.sam3_mask_threshold,
                "iou_threshold": self.config.iou_threshold,
                "inpaint_backend": self.config.inpaint_backend,
                "save_debug": self.config.save_debug,
                "save_individual_masks": self.config.save_individual_masks,
                "save_individual_masks_include_robot_gripper": (
                    self.config.save_individual_masks_include_robot_gripper
                ),
                "save_individual_transparent_masks": (
                    self.config.save_individual_transparent_masks
                ),
                "save_individual_transparent_masks_include_robot_gripper": (
                    self.config.save_individual_transparent_masks_include_robot_gripper
                ),
            },
        }
        with open(output_path / "report.json", "w") as f:
            json.dump(report, f, indent=2)

        print(f"Results saved to: {output_path}")
        return report
