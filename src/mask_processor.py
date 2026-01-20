"""Mask processing utilities: deduplication, merging, and analysis."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


@dataclass
class SegmentedObject:
    """A segmented object with its mask and metadata."""

    id: int
    mask: np.ndarray  # Boolean mask (H, W)
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    score: float
    labels: List[str] = field(default_factory=list)  # May have multiple labels if merged
    area: int = 0

    def __post_init__(self):
        if self.area == 0:
            self.area = int(np.sum(self.mask))


def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute Intersection over Union between two masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def compute_overlap_ratio(mask1: np.ndarray, mask2: np.ndarray) -> Tuple[float, float]:
    """
    Compute overlap ratios for both masks.

    Returns:
        (ratio1, ratio2) where ratio1 = intersection / mask1_area
    """
    intersection = np.logical_and(mask1, mask2).sum()
    area1 = mask1.sum()
    area2 = mask2.sum()

    ratio1 = float(intersection / area1) if area1 > 0 else 0.0
    ratio2 = float(intersection / area2) if area2 > 0 else 0.0

    return ratio1, ratio2


def mask_boundary(mask: np.ndarray) -> np.ndarray:
    """Return a 1-pixel boundary mask."""
    mask_bool = mask.astype(bool) if mask.dtype != bool else mask
    if not mask_bool.any():
        return mask_bool
    mask_u8 = (mask_bool.astype(np.uint8) * 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    eroded = cv2.erode(mask_u8, kernel, iterations=1)
    boundary = cv2.bitwise_xor(mask_u8, eroded)
    return boundary > 0


def contour_overlap_ratio(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Compute overlap ratio of two mask boundaries.

    Uses overlap / min(boundary_pixels) to be conservative.
    """
    boundary1 = mask_boundary(mask1)
    boundary2 = mask_boundary(mask2)
    denom = min(boundary1.sum(), boundary2.sum())
    if denom == 0:
        return 0.0
    overlap = np.logical_and(boundary1, boundary2).sum()
    return float(overlap / denom)


def is_contained_split(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    overlap_ratio_thresh: float,
    contour_overlap_thresh: float,
) -> bool:
    """Detect likely split masks via containment + contour overlap."""
    area_a = mask_a.sum()
    area_b = mask_b.sum()
    if area_a == 0 or area_b == 0:
        return False

    if area_a <= area_b:
        small, large = mask_a, mask_b
        small_area = area_a
    else:
        small, large = mask_b, mask_a
        small_area = area_b

    overlap_ratio = np.logical_and(small, large).sum() / small_area
    if overlap_ratio < overlap_ratio_thresh:
        return False

    contour_ratio = contour_overlap_ratio(small, large)
    return contour_ratio >= contour_overlap_thresh


def merge_masks(mask1: np.ndarray, mask2: np.ndarray) -> np.ndarray:
    """Merge two masks using logical OR."""
    return np.logical_or(mask1, mask2)


def deduplicate_objects(
    objects: List[SegmentedObject],
    iou_threshold: float = 0.5,
    containment_overlap_ratio: float = 0.9,
    contour_overlap_ratio: float = 0.3,
) -> List[SegmentedObject]:
    """
    Deduplicate objects by merging those with high IoU.

    Objects with IoU above threshold are considered the same object
    (e.g., "red cup" and "coffee mug" pointing to the same thing).
    Also merges when a smaller mask is mostly contained within a larger one
    and their contours overlap significantly (likely split).

    Args:
        objects: List of segmented objects
        iou_threshold: IoU threshold for merging

    Returns:
        Deduplicated list of objects
    """
    if not objects:
        return []

    # Sort by score (highest first)
    sorted_objects = sorted(objects, key=lambda x: x.score, reverse=True)

    kept = []
    merged_indices = set()

    for i, obj_i in enumerate(sorted_objects):
        if i in merged_indices:
            continue

        # Find all objects that should be merged with this one
        to_merge = [obj_i]
        merge_labels = list(obj_i.labels)
        merge_scores = [obj_i.score]
        merged_mask = obj_i.mask

        for j, obj_j in enumerate(sorted_objects[i + 1 :], start=i + 1):
            if j in merged_indices:
                continue

            iou = compute_iou(merged_mask, obj_j.mask)
            if iou >= iou_threshold or is_contained_split(
                merged_mask,
                obj_j.mask,
                overlap_ratio_thresh=containment_overlap_ratio,
                contour_overlap_thresh=contour_overlap_ratio,
            ):
                to_merge.append(obj_j)
                merge_labels.extend(obj_j.labels)
                merge_scores.append(obj_j.score)
                merged_mask = merge_masks(merged_mask, obj_j.mask)
                merged_indices.add(j)

        # Create merged object
        if len(to_merge) == 1:
            kept.append(obj_i)
        else:
            # Merge masks
            # Compute new bbox from merged mask
            rows = np.any(merged_mask, axis=1)
            cols = np.any(merged_mask, axis=0)
            y1, y2 = np.where(rows)[0][[0, -1]]
            x1, x2 = np.where(cols)[0][[0, -1]]

            # Average score, keep highest
            avg_score = max(merge_scores)

            # Remove duplicate labels
            unique_labels = list(dict.fromkeys(merge_labels))

            merged_obj = SegmentedObject(
                id=obj_i.id,
                mask=merged_mask,
                bbox=(int(x1), int(y1), int(x2), int(y2)),
                score=avg_score,
                labels=unique_labels,
            )
            kept.append(merged_obj)

    # Re-assign IDs
    for i, obj in enumerate(kept):
        obj.id = i

    return kept


def combine_all_masks(objects: List[SegmentedObject]) -> Optional[np.ndarray]:
    """Combine all object masks into a single mask."""
    if not objects:
        return None

    combined = np.zeros_like(objects[0].mask, dtype=bool)
    for obj in objects:
        combined = np.logical_or(combined, obj.mask)

    return combined


def extract_object_crop(
    image: np.ndarray,
    mask: np.ndarray,
    bbox: Optional[Tuple[int, int, int, int]] = None,
    padding: int = 0,
) -> Image.Image:
    """
    Extract an object from the image using its mask.

    Returns an RGBA image with transparent background.
    """
    h, w = image.shape[:2]

    if bbox is None:
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not rows.any() or not cols.any():
            return Image.new("RGBA", (1, 1), (0, 0, 0, 0))
        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]
    else:
        x1, y1, x2, y2 = bbox

    # Apply padding
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding + 1)
    y2 = min(h, y2 + padding + 1)

    # Crop image and mask
    cropped_image = image[y1:y2, x1:x2]
    cropped_mask = mask[y1:y2, x1:x2]

    # Create RGBA image
    rgba = np.zeros((y2 - y1, x2 - x1, 4), dtype=np.uint8)
    rgba[:, :, :3] = cropped_image
    rgba[:, :, 3] = (cropped_mask * 255).astype(np.uint8)

    return Image.fromarray(rgba, mode="RGBA")


def mask_to_image(mask: np.ndarray) -> Image.Image:
    """Convert a boolean mask to a PIL Image."""
    return Image.fromarray((mask.astype(np.uint8) * 255))


def mask_to_rgb(image: np.ndarray, mask: np.ndarray) -> Image.Image:
    """
    Create an RGB image showing the masked region with original colors.

    Args:
        image: Original RGB image (H, W, 3)
        mask: Boolean mask (H, W)

    Returns:
        RGB image with black background and object in original colors
    """
    # Ensure mask is boolean
    mask_bool = mask.astype(bool) if mask.dtype != bool else mask

    # Create black background
    result = np.zeros_like(image)

    # Copy original pixels where mask is True
    result[mask_bool] = image[mask_bool]

    return Image.fromarray(result)


def dilate_mask(mask: np.ndarray, pixels: int = 5) -> np.ndarray:
    """
    Dilate a mask by a number of pixels.

    This is useful for inpainting to avoid leaving edge artifacts.

    Args:
        mask: Boolean or uint8 mask
        pixels: Number of pixels to dilate

    Returns:
        Dilated mask (same dtype as input)
    """
    if pixels <= 0:
        return mask

    # Convert to uint8 if needed
    was_bool = mask.dtype == bool
    mask_u8 = mask.astype(np.uint8) * 255 if was_bool else mask

    # Create circular kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pixels * 2 + 1, pixels * 2 + 1))

    # Dilate
    dilated = cv2.dilate(mask_u8, kernel, iterations=1)

    # Convert back if needed
    if was_bool:
        return dilated > 0
    return dilated
