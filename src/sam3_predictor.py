"""SAM3 (Segment Anything 3) text-prompted segmentation.

Uses Meta's official `sam3` package (github.com/facebookresearch/sam3).
Vision features are cached in `state` between `set_image()` and `detect()`,
so multiple prompts on the same image only pay the image-encoder cost once.
"""

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model


@dataclass
class Detection:
    """A single detection: bounding box, confidence, label, and binary mask."""

    bbox: Tuple[int, int, int, int]
    score: float
    label: str
    mask: Optional[np.ndarray] = None


class SAM3Predictor:
    """Text-prompted concept segmentation using SAM3."""

    def __init__(
        self,
        model_id: str = "facebook/sam3",
        device: str = "cuda",
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
        compile_model: bool = False,
    ) -> None:
        self.device = device if torch.cuda.is_available() else "cpu"
        self.threshold = threshold
        self.mask_threshold = mask_threshold

        if self.device.startswith("cuda") and torch.cuda.is_bf16_supported():
            self.autocast_dtype: Optional[torch.dtype] = torch.bfloat16
        else:
            self.autocast_dtype = None

        checkpoint_path = self._resolve_checkpoint(model_id)
        load_from_hf = checkpoint_path is None

        print(f"Loading SAM3 (checkpoint={checkpoint_path or 'auto from HF'})...")
        self.model = build_sam3_image_model(
            device=self.device,
            checkpoint_path=checkpoint_path,
            load_from_HF=load_from_hf,
            compile=compile_model,
        )
        self.processor = Sam3Processor(self.model)
        if hasattr(self.processor, "set_confidence_threshold"):
            self.processor.set_confidence_threshold(self.threshold)
        print("SAM3 loaded.")

        self._state = None
        self._image_size: Optional[Tuple[int, int]] = None

    @staticmethod
    def _resolve_checkpoint(model_id: str) -> Optional[str]:
        """Return a local checkpoint path if available; else None to let
        sam3 auto-download from HuggingFace Hub.

        Lookup order:
          1. Direct file path
          2. <repo>/checkpoints/<basename>.pt
          3. <repo>/checkpoints/sam3.pt
          4. Download from HF Hub into <repo>/checkpoints/sam3.pt
        """
        path = Path(model_id)
        if path.is_file():
            return str(path)

        checkpoints_dir = Path(__file__).parent.parent / "checkpoints"
        if "/" in model_id:
            basename = model_id.split("/", 1)[1]
        else:
            basename = model_id
        candidate = checkpoints_dir / f"{basename}.pt"
        if candidate.is_file():
            return str(candidate)
        default = checkpoints_dir / "sam3.pt"
        if default.is_file():
            return str(default)

        if "/" in model_id:
            try:
                from huggingface_hub import hf_hub_download
            except ImportError:
                return None
            checkpoints_dir.mkdir(parents=True, exist_ok=True)
            print(
                f"Downloading {model_id}/sam3.pt into {checkpoints_dir} "
                "(first-time setup, ~3.4GB)..."
            )
            downloaded = hf_hub_download(
                repo_id=model_id,
                filename="sam3.pt",
                local_dir=str(checkpoints_dir),
            )
            return downloaded
        return None

    def _autocast(self):
        if self.autocast_dtype is None or not self.device.startswith("cuda"):
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.autocast_dtype)

    def set_image(self, image: np.ndarray) -> None:
        """Compute and cache vision features for one image (RGB numpy array)."""
        pil = Image.fromarray(image)
        self._image_size = pil.size  # (W, H)
        with torch.no_grad(), self._autocast():
            self._state = self.processor.set_image(pil)

    def reset_image(self) -> None:
        self._state = None
        self._image_size = None

    def detect(self, prompt: str) -> List[Detection]:
        """Detect and segment all instances matching `prompt`."""
        if self._state is None:
            raise RuntimeError("Must call set_image() before detect()")

        if hasattr(self.processor, "reset_all_prompts"):
            self.processor.reset_all_prompts(self._state)

        with torch.no_grad(), self._autocast():
            output = self.processor.set_text_prompt(prompt=prompt, state=self._state)

        return self._output_to_detections(output, prompt)

    def _output_to_detections(self, output: dict, label: str) -> List[Detection]:
        masks = output.get("masks")
        boxes = output.get("boxes")
        scores = output.get("scores")
        if masks is None or boxes is None or scores is None:
            return []

        masks_np = self._to_numpy(masks)
        boxes_np = self._to_numpy(boxes)
        scores_np = self._to_numpy(scores)

        if masks_np.size == 0:
            return []

        detections: List[Detection] = []
        for mask, box, score in zip(masks_np, boxes_np, scores_np):
            float_score = float(score)
            if float_score < self.threshold:
                continue

            mask_arr = mask
            if mask_arr.ndim == 3:
                mask_arr = mask_arr[0]
            if mask_arr.dtype == bool:
                mask_bool = mask_arr
            else:
                mask_bool = mask_arr > self.mask_threshold

            x1, y1, x2, y2 = self._box_to_pixels(box)
            detections.append(
                Detection(
                    bbox=(x1, y1, x2, y2),
                    score=float_score,
                    label=label,
                    mask=mask_bool,
                )
            )
        return detections

    @staticmethod
    def _to_numpy(tensor) -> np.ndarray:
        """Convert torch tensor to numpy, casting bf16/fp16 to fp32 first."""
        if hasattr(tensor, "detach"):
            t = tensor.detach()
            if hasattr(t, "to") and hasattr(t, "dtype"):
                if t.dtype in (torch.bfloat16, torch.float16):
                    t = t.to(torch.float32)
            t = t.cpu()
            return t.numpy()
        return np.asarray(tensor)

    def _box_to_pixels(self, box: np.ndarray) -> Tuple[int, int, int, int]:
        """Convert SAM3 box to absolute (x1, y1, x2, y2) pixel ints.

        SAM3 returns boxes in xyxy. They may be in pixels or normalized [0,1] —
        detect by magnitude and rescale if needed.
        """
        x1, y1, x2, y2 = (float(v) for v in box[:4])
        if self._image_size is not None:
            w, h = self._image_size
            if max(x1, y1, x2, y2) <= 1.5:
                x1 *= w
                x2 *= w
                y1 *= h
                y2 *= h
        return int(x1), int(y1), int(x2), int(y2)
