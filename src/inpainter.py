"""Background inpainting using iopaint or OpenCV."""

import os
from typing import Optional

import numpy as np
from PIL import Image


class Inpainter:
    """Background inpainter supporting multiple backends."""

    def __init__(self, backend: str = "iopaint", device: str = "cuda"):
        """
        Initialize the inpainter.

        Args:
            backend: Inpainting backend ("iopaint", "opencv", "none")
            device: Device for iopaint
        """
        self.backend = backend
        self.device = device
        self._model = None

    def _init_iopaint(self):
        """Initialize iopaint model."""
        if self._model is not None:
            return

        from iopaint.model_manager import ModelManager

        # Set torch home for model downloads
        if "TORCH_HOME" not in os.environ:
            checkpoint_folder = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
            os.environ["TORCH_HOME"] = os.path.abspath(checkpoint_folder)

        # Ensure LaMa model is downloaded
        from iopaint.model.lama import LaMa

        if not LaMa.is_downloaded():
            print("Downloading LaMa inpainting model...")
            LaMa.download()

        self._model = ModelManager(name="lama", device=self.device)

    def inpaint(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """
        Inpaint the masked regions of an image.

        Args:
            image: RGB image as numpy array (H, W, 3)
            mask: Binary mask where True/255 indicates regions to inpaint

        Returns:
            Inpainted RGB image
        """
        if self.backend == "none":
            return image.copy()

        # Ensure mask is uint8 0-255
        if mask.dtype == bool:
            mask_u8 = mask.astype(np.uint8) * 255
        elif mask.max() <= 1:
            mask_u8 = (mask * 255).astype(np.uint8)
        else:
            mask_u8 = mask.astype(np.uint8)

        if self.backend == "iopaint":
            return self._inpaint_iopaint(image, mask_u8)
        elif self.backend == "opencv":
            return self._inpaint_opencv(image, mask_u8)
        else:
            raise ValueError(f"Unknown inpaint backend: {self.backend}")

    def _inpaint_iopaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Inpaint using iopaint/LaMa."""
        self._init_iopaint()

        try:
            from iopaint.schema import Config as IOPaintConfig

            config = IOPaintConfig(
                hd_strategy="Original",
                hd_strategy_crop_margin=32,
            )
        except ImportError:
            from iopaint.schema import HDStrategy, InpaintRequest

            config = InpaintRequest(
                hd_strategy=HDStrategy.ORIGINAL,
                hd_strategy_crop_margin=32,
            )

        result = self._model(image, mask, config=config)

        # iopaint returns BGR, convert to RGB
        return result[:, :, ::-1].copy()

    def _inpaint_opencv(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Inpaint using OpenCV."""
        import cv2

        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        inpainted = cv2.inpaint(bgr, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        return cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
