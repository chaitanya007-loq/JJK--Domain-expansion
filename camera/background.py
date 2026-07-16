"""
camera/background.py — AI background removal via MediaPipe ImageSegmenter (Tasks API).

Compatible with mediapipe >= 0.10.30 (Python 3.13).
Uses the selfie_segmenter.tflite model in models/ directory.

Isolates the user's silhouette from the webcam frame, returning:
  • a binary mask (uint8, 0/255)
  • a composited BGR frame (person over replacement background)
"""

import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from utils.logger import get_logger
import config

log = get_logger(__name__)

# Model path (downloaded once into models/)
_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "selfie_segmenter.tflite")


class BackgroundRemover:
    """
    Uses MediaPipe's ImageSegmenter (Tasks API) to separate the person
    from the background in real-time.
    """

    def __init__(self):
        self._segmenter = None
        self._last_mask  = None   # cached mask from latest callback
        log.info("BackgroundRemover initialised (model not loaded yet).")

    # ------------------------------------------------------------------ lifecycle

    def start(self):
        """Load the segmentation model (call once before processing frames)."""
        if not os.path.isfile(_MODEL_PATH):
            log.error(f"Segmentation model not found: {_MODEL_PATH}")
            log.error("Run: python -c \"import urllib.request; urllib.request.urlretrieve("
                      "'https://storage.googleapis.com/mediapipe-models/image_segmenter/"
                      "selfie_segmenter/float16/latest/selfie_segmenter.tflite', "
                      "'models/selfie_segmenter.tflite')\"")
            return

        base_options    = mp_python.BaseOptions(model_asset_path=_MODEL_PATH)
        options         = mp_vision.ImageSegmenterOptions(
            base_options  = base_options,
            running_mode  = mp_vision.RunningMode.IMAGE,
            output_confidence_masks = True,
        )
        self._segmenter = mp_vision.ImageSegmenter.create_from_options(options)
        log.info(f"Selfie segmentation model loaded: {_MODEL_PATH}")

    def stop(self):
        """Release model resources."""
        if self._segmenter:
            self._segmenter.close()
            self._segmenter = None
            log.info("Selfie segmentation stopped.")

    # ------------------------------------------------------------------ processing

    def get_mask(self, bgr_frame: np.ndarray) -> np.ndarray:
        """
        Compute a binary person-mask for the given BGR frame.

        Parameters
        ----------
        bgr_frame : np.ndarray — Input webcam frame in BGR colour space.

        Returns
        -------
        mask : np.ndarray (H×W, uint8) — 255 = person, 0 = background.
        """
        if self._segmenter is None:
            # Fallback: return full-frame mask (no segmentation)
            h, w = bgr_frame.shape[:2]
            return np.full((h, w), 255, dtype=np.uint8)

        rgb       = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result    = self._segmenter.segment(mp_image)

        # confidence_masks[0]: float32 [0,1] confidence map
        conf_mask = result.confidence_masks[0].numpy_view()
        if len(conf_mask.shape) == 3:
            conf_mask = conf_mask[:, :, 0]

        # threshold to create binary mask
        binary = (conf_mask > config.BG_THRESHOLD).astype(np.uint8) * 255

        # Morphological cleanup (3x3 kernel preserves fingers and hair details)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel)

        return binary

    def remove_background(
        self,
        bgr_frame:   np.ndarray,
        replacement: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Replace the background with `replacement` (or black if None).

        Returns
        -------
        (composited_bgr, binary_mask)
        """
        mask = self.get_mask(bgr_frame)

        if replacement is None:
            replacement = np.zeros_like(bgr_frame)
        else:
            replacement = cv2.resize(
                replacement,
                (bgr_frame.shape[1], bgr_frame.shape[0])
            )

        # Smooth the mask edges for a cleaner composite
        smooth_mask = cv2.GaussianBlur(mask, (21, 21), 0)
        alpha       = smooth_mask[:, :, np.newaxis].astype(np.float32) / 255.0

        person = bgr_frame.astype(np.float32)
        bg     = replacement.astype(np.float32)

        composited = (person * alpha + bg * (1.0 - alpha)).astype(np.uint8)
        return composited, mask
