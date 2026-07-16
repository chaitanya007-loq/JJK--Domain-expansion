"""
effects/flash.py — Screen flash effect triggered on domain expansion.

The flash ramps up to full brightness in the first half of its duration,
then fades to zero in the second half (cosine easing for smoothness).
"""

import cv2
import numpy as np
import config
from utils.logger import get_logger

log = get_logger(__name__)


class FlashEffect:
    """
    Overlay a coloured flash that fades in and out over N frames.

    Usage
    -----
    flash = FlashEffect()
    flash.trigger(color=(255, 255, 255))   # call once to start

    # In your render loop:
    frame = flash.apply(frame)             # call every frame
    """

    def __init__(self):
        self._active    = False
        self._frame_idx = 0
        self._duration  = config.FLASH_DURATION_FRAMES
        self._color     = (255, 255, 255)   # BGR
        self._peak      = config.FLASH_PEAK_ALPHA

    # ------------------------------------------------------------------ control

    def trigger(self, color: tuple = (255, 255, 255)):
        """Start a new flash with the given colour."""
        self._color     = color
        self._frame_idx = 0
        self._active    = True
        log.debug(f"Flash triggered: color={color}")

    def stop(self):
        self._active    = False
        self._frame_idx = 0

    @property
    def is_active(self) -> bool:
        return self._active

    # ------------------------------------------------------------------ render

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """
        Blend the flash colour onto `frame`.
        Returns the frame unmodified if the flash is not active.
        """
        if not self._active:
            return frame

        t       = self._frame_idx / self._duration
        half    = 0.5

        # Ramp up to peak in first half, ramp down in second half
        if t < half:
            alpha = self._peak * (t / half)
        else:
            alpha = self._peak * (1.0 - (t - half) / half)

        # Cosine smoothing
        alpha = alpha * (1.0 - np.cos(alpha * np.pi)) / 2.0 + alpha / 2.0
        alpha = float(np.clip(alpha, 0.0, 1.0))

        flash_layer = np.full_like(frame, self._color, dtype=np.uint8)
        output      = cv2.addWeighted(frame, 1.0 - alpha, flash_layer, alpha, 0)

        self._frame_idx += 1
        if self._frame_idx >= self._duration:
            self._active = False
            log.debug("Flash complete.")

        return output


# ─── Secondary: domain-enter burst (instantaneous bright frame) ───────────────

class BurstFlash(FlashEffect):
    """
    A short 3-frame burst used as the very first frame of a domain activation
    before the regular flash takes over.
    """
    def __init__(self, color: tuple = (255, 255, 255)):
        super().__init__()
        self._duration = 3
        self._peak     = 1.0
        self._color    = color
