"""
effects/shake.py — Camera shake effect applied on domain expansion entry.

Uses exponentially-decaying random pixel offsets to simulate an earthquake-
like impact, then smoothly returns to center.
"""

import numpy as np
import cv2
import config
from utils.logger import get_logger

log = get_logger(__name__)


class ShakeEffect:
    """
    Applies random translational offsets to the output frame to simulate
    a camera shake.

    Usage
    -----
    shake = ShakeEffect()
    shake.trigger()              # call once to start shaking

    # In your render loop:
    frame = shake.apply(frame)   # call every frame
    """

    def __init__(self):
        self._active    = False
        self._frame_idx = 0
        self._duration  = config.SHAKE_DURATION_FRAMES
        self._intensity = config.SHAKE_INTENSITY
        self._rng       = np.random.default_rng()

    # ------------------------------------------------------------------ control

    def trigger(self, intensity: float | None = None):
        """Start the shake effect."""
        self._frame_idx = 0
        self._active    = True
        if intensity is not None:
            self._intensity = intensity
        log.debug("Shake triggered.")

    def stop(self):
        self._active    = False
        self._frame_idx = 0

    @property
    def is_active(self) -> bool:
        return self._active

    # ------------------------------------------------------------------ render

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """
        Offset `frame` by a decaying random amount and return the result.
        Black borders are filled in at the edges.
        """
        if not self._active:
            return frame

        # Exponential decay: shake starts strong and dies out
        progress = self._frame_idx / self._duration
        decay    = np.exp(-4.0 * progress)    # 1.0 → ~0.02 over duration

        max_offset = int(self._intensity * decay)
        if max_offset < 1:
            self._active = False
            return frame

        dx = int(self._rng.integers(-max_offset, max_offset + 1))
        dy = int(self._rng.integers(-max_offset, max_offset + 1))

        h, w   = frame.shape[:2]
        M      = np.float32([[1, 0, dx], [0, 1, dy]])
        output = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        self._frame_idx += 1
        if self._frame_idx >= self._duration:
            self._active = False
            log.debug("Shake complete.")

        return output
