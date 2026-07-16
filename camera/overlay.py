"""
camera/overlay.py — Domain video preloader + person compositor.

DomainVideoPlayer
-----------------
  Preloads domain video frames into RAM in a background thread the moment
  the object is created. next_frame() never blocks on disk I/O.

  Storage: frames at INFERENCE_WIDTH × INFERENCE_HEIGHT (640×360) with
  tint applied once at load time. next_frame(w,h) does a fast bilinear
  resize to display resolution.

Overlay
-------
  Composites the user silhouette over the domain background using a
  pre-allocated output buffer to avoid per-frame memory allocation.
"""

import os
import threading
import cv2
import numpy as np

import config
from utils.logger import get_logger

log = get_logger(__name__)

# Resolution at which frames are stored (full webcam display resolution to avoid real-time resizing)
_PRE_W = config.CAMERA_WIDTH
_PRE_H = config.CAMERA_HEIGHT


class DomainVideoPlayer:
    """
    Loads all domain video frames into memory at construction time
    (background thread). Provides instant, non-blocking next_frame().
    """

    MAX_FRAMES: int = config.VIDEO_PRELOAD_MAX_FRAMES

    def __init__(self, video_path: str, tint_color: tuple = (255, 255, 255)):
        self._path   = video_path
        self._tint   = np.array(tint_color, dtype=np.float32) / 255.0

        self._frames: list[np.ndarray] = []   # preloaded, tinted BGR frames
        self._lock   = threading.Lock()
        self._idx    = 0
        self._ready  = False         # True once preloading is complete
        self._tick   = 0             # for fallback animation

        # Start loading immediately so frames are ready before first activation
        t = threading.Thread(target=self._preload_worker,
                             daemon=True, name="VideoPreloader")
        t.start()

    # ------------------------------------------------------------------ preloading

    def _preload_worker(self):
        """Runs once in background: decode → resize → tint → store."""
        if not os.path.isfile(self._path):
            log.warning(f"Domain video not found: {self._path}  --> animated fallback.")
            return

        cap = cv2.VideoCapture(self._path)
        if not cap.isOpened():
            log.warning(f"Could not open domain video: {self._path}")
            return

        frames = []
        while len(frames) < self.MAX_FRAMES:
            ret, frame = cap.read()
            if not ret:
                break
            small  = cv2.resize(frame, (_PRE_W, _PRE_H),
                                 interpolation=cv2.INTER_LINEAR)
            # Keep original colors (no tinting overlay)
            frames.append(small)

        cap.release()

        with self._lock:
            self._frames = frames
            self._ready  = bool(frames)

        log.info(f"Preloaded {len(frames)} frames  ← {os.path.basename(self._path)}")

    # ------------------------------------------------------------------ lifecycle

    def open(self):
        """Called at domain activation — rewind to first frame."""
        with self._lock:
            self._idx = 0

    def release(self):
        """Called at domain deactivation — keep frames in RAM for reuse."""
        pass   # intentional no-op

    # ------------------------------------------------------------------ frame access

    def next_frame(self, width: int, height: int) -> np.ndarray:
        """
        Return next preloaded frame resized to (width, height).
        Falls back to an animated gradient if not yet loaded.
        """
        with self._lock:
            if self._ready and self._frames:
                frame     = self._frames[self._idx % len(self._frames)]
                self._idx += 1
                # Fast bilinear resize from preload res → display res
                if width != _PRE_W or height != _PRE_H:
                    return cv2.resize(frame, (width, height),
                                      interpolation=cv2.INTER_LINEAR)
                return frame.copy()

        return self._generate_fallback(width, height)

    # ------------------------------------------------------------------ fallback

    def _generate_fallback(self, width: int, height: int) -> np.ndarray:
        """Animated gradient while preloading or when video is absent."""
        self._tick = (self._tick + 1) % 360
        b, g, r   = (int(c * 255) for c in self._tint)

        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(height):
            ratio      = (i / height + self._tick / 360.0) % 1.0
            canvas[i] = [int(b * ratio * 0.6),
                         int(g * ratio * 0.3),
                         int(r * ratio * 0.8)]

        cx, cy = width // 2, height // 2
        Y, X   = np.ogrid[:height, :width]
        dist   = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        vig    = (1.0 - np.clip(dist / (max(cx, cy) * 1.1), 0, 1))[:, :, np.newaxis]
        return (canvas.astype(np.float32) * vig).clip(0, 255).astype(np.uint8)


# ────────────────────────────────────────────────────────────────────────────────

class Overlay:
    """
    Composites the user silhouette over a domain background frame.
    Uses a class-level pre-allocated output buffer to avoid per-frame malloc.
    """

    _out_buf:   np.ndarray | None = None
    _buf_shape: tuple             = ()

    @classmethod
    def composite(
        cls,
        user_frame:   np.ndarray,
        person_mask:  np.ndarray,
        domain_frame: np.ndarray,
    ) -> np.ndarray:
        """
        Alpha-blend the user over the domain background using person_mask.

        Parameters
        ----------
        user_frame   : BGR webcam frame
        person_mask  : uint8 H×W (255=person, 0=background)
        domain_frame : BGR domain background, any size (resized internally)

        Returns
        -------
        BGR composited frame (same size as user_frame)
        """
        h, w = user_frame.shape[:2]

        # Ensure domain_frame matches display size
        if domain_frame.shape[:2] != (h, w):
            domain_frame = cv2.resize(domain_frame, (w, h),
                                      interpolation=cv2.INTER_LINEAR)

        # Pre-allocate output buffer (only once, or when size changes)
        if cls._buf_shape != (h, w):
            cls._out_buf  = np.empty((h, w, 3), dtype=np.float32)
            cls._buf_shape = (h, w)

        # Use soft contrast mask thresholding (contrasts feathered edge, removes halo)
        alpha = np.clip((person_mask.astype(np.float32) - 100.0) / 155.0, 0.0, 1.0)
        alpha_f = alpha[:, :, np.newaxis]

        # In-place blend into pre-allocated buffer
        user_f   = user_frame.astype(np.float32)
        domain_f = domain_frame.astype(np.float32)
        np.multiply(user_f,   alpha_f,       out=cls._out_buf)
        np.add(cls._out_buf, domain_f * (1.0 - alpha_f), out=cls._out_buf)
        np.clip(cls._out_buf, 0, 255, out=cls._out_buf)

        return cls._out_buf.astype(np.uint8)
