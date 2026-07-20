"""
camera/inference_thread.py — Background MediaPipe inference thread.

Supports two operating modes:
  "full"     → segmentation + hand detection  (idle / cooldown state)
  "seg_only" → segmentation only, skip hands  (domain active — throttled)

Includes:
  - Temporal mask smoothing (flicker reduction) for clean body edges.
  - Exponential Moving Average (EMA) landmark smoothing for stable gesture tracking.
"""

import threading
import cv2
import numpy as np
from typing import List, Tuple, Optional

from utils.logger import get_logger
from gesture.detector import HandLandmarks
import config

log = get_logger(__name__)


class InferenceThread(threading.Thread):
    """
    Background thread for all MediaPipe AI processing.

    Modes
    -----
    "full"     — Run both segmentation and hand detection every submitted frame.
    "seg_only" — Run segmentation only; return empty hand list.  Used during
                 active domain states to save CPU (Dynamic AI Throttling).
    """

    def __init__(self, bg_remover, hand_detector):
        super().__init__(daemon=True, name="InferenceThread")

        self._bg_remover  = bg_remover
        self._detector    = hand_detector

        # Input buffers
        self._pending_frame: np.ndarray | None = None
        self._pending_w    = config.CAMERA_WIDTH
        self._pending_h    = config.CAMERA_HEIGHT
        self._pending_lock = threading.Lock()
        self._frame_event  = threading.Event()

        # Output buffers
        self._result: Tuple[Optional[np.ndarray], List] = (None, [])
        self._result_lock  = threading.Lock()

        self._running      = False

        # Operating mode
        self._mode      = "full"   # "full" | "seg_only"
        self._mode_lock = threading.Lock()

        # Temporal smoothing caches
        self._prev_mask: np.ndarray | None = None
        self._prev_hands: dict[str, list] = {}  # side -> landmark list

    # ------------------------------------------------------------------ control

    def start_inference(self):
        self._running = True
        self.start()
        log.info(f"Inference thread started  "
                 f"({config.INFERENCE_WIDTH}x{config.INFERENCE_HEIGHT} per frame).")

    def stop(self):
        self._running = False
        self._frame_event.set()

    # ------------------------------------------------------------------ mode

    def set_mode(self, mode: str):
        """
        Switch between inference modes.

        Parameters
        ----------
        mode : "full" or "seg_only"
        """
        with self._mode_lock:
            if self._mode != mode:
                log.info(f"Inference mode → {mode}")
                self._mode = mode

    def get_mode(self) -> str:
        with self._mode_lock:
            return self._mode

    # ------------------------------------------------------------------ API

    def submit_frame(self, frame: np.ndarray, display_w: int, display_h: int):
        with self._pending_lock:
            self._pending_frame = frame
            self._pending_w     = display_w
            self._pending_h     = display_h
        self._frame_event.set()

    def get_result(self) -> Tuple[Optional[np.ndarray], List]:
        with self._result_lock:
            mask, hands = self._result
            return (mask.copy() if mask is not None else None), list(hands)

    # ------------------------------------------------------------------ smoothing

    def _smooth_landmarks(self, current_hands: List[HandLandmarks]) -> List[HandLandmarks]:
        """Apply Exponential Moving Average (EMA) to prevent finger jitter."""
        alpha = 0.40  # weight of new frame coordinates (0.4 = smooth, 1.0 = raw)
        smoothed = []

        # Keep track of active sides in this frame to clear old cached sides
        active_sides = set()

        for hand in current_hands:
            side = hand.handedness
            active_sides.add(side)
            pts = np.array(hand.landmarks)

            if side in self._prev_hands:
                prev_pts = np.array(self._prev_hands[side])
                if prev_pts.shape == pts.shape:
                    pts = alpha * pts + (1.0 - alpha) * prev_pts

            self._prev_hands[side] = pts.tolist()
            smoothed.append(HandLandmarks(landmarks=pts.tolist(), handedness=side))

        # Clean up cached hands for sides that disappeared
        for cached_side in list(self._prev_hands.keys()):
            if cached_side not in active_sides:
                self._prev_hands.pop(cached_side, None)

        return smoothed

    # ------------------------------------------------------------------ thread body

    def run(self):
        INF_W = config.INFERENCE_WIDTH
        INF_H = config.INFERENCE_HEIGHT

        while self._running:
            fired = self._frame_event.wait(timeout=0.05)
            self._frame_event.clear()

            if not fired or not self._running:
                continue

            with self._pending_lock:
                frame   = self._pending_frame
                disp_w  = self._pending_w
                disp_h  = self._pending_h

            if frame is None:
                continue

            # Read current mode
            with self._mode_lock:
                mode = self._mode

            try:
                small = cv2.resize(frame, (INF_W, INF_H),
                                   interpolation=cv2.INTER_LINEAR)

                # 1. Segmentation (always runs)
                mask_small = self._bg_remover.get_mask(small)

                # The mask is now float32 [0,1] — upscale and blur
                mask_full = cv2.resize(mask_small, (disp_w, disp_h),
                                       interpolation=cv2.INTER_LINEAR)
                mask_full = cv2.GaussianBlur(mask_full, (11, 11), 0)

                # Temporal mask smoothing to eliminate edge flicker
                if self._prev_mask is not None and self._prev_mask.shape == mask_full.shape:
                    cv2.addWeighted(mask_full, 0.45, self._prev_mask, 0.55, 0, dst=mask_full)
                self._prev_mask = mask_full.copy()

                # 2. Hand tracking (only in "full" mode)
                if mode == "full":
                    raw_hands = self._detector.detect(small)
                    hands = self._smooth_landmarks(raw_hands)
                else:
                    hands = []

                # Publish
                with self._result_lock:
                    self._result = (mask_full, hands)

            except Exception as exc:
                log.warning(f"Inference error (frame skipped): {exc}")

        log.info("Inference thread stopped.")
