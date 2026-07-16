"""
gesture/detector.py — MediaPipe HandLandmarker wrapper (Tasks API).

Compatible with mediapipe >= 0.10.30 (Python 3.13).
Uses the hand_landmarker.task model in models/ directory.

Provides normalised landmark positions and per-hand handedness labels.
"""

import os
import cv2
import numpy as np
# pyrefly: ignore [missing-import]
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from dataclasses import dataclass
from typing import List, Optional

from utils.logger import get_logger
import config

log = get_logger(__name__)

# Model path (downloaded once into models/)
_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "hand_landmarker.task")


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class HandLandmarks:
    """Normalised (x, y, z) landmarks for one hand, plus its handedness."""
    landmarks:  list   # list of (x, y, z) tuples, 21 points
    handedness: str    # "Left" or "Right"


# ─── Detector ─────────────────────────────────────────────────────────────────

class HandDetector:
    """
    Wraps MediaPipe HandLandmarker (Tasks API) for multi-hand landmark detection.

    Usage
    -----
    detector = HandDetector()
    detector.start()
    hands = detector.detect(bgr_frame)   # → list[HandLandmarks]
    detector.stop()
    """

    def __init__(self):
        self._landmarker = None

    # ------------------------------------------------------------------ lifecycle

    def start(self):
        """Load the hand landmark model."""
        if not os.path.isfile(_MODEL_PATH):
            log.error(f"Hand model not found: {_MODEL_PATH}")
            return

        base_options = mp_python.BaseOptions(model_asset_path=_MODEL_PATH)
        options      = mp_vision.HandLandmarkerOptions(
            base_options            = base_options,
            running_mode            = mp_vision.RunningMode.IMAGE,
            num_hands               = config.GESTURE_MAX_HANDS,
            min_hand_detection_confidence = config.GESTURE_DETECTION_CONFIDENCE,
            min_hand_presence_confidence  = config.GESTURE_DETECTION_CONFIDENCE,
            min_tracking_confidence       = config.GESTURE_TRACKING_CONFIDENCE,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
        log.info(f"MediaPipe HandLandmarker loaded: {_MODEL_PATH}")

    def stop(self):
        """Release model resources."""
        if self._landmarker:
            self._landmarker.close()
            self._landmarker = None
            log.info("MediaPipe HandLandmarker closed.")

    # ------------------------------------------------------------------ detection

    def detect(self, bgr_frame: np.ndarray) -> List[HandLandmarks]:
        """
        Detect hands in `bgr_frame`.

        Returns
        -------
        List of HandLandmarks (one per detected hand, up to num_hands).
        """
        if self._landmarker is None:
            return []

        rgb      = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = self._landmarker.detect(mp_image)

        if not result.hand_landmarks:
            return []

        detected = []
        for hand_lm, handedness_list in zip(result.hand_landmarks, result.handedness):
            pts  = [(lm.x, lm.y, lm.z) for lm in hand_lm]
            side = handedness_list[0].display_name   # "Left" or "Right"
            detected.append(HandLandmarks(landmarks=pts, handedness=side))

        return detected

    # ------------------------------------------------------------------ drawing

    def draw_landmarks(self, bgr_frame: np.ndarray, _hands: List[HandLandmarks]) -> np.ndarray:
        """
        Draw skeleton landmarks onto the frame (for debug mode).
        Re-detects from the frame so we have the raw mp result for drawing.
        """
        if self._landmarker is None:
            return bgr_frame

        rgb      = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = self._landmarker.detect(mp_image)

        if not result.hand_landmarks:
            return bgr_frame

        h, w = bgr_frame.shape[:2]
        for hand_lm in result.hand_landmarks:
            # Draw each landmark as a circle
            for lm in hand_lm:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(bgr_frame, (cx, cy), 5, (0, 255, 0), -1)

            # Draw connections from the HAND_CONNECTIONS constant
            connections = mp_vision.HandLandmarksConnections.HAND_CONNECTIONS
            for conn in connections:
                s = conn.start
                e = conn.end
                sx, sy = int(hand_lm[s].x * w), int(hand_lm[s].y * h)
                ex, ey = int(hand_lm[e].x * w), int(hand_lm[e].y * h)
                cv2.line(bgr_frame, (sx, sy), (ex, ey), (0, 200, 0), 2)

        return bgr_frame


# ─── Landmark index constants (MediaPipe 21-point scheme) ─────────────────────
class LM:
    WRIST           = 0
    THUMB_CMC       = 1
    THUMB_MCP       = 2
    THUMB_IP        = 3
    THUMB_TIP       = 4
    INDEX_MCP       = 5
    INDEX_PIP       = 6
    INDEX_DIP       = 7
    INDEX_TIP       = 8
    MIDDLE_MCP      = 9
    MIDDLE_PIP      = 10
    MIDDLE_DIP      = 11
    MIDDLE_TIP      = 12
    RING_MCP        = 13
    RING_PIP        = 14
    RING_DIP        = 15
    RING_TIP        = 16
    PINKY_MCP       = 17
    PINKY_PIP       = 18
    PINKY_DIP       = 19
    PINKY_TIP       = 20
