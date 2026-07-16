"""
camera/webcam.py — Webcam capture wrapper.

Handles opening/closing the camera, enforcing resolution & FPS settings,
and providing a clean frame-reading interface to the rest of the app.
"""

import cv2
from utils.logger import get_logger
import config

log = get_logger(__name__)


class Webcam:
    """Thin wrapper around cv2.VideoCapture."""

    def __init__(self):
        self._cap = None
        self.width  = config.CAMERA_WIDTH
        self.height = config.CAMERA_HEIGHT

    # ------------------------------------------------------------------ open/close

    def open(self) -> bool:
        """Open the webcam device. Returns True on success."""
        log.info(f"Opening camera index {config.CAMERA_INDEX} ...")
        self._cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_DSHOW)

        if not self._cap.isOpened():
            log.error("Failed to open webcam!")
            return False

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS_TARGET)

        # Read back actual values (driver may cap them)
        self.width  = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_actual  = self._cap.get(cv2.CAP_PROP_FPS)
        log.info(f"Camera opened: {self.width}×{self.height} @ {fps_actual:.0f} FPS")
        return True

    def release(self):
        """Release the webcam device."""
        if self._cap and self._cap.isOpened():
            self._cap.release()
            log.info("Camera released.")

    # ------------------------------------------------------------------ reading

    def get_frame(self):
        """
        Read one frame from the webcam.

        Returns
        -------
        frame : np.ndarray | None
            BGR frame, or None if read failed.
        """
        if not self._cap or not self._cap.isOpened():
            return None

        ret, frame = self._cap.read()
        if not ret:
            log.warning("Failed to read frame from camera.")
            return None

        # Mirror so the user's left hand appears on the left of the screen
        return cv2.flip(frame, 1)

    # ------------------------------------------------------------------ properties

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()
