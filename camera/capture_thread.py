"""
camera/capture_thread.py — Dedicated webcam capture thread.

Reads frames from the webcam continuously in the background so the
main render loop never blocks on I/O or camera latency.
"""

import threading
import numpy as np
from utils.logger import get_logger

log = get_logger(__name__)


class CaptureThread(threading.Thread):
    """
    Runs in the background, continuously reading from the webcam and storing
    the most recent frame. The main loop calls get_latest_frame() which
    returns immediately — no blocking, no dropped render frames.
    """

    def __init__(self, webcam):
        super().__init__(daemon=True, name="CaptureThread")
        self._cam          = webcam
        self._latest_frame: np.ndarray | None = None
        self._lock         = threading.Lock()
        self._running      = False

    # ------------------------------------------------------------------ control

    def start_capture(self):
        """Start reading webcam frames in the background."""
        self._running = True
        self.start()
        log.info("Capture thread started.")

    def stop(self):
        """Signal the thread to exit after the current frame read."""
        self._running = False

    # ------------------------------------------------------------------ access

    def get_latest_frame(self) -> np.ndarray | None:
        """
        Return a copy of the most recently captured frame, or None if no
        frame has been captured yet.
        """
        with self._lock:
            if self._latest_frame is not None:
                return self._latest_frame.copy()
            return None

    # ------------------------------------------------------------------ thread body

    def run(self):
        while self._running:
            frame = self._cam.get_frame()
            if frame is not None:
                with self._lock:
                    self._latest_frame = frame   # always hold latest
        log.info("Capture thread stopped.")
