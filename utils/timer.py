"""
utils/timer.py — Frame-based and wall-clock timer utilities.
"""

import time
from utils.logger import get_logger

log = get_logger(__name__)


class Cooldown:
    """
    Simple wall-clock cooldown.

    Usage
    -----
    cd = Cooldown(seconds=8)
    if cd.ready():
        cd.reset()
        # do the thing
    """

    def __init__(self, seconds: float):
        self._duration = seconds
        self._last     = 0.0    # epoch time of last reset

    def ready(self) -> bool:
        """Returns True if the cooldown has elapsed."""
        return (time.monotonic() - self._last) >= self._duration

    def reset(self):
        """Restart the cooldown from now."""
        self._last = time.monotonic()

    def remaining(self) -> float:
        """Seconds left until ready (0.0 if already ready)."""
        elapsed = time.monotonic() - self._last
        return max(0.0, self._duration - elapsed)

    def progress(self) -> float:
        """0.0 → 1.0 progress through the cooldown period."""
        elapsed = time.monotonic() - self._last
        return min(1.0, elapsed / self._duration)


class CountdownTimer:
    """
    Counts down from `seconds` to zero.

    Usage
    -----
    t = CountdownTimer(15)
    t.start()
    while not t.expired():
        ...
    """

    def __init__(self, seconds: float):
        self._duration  = seconds
        self._start     = None

    def start(self):
        self._start = time.monotonic()

    def expired(self) -> bool:
        if self._start is None:
            return True
        return (time.monotonic() - self._start) >= self._duration

    def remaining(self) -> float:
        if self._start is None:
            return 0.0
        return max(0.0, self._duration - (time.monotonic() - self._start))

    def progress(self) -> float:
        """0.0 (just started) → 1.0 (expired)."""
        if self._start is None:
            return 1.0
        return min(1.0, (time.monotonic() - self._start) / self._duration)

    def reset(self):
        self._start = None


class FPSCounter:
    """Calculates rolling-average FPS, instantaneous FPS, and frame processing times."""

    def __init__(self, window: int = 30):
        self._times = []
        self._window = window
        self._last_tick = None
        self.frame_time_ms = 0.0

    def tick(self):
        now = time.monotonic()
        if self._last_tick is not None:
            self.frame_time_ms = (now - self._last_tick) * 1000.0
        self._last_tick = now
        
        self._times.append(now)
        if len(self._times) > self._window:
            self._times.pop(0)

    @property
    def fps(self) -> float:
        """Rolling average FPS."""
        if len(self._times) < 2:
            return 0.0
        span = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / span if span > 0 else 0.0

    @property
    def current_fps(self) -> float:
        """Instantaneous FPS based on the last frame duration."""
        if self.frame_time_ms > 0:
            return 1000.0 / self.frame_time_ms
        return 0.0


class Profiler:
    """Lightweight code block performance profiler."""

    def __init__(self):
        self._starts = {}
        self.durations = {}

    def start(self, name: str):
        self._starts[name] = time.perf_counter()

    def stop(self, name: str):
        if name in self._starts:
            self.durations[name] = (time.perf_counter() - self._starts[name]) * 1000.0

