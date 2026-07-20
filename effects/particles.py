import cv2
import numpy as np
import numba
from numba import njit, prange

import config
from utils.logger import get_logger

log = get_logger(__name__)

@njit(cache=True)
def _update_particles_kernel(
    x, y, vx, vy, life, max_life, alive,
    gravity: float, drag: float,
    w: int, h: int,
):
    n = x.shape[0]
    alpha = np.zeros(n, dtype=np.float64)
    alive_count = 0

    for i in range(n):
        if alive[i] == 0:
            continue

        x[i]  += vx[i]
        y[i]  += vy[i]
        vy[i] -= gravity
        vx[i] *= drag
        life[i] += 1

        if life[i] >= max_life[i] or x[i] < 0 or x[i] >= w or y[i] < 0 or y[i] >= h:
            alive[i] = 0
            continue

        alive_count += 1
        t = life[i] / max_life[i]
        alpha[i] = np.sin(t * np.pi)

    return alpha, alive_count

@njit(cache=True, parallel=True)
def _render_particles_kernel(
    out,
    x, y, radius, alive, alpha,
    color_b: int, color_g: int, color_r: int,
    intensity: float,
):
    h = out.shape[0]
    w = out.shape[1]
    n = x.shape[0]

    for idx in prange(n):
        if alive[idx] == 0 or alpha[idx] < 0.05:
            continue

        a  = alpha[idx] * intensity
        cx = int(x[idx])
        cy = int(y[idx])
        r  = int(radius[idx])

        x1 = max(0, cx - r)
        x2 = min(w - 1, cx + r)
        y1 = max(0, cy - r)
        y2 = min(h - 1, cy + r)
        r2 = r * r

        for py in range(y1, y2 + 1):
            for px in range(x1, x2 + 1):
                dx = px - cx
                dy = py - cy
                if dx * dx + dy * dy <= r2:
                    ob = out[py, px, 0]
                    og = out[py, px, 1]
                    orr = out[py, px, 2]
                    nb = int(ob + color_b * a)
                    ng = int(og + color_g * a)
                    nr = int(orr + color_r * a)
                    out[py, px, 0] = min(255, nb) if nb > ob else ob
                    out[py, px, 1] = min(255, ng) if ng > og else og
                    out[py, px, 2] = min(255, nr) if nr > orr else orr

class ParticleSystem:
    """Manages particle pools using preallocated numpy arrays for speed."""
    def __init__(self, color: tuple = (200, 200, 255)):
        self.color = color
        n = config.PARTICLE_COUNT

        self._x        = np.zeros(n, dtype=np.float64)
        self._y        = np.zeros(n, dtype=np.float64)
        self._vx       = np.zeros(n, dtype=np.float64)
        self._vy       = np.zeros(n, dtype=np.float64)
        self._radius   = np.full(n, 3.0, dtype=np.float64)
        self._life     = np.zeros(n, dtype=np.float64)
        self._max_life = np.full(n, float(config.PARTICLE_LIFETIME), dtype=np.float64)
        self._alive    = np.zeros(n, dtype=np.int32)

        self._rng      = np.random.default_rng()
        self._capacity = n

    def update_and_draw(
        self,
        frame:       np.ndarray,
        person_mask: np.ndarray,
        intensity:   float = 1.0,
    ) -> np.ndarray:
        h, w = frame.shape[:2]

        edge_px = self._get_edge_pixels(person_mask)
        target  = int(self._capacity * max(0.0, min(1.0, intensity)))

        alive_count = int(np.sum(self._alive))
        need = target - alive_count
        if need > 0 and len(edge_px) > 0:
            self._spawn_batch(edge_px, min(need, len(edge_px)), h, w)

        alpha, _ = _update_particles_kernel(
            self._x, self._y, self._vx, self._vy,
            self._life, self._max_life, self._alive,
            gravity=0.04, drag=0.99,
            w=w, h=h,
        )

        output = frame.copy()
        _render_particles_kernel(
            output,
            self._x, self._y, self._radius, self._alive, alpha,
            color_b=self.color[0], color_g=self.color[1], color_r=self.color[2],
            intensity=float(intensity),
        )

        return output

    def _get_edge_pixels(self, mask: np.ndarray) -> np.ndarray:
        edges = cv2.Canny(mask, 30, 100)
        ys, xs = np.where(edges > 0)
        if len(xs) == 0:
            return np.empty((0, 2), dtype=np.int64)
        return np.column_stack([xs, ys])

    def _spawn_batch(self, edge_px: np.ndarray, count: int, h: int, w: int):
        rng  = self._rng
        dead = np.where(self._alive == 0)[0]
        if len(dead) == 0:
            return
        slots = dead[:count]
        n     = len(slots)

        idx = rng.integers(0, len(edge_px), size=n)
        self._x[slots] = edge_px[idx, 0].astype(np.float64)
        self._y[slots] = edge_px[idx, 1].astype(np.float64)

        speed  = rng.uniform(config.PARTICLE_SPEED_MIN, config.PARTICLE_SPEED_MAX, size=n)
        angle  = rng.uniform(0, 2.0 * np.pi, size=n)
        self._vx[slots] = np.cos(angle) * speed
        self._vy[slots] = np.sin(angle) * speed - 1.0

        self._radius[slots]   = rng.integers(
            config.PARTICLE_RADIUS_MIN, config.PARTICLE_RADIUS_MAX + 1, size=n
        ).astype(np.float64)
        self._life[slots]     = rng.integers(
            0, config.PARTICLE_LIFETIME // 3, size=n
        ).astype(np.float64)
        self._max_life[slots] = float(config.PARTICLE_LIFETIME)
        self._alive[slots]    = 1

    def clear(self):
        self._alive[:] = 0

    @classmethod
    def warm_up(cls):
        log.info("Warming up Numba particle kernels...")
        n = 8
        x  = np.zeros(n, dtype=np.float64)
        y  = np.zeros(n, dtype=np.float64)
        vx = np.zeros(n, dtype=np.float64)
        vy = np.zeros(n, dtype=np.float64)
        life     = np.zeros(n, dtype=np.float64)
        max_life = np.full(n, 10.0, dtype=np.float64)
        alive    = np.ones(n, dtype=np.int32)
        radius   = np.full(n, 2.0, dtype=np.float64)

        _update_particles_kernel(x, y, vx, vy, life, max_life, alive,
                                 0.04, 0.99, 64, 64)

        dummy_frame = np.zeros((64, 64, 3), dtype=np.uint8)
        alpha = np.ones(n, dtype=np.float64) * 0.5
        _render_particles_kernel(dummy_frame, x, y, radius, alive, alpha,
                                 200, 200, 255, 1.0)
        log.info("Numba particle kernels compiled OK")
