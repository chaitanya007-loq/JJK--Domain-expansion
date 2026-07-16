"""
effects/particles.py — Energy particle system that orbits the user silhouette.

Each particle is born near the silhouette edge, moves outward with some
random velocity, fades over its lifetime, and is recycled when it dies.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List

import config
from utils.logger import get_logger

log = get_logger(__name__)


# ─── Particle data ────────────────────────────────────────────────────────────

@dataclass
class Particle:
    x:        float = 0.0
    y:        float = 0.0
    vx:       float = 0.0
    vy:       float = 0.0
    radius:   int   = 3
    life:     int   = 0      # current age (frames)
    max_life: int   = config.PARTICLE_LIFETIME
    color:    tuple = (255, 255, 255)   # BGR


# ─── System ───────────────────────────────────────────────────────────────────

class ParticleSystem:
    """
    Maintains a pool of `config.PARTICLE_COUNT` particles that are spawned
    along the person silhouette edge and drift outward.
    """

    def __init__(self, color: tuple = (200, 200, 255)):
        self.color     = color
        self._particles: List[Particle] = []
        self._rng       = np.random.default_rng()

    # ------------------------------------------------------------------ update + draw

    def update_and_draw(
        self,
        frame:       np.ndarray,
        person_mask: np.ndarray,
        intensity:   float = 1.0,
    ) -> np.ndarray:
        """
        Spawn new particles along the mask edge, advance existing ones, and
        render them onto `frame`.

        Parameters
        ----------
        frame       : BGR frame to draw on (copied internally)
        person_mask : uint8 person mask
        intensity   : 0.0–1.0 (scales alpha and count)

        Returns
        -------
        frame with particles drawn
        """
        output = frame.copy()
        h, w   = frame.shape[:2]

        # Find silhouette edge pixels as spawn candidates
        edge_px = self._get_edge_pixels(person_mask)

        # Spawn new particles up to the configured count
        target_count = int(config.PARTICLE_COUNT * max(0.0, min(1.0, intensity)))
        while len(self._particles) < target_count and len(edge_px) > 0:
            self._particles.append(self._spawn(edge_px, h, w))

        # Update + draw each particle
        alive = []
        for p in self._particles:
            p.x     += p.vx
            p.y     += p.vy
            p.vy    -= 0.04    # gentle upward drift (gravity reversed)
            p.vx    *= 0.99    # drag
            p.life  += 1

            if p.life >= p.max_life or not (0 <= p.x < w and 0 <= p.y < h):
                continue       # particle dies — will be replaced next frame
            alive.append(p)

            # Alpha fades in then out over lifetime
            t     = p.life / p.max_life
            alpha = np.sin(t * np.pi) * intensity   # 0 → 1 → 0
            if alpha < 0.05:
                continue

            # Draw with local blend (faster, no full frame copy)
            cx, cy, r = int(p.x), int(p.y), p.radius
            # Crop bounds to avoid going outside frame edges
            x1, y1 = max(0, cx - r), max(0, cy - r)
            x2, y2 = min(w, cx + r + 1), min(h, cy + r + 1)
            if x2 > x1 and y2 > y1:
                patch = output[y1:y2, x1:x2].copy()
                cv2.circle(patch, (cx - x1, cy - y1), r, p.color, -1)
                cv2.addWeighted(patch, alpha, output[y1:y2, x1:x2], 1.0 - alpha, 0, dst=output[y1:y2, x1:x2])

            # Small bloom ring (drawn directly on output)
            if p.radius > 2:
                cv2.circle(
                    output,
                    (cx, cy),
                    p.radius + 2,
                    p.color,
                    1,
                )

        self._particles = alive
        return output

    # ------------------------------------------------------------------ helpers

    def _get_edge_pixels(self, mask: np.ndarray) -> np.ndarray:
        """Return (N,2) array of (x,y) edge pixel coordinates."""
        edges = cv2.Canny(mask, 30, 100)
        ys, xs = np.where(edges > 0)
        if len(xs) == 0:
            return np.empty((0, 2), dtype=int)
        return np.column_stack([xs, ys])

    def _spawn(self, edge_px: np.ndarray, h: int, w: int) -> Particle:
        """Create a new particle at a random edge pixel with random velocity."""
        rng    = self._rng
        idx    = rng.integers(0, len(edge_px))
        x, y   = float(edge_px[idx, 0]), float(edge_px[idx, 1])

        speed  = rng.uniform(config.PARTICLE_SPEED_MIN, config.PARTICLE_SPEED_MAX)
        angle  = rng.uniform(0, 2 * np.pi)
        vx     = np.cos(angle) * speed
        vy     = np.sin(angle) * speed - 1.0    # bias upward

        radius = rng.integers(config.PARTICLE_RADIUS_MIN, config.PARTICLE_RADIUS_MAX + 1)
        life   = rng.integers(0, config.PARTICLE_LIFETIME // 3)   # stagger ages

        return Particle(
            x=x, y=y, vx=vx, vy=vy,
            radius=int(radius),
            life=int(life),
            max_life=config.PARTICLE_LIFETIME,
            color=self.color,
        )

    def clear(self):
        """Remove all particles (e.g., when domain ends)."""
        self._particles.clear()
