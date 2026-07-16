"""
effects/aura.py — Cinematic Multi-Layer Aura & Environmental Lighting.

Features:
- Environmental lighting (color matching): Subtly tints the user's body
  to match the domain color palette.
- Multi-layer animated energy glow (bloom) run at 1/4 resolution for high FPS.
- Smooth rim lighting outlining the user's silhouette.
"""

import cv2
import numpy as np
from utils.logger import get_logger
import config

log = get_logger(__name__)


class AuraEffect:
    """
    Applies real-time environmental lighting match + cinematic rim glow.
    """

    def __init__(self, color: tuple = (255, 200, 100)):
        self.color    = color
        self._tick    = 0

        # Buffers
        self._h: int | None      = None
        self._w: int | None      = None
        self._sh: int | None     = None
        self._sw: int | None     = None

        self._glow_buf_small = None
        self._sil_buf_small  = None
        self._rim_color_small = None
        self._rim_color = None
        self._tmp_f32  = None

    def _setup(self, h: int, w: int):
        self._h       = h
        self._w       = w
        self._sh      = h // 4
        self._sw      = w // 4

        self._glow_buf_small = np.zeros((self._sh, self._sw, 3), dtype=np.float32)
        self._sil_buf_small  = np.zeros((self._sh, self._sw, 3), dtype=np.uint8)
        self._rim_color_small = np.full((self._sh, self._sw, 3), self.color, dtype=np.uint8)
        self._rim_color = np.full((h, w, 3), self.color, dtype=np.uint8)
        self._tmp_f32  = np.empty((h, w, 3), dtype=np.float32)

    def apply(
        self,
        frame:       np.ndarray,
        person_mask: np.ndarray,
        intensity:   float = 1.0,
    ) -> np.ndarray:
        """
        Applies environmental color match to the user and a soft cinematic rim glow.
        """
        self._tick += 1
        h, w = frame.shape[:2]

        if self._h != h or self._w != w:
            self._setup(h, w)

        pulse     = 0.7 + 0.3 * abs(np.sin(self._tick * 0.05))
        eff_alpha = float(np.clip(intensity * pulse * config.AURA_INTENSITY, 0.0, 1.0))
        if eff_alpha < 0.01:
            return frame

        output = frame.copy()

        # ── 1. Color Matching / Environmental Lighting (15% theme color tint on user) ──
        # Get feathered mask to apply color matching only on the person
        mask_f = person_mask.astype(np.float32)[:, :, np.newaxis] / 255.0
        tint_strength = 0.15 * eff_alpha

        # Blend user pixels with theme color
        tinted_user = (output.astype(np.float32) * (1.0 - tint_strength) + 
                       self._rim_color.astype(np.float32) * tint_strength)
        np.clip(tinted_user, 0, 255, out=tinted_user)
        
        # Apply only to the user area
        output = (tinted_user * mask_f + output.astype(np.float32) * (1.0 - mask_f)).astype(np.uint8)

        # ── 2. Build low-res coloured silhouette for glow ──────────────────
        person_mask_small = cv2.resize(person_mask, (self._sw, self._sh), interpolation=cv2.INTER_NEAREST)
        person_3ch_small  = cv2.merge([person_mask_small, person_mask_small, person_mask_small])
        np.copyto(self._sil_buf_small, self._rim_color_small)
        cv2.bitwise_and(self._sil_buf_small, person_3ch_small, dst=self._sil_buf_small)

        # ── 3. Multi-layer Gaussian glow (bloom) ──────────────────────────
        self._glow_buf_small[:] = 0.0
        n = config.AURA_LAYERS
        for i in range(1, n + 1):
            ksize = (config.AURA_BLUR_KERNEL * i) // 4
            if ksize % 2 == 0:
                ksize += 1
            if ksize < 1:
                ksize = 1

            blurred = cv2.GaussianBlur(self._sil_buf_small, (ksize, ksize), 0)
            weight  = (n + 1 - i) / n
            np.add(self._glow_buf_small,
                   blurred.astype(np.float32) * weight,
                   out=self._glow_buf_small)

        np.clip(self._glow_buf_small, 0, 255, out=self._glow_buf_small)
        glow_u8_small = self._glow_buf_small.astype(np.uint8)
        glow_u8 = cv2.resize(glow_u8_small, (w, h), interpolation=cv2.INTER_LINEAR)

        # Restrict glow to the area OUTSIDE the silhouette
        inv_mask = cv2.bitwise_not(person_mask)
        inv_3ch  = cv2.merge([inv_mask, inv_mask, inv_mask])
        outer    = cv2.bitwise_and(glow_u8, inv_3ch)

        # Screen blend the soft background glow onto the frame
        out_f  = output.astype(np.float32)
        out_f  = 255.0 - ((255.0 - out_f) * (255.0 - outer.astype(np.float32)) / 255.0)
        output = np.clip(out_f, 0, 255).astype(np.uint8)

        # ── 4. Cinematic Rim Light (glow edge overlay) ──────────────────────
        edges    = cv2.Canny(person_mask, 50, 150)
        rim_mask = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        rim_a    = rim_mask.astype(np.float32)[:, :, np.newaxis] / 255.0 * eff_alpha * 0.8

        np.multiply(output.astype(np.float32), 1.0 - rim_a, out=self._tmp_f32)
        np.add(self._tmp_f32,
               self._rim_color.astype(np.float32) * rim_a,
               out=self._tmp_f32)
        np.clip(self._tmp_f32, 0, 255, out=self._tmp_f32)
        output = self._tmp_f32.astype(np.uint8)

        return output

    def reset(self):
        self._tick = 0
