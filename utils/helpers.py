"""
utils/helpers.py — General-purpose frame manipulation helpers.
"""

import cv2
import numpy as np


# ─── Frame utilities ──────────────────────────────────────────────────────────

def resize_to_fit(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize frame to (width, height), preserving aspect ratio with letterboxing."""
    h, w = frame.shape[:2]
    scale = min(width / w, height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    y_off  = (height - new_h) // 2
    x_off  = (width  - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def alpha_blend(
    background: np.ndarray,
    overlay:    np.ndarray,
    alpha:      float,
) -> np.ndarray:
    """
    Simple alpha blend: output = overlay * alpha + background * (1 - alpha).
    Both arrays must be the same shape (H, W, 3).
    """
    return cv2.addWeighted(background, 1.0 - alpha, overlay, alpha, 0)


# ─── Vignette cache ───────────────────────────────────────────────────────────
# The vignette mask depends only on (h, w, strength) and never changes between
# frames.  We cache it as a uint8 3-channel multiplier so the per-frame cost
# is a single cv2.multiply call (SIMD-optimised, ~2 ms vs ~47 ms).

_vig_cache: dict[tuple, np.ndarray] = {}


def add_vignette(frame: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """
    Darken the frame edges (vignette).

    Parameters
    ----------
    strength : 0.0 (no effect) → 1.0 (very dark edges)
    """
    h, w = frame.shape[:2]
    key  = (h, w, round(strength, 2))

    if key not in _vig_cache:
        cx, cy = w / 2, h / 2
        Y, X   = np.ogrid[:h, :w]
        dist   = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        norm   = dist / np.sqrt(cx ** 2 + cy ** 2)
        mask   = (1.0 - np.clip(norm * strength, 0, 1))
        # Store as uint8 scaled 0–255 for fast cv2.multiply
        mask_u8 = (mask * 255).clip(0, 255).astype(np.uint8)
        _vig_cache[key] = cv2.merge([mask_u8, mask_u8, mask_u8])

    # cv2.multiply with scale=1/255 is SIMD-optimised — no float32 copy needed
    return cv2.multiply(frame, _vig_cache[key], scale=1.0 / 255.0, dtype=cv2.CV_8U)


def draw_hud_text(
    frame:     np.ndarray,
    text:      str,
    pos:       tuple,
    color:     tuple  = (255, 255, 255),
    scale:     float  = 0.8,
    thickness: int    = 2,
    shadow:    bool   = True,
) -> np.ndarray:
    """Draw text with an optional dark shadow for legibility on any background."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = pos

    if shadow:
        cv2.putText(frame, text, (x + 2, y + 2), font, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)
    return frame


def draw_cooldown_bar(
    frame:    np.ndarray,
    progress: float,         # 0.0–1.0
    color:    tuple,
    label:    str  = "",
) -> np.ndarray:
    """
    Draw a small cooldown progress bar at the bottom of the frame.

    Parameters
    ----------
    progress : 0.0 = empty, 1.0 = full / ready
    """
    h, w   = frame.shape[:2]
    bar_h  = 6
    bar_w  = int(w * 0.4)
    x_off  = (w - bar_w) // 2
    y_off  = h - 30

    # Background track
    cv2.rectangle(frame, (x_off, y_off), (x_off + bar_w, y_off + bar_h),
                  (40, 40, 40), -1)
    # Fill
    fill_w = int(bar_w * min(1.0, max(0.0, progress)))
    if fill_w > 0:
        cv2.rectangle(frame, (x_off, y_off), (x_off + fill_w, y_off + bar_h),
                      color, -1)
    # Border
    cv2.rectangle(frame, (x_off, y_off), (x_off + bar_w, y_off + bar_h),
                  (120, 120, 120), 1)

    if label:
        draw_hud_text(frame, label, (x_off, y_off - 8), color, scale=0.45, thickness=1)

    return frame


# ─── Color utilities ──────────────────────────────────────────────────────────

def bgr_to_normalized(bgr: tuple) -> np.ndarray:
    """Convert a (B,G,R) integer tuple to a float32 array in [0,1]."""
    return np.array(bgr, dtype=np.float32) / 255.0


def lerp_color(a: tuple, b: tuple, t: float) -> tuple:
    """Linearly interpolate between two BGR colours."""
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))
