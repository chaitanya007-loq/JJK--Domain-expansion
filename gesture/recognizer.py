"""
gesture/recognizer.py — JJK-accurate gesture detection.

Flawless Single-Hand and Crossed-Finger Distinction:
  - Gojo: Triggered if any hand has:
    1. Only the INDEX finger pointing up (Middle, Ring, Pinky curled).
    2. OR both INDEX and MIDDLE fingers extended but CROSSED/TOUCHING (distance < 0.045).
  - Sukuna: Triggered if any hand has:
    - Both INDEX and MIDDLE fingers extended and SEPARATED (distance >= 0.045).
    - This allows Sukuna's domain to trigger reliably even if MediaPipe only detects
      one of the two hands due to occlusion/touching.
"""

import math
from typing import List, Optional, Tuple

from gesture.detector import HandLandmarks, LM
from utils.logger import get_logger

log = get_logger(__name__)


# ─── Geometry helpers ─────────────────────────────────────────────────────────

def _dist(a: tuple, b: tuple) -> float:
    """2-D Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


# Map finger tip to its corresponding MCP (base) joint
_BASE_MCP = {
    LM.INDEX_TIP:  LM.INDEX_MCP,
    LM.MIDDLE_TIP: LM.MIDDLE_MCP,
    LM.RING_TIP:   LM.RING_MCP,
    LM.PINKY_TIP:  LM.PINKY_MCP,
}


def _is_extended(lm: list, tip: int, pip: int) -> bool:
    """True if the finger is extended outward from its base (MCP)."""
    mcp = _BASE_MCP.get(tip, LM.WRIST)
    return _dist(lm[tip], lm[mcp]) > _dist(lm[pip], lm[mcp])


def _is_curled(lm: list, tip: int, pip: int) -> bool:
    """True if the finger is curled back towards its base (MCP)."""
    mcp = _BASE_MCP.get(tip, LM.WRIST)
    return _dist(lm[tip], lm[mcp]) <= _dist(lm[pip], lm[mcp])


# ─── Per-gesture hand checks ──────────────────────────────────────────────────

def _gojo_hand(lm: list) -> bool:
    """
    Gojo "Infinity" pose checks (supports crossed fingers or index-only):
    """
    index_up      = _is_extended(lm, LM.INDEX_TIP,  LM.INDEX_PIP)
    middle_up     = _is_extended(lm, LM.MIDDLE_TIP, LM.MIDDLE_PIP)
    ring_curled   = _is_curled  (lm, LM.RING_TIP,   LM.RING_PIP)
    pinky_curled  = _is_curled  (lm, LM.PINKY_TIP,  LM.PINKY_PIP)

    # Base curl requirements for ring and pinky
    if not (ring_curled and pinky_curled):
        return False

    # Case 1: Classic index-only extended (middle curled)
    if index_up and _is_curled(lm, LM.MIDDLE_TIP, LM.MIDDLE_PIP):
        return True

    # Case 2: Index and middle both extended, but crossed/touching (Gojo crossed fingers)
    if index_up and middle_up:
        # Distance between tips is extremely small when crossed
        return _dist(lm[LM.INDEX_TIP], lm[LM.MIDDLE_TIP]) < 0.045

    return False


def _sukuna_hand(lm: list) -> bool:
    """
    Sukuna "Malevolent Shrine" pose check:
    - Index and Middle extended and separated (side-by-side).
    - Ring and Pinky curled.
    """
    index_up      = _is_extended(lm, LM.INDEX_TIP,  LM.INDEX_PIP)
    middle_up     = _is_extended(lm, LM.MIDDLE_TIP, LM.MIDDLE_PIP)
    ring_curled   = _is_curled  (lm, LM.RING_TIP,   LM.RING_PIP)
    pinky_curled  = _is_curled  (lm, LM.PINKY_TIP,  LM.PINKY_PIP)

    if index_up and middle_up and ring_curled and pinky_curled:
        # Fingers must be side-by-side and separated, not crossed/touching
        return _dist(lm[LM.INDEX_TIP], lm[LM.MIDDLE_TIP]) >= 0.045

    return False


# ─── Multi-hand gesture checks ────────────────────────────────────────────────

def _is_sukuna(hands: List[HandLandmarks]) -> bool:
    """Trigger Sukuna if at least one hand is detected showing the side-by-side pose."""
    return any(_sukuna_hand(h.landmarks) for h in hands)


def _is_gojo(hands: List[HandLandmarks]) -> bool:
    """
    Trigger Gojo if:
    - Any hand matches the Gojo pose (crossed fingers or index-only).
    - No hand matches the Sukuna pose.
    """
    if _is_sukuna(hands):
        return False
    return any(_gojo_hand(h.landmarks) for h in hands)


# ─── Recognizer ───────────────────────────────────────────────────────────────

class GestureRecognizer:
    """
    Converts hand landmarks into domain activation commands.
    """

    REQUIRED_FRAMES: int = 7

    def __init__(self):
        self._streak: dict[str, int] = {"gojo": 0, "sukuna": 0}

    def recognize(self, hands: List[HandLandmarks]) -> Optional[str]:
        if not hands:
            self._reset_all()
            return None

        gojo_match   = _is_gojo(hands)
        sukuna_match = _is_sukuna(hands)

        if sukuna_match:
            self._streak["sukuna"] += 1
            self._streak["gojo"]    = 0
            if self._streak["sukuna"] >= 4:  # Trigger Sukuna fast
                log.info("Gesture confirmed: sukuna")
                self._reset_all()
                return "sukuna"
        elif gojo_match:
            self._streak["gojo"]   += 1
            self._streak["sukuna"]  = 0
            if self._streak["gojo"] >= self.REQUIRED_FRAMES:
                log.info("Gesture confirmed: gojo")
                self._reset_all()
                return "gojo"
        else:
            self._reset_all()
            return None

        return None

    def progress(self) -> Tuple[Optional[str], float]:
        best = max(self._streak, key=self._streak.get)
        val  = self._streak[best]
        if val == 0:
            return None, 0.0
        return best, min(1.0, val / self.REQUIRED_FRAMES)

    def debug_info(self, hands: List[HandLandmarks]) -> dict:
        if not hands:
            return {"hands": 0}
        info = {
            "hands":         len(hands),
            "gojo_streak":   self._streak["gojo"],
            "sukuna_streak": self._streak["sukuna"],
        }
        for i, h in enumerate(hands[:2]):
            lm = h.landmarks
            info[f"h{i}_gojo"]   = _gojo_hand(lm)
            info[f"h{i}_sukuna"] = _sukuna_hand(lm)
            if len(lm) > LM.MIDDLE_TIP:
                info[f"h{i}_dist"] = _dist(lm[LM.INDEX_TIP], lm[LM.MIDDLE_TIP])
        return info

    def _reset_all(self):
        self._streak = {"gojo": 0, "sukuna": 0}
