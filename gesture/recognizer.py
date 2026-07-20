import math
from typing import List, Optional, Tuple

from gesture.detector import HandLandmarks, LM
from utils.logger import get_logger

log = get_logger(__name__)

# Map finger tip to its base (MCP) and middle (PIP) joints
_BASE_MCP = {
    LM.INDEX_TIP:  LM.INDEX_MCP,
    LM.MIDDLE_TIP: LM.MIDDLE_MCP,
    LM.RING_TIP:   LM.RING_MCP,
    LM.PINKY_TIP:  LM.PINKY_MCP,
}

_FINGER_PIP = {
    LM.INDEX_TIP:  LM.INDEX_PIP,
    LM.MIDDLE_TIP: LM.MIDDLE_PIP,
    LM.RING_TIP:   LM.RING_PIP,
    LM.PINKY_TIP:  LM.PINKY_PIP,
}

def _dist(a: tuple, b: tuple) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def _hand_size(lm: list) -> float:
    d = _dist(lm[LM.WRIST], lm[LM.MIDDLE_MCP])
    return d if d > 0.001 else 0.001

def _is_extended(lm: list, tip: int) -> bool:
    """True if finger is straight (tip-to-knuckle > middle-joint-to-knuckle)."""
    mcp = _BASE_MCP.get(tip, LM.WRIST)
    pip = _FINGER_PIP.get(tip, LM.WRIST)
    return _dist(lm[tip], lm[mcp]) > _dist(lm[pip], lm[mcp])

def _is_curled(lm: list, tip: int) -> bool:
    """True if finger is bent (tip is closer to knuckle than middle joint is)."""
    mcp = _BASE_MCP.get(tip, LM.WRIST)
    pip = _FINGER_PIP.get(tip, LM.WRIST)
    return _dist(lm[tip], lm[mcp]) <= _dist(lm[pip], lm[mcp])

def _thumb_tucked(lm: list) -> bool:
    thumb_to_index = _dist(lm[LM.THUMB_TIP], lm[LM.INDEX_MCP])
    hs = _hand_size(lm)
    return (thumb_to_index / hs) < 1.2

def _fingers_separated(lm: list) -> bool:
    tip_dist = _dist(lm[LM.INDEX_TIP], lm[LM.MIDDLE_TIP])
    hs = _hand_size(lm)
    return (tip_dist / hs) >= 0.15

def _fingers_crossed(lm: list) -> bool:
    tip_dist = _dist(lm[LM.INDEX_TIP], lm[LM.MIDDLE_TIP])
    hs = _hand_size(lm)
    return (tip_dist / hs) < 0.15

def _gojo_hand(lm: list) -> bool:
    index_up     = _is_extended(lm, LM.INDEX_TIP)
    middle_up    = _is_extended(lm, LM.MIDDLE_TIP)
    ring_curled  = _is_curled(lm, LM.RING_TIP)
    pinky_curled = _is_curled(lm, LM.PINKY_TIP)

    if not (ring_curled and pinky_curled):
        return False

    if index_up and _is_curled(lm, LM.MIDDLE_TIP):
        return True

    if index_up and middle_up and _fingers_crossed(lm):
        return True

    return False

def _sukuna_hand_vsign(lm: list) -> bool:
    index_up     = _is_extended(lm, LM.INDEX_TIP)
    middle_up    = _is_extended(lm, LM.MIDDLE_TIP)
    ring_curled  = _is_curled(lm, LM.RING_TIP)
    pinky_curled = _is_curled(lm, LM.PINKY_TIP)

    if not (index_up and middle_up and ring_curled and pinky_curled):
        return False

    return _fingers_separated(lm)

def _sukuna_hand_fist(lm: list) -> bool:
    index_curled  = _is_curled(lm, LM.INDEX_TIP)
    middle_curled = _is_curled(lm, LM.MIDDLE_TIP)
    ring_curled   = _is_curled(lm, LM.RING_TIP)
    pinky_curled  = _is_curled(lm, LM.PINKY_TIP)

    if not (index_curled and middle_curled and ring_curled and pinky_curled):
        return False

    return _thumb_tucked(lm)

def _sukuna_hand(lm: list) -> bool:
    return _sukuna_hand_vsign(lm) or _sukuna_hand_fist(lm)

def _is_sukuna(hands: List[HandLandmarks]) -> bool:
    return any(_sukuna_hand(h.landmarks) for h in hands)

def _is_gojo(hands: List[HandLandmarks]) -> bool:
    if _is_sukuna(hands):
        return False
    return any(_gojo_hand(h.landmarks) for h in hands)

class GestureRecognizer:
    GOJO_REQUIRED_FRAMES:   int = 8
    SUKUNA_REQUIRED_FRAMES: int = 5
    DRAIN_RATE:             int = 2

    def __init__(self):
        self._streak = {"gojo": 0, "sukuna": 0}

    def recognize(self, hands: List[HandLandmarks]) -> Optional[str]:
        if not hands:
            self._drain_all()
            return None

        gojo_match   = _is_gojo(hands)
        sukuna_match = _is_sukuna(hands)

        if sukuna_match:
            self._streak["sukuna"] += 1
            self._streak["gojo"]    = max(0, self._streak["gojo"] - self.DRAIN_RATE)
            if self._streak["sukuna"] >= self.SUKUNA_REQUIRED_FRAMES:
                log.info("Gesture confirmed: sukuna")
                self._reset_all()
                return "sukuna"
        elif gojo_match:
            self._streak["gojo"]   += 1
            self._streak["sukuna"]  = max(0, self._streak["sukuna"] - self.DRAIN_RATE)
            if self._streak["gojo"] >= self.GOJO_REQUIRED_FRAMES:
                log.info("Gesture confirmed: gojo")
                self._reset_all()
                return "gojo"
        else:
            self._drain_all()
            return None

        return None

    def progress(self) -> Tuple[Optional[str], float]:
        best = max(self._streak, key=self._streak.get)
        val  = self._streak[best]
        if val == 0:
            return None, 0.0
        req = (self.GOJO_REQUIRED_FRAMES if best == "gojo"
               else self.SUKUNA_REQUIRED_FRAMES)
        return best, min(1.0, val / req)

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
            hs = _hand_size(lm)
            info[f"h{i}_gojo"]    = _gojo_hand(lm)
            info[f"h{i}_suk_v"]   = _sukuna_hand_vsign(lm)
            info[f"h{i}_suk_f"]   = _sukuna_hand_fist(lm)
            info[f"h{i}_scale"]   = f"{hs:.3f}"
            if len(lm) > LM.MIDDLE_TIP:
                tip_d = _dist(lm[LM.INDEX_TIP], lm[LM.MIDDLE_TIP])
                info[f"h{i}_sep"]  = f"{tip_d/hs:.2f}"
        return info

    def _drain_all(self):
        self._streak["gojo"]   = max(0, self._streak["gojo"]   - self.DRAIN_RATE)
        self._streak["sukuna"] = max(0, self._streak["sukuna"] - self.DRAIN_RATE)

    def _reset_all(self):
        self._streak = {"gojo": 0, "sukuna": 0}
