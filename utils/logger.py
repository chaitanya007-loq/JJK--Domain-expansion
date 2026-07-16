"""
utils/logger.py — ANSI-coloured console logger.

Fix: always copy the LogRecord before mutating levelname/name so we don't
corrupt shared state. format() is also wrapped in try/except so a
KeyboardInterrupt or any other exception during shutdown never crashes
the app inside the logging machinery.
"""

import copy
import logging
import sys
import config

# ─── ANSI colour codes ────────────────────────────────────────────────────────
_RESET  = "\033[0m"
_GREY   = "\033[90m"
_CYAN   = "\033[96m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"
_BOLD   = "\033[1m"

_LEVEL_COLORS = {
    "DEBUG":    _GREY,
    "INFO":     _CYAN,
    "WARNING":  _YELLOW,
    "ERROR":    _RED,
    "CRITICAL": _BOLD + _RED,
}


class _ColorFormatter(logging.Formatter):
    """Custom formatter: ANSI colours + clean timestamp. Never raises."""

    FMT = "[{asctime}] [{name:<20}] {levelname:<8} {message}"

    def format(self, record: logging.LogRecord) -> str:
        try:
            # Work on a shallow copy so we don't mutate the original record
            # (mutating the original causes issues when the same record is
            # passed to multiple handlers or when formatting is retried).
            rec          = copy.copy(record)
            color        = _LEVEL_COLORS.get(rec.levelname, _RESET)
            rec.levelname = f"{color}{rec.levelname}{_RESET}"
            rec.name      = f"{_GREY}{rec.name}{_RESET}"
            formatter     = logging.Formatter(self.FMT, datefmt="%H:%M:%S", style="{")
            return formatter.format(rec)
        except Exception:
            # Absolute fallback — never crash the app because of a log call
            try:
                return logging.Formatter().format(record)
            except Exception:
                return f"[LOG ERROR] {getattr(record, 'msg', '')}"


# ─── Root logger setup ────────────────────────────────────────────────────────
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(_ColorFormatter())

_root_logger = logging.getLogger("domainexpansion")
_root_logger.handlers.clear()
_root_logger.addHandler(_handler)
_root_logger.propagate = False
_root_logger.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))


def get_logger(name: str) -> logging.Logger:
    """Return a child logger scoped to `name` (typically __name__)."""
    return _root_logger.getChild(name)
