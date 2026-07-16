"""
domains/manager.py — Controls which domain is active and handles transitions.

Responsibilities
----------------
  • Owns GojoDomain and SukunaDomain instances
  • Enforces the global cooldown between expansions
  • Exposes a single `trigger(name)` and `update()` API to main.py
"""

from typing import Optional

import config
from domains.gojo   import GojoDomain
from domains.sukuna import SukunaDomain
from audio.player   import AudioPlayer
from utils.timer    import Cooldown
from utils.logger   import get_logger

log = get_logger(__name__)


class DomainManager:
    """
    Manages the lifecycle of all domains.

    Usage (in main loop)
    --------------------
    manager = DomainManager(audio)
    manager.trigger("gojo")        # call when gesture detected

    # Every frame:
    domain = manager.active_domain # None | GojoDomain | SukunaDomain
    manager.update()               # handles expiry / cleanup
    """

    def __init__(self, audio: AudioPlayer):
        self._audio    = audio
        self._gojo     = GojoDomain(audio)
        self._sukuna   = SukunaDomain(audio)
        self._active   = None          # currently active domain object
        self._cooldown = Cooldown(config.DOMAIN_COOLDOWN_SECONDS)

    # ------------------------------------------------------------------ trigger

    def trigger(self, name: str) -> bool:
        """
        Attempt to expand a domain named `name` ("gojo" | "sukuna").

        Returns True if the domain was successfully triggered,
        False if a cooldown or active domain blocked it.
        """
        if not self._cooldown.ready():
            log.info(f"Cooldown active — {self._cooldown.remaining():.1f}s remaining.")
            return False

        if self._active and self._active.is_active:
            log.info("A domain is already active — closing it first.")
            self._deactivate_current()

        if name == "gojo":
            domain = self._gojo
        elif name == "sukuna":
            domain = self._sukuna
        else:
            log.warning(f"Unknown domain name: '{name}'")
            return False

        domain.activate()
        self._active = domain
        self._cooldown.reset()
        log.info(f"Domain '{name}' activated. Cooldown started.")
        return True

    # ------------------------------------------------------------------ update

    def update(self):
        """
        Call once per frame to auto-expire the active domain when its
        timer runs out.
        """
        if self._active and self._active.is_active:
            if self._active.expired:
                log.info("Domain expired naturally.")
                self._deactivate_current()

    # ------------------------------------------------------------------ helpers

    def _deactivate_current(self):
        if self._active:
            self._active.deactivate()
            self._active = None

    def close_domain(self):
        """Manually close the active domain (e.g., user pressed ESC)."""
        self._deactivate_current()

    # ------------------------------------------------------------------ properties

    @property
    def active_domain(self):
        """Returns the active domain object, or None."""
        return self._active if (self._active and self._active.is_active) else None

    @property
    def active_name(self) -> Optional[str]:
        """Returns "gojo", "sukuna", or None."""
        d = self.active_domain
        if d is self._gojo:
            return "gojo"
        if d is self._sukuna:
            return "sukuna"
        return None

    @property
    def cooldown_progress(self) -> float:
        """0.0 = just triggered, 1.0 = ready."""
        return self._cooldown.progress()

    @property
    def cooldown_ready(self) -> bool:
        return self._cooldown.ready()
