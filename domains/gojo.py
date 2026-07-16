"""
domains/gojo.py — "Unlimited Void" domain state for Satoru Gojo.

Responsible for:
  • Configuring all Gojo-specific colours and assets
  • Triggering the entry sequence (burst flash → shake → voice → BGM)
  • Exposing per-frame render parameters to the manager
"""

import config
from effects.aura      import AuraEffect
from effects.particles import ParticleSystem
from effects.flash     import FlashEffect, BurstFlash
from effects.shake     import ShakeEffect
from audio.player      import AudioPlayer
from camera.overlay    import DomainVideoPlayer
from utils.logger      import get_logger
from utils.timer       import CountdownTimer

log = get_logger(__name__)

NAME         = "gojo"
DISPLAY_NAME = "Unlimited Void"


class GojoDomain:
    """
    Encapsulates everything needed to render the Unlimited Void domain.

    Shared instances (audio, effects) are injected by the manager so they
    are never duplicated across domain objects.
    """

    def __init__(self, audio: AudioPlayer):
        self._audio    = audio
        self._timer    = CountdownTimer(config.DOMAIN_DURATION_SECONDS)

        # Effects wired to Gojo's colour palette
        self.aura      = AuraEffect(color=config.GOJO_AURA_COLOR)
        self.particles = ParticleSystem(color=config.GOJO_PARTICLE_COLOR)
        self.flash     = FlashEffect()
        self.burst     = BurstFlash(color=config.GOJO_FLASH_COLOR)
        self.shake     = ShakeEffect()

        # Background video player
        self.video     = DomainVideoPlayer(
            video_path  = config.GOJO_BG_VIDEO,
            tint_color  = config.GOJO_AURA_COLOR,
        )

        self._active   = False

    # ------------------------------------------------------------------ lifecycle

    def activate(self):
        """Trigger the full domain entry sequence."""
        log.info(f"[{NAME}] Domain expansion: {DISPLAY_NAME}!")
        self._active = True
        self._timer.start()

        # Entry FX sequence
        self.burst.trigger(color=config.GOJO_FLASH_COLOR)
        self.flash.trigger(color=config.GOJO_FLASH_COLOR)
        self.shake.trigger()
        self.aura.reset()
        self.particles.clear()

        # Audio
        self._audio.play_voice(config.GOJO_VOICE)
        self._audio.play_bgm(config.GOJO_BGM)

        self.video.open()

    def deactivate(self):
        """Clean up when the domain ends."""
        log.info(f"[{NAME}] Domain closing.")
        self._active = False
        self._audio.fade_out_bgm(ms=2500)
        self.video.release()
        self.particles.clear()

    # ------------------------------------------------------------------ state

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def expired(self) -> bool:
        """True when the domain timer has run out."""
        return self._timer.expired()

    @property
    def time_remaining(self) -> float:
        return self._timer.remaining()

    @property
    def progress(self) -> float:
        """0.0 = just started, 1.0 = about to expire."""
        return self._timer.progress()

    # ------------------------------------------------------------------ per-frame

    def get_bg_frame(self, width: int, height: int):
        """Return the next domain background video frame."""
        return self.video.next_frame(width, height)

    @property
    def intensity(self) -> float:
        """
        Fade the effects out gracefully in the last 3 seconds of the domain.
        """
        remaining = self.time_remaining
        if remaining < 3.0:
            return remaining / 3.0
        return 1.0
