"""
domains/sukuna.py — "Malevolent Shrine" domain state for Ryomen Sukuna.

Mirror of gojo.py but wired to Sukuna's crimson colour palette and assets.
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

NAME         = "sukuna"
DISPLAY_NAME = "Malevolent Shrine"


class SukunaDomain:
    """
    Encapsulates everything needed to render the Malevolent Shrine domain.
    """

    def __init__(self, audio: AudioPlayer):
        self._audio    = audio
        self._timer    = CountdownTimer(config.DOMAIN_DURATION_SECONDS)

        # Effects wired to Sukuna's colour palette
        self.aura      = AuraEffect(color=config.SUKUNA_AURA_COLOR)
        self.particles = ParticleSystem(color=config.SUKUNA_PARTICLE_COLOR)
        self.flash     = FlashEffect()
        self.burst     = BurstFlash(color=config.SUKUNA_FLASH_COLOR)
        self.shake     = ShakeEffect()

        # Background video player
        self.video     = DomainVideoPlayer(
            video_path  = config.SUKUNA_BG_VIDEO,
            tint_color  = config.SUKUNA_AURA_COLOR,
        )

        self._active   = False

    # ------------------------------------------------------------------ lifecycle

    def activate(self):
        """Trigger the full domain entry sequence."""
        log.info(f"[{NAME}] Domain expansion: {DISPLAY_NAME}!")
        self._active = True
        self._timer.start()

        # Entry FX sequence
        self.burst.trigger(color=config.SUKUNA_FLASH_COLOR)
        self.flash.trigger(color=config.SUKUNA_FLASH_COLOR)
        self.shake.trigger(intensity=config.SHAKE_INTENSITY * 1.3)   # Sukuna hits harder
        self.aura.reset()
        self.particles.clear()

        # Audio
        self._audio.play_voice(config.SUKUNA_VOICE)
        self._audio.play_bgm(config.SUKUNA_BGM)

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
        return self._timer.expired()

    @property
    def time_remaining(self) -> float:
        return self._timer.remaining()

    @property
    def progress(self) -> float:
        return self._timer.progress()

    # ------------------------------------------------------------------ per-frame

    def get_bg_frame(self, width: int, height: int):
        return self.video.next_frame(width, height)

    @property
    def intensity(self) -> float:
        remaining = self.time_remaining
        if remaining < 3.0:
            return remaining / 3.0
        return 1.0
