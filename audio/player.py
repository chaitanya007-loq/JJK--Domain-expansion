"""
audio/player.py — pygame.mixer wrapper with startup preloading.

All audio assets are loaded into pygame.Sound objects once at startup via
preload_all(). Triggers then just call .play() — zero disk I/O at runtime.
"""

import os
import pygame
from utils.logger import get_logger
import config

log = get_logger(__name__)


class AudioPlayer:
    """
    Manages two audio channels:
      Channel 0 → Voice (one-shot character clip)
      Channel 1 → BGM   (looping background music)

    Call preload_all() after start() to cache all assets into RAM.
    """

    VOICE_CHANNEL = 0
    BGM_CHANNEL   = 1

    def __init__(self):
        self._initialised   = False
        self._voice_channel: pygame.mixer.Channel | None = None
        self._bgm_channel:   pygame.mixer.Channel | None = None
        self._sound_cache:   dict[str, pygame.mixer.Sound] = {}   # path → Sound
        self._current_bgm_path: str | None = None

    # ------------------------------------------------------------------ lifecycle

    def start(self):
        """Initialise pygame mixer."""
        try:
            pygame.mixer.init(
                frequency = config.AUDIO_FREQUENCY,
                size      = -16,
                channels  = 2,
                buffer    = config.AUDIO_BUFFER,
            )
            pygame.mixer.set_num_channels(4)
            self._voice_channel = pygame.mixer.Channel(self.VOICE_CHANNEL)
            self._bgm_channel   = pygame.mixer.Channel(self.BGM_CHANNEL)
            self._initialised   = True
            log.info("pygame.mixer initialised.")
        except Exception as e:
            log.error(f"pygame.mixer failed: {e}")

    def preload_all(self):
        """
        Load all known audio assets into pygame.Sound objects.
        Call once after start() so all runtime plays are instant.
        """
        if not self._initialised:
            return

        paths = [
            config.GOJO_VOICE,  config.SUKUNA_VOICE,
            config.GOJO_BGM,    config.SUKUNA_BGM,
        ]
        loaded = 0
        for path in paths:
            if not os.path.isfile(path):
                log.warning(f"Asset missing (will skip): {path}")
                continue
            if path in self._sound_cache:
                continue
            try:
                self._sound_cache[path] = pygame.mixer.Sound(path)
                log.info(f"Preloaded audio: {os.path.basename(path)}")
                loaded += 1
            except Exception as e:
                log.warning(f"Could not load {os.path.basename(path)}: {e}")

        log.info(f"Audio preload complete — {loaded}/{len(paths)} assets cached.")

    def stop_all(self):
        if not self._initialised:
            return
        pygame.mixer.stop()

    def shutdown(self):
        if self._initialised:
            self.stop_all()
            pygame.mixer.quit()
            log.info("pygame.mixer shut down.")

    # ------------------------------------------------------------------ playback

    def _get_sound(self, path: str) -> pygame.mixer.Sound | None:
        """Return cached Sound object, or load on demand if not preloaded."""
        if path in self._sound_cache:
            return self._sound_cache[path]
        if not os.path.isfile(path):
            log.warning(f"Audio file not found: {path}")
            return None
        try:
            snd = pygame.mixer.Sound(path)
            self._sound_cache[path] = snd
            return snd
        except Exception as e:
            log.error(f"Failed to load {path}: {e}")
            return None

    def play_voice(self, path: str):
        """Play voice clip once on channel 0. Instant if preloaded."""
        if not self._initialised:
            return
        snd = self._get_sound(path)
        if snd is None:
            return
        snd.set_volume(config.AUDIO_VOICE_VOLUME)
        self._voice_channel.stop()
        self._voice_channel.play(snd)

    def play_bgm(self, path: str, loops: int = -1):
        """Play BGM on channel 1 (loops infinitely by default). Instant if preloaded."""
        if not self._initialised:
            return
        # Avoid restarting the same track
        if self._current_bgm_path == path and self._bgm_channel.get_busy():
            return
        snd = self._get_sound(path)
        if snd is None:
            return
        snd.set_volume(config.AUDIO_BGM_VOLUME)
        self._bgm_channel.stop()
        self._bgm_channel.play(snd, loops=loops)
        self._current_bgm_path = path

    def stop_bgm(self):
        if self._initialised and self._bgm_channel:
            self._bgm_channel.stop()
            self._current_bgm_path = None

    def fade_out_bgm(self, ms: int = 2000):
        if self._initialised and self._bgm_channel and self._bgm_channel.get_busy():
            self._bgm_channel.fadeout(ms)

    # ------------------------------------------------------------------ volume

    def set_voice_volume(self, vol: float):
        if self._initialised and self._voice_channel:
            self._voice_channel.set_volume(max(0.0, min(1.0, vol)))

    def set_bgm_volume(self, vol: float):
        """Update volume of the currently playing BGM Sound object."""
        if not self._initialised:
            return
        if self._current_bgm_path and self._current_bgm_path in self._sound_cache:
            self._sound_cache[self._current_bgm_path].set_volume(
                max(0.0, min(1.0, vol))
            )
