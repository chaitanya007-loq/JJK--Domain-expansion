import os
import threading
import pygame
from utils.logger import get_logger
import config

log = get_logger(__name__)

class AudioPlayer:
    """Non-blocking game audio player that loads sound assets to memory on start."""
    VOICE_CHANNEL = 0
    BGM_CHANNEL   = 1
    SFX_CHANNEL   = 2

    _DUCK_VOLUME   = 0.30
    _DUCK_POLL_MS  = 200

    def __init__(self):
        self._initialised = False
        self._voice_channel = None
        self._bgm_channel = None
        self._sfx_channel = None
        self._sound_cache = {}
        self._current_bgm_path = None
        self._original_bgm_vol = config.AUDIO_BGM_VOLUME
        self._duck_timer = None
        self._lock = threading.Lock()

    def start(self):
        with self._lock:
            try:
                pygame.mixer.init(
                    frequency = config.AUDIO_FREQUENCY,
                    size      = -16,
                    channels  = 2,
                    buffer    = config.AUDIO_BUFFER,
                )
                pygame.mixer.set_num_channels(8)
                self._voice_channel = pygame.mixer.Channel(self.VOICE_CHANNEL)
                self._bgm_channel   = pygame.mixer.Channel(self.BGM_CHANNEL)
                self._sfx_channel   = pygame.mixer.Channel(self.SFX_CHANNEL)
                self._initialised   = True
                log.info("pygame.mixer initialised (8 channels).")
            except Exception as e:
                log.error(f"pygame.mixer failed: {e}")

    def preload_all(self):
        with self._lock:
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
        with self._lock:
            if not self._initialised:
                return
            self._cancel_duck_timer()
            pygame.mixer.stop()

    def shutdown(self):
        with self._lock:
            if self._initialised:
                self._cancel_duck_timer()
                pygame.mixer.stop()
                pygame.mixer.quit()
                self._initialised = False
                log.info("pygame.mixer shut down.")

    def _get_sound(self, path: str) -> pygame.mixer.Sound | None:
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
        with self._lock:
            if not self._initialised:
                return
            snd = self._get_sound(path)
            if snd is None:
                return
            snd.set_volume(config.AUDIO_VOICE_VOLUME)
            self._voice_channel.stop()
            self._voice_channel.play(snd)
            self._start_duck()

    def play_bgm(self, path: str, loops: int = -1):
        with self._lock:
            if not self._initialised:
                return
            if self._current_bgm_path == path and self._bgm_channel.get_busy():
                return
            snd = self._get_sound(path)
            if snd is None:
                return
            snd.set_volume(config.AUDIO_BGM_VOLUME)
            self._original_bgm_vol = config.AUDIO_BGM_VOLUME
            self._bgm_channel.stop()
            self._bgm_channel.play(snd, loops=loops)
            self._current_bgm_path = path

    def play_sfx(self, path: str, volume: float = 0.8):
        with self._lock:
            if not self._initialised:
                return
            snd = self._get_sound(path)
            if snd is None:
                return
            snd.set_volume(max(0.0, min(1.0, volume)))
            self._sfx_channel.stop()
            self._sfx_channel.play(snd)

    def stop_bgm(self):
        with self._lock:
            if self._initialised and self._bgm_channel:
                self._bgm_channel.stop()
                self._current_bgm_path = None

    def fade_out_bgm(self, ms: int = 2000):
        with self._lock:
            if self._initialised and self._bgm_channel and self._bgm_channel.get_busy():
                self._cancel_duck_timer()
                self._bgm_channel.fadeout(ms)

    def _start_duck(self):
        self._cancel_duck_timer()
        if self._current_bgm_path and self._current_bgm_path in self._sound_cache:
            self._sound_cache[self._current_bgm_path].set_volume(self._DUCK_VOLUME)
        self._schedule_duck_restore()

    def _schedule_duck_restore(self):
        self._duck_timer = threading.Timer(
            self._DUCK_POLL_MS / 1000.0,
            self._duck_restore_tick,
        )
        self._duck_timer.daemon = True
        self._duck_timer.start()

    def _duck_restore_tick(self):
        with self._lock:
            if not self._initialised:
                return
            if self._voice_channel and self._voice_channel.get_busy():
                self._schedule_duck_restore()
            else:
                self._restore_bgm_volume()

    def _restore_bgm_volume(self):
        if self._current_bgm_path and self._current_bgm_path in self._sound_cache:
            self._sound_cache[self._current_bgm_path].set_volume(self._original_bgm_vol)

    def _cancel_duck_timer(self):
        if self._duck_timer is not None:
            self._duck_timer.cancel()
            self._duck_timer = None

    def is_voice_playing(self) -> bool:
        with self._lock:
            if not self._initialised or not self._voice_channel:
                return False
            return self._voice_channel.get_busy()

    def set_voice_volume(self, vol: float):
        with self._lock:
            if self._initialised and self._voice_channel:
                self._voice_channel.set_volume(max(0.0, min(1.0, vol)))

    def set_bgm_volume(self, vol: float):
        with self._lock:
            if not self._initialised:
                return
            self._original_bgm_vol = max(0.0, min(1.0, vol))
            if not (self._voice_channel and self._voice_channel.get_busy()):
                if self._current_bgm_path and self._current_bgm_path in self._sound_cache:
                    self._sound_cache[self._current_bgm_path].set_volume(
                        self._original_bgm_vol
                    )
