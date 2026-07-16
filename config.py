"""
config.py — Central configuration for JJK Domain Expansion.
Edit this file to tune camera, effects, and asset paths.
"""

import os

# ─── Base Paths ───────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
AUDIO_DIR  = os.path.join(BASE_DIR, "audio")

# ─── Camera ───────────────────────────────────────────────────────────────────
CAMERA_INDEX      = 0          # webcam device index
CAMERA_WIDTH      = 1280       # capture width  (px)
CAMERA_HEIGHT     = 720        # capture height (px)
CAMERA_FPS_TARGET = 30         # target FPS for the main loop

# ─── Display ──────────────────────────────────────────────────────────────────
WINDOW_NAME  = "JJK — Domain Expansion"
FULLSCREEN   = False           # set True for borderless fullscreen

# ─── Background Removal (MediaPipe Selfie Segmentation) ───────────────────────
BG_MODEL_SELECTION = 1        # 0 = general, 1 = landscape (wider FoV)
BG_THRESHOLD       = 0.6      # confidence threshold for segmentation mask

# ─── Gesture (MediaPipe Hands) ────────────────────────────────────────────────
GESTURE_MAX_HANDS             = 2
GESTURE_DETECTION_CONFIDENCE  = 0.7
GESTURE_TRACKING_CONFIDENCE   = 0.5
# Keyboard fallback: press G → Gojo, S → Sukuna, ESC → quit
KEYBOARD_TRIGGER_ENABLED      = True

# ─── Domain Cooldown ──────────────────────────────────────────────────────────
DOMAIN_COOLDOWN_SECONDS = 8    # seconds before another expansion can trigger
DOMAIN_DURATION_SECONDS = 15   # how long the domain stays active

# ─── Flash Effect ─────────────────────────────────────────────────────────────
FLASH_DURATION_FRAMES = 18     # total frames for one flash (in + out)
FLASH_PEAK_ALPHA      = 0.95   # max opacity of the white/colour flash

# ─── Shake Effect ─────────────────────────────────────────────────────────────
SHAKE_DURATION_FRAMES = 25     # frames the shake lasts
SHAKE_INTENSITY       = 22     # max pixel offset

# ─── Aura Effect ──────────────────────────────────────────────────────────────
AURA_BLUR_KERNEL = 41          # must be odd; larger = softer glow
AURA_LAYERS      = 4           # number of glow layers stacked
AURA_INTENSITY   = 0.75        # overall opacity of the aura

# ─── Particle System ──────────────────────────────────────────────────────────
PARTICLE_COUNT      = 120      # number of active particles
PARTICLE_SPEED_MIN  = 1.0
PARTICLE_SPEED_MAX  = 4.5
PARTICLE_RADIUS_MIN = 2
PARTICLE_RADIUS_MAX = 7
PARTICLE_LIFETIME   = 60       # frames before a particle is recycled

# ─── Gojo — "Unlimited Void" ──────────────────────────────────────────────────
GOJO_AURA_COLOR      = (255, 160,  30)   # BGR — electric blue-white
GOJO_PARTICLE_COLOR  = (255, 220, 100)   # BGR — pale gold/white
GOJO_FLASH_COLOR     = (255, 255, 255)   # BGR — pure white
GOJO_BG_VIDEO        = os.path.join(ASSETS_DIR, "videos", "gojo.mp4")
GOJO_VOICE           = os.path.join(AUDIO_DIR,  "voices", "Gojo Domain Expansion sound effect.mp3")
GOJO_BGM             = os.path.join(AUDIO_DIR,  "bgm",    "unlimited_void.mp3")

# ─── Sukuna — "Malevolent Shrine" ─────────────────────────────────────────────
SUKUNA_AURA_COLOR     = ( 40,  20, 220)   # BGR — deep crimson red
SUKUNA_PARTICLE_COLOR = (  0,   0, 180)   # BGR — dark red
SUKUNA_FLASH_COLOR    = ( 60,   0, 180)   # BGR — dark-red flash
SUKUNA_BG_VIDEO       = os.path.join(ASSETS_DIR, "videos", "sukuna.mp4")
SUKUNA_VOICE          = os.path.join(AUDIO_DIR,  "voices", "Sukuna's Domain Expansion sound effect.mp3")
SUKUNA_BGM            = os.path.join(AUDIO_DIR,  "bgm",    "malevolent_shrine.mp3")

# ─── Audio ────────────────────────────────────────────────────────────────────
AUDIO_VOICE_VOLUME = 1.0       # 0.0 – 1.0
AUDIO_BGM_VOLUME   = 0.55      # 0.0 – 1.0
AUDIO_FREQUENCY    = 44100
AUDIO_BUFFER       = 512

# ─── Performance / Threading ─────────────────────────────────────────────────
# MediaPipe inference runs at this resolution (4× fewer pixels than 1280×720)
INFERENCE_WIDTH          = 640
INFERENCE_HEIGHT         = 360
# Max frames preloaded per domain video into RAM (150 @ 640×360 ≈ 104 MB each)
VIDEO_PRELOAD_MAX_FRAMES = 150

# ─── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"             # DEBUG | INFO | WARNING | ERROR
