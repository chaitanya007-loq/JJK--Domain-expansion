import os

# Camera settings
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS_TARGET = 30

WINDOW_NAME = "JJK — Domain Expansion"
FULLSCREEN = False

# Background removal backend and config
BG_BACKEND = "rembg"  # "rembg" or "mediapipe"
BG_MODEL_SELECTION = 1
BG_THRESHOLD = 0.6
REMBG_MODEL = "u2net_human_seg"

# Gesture detection thresholds
GESTURE_MAX_HANDS = 2
GESTURE_DETECTION_CONFIDENCE = 0.7
GESTURE_TRACKING_CONFIDENCE = 0.5
KEYBOARD_TRIGGER_ENABLED = True

# Domain limits
DOMAIN_COOLDOWN_SECONDS = 8
DOMAIN_DURATION_SECONDS = 15

# Visual FX settings
FLASH_DURATION_FRAMES = 18
FLASH_PEAK_ALPHA = 0.95

SHAKE_DURATION_FRAMES = 25
SHAKE_INTENSITY = 22

AURA_BLUR_KERNEL = 41
AURA_LAYERS = 4
AURA_INTENSITY = 0.75

PARTICLE_COUNT = 120
PARTICLE_SPEED_MIN = 1.0
PARTICLE_SPEED_MAX = 4.5
PARTICLE_RADIUS_MIN = 2
PARTICLE_RADIUS_MAX = 7
PARTICLE_LIFETIME = 60

# Gojo theme config
GOJO_AURA_COLOR = (255, 160, 30)
GOJO_PARTICLE_COLOR = (255, 220, 100)
GOJO_FLASH_COLOR = (255, 255, 255)
GOJO_BG_VIDEO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "videos", "gojo.mp4")
GOJO_VOICE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio", "voices", "Gojo Domain Expansion sound effect.mp3")
GOJO_BGM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio", "bgm", "unlimited_void.mp3")

# Sukuna theme config
SUKUNA_AURA_COLOR = (40, 20, 220)
SUKUNA_PARTICLE_COLOR = (0, 0, 180)
SUKUNA_FLASH_COLOR = (60, 0, 180)
SUKUNA_BG_VIDEO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "videos", "sukuna.mp4")
SUKUNA_VOICE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio", "voices", "Sukuna's Domain Expansion sound effect.mp3")
SUKUNA_BGM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio", "bgm", "malevolent_shrine.mp3")

# Audio Mixer config
AUDIO_VOICE_VOLUME = 1.0
AUDIO_BGM_VOLUME = 0.55
AUDIO_FREQUENCY = 44100
AUDIO_BUFFER = 512

# Inference and threading bounds
INFERENCE_WIDTH = 640
INFERENCE_HEIGHT = 360
VIDEO_PRELOAD_MAX_FRAMES = 150

THROTTLE_MASK_REFRESH_INTERVAL = 15
COMPOSITOR_USE_GPU = True
PARTICLE_WARM_UP_AT_START = True

LOG_LEVEL = "INFO"
