# JJK — Domain Expansion 🌀

A real-time Python desktop app that uses your **webcam** and **AI hand tracking** to trigger immersive *Jujutsu Kaisen* Domain Expansion experiences — complete with animated backgrounds, glowing auras, energy particles, screen flash, camera shake, and character voice lines.

---

## ✨ Features

| Feature | Details |
|---|---|
| 🎥 AI background removal | MediaPipe Selfie Segmentation isolates you from your background |
| 🤚 Gesture recognition | Detects Gojo (pinch both hands) and Sukuna (open palms) poses |
| 🌌 Domain video | Your video clips play as the animated background behind you |
| 💠 Aura glow | Multi-layer coloured outline pulsates around your silhouette |
| ✨ Particles | Energy particles spawn along your body edge and drift outward |
| ⚡ Flash + Shake | Cinematic impact effects on domain entry |
| 🔊 Audio | Voice lines + looping background music via pygame |
| ⌨️ Keyboard fallback | `G` / `S` keys trigger domains without needing gestures |

---

## 📁 Project Structure

```
domainexpansion/
├── main.py              # Entry point — run this
├── config.py            # All settings (edit me!)
├── camera/
│   ├── webcam.py        # Webcam capture
│   ├── background.py    # AI background removal
│   └── overlay.py       # Domain video compositor
├── gesture/
│   ├── detector.py      # Hand landmark detection
│   ├── recognizer.py    # Gojo / Sukuna gesture logic
│   └── trainer.py       # Record custom gesture data
├── effects/
│   ├── aura.py          # Pulsating glow effect
│   ├── particles.py     # Energy particle system
│   ├── flash.py         # Screen flash
│   └── shake.py         # Camera shake
├── audio/
│   ├── player.py        # Voice + BGM player
│   ├── voices/          # ← Put gojo.wav & sukuna.wav here
│   └── bgm/             # ← Put unlimited_void.mp3 & malevolent_shrine.mp3 here
├── domains/
│   ├── gojo.py          # Unlimited Void domain
│   ├── sukuna.py        # Malevolent Shrine domain
│   └── manager.py       # Domain lifecycle controller
├── assets/
│   └── videos/          # ← Put gojo.mp4 & sukuna.mp4 here
├── utils/
│   ├── timer.py
│   ├── helpers.py
│   └── logger.py
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Python 3.10+** required (uses `match`, `|` union types).

### 2. Add your assets

Place your media files in the correct directories:

```
assets/videos/gojo.mp4
assets/videos/sukuna.mp4
audio/voices/gojo.wav
audio/voices/sukuna.wav
audio/bgm/unlimited_void.mp3
audio/bgm/malevolent_shrine.mp3
```

> The app will still run **without** assets — animated fallback backgrounds and silent mode are used automatically.

### 3. Run!

```bash
python main.py
```

---

## ⌨️ Controls

| Key | Action |
|---|---|
| `G` | Expand Gojo's **Unlimited Void** |
| `S` | Expand Sukuna's **Malevolent Shrine** |
| `D` | Toggle debug overlay (landmarks, gesture streaks) |
| `ESC` | Close active domain / quit |

---

## 🤚 Gesture Guide

| Character | Gesture |
|---|---|
| **Gojo** | Pinch both hands (touch thumb + index finger tips together on both hands, hold for ~0.3s) |
| **Sukuna** | Open both palms flat toward the camera, all fingers extended, hold for ~0.3s |

---

## ⚙️ Configuration

Edit [`config.py`](config.py) to tune everything:

```python
CAMERA_INDEX      = 0      # change if using external webcam
FULLSCREEN        = False  # set True for fullscreen mode
DOMAIN_DURATION_SECONDS = 15   # how long each expansion lasts
DOMAIN_COOLDOWN_SECONDS = 8    # wait time between expansions
GESTURE_DETECTION_CONFIDENCE = 0.7
```

---

## 🎓 Training Custom Gestures

Record your own gesture data:

```bash
python -m gesture.trainer --label gojo --count 200 --out data/gestures.jsonl
```

Then use the saved `.jsonl` file to train a classifier and swap it into `gesture/recognizer.py`.

---

## 📦 Requirements

```
opencv-python>=4.8.0
mediapipe>=0.10.0
numpy>=1.24.0
pygame>=2.5.0
Pillow>=10.0.0
```

---

## 🙏 Credits

- [Jujutsu Kaisen](https://en.wikipedia.org/wiki/Jujutsu_Kaisen) — Gege Akutami / MAPPA
- [MediaPipe](https://mediapipe.dev/) — Google
- [OpenCV](https://opencv.org/)

> ⚠️ This is a fan project for personal use. All JJK characters, names, and media are property of their respective owners.
