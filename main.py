import os
# Suppress MediaPipe and TensorFlow Lite C++ log warnings
os.environ['GLOG_minloglevel'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import cv2
import numpy as np

import config
from camera.webcam          import Webcam
from camera.capture_thread  import CaptureThread
from camera.inference_thread import InferenceThread
from camera.background      import BackgroundRemover
from camera.overlay         import Overlay
from gesture.detector       import HandDetector
from gesture.recognizer     import GestureRecognizer
from domains.manager        import DomainManager
from audio.player           import AudioPlayer
from effects.particles      import ParticleSystem
from utils.timer            import FPSCounter, Profiler
from utils.helpers          import draw_hud_text, draw_cooldown_bar, add_vignette
from utils.logger           import get_logger

log = get_logger(__name__)

_EMPTY_MASK = np.zeros(
    (config.CAMERA_HEIGHT, config.CAMERA_WIDTH), dtype=np.float32
)

def _draw_hud(
    frame:          np.ndarray,
    avg_fps:        float,
    curr_fps:       float,
    frame_time:     float,
    domain_name:    str | None,
    time_remaining: float,
    cooldown_prog:  float,
    cooldown_ready: bool,
    gesture_prog:   tuple,
    debug_info:     dict,
    debug_mode:     bool,
) -> np.ndarray:
    h, w = frame.shape[:2]

    hud_text = f"Avg FPS: {avg_fps:.1f} | Instant FPS: {curr_fps:.1f} | Latency: {frame_time:.1f}ms"
    draw_hud_text(frame, hud_text, (15, 30),
                  color=(200, 255, 200), scale=0.5, thickness=1)

    if domain_name:
        banner = "UNLIMITED VOID" if domain_name == "gojo" else "MALEVOLENT SHRINE"
        col    = config.GOJO_AURA_COLOR if domain_name == "gojo" else config.SUKUNA_AURA_COLOR

        (tw, _), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
        cx = (w - tw) // 2

        strip = frame.copy()
        cv2.rectangle(strip, (0, h - 80), (w, h - 45), (0, 0, 0), -1)
        cv2.addWeighted(strip, 0.5, frame, 0.5, 0, frame)

        draw_hud_text(frame, banner,          (cx, h - 55),   col,          1.2, 2)
        draw_hud_text(frame, f"{time_remaining:.1f}s",
                      (w - 80, h - 55), (200, 200, 200), 0.7)
    else:
        bar_col = (120, 200, 255) if cooldown_ready else (80, 80, 120)
        label   = "READY" if cooldown_ready else f"COOLDOWN {100 * cooldown_prog:.0f}%"
        draw_cooldown_bar(frame, cooldown_prog, bar_col, label=label)

        if cooldown_ready:
            for i, (key, name, col) in enumerate([
                ("G", "Gojo",   config.GOJO_AURA_COLOR),
                ("S", "Sukuna", config.SUKUNA_AURA_COLOR),
            ]):
                draw_hud_text(frame, f"[{key}] {name}",
                              (w - 140, 30 + i * 25), col, 0.55)

        g_name, g_prog = gesture_prog
        if g_name and g_prog > 0.0:
            g_col = config.GOJO_AURA_COLOR if g_name == "gojo" else config.SUKUNA_AURA_COLOR
            g_label = "Gojo" if g_name == "gojo" else "Sukuna"

            bar_w  = int(w * 0.5)
            bar_h  = 12
            x_off  = (w - bar_w) // 2
            y_off  = h - 60

            cv2.rectangle(frame, (x_off, y_off), (x_off + bar_w, y_off + bar_h),
                          (30, 30, 30), -1)
            fill_w = int(bar_w * g_prog)
            if fill_w > 0:
                cv2.rectangle(frame, (x_off, y_off), (x_off + fill_w, y_off + bar_h),
                              g_col, -1)
            cv2.rectangle(frame, (x_off, y_off), (x_off + bar_w, y_off + bar_h),
                          (150, 150, 150), 1)

            draw_hud_text(frame,
                          f"{g_label} gesture detected... hold it!",
                          (x_off, y_off - 10),
                          color=g_col, scale=0.55, thickness=1)

    if debug_mode and debug_info:
        y = 60
        for k, v in debug_info.items():
            draw_hud_text(frame, f"{k}: {v}", (15, y),
                          color=(180, 255, 180), scale=0.45, thickness=1)
            y += 18

    return frame

def _make_idle_frame(width: int, height: int, tick: int) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    t     = tick * 0.02
    cx, cy = width // 2, height // 2
    for r in range(50, min(cx, cy), 60):
        pulse = abs(np.sin(t + r * 0.05))
        cv2.circle(frame, (cx, cy), r,
                   (int(40 * pulse), int(20 * pulse), int(80 * pulse)), 1)
    draw_hud_text(frame, "JJK — Domain Expansion",
                  (cx - 200, cy - 20), (180, 120, 255), 1.0, 2)
    draw_hud_text(frame, "Press G (Gojo) or S (Sukuna) to expand",
                  (cx - 230, cy + 20), (140, 140, 180), 0.6, 1)
    return frame

def _mask_to_uint8(mask: np.ndarray) -> np.ndarray:
    if mask.dtype == np.float32 or mask.dtype == np.float64:
        return (mask * 255.0).clip(0, 255).astype(np.uint8)
    return mask

def main():
    log.info("=== JJK Domain Expansion starting ===")

    cam         = Webcam()
    bg_remover  = BackgroundRemover()
    detector    = HandDetector()
    recognizer  = GestureRecognizer()
    audio       = AudioPlayer()
    fps_counter = FPSCounter(window=30)

    audio.start()
    audio.preload_all()

    bg_remover.start()
    detector.start()

    if config.PARTICLE_WARM_UP_AT_START:
        ParticleSystem.warm_up()

    cam_ok = cam.open()
    if not cam_ok:
        log.warning("Webcam unavailable — keyboard-only mode.")

    W = cam.width  if cam_ok else config.CAMERA_WIDTH
    H = cam.height if cam_ok else config.CAMERA_HEIGHT

    manager = DomainManager(audio)

    capture_thread   = CaptureThread(cam)
    inference_thread = InferenceThread(bg_remover, detector)

    if cam_ok:
        capture_thread.start_capture()

    inference_thread.start_inference()

    profiler = Profiler()

    flags = cv2.WINDOW_NORMAL
    cv2.namedWindow(config.WINDOW_NAME, flags)
    if config.FULLSCREEN:
        cv2.setWindowProperty(config.WINDOW_NAME,
                              cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    debug_mode = False
    tick       = 0

    throttle_active       = False
    throttle_frame_count  = 0
    cached_mask           = _EMPTY_MASK.copy()
    MASK_REFRESH_INTERVAL = config.THROTTLE_MASK_REFRESH_INTERVAL

    log.info("Main loop started. Controls: G=Gojo  S=Sukuna  D=Debug  ESC=Quit")

    try:
        while True:
            tick += 1
            fps_counter.tick()

            profiler.start("camera_capture")
            if cam_ok:
                frame = capture_thread.get_latest_frame()
            else:
                frame = None
            profiler.stop("camera_capture")

            profiler.start("segmentation")
            if frame is None:
                frame        = _make_idle_frame(W, H, tick)
                person_mask  = _EMPTY_MASK
                hands        = []
                gesture_prog = (None, 0.0)
            else:
                if throttle_active:
                    throttle_frame_count += 1
                    if throttle_frame_count >= MASK_REFRESH_INTERVAL:
                        inference_thread.submit_frame(frame, W, H)
                        person_mask, _ = inference_thread.get_result()
                        if person_mask is not None:
                            cached_mask = person_mask
                        else:
                            person_mask = cached_mask
                        throttle_frame_count = 0
                    else:
                        person_mask = cached_mask
                    hands = []
                else:
                    inference_thread.submit_frame(frame, W, H)
                    person_mask, hands = inference_thread.get_result()
                    if person_mask is None:
                        person_mask = _EMPTY_MASK
                    else:
                        cached_mask = person_mask
            profiler.stop("segmentation")

            profiler.start("gesture_recognition")
            if throttle_active:
                gesture      = None
                gesture_prog = (None, 0.0)
            else:
                gesture      = recognizer.recognize(hands)
                gesture_prog = recognizer.progress()
                if gesture and config.KEYBOARD_TRIGGER_ENABLED:
                    manager.trigger(gesture)
            profiler.stop("gesture_recognition")

            manager.update()
            domain = manager.active_domain
            curr_active = domain is not None

            if curr_active and not throttle_active:
                throttle_active      = True
                throttle_frame_count = 0
                inference_thread.set_mode("seg_only")
                log.info("Dynamic AI Throttling: ACTIVE (hand tracking disabled)")
            elif not curr_active and throttle_active:
                throttle_active = False
                inference_thread.set_mode("full")
                log.info("Dynamic AI Throttling: INACTIVE (full inference resumed)")

            person_mask_u8 = _mask_to_uint8(person_mask)

            if domain:
                profiler.start("video_playback")
                bg_frame  = domain.get_bg_frame(W, H)
                intensity = domain.intensity
                profiler.stop("video_playback")

                tint_color = None
                if manager.active_name == "gojo":
                    tint_color = config.GOJO_AURA_COLOR
                elif manager.active_name == "sukuna":
                    tint_color = config.SUKUNA_AURA_COLOR

                profiler.start("compositing")
                frame = Overlay.composite(frame, person_mask, bg_frame,
                                          tint_color=tint_color,
                                          tint_strength=0.12)
                profiler.stop("compositing")

                frame = domain.burst.apply(frame)
                frame = domain.flash.apply(frame)

                profiler.start("aura_rendering")
                frame = domain.aura.apply(frame, person_mask_u8, intensity)
                profiler.stop("aura_rendering")

                profiler.start("particle_rendering")
                frame = domain.particles.update_and_draw(frame, person_mask_u8,
                                                         intensity)
                profiler.stop("particle_rendering")

                frame = domain.shake.apply(frame)
                frame = add_vignette(frame, strength=0.6)
            else:
                for key in ["video_playback", "compositing", "aura_rendering", "particle_rendering"]:
                    profiler.durations[key] = 0.0
                frame = add_vignette(frame, strength=0.35)

            debug_info = recognizer.debug_info(hands) if debug_mode else {}
            if debug_mode:
                for key, val in profiler.durations.items():
                    debug_info[f"CPU_{key}_ms"] = f"{val:.2f}"
                debug_info["throttle_mode"] = "ACTIVE" if throttle_active else "idle"
                if throttle_active:
                    debug_info["mask_refresh_in"] = f"{MASK_REFRESH_INTERVAL - throttle_frame_count} frames"
                debug_info["compositor"] = "GPU (CuPy)" if BackgroundRemover._HAS_CUPY else "CPU (NumPy)"

            frame = _draw_hud(
                frame,
                avg_fps        = fps_counter.fps,
                curr_fps       = fps_counter.current_fps,
                frame_time     = fps_counter.frame_time_ms,
                domain_name    = manager.active_name,
                time_remaining = domain.time_remaining if domain else 0.0,
                cooldown_prog  = manager.cooldown_progress,
                cooldown_ready = manager.cooldown_ready,
                gesture_prog   = gesture_prog if not domain else (None, 0.0),
                debug_info     = debug_info,
                debug_mode     = debug_mode,
            )

            profiler.start("final_display")
            cv2.imshow(config.WINDOW_NAME, frame)
            profiler.stop("final_display")

            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                if domain:
                    manager.close_domain()
                else:
                    break

            elif key in (ord('g'), ord('G')):
                if config.KEYBOARD_TRIGGER_ENABLED:
                    manager.trigger("gojo")

            elif key in (ord('s'), ord('S')):
                if config.KEYBOARD_TRIGGER_ENABLED:
                    manager.trigger("sukuna")

            elif key in (ord('d'), ord('D')):
                debug_mode = not debug_mode
                log.info(f"Debug mode: {'ON' if debug_mode else 'OFF'}")

    except KeyboardInterrupt:
        pass
    finally:
        for step_name, step_fn in [
            ("close domain",     manager.close_domain),
            ("stop inference",   inference_thread.stop),
            ("stop capture",     capture_thread.stop),
            ("shutdown audio",   audio.shutdown),
            ("stop detector",    detector.stop),
            ("stop bg_remover",  bg_remover.stop),
            ("release camera",   cam.release),
            ("destroy windows",  cv2.destroyAllWindows),
        ]:
            try:
                step_fn()
            except Exception as exc:
                sys.stderr.write(f"[shutdown] {step_name}: {exc}\n")

        sys.stdout.write("=== Bye! ===\n")

if __name__ == "__main__":
    main()
