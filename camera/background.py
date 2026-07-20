import os
import cv2
import numpy as np

from utils.logger import get_logger
import config

log = get_logger(__name__)

_HAS_CUPY = False
cp = None

if config.COMPOSITOR_USE_GPU:
    try:
        import cupy as _cp
        _cp.array([1.0])
        cp = _cp
        _HAS_CUPY = True
        log.info(f"CuPy GPU compositor enabled (CUDA device: {_cp.cuda.runtime.getDevice()}).")
    except Exception as e:
        log.info(f"CuPy unavailable -- using NumPy CPU fallback ({e}).")

_HAS_REMBG = False
_rembg_session = None

if config.BG_BACKEND == "rembg":
    try:
        from rembg import new_session, remove
        _HAS_REMBG = True
        log.info("rembg library detected.")
    except ImportError:
        log.warning("rembg not installed -- falling back to MediaPipe backend. Install with: pip install rembg[cpu]")

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "models", "selfie_segmenter.tflite",
)

class BackgroundRemover:
    """Handles background removal with rembg or mediapipe fallback."""
    _HAS_CUPY  = _HAS_CUPY
    _HAS_REMBG = _HAS_REMBG

    def __init__(self):
        self._segmenter = None
        self._rembg_session = None
        self._backend = "none"
        log.info("BackgroundRemover initialised.")

    def start(self):
        if _HAS_REMBG and config.BG_BACKEND == "rembg":
            try:
                self._rembg_session = new_session(model_name=config.REMBG_MODEL)
                self._backend = "rembg"
                log.info(f"rembg backend loaded (model: {config.REMBG_MODEL})")
                return
            except Exception as e:
                log.warning(f"rembg session failed: {e} -- falling back to MediaPipe")

        self._start_mediapipe()

    def _start_mediapipe(self):
        if not os.path.isfile(_MODEL_PATH):
            log.error(f"Segmentation model not found: {_MODEL_PATH}")
            return

        base_options = mp_python.BaseOptions(model_asset_path=_MODEL_PATH)
        options = mp_vision.ImageSegmenterOptions(
            base_options = base_options,
            running_mode = mp_vision.RunningMode.IMAGE,
            output_confidence_masks = True,
        )
        self._segmenter = mp_vision.ImageSegmenter.create_from_options(options)
        self._backend = "mediapipe"
        log.info(f"MediaPipe Selfie Segmenter loaded: {_MODEL_PATH}")

    def stop(self):
        if self._segmenter:
            self._segmenter.close()
            self._segmenter = None
        self._rembg_session = None
        self._backend = "none"
        log.info("Background segmentation stopped.")

    @property
    def backend_name(self) -> str:
        return self._backend

    def get_mask(self, bgr_frame: np.ndarray) -> np.ndarray:
        if self._backend == "rembg":
            return self._get_mask_rembg(bgr_frame)
        elif self._backend == "mediapipe":
            return self._get_mask_mediapipe(bgr_frame)
        else:
            h, w = bgr_frame.shape[:2]
            return np.ones((h, w), dtype=np.float32)

    def _get_mask_rembg(self, bgr_frame: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgba = remove(
            rgb,
            session=self._rembg_session,
            only_mask=True,
            post_process_mask=True,
        )
        if len(rgba.shape) == 3:
            mask_u8 = rgba[:, :, 0]
        else:
            mask_u8 = rgba

        soft = mask_u8.astype(np.float32) / 255.0
        return cv2.GaussianBlur(soft, (15, 15), 0)

    def _get_mask_mediapipe(self, bgr_frame: np.ndarray) -> np.ndarray:
        if self._segmenter is None:
            h, w = bgr_frame.shape[:2]
            return np.ones((h, w), dtype=np.float32)

        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._segmenter.segment(mp_image)

        conf_mask = result.confidence_masks[0].numpy_view().copy()
        if len(conf_mask.shape) == 3:
            conf_mask = conf_mask[:, :, 0]

        margin = 0.15
        lo = config.BG_THRESHOLD - margin
        hi = config.BG_THRESHOLD + margin
        soft = np.clip((conf_mask - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)

        return cv2.GaussianBlur(soft, (21, 21), 0)

    def get_mask_uint8(self, bgr_frame: np.ndarray) -> np.ndarray:
        soft = self.get_mask(bgr_frame)
        return (soft * 255.0).clip(0, 255).astype(np.uint8)

    def remove_background(
        self,
        bgr_frame:   np.ndarray,
        replacement: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        soft_mask = self.get_mask(bgr_frame)

        if replacement is None:
            replacement = np.zeros_like(bgr_frame)
        else:
            replacement = cv2.resize(
                replacement,
                (bgr_frame.shape[1], bgr_frame.shape[0])
            )

        composited = composite_frames(bgr_frame, replacement, soft_mask)
        mask_u8    = (soft_mask * 255.0).clip(0, 255).astype(np.uint8)
        return composited, mask_u8

def composite_frames(
    user_frame:   np.ndarray,
    bg_frame:     np.ndarray,
    soft_mask:    np.ndarray,
    tint_color:   tuple | None = None,
    tint_strength: float = 0.12,
) -> np.ndarray:
    if _HAS_CUPY:
        return _composite_gpu(user_frame, bg_frame, soft_mask, tint_color, tint_strength)
    else:
        return _composite_cpu(user_frame, bg_frame, soft_mask, tint_color, tint_strength)

def _composite_gpu(
    user_frame: np.ndarray,
    bg_frame:   np.ndarray,
    soft_mask:  np.ndarray,
    tint_color: tuple | None,
    tint_strength: float,
) -> np.ndarray:
    user_g  = cp.asarray(user_frame, dtype=cp.float32)
    bg_g    = cp.asarray(bg_frame,   dtype=cp.float32)
    mask_g  = cp.asarray(soft_mask,  dtype=cp.float32)[:, :, cp.newaxis]

    if tint_color is not None and tint_strength > 0.0:
        tint_g  = cp.array(tint_color, dtype=cp.float32).reshape(1, 1, 3)
        ts      = cp.float32(tint_strength)
        user_g  = user_g * (1.0 - ts) + tint_g * ts

    out_g = user_g * mask_g + bg_g * (1.0 - mask_g)
    cp.clip(out_g, 0, 255, out=out_g)
    return out_g.astype(cp.uint8).get()

_cpu_buf = None
_cpu_buf_shape = ()

def _composite_cpu(
    user_frame: np.ndarray,
    bg_frame:   np.ndarray,
    soft_mask:  np.ndarray,
    tint_color: tuple | None,
    tint_strength: float,
) -> np.ndarray:
    global _cpu_buf, _cpu_buf_shape

    h, w = user_frame.shape[:2]

    if _cpu_buf_shape != (h, w):
        _cpu_buf       = np.empty((h, w, 3), dtype=np.float32)
        _cpu_buf_shape = (h, w)

    user_f = user_frame.astype(np.float32)
    bg_f   = bg_frame.astype(np.float32)
    mask_f = soft_mask[:, :, np.newaxis]

    if tint_color is not None and tint_strength > 0.0:
        tint_arr = np.array(tint_color, dtype=np.float32).reshape(1, 1, 3)
        np.multiply(user_f, 1.0 - tint_strength, out=user_f)
        user_f += tint_arr * tint_strength

    np.multiply(user_f, mask_f, out=_cpu_buf)
    np.add(_cpu_buf, bg_f * (1.0 - mask_f), out=_cpu_buf)
    np.clip(_cpu_buf, 0, 255, out=_cpu_buf)

    return _cpu_buf.astype(np.uint8)
