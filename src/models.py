"""Model loading: every weight file is loaded once per process and reused across videos."""
from __future__ import annotations

import torch
from ultralytics import YOLO

from src.paths import WEIGHTS_DIR

_MODEL_CACHE: dict[str, YOLO] = {}

USE_HALF = torch.cuda.is_available()   # FP16 inference on GPU, FP32 on CPU
DEVICE = 0 if torch.cuda.is_available() else "cpu"


def load_yolo(name: str) -> YOLO:
    """Load (and memoize) a YOLO model from weights/. Never downloads anything."""
    weight_path = WEIGHTS_DIR / name
    if not weight_path.exists():
        raise FileNotFoundError(
            f"Weight file {weight_path} not found. Run `bash weights/download.sh` "
            f"(or `python weights/download.py`) once before evaluation."
        )
    key = str(weight_path)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = YOLO(key)
    return _MODEL_CACHE[key]


def predict(model: YOLO, frame, **kwargs):
    """Single-image inference with the project-wide device/precision settings."""
    return model.predict(frame, verbose=False, quantize=16 if USE_HALF else None, device=DEVICE, **kwargs)[0]


# Backwards-compatible alias used by older tooling.
_load_yolo = load_yolo
