from ultralytics import YOLO
from src.paths import WEIGHTS_DIR

_MODEL_CACHE: dict[str, YOLO] = {}

def _load_yolo(name: str) -> YOLO:
    """Load (and memoize) a YOLO model so folders of videos don't reload weights."""
    weight_path = WEIGHTS_DIR / name
    if not weight_path.exists():
        raise FileNotFoundError(f"Weight file {name} not found. Please run: bash weights/download.sh (or python weights/download.py on Windows)")
    
    path_str = str(weight_path)
    if path_str not in _MODEL_CACHE:
        # Load with half=True if CUDA is available, as suggested in Stage 6 (but we can add half later).
        # We will keep it simple for now as per stage 1.
        _MODEL_CACHE[path_str] = YOLO(path_str)
    return _MODEL_CACHE[path_str]
