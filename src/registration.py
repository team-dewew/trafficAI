"""Per-video scene registration.

The scene layout in ``src/scene.py`` was drawn on the first frame of the
reference clip (stored as ``assets/scene_ref.jpg`` at 1920x1080). The sample
videos show that the camera pose drifts slightly between recordings (up to
~90 px of shift and ~1 degree of rotation at 4K), which is enough to move a stop
line or the traffic-light lamps off their calibrated positions.

``estimate_scene_transform`` matches SIFT features between the reference frame
and a few frames of the video and returns a 2x3 similarity transform that maps
reference 4K coordinates to the video's native pixel coordinates. Anything that
looks implausible falls back to a pure resolution scale (no drift correction).
"""
from __future__ import annotations

import cv2
import numpy as np

from src.paths import REPO_ROOT

REF_IMAGE_PATH = REPO_ROOT / "assets" / "scene_ref.jpg"
REF_W, REF_H = 3840, 2160          # coordinate system of SCENE_CONFIG
WORK_W, WORK_H = 1920, 1080        # resolution used for feature matching

MIN_INLIERS = 40
MAX_SHIFT_PX = 400.0               # in reference (4K) pixels
MAX_ROT_DEG = 5.0
SCALE_RANGE = (0.9, 1.1)

_REF_CACHE: dict[str, tuple] = {}
_CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))


def _prep(frame: np.ndarray) -> np.ndarray:
    small = cv2.resize(frame, (WORK_W, WORK_H), interpolation=cv2.INTER_AREA)
    return _CLAHE.apply(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))


def _ref_features():
    if "ref" not in _REF_CACHE:
        ref = cv2.imread(str(REF_IMAGE_PATH))
        if ref is None:
            raise FileNotFoundError(f"missing scene reference image {REF_IMAGE_PATH}")
        sift = cv2.SIFT_create(nfeatures=6000)
        kp, des = sift.detectAndCompute(_prep(ref), None)
        _REF_CACHE["ref"] = (kp, des)
    return _REF_CACHE["ref"]


def _scale_only(width: int, height: int) -> np.ndarray:
    return np.array([[width / REF_W, 0.0, 0.0], [0.0, height / REF_H, 0.0]], dtype=np.float64)


def _match_one(frame: np.ndarray) -> tuple[np.ndarray, int] | None:
    """Similarity transform ref_work -> video_work for one frame, or None."""
    kp_r, des_r = _ref_features()
    sift = cv2.SIFT_create(nfeatures=6000)
    kp_v, des_v = sift.detectAndCompute(_prep(frame), None)
    if des_v is None or len(kp_v) < MIN_INLIERS:
        return None
    pairs = cv2.BFMatcher().knnMatch(des_r, des_v, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.7 * n.distance]
    if len(good) < MIN_INLIERS:
        return None
    src = np.float32([kp_r[m.queryIdx].pt for m in good])
    dst = np.float32([kp_v[m.trainIdx].pt for m in good])
    A, inliers = cv2.estimateAffinePartial2D(src, dst, ransacReprojThreshold=2.0, maxIters=4000, confidence=0.999)
    if A is None or inliers is None:
        return None
    return A, int(inliers.sum())


def _plausible(A_work: np.ndarray) -> bool:
    scale = float(np.hypot(A_work[0, 0], A_work[1, 0]))
    rot = float(np.degrees(np.arctan2(A_work[1, 0], A_work[0, 0])))
    shift = float(np.hypot(A_work[0, 2], A_work[1, 2])) * (REF_W / WORK_W)
    return SCALE_RANGE[0] <= scale <= SCALE_RANGE[1] and abs(rot) <= MAX_ROT_DEG and shift <= MAX_SHIFT_PX


def estimate_scene_transform(frames: list[np.ndarray]) -> tuple[np.ndarray, dict]:
    """Return (A, info): A maps reference 4K coords -> native video pixel coords."""
    if not frames:
        raise ValueError("need at least one frame")
    h, w = frames[0].shape[:2]
    fallback = _scale_only(w, h)
    estimates = []
    for f in frames:
        res = _match_one(f)
        if res is not None and res[1] >= MIN_INLIERS and _plausible(res[0]):
            estimates.append(res)
    if not estimates:
        return fallback, {"status": "fallback", "inliers": 0}

    # Element-wise median over the per-frame estimates is robust to one bad frame.
    A_work = np.median(np.stack([a for a, _ in estimates]), axis=0)
    # ref 4K -> ref work -> video work -> video native
    s_in = np.diag([WORK_W / REF_W, WORK_H / REF_H, 1.0])
    s_out = np.diag([w / WORK_W, h / WORK_H, 1.0])
    A3 = np.vstack([A_work, [0.0, 0.0, 1.0]])
    A = (s_out @ A3 @ s_in)[:2]
    info = {
        "status": "ok",
        "inliers": int(np.median([n for _, n in estimates])),
        "dx_4k": round(float(A_work[0, 2] * REF_W / WORK_W), 1),
        "dy_4k": round(float(A_work[1, 2] * REF_H / WORK_H), 1),
        "scale": round(float(np.hypot(A_work[0, 0], A_work[1, 0])), 4),
        "rot_deg": round(float(np.degrees(np.arctan2(A_work[1, 0], A_work[0, 0]))), 3),
    }
    return A, info


def transform_points(points: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Apply a 2x3 affine to an (N, 2) array of points."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return pts @ A[:, :2].T + A[:, 2]


def sample_frames(video_path: str, times_sec: tuple[float, ...] = (0.0, 2.0, 4.0)) -> list[np.ndarray]:
    """Read a few frames for registration (random access; the video is closed after)."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    try:
        for t in times_sec:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, f = cap.read()
            if ok and f is not None:
                frames.append(f)
    finally:
        cap.release()
    return frames
