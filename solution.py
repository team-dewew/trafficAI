"""
solution.py — the interface the organizers' harness (run_submission.py) imports.

    detect_events(video_path)  -> [[start_sec, end_sec, label], ...]    # Part A
    RiskEstimator().reset(meta); .step(frame, t_sec) -> float           # Part B

The implementation lives in src/:
    src/registration.py   per-video alignment of the scene layout (SIFT, similarity)
    src/scene.py          hand-calibrated scene layout of the camera
    src/traffic_light.py  lamp-based signal state
    src/perception.py     YOLO11-L detector + duplicate suppression + ByteTrack
    src/rules.py          event rules (one method per class)
    src/postprocess.py    merging / clipping of segments
    src/events.py         Part A pipeline
    src/risk.py           Part B causal risk estimator
    src/config.py         every threshold, the enabled classes and the seed
"""
from __future__ import annotations

import traceback
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np  # noqa: E402

from src.config import seed_everything  # noqa: E402

seed_everything()

from src.events import run_part_a  # noqa: E402
from src.risk import RiskCore  # noqa: E402


def _load_models() -> None:
    """Load every model once, when the harness imports this module, so that loading
    (a few seconds, mostly the 1B verifier) is not charged to the first video's time budget.
    Missing weights are reported by the first detect_events call instead."""
    from src.config import PERCEPTION
    from src.models import load_yolo

    try:
        for name in (PERCEPTION["detector"], PERCEPTION["anomaly_model"], "yolov8n.pt"):
            load_yolo(name)
        if PERCEPTION.get("use_vlm"):
            from src.vlm import get_verifier

            get_verifier()
    except Exception as exc:  # noqa: BLE001
        print(f"[solution] model preload skipped: {type(exc).__name__}: {exc}")


_load_models()

# Official class ids (14). Classes switched off in src/config.py:ENABLED_CLASSES
# are simply never emitted.
CLASSES: list[str] = [
    "accident",            # collision between road users / with a fixed object
    "near_miss",           # sharp braking or swerving to avoid a collision, no contact
    "red_light",           # crossing the stop line on red
    "wrong_way",           # driving against the traffic direction / in the oncoming lane
    "illegal_u_turn",      # U-turn where prohibited
    "stopped_vehicle",     # stationary on the carriageway >= 10 s, not queued at a signal
    "jaywalking",          # pedestrian on the carriageway outside a crossing
    "failure_to_yield",    # driving through a crossing while a pedestrian is on it
    "illegal_turn",        # turn from the wrong lane or in a prohibited direction
    "solid_line_crossing", # lane change / manoeuvre across a solid marking
    "stop_line",           # stopped past the stop line on red
    "congestion",          # standstill / crawling traffic across all lanes of a direction
    "road_obstacle",       # debris, animal or fallen object on the carriageway
    "fire_smoke",          # visible fire or smoke from a vehicle or on the road
]

RISK_HORIZON_SEC = 5.0


def detect_events(video_path: str, progress_callback=None) -> list[list]:
    """Part A. Return [[start_sec, end_sec, label], ...] for one .mp4.

    `progress_callback(frame_idx, n_frames)` is optional (used by the website).
    """
    events, diag = run_part_a(video_path, progress_callback=progress_callback)
    vlm = diag.get("vlm") or {}
    if vlm.get("enabled"):
        ps = [q["p"] for q in vlm["log"] if q.get("p") is not None]
        print(f"[vlm] {len(ps)} questions in {vlm['sec']}s, max p(yes) {max(ps, default=0.0):.2f}, "
              f"top: {sorted(((q['p'], q['kind'], q['t']) for q in vlm['log'] if q.get('p') is not None), reverse=True)[:3]}")
    return events


class RiskEstimator:
    """Part B. Causal: step() sees frames in order and nothing else."""

    def reset(self, meta: dict) -> None:
        # meta = {"video_id", "fps", "width", "height", "n_frames"}
        self._core = RiskCore()
        self._core.reset(meta)

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        # frame: BGR uint8 (H, W, 3). Return P(accident starts within 5 s) in [0, 1].
        try:
            return self._core.step(frame, t_sec)
        except Exception:  # one bad frame must not wipe the whole risk curve
            if not getattr(self, "_warned", False):
                self._warned = True
                traceback.print_exc()
            return float(getattr(self._core, "ema", 0.0))
