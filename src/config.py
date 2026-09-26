"""Every tunable constant of the pipeline in one place.

Speeds are in body-diagonals per second ("diag/s") so the same threshold works
for a car next to the camera and one 150 m away, and at any video resolution.
Durations are in seconds of video time.
"""
from __future__ import annotations

SEED = 42

OFFICIAL_CLASSES = [
    "accident", "near_miss", "red_light", "wrong_way", "illegal_u_turn", "stopped_vehicle",
    "jaywalking", "failure_to_yield", "illegal_turn", "solid_line_crossing", "stop_line",
    "congestion", "road_obstacle", "fire_smoke",
]

PERCEPTION = {
    "detector": "yolo11l.pt",
    "anomaly_model": "accident_model.pt",
    "imgsz": 960,              # detector input size (4K frame is letterboxed to this)
    "conf": 0.25,
    "dup_iou": 0.6,            # car/bus/truck boxes overlapping more than this are one vehicle
    "stride": 3,               # Part A runs perception on every 3rd frame (~10 Hz)
    "anomaly_every_sec": 1.0,
    "budget_factor": 1.6,      # if Part A's frame loop runs slower than this x real time, stride is doubled
}

# The traffic signal is read from its lamps (see src/traffic_light.py).
# Thresholds from the 4 sample videos, 1 sample/s (day and dusk): an unlit lamp
# scores <= 10 (red) / <= 17 (green); a lit one >= 45 (red) / >= 126 (green).
SIGNAL = {
    "lamp_radius_4k": 9,       # search half-window around each lamp centre (reference px)
    "red_on": 30.0,            # top-k mean of R - max(G, B) inside the red lamp window
    "green_on": 60.0,          # top-k mean of G - R inside the green lamp window
    "yellow_on": 60.0,         # top-k mean of min(R, G) - B inside the yellow lamp window
    "hold_sec": 0.4,           # a new state must persist this long before it is accepted
    "unknown_after_sec": 3.0,  # no lamp readable for this long -> UNKNOWN
}

RULES = {
    "jaywalking": {
        "min_duration": 1.5, "gap": 1.0,
        "road_margin_diag": 0.10,    # must be this far inside the carriageway
        "safe_margin_diag": 0.4,     # within ~1 m of a zebra / island / kerb still counts as crossing
    },
    "failure_to_yield": {
        "min_duration": 0.4,
        "ped_min_age": 0.5,          # pedestrian present on the zebra at least this long
        "vehicle_min_speed": 0.4,    # diag/s: the vehicle is driving through, not waiting
        "ped_inside_diag": 0.25,     # pedestrian at least this far inside the zebra (not at the kerb)
        "ahead_diag": 1.5,           # pedestrian ahead of the vehicle along its path ...
        "lateral_diag": 0.6,         # ... and within its swept width
    },
    "congestion": {
        "min_duration": 8.0, "gap": 3.0,
        "min_vehicles": 4,
        "crawl_speed": 0.12,         # median diag/s below this = standstill / crawling
        "min_green_sec": 12.0,       # lane_ltr: must persist through >= this much green
    },
    "red_light": {
        "min_red_age": 1.0,          # red must have been on this long (excludes amber->red edge)
        "min_speed": 0.3,
        "confirm_sec": 5.0,          # must drive into the junction within this time after the line
        "early_start_sec": 2.0,      # crossing <= this long before green = anticipating the green
        "max_duration": 15.0,
    },
    "stop_line": {
        "min_duration": 1.0, "gap": 0.6,
        "stopped_speed": 0.08,
    },
    "stopped_vehicle": {
        "min_duration": 10.0, "gap": 1.5,
        "stopped_speed": 0.06,
        "moving_speed": 0.5,         # a track must have driven at least this fast once (else: parked)
        "crossing_clear_diag": 1.0,  # stops closer than this to a zebra are yielding / queueing
        "queue_ahead_diag": 1.8,     # a stopped vehicle this close ahead means "queue"
    },
    "wrong_way": {
        "min_duration": 1.5, "gap": 1.0,
        "min_speed": 0.4,
        "max_cos": -0.6,             # heading vs lane flow
    },
    "near_miss": {
        "min_duration": 1.0,
        "close_diag": 0.8,           # ground-point gap / larger diagonal
        "clear_diag": 1.5,
        "brake_ratio": 0.4,          # speed over last 0.5 s < ratio * speed over the second before
        "brake_min_speed": 1.0,      # diag/s before braking
    },
    "accident": {
        "min_duration": 1.5, "gap": 2.0,
        "conf": 0.6,
        "hits": 3, "window": 4,      # >= hits positive checks among the last `window`
    },
    "fire_smoke": {
        "min_duration": 2.0, "gap": 2.0,
        "conf": 0.6,
        "hits": 3, "window": 4,
    },
    "road_obstacle": {
        "min_duration": 2.0, "gap": 1.0,
        "max_speed": 0.3,
    },
}

# Classes emitted by detect_events. A predicted class that never occurs in the
# test set enters macro-F1 as a 0, so classes without a trustworthy rule stay off
# (see docs/class_policy.md for the reasoning behind each decision).
ENABLED_CLASSES = [
    "red_light",
    "stop_line",
    "jaywalking",
    "failure_to_yield",
    "stopped_vehicle",
    "congestion",
    "wrong_way",
    "near_miss",
    "accident",
    "fire_smoke",
    "road_obstacle",
]

# Merge gap for same-class segments in post-processing.
MERGE_GAP = {"congestion": 3.0, "stopped_vehicle": 2.0, "jaywalking": 1.0}
DEFAULT_MERGE_GAP = 0.5
MIN_SEGMENT = 0.5


def seed_everything(seed: int = SEED) -> None:
    import os
    import random

    import numpy as np
    import torch

    random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
