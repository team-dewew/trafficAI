"""Configuration for rules and heuristics thresholds."""

RULES = {
    "red_light": {
        "min_duration": 0.5,
    },
    "stop_line": {
        "min_duration": 0.5,
        "speed_rel_thresh": 0.05,  # Speed relative to bbox diagonal
    },
    "jaywalking": {
        "min_duration": 1.0,
    },
    "failure_to_yield": {
        "min_duration": 0.5,
    },
    "wrong_way": {
        "min_duration": 1.5,
        "dx_rel_thresh": 0.3, # Displacement dx relative to diag
    },
    "solid_line_crossing": {
        "min_duration": 0.5,
    },
    "stopped_vehicle": {
        "min_duration": 10.0,
        "speed_rel_thresh": 0.05,
    },
    "illegal_turn": {
        "min_duration": 0.5,
    },
    "illegal_u_turn": {
        "min_duration": 0.5,
    },
    "congestion": {
        "min_duration": 5.0,
        "speed_rel_thresh": 0.05,
        "min_vehicles": 4,
    },
    "road_obstacle": {
        "min_duration": 1.0,
        "speed_rel_thresh": 0.05,
    },
    "near_miss": {
        "min_duration": 0.8,
        "ttc_thresh": 1.5,
        "proximity_diag_ratio": 0.85,
    },
    "accident": {
        "min_duration": 1.0,
    },
    "fire_smoke": {
        "min_duration": 1.0,
    }
}

ENABLED_CLASSES = [
    "accident", "near_miss", "red_light", "wrong_way", "illegal_u_turn",
    "stopped_vehicle", "jaywalking", "failure_to_yield", "illegal_turn",
    "solid_line_crossing", "stop_line", "congestion", "road_obstacle",
    "fire_smoke"
]

def seed_everything(seed=42):
    import random
    import os
    import numpy as np
    import torch
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
