"""Print the signal lamp scores and a 1-sample-per-second state timeline per video.

This is how the thresholds in src/config.py:SIGNAL were chosen:
    python scripts/signal_timeline.py samples/*.MP4
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.events import open_scene  # noqa: E402
from src.traffic_light import classify, lamp_scores  # noqa: E402

SYMBOL = {"RED": "R", "GREEN": "G", "YELLOW": "Y", "UNKNOWN": "."}


def main(paths: list[str]) -> None:
    for path in paths:
        scene, info = open_scene(path)
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        scores, states, idx = [], [], 0
        while True:                              # sequential read (seeking 4K H.264 is slow)
            ok = cap.grab()
            if not ok:
                break
            if idx % int(round(fps)) == 0:
                _, frame = cap.retrieve()
                sc = lamp_scores(frame, scene["main_signal_lamps"], scene["px_scale"])
                scores.append(sc)
                states.append(classify(sc))
            idx += 1
        cap.release()
        red = np.array([s["red"] for s in scores])
        green = np.array([s["green"] for s in scores])
        print(f"{Path(path).name}  registration={info}")
        print(f"  red   p5/p50/p95 = {np.percentile(red, [5, 50, 95]).round(0)}")
        print(f"  green p5/p50/p95 = {np.percentile(green, [5, 50, 95]).round(0)}")
        print("  " + "".join(SYMBOL[s] for s in states))


if __name__ == "__main__":
    main(sys.argv[1:])
