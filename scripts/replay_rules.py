"""Development tool: cache perception output once, then re-run only the rules.

    python scripts/replay_rules.py cache  samples/C3896.MP4 [...] --cache-dir cache/
    python scripts/replay_rules.py replay samples/C3896.MP4 [...] --cache-dir cache/ [--out preds.json]

`replay` prints per-class event counts and optionally writes a predictions file
(same shape as run_submission.py output, events only) for evaluate.py.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402

from src.events import replay_rules, run_part_a  # noqa: E402


def _duration(path: str) -> float:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n / fps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["cache", "replay"])
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    preds = {"team": "dewew-dev", "videos": {}}
    for video in args.videos:
        name = Path(video).name
        cache_file = cache_dir / f"{Path(video).stem}.pkl"
        if args.mode == "cache":
            sink: list = []
            t0 = time.perf_counter()
            events, diag = run_part_a(video, obs_sink=sink)
            with open(cache_file, "wb") as f:
                pickle.dump({"obs": sink, "diag": diag}, f)
            print(f"{name}: cached {len(sink)} frames in {time.perf_counter() - t0:.0f}s; diag={json.dumps(diag)[:300]}")
        else:
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
            events = replay_rules(video, data["obs"], _duration(video))
        preds["videos"][name] = {"events": events, "risk": []}
        print(f"{name}: {len(events)} events {dict(Counter(e[2] for e in events))}")
    if args.out:
        Path(args.out).write_text(json.dumps(preds, indent=1))


if __name__ == "__main__":
    main()
