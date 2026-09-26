"""Render one or two example frames for every event class detected on the samples.

The frame is taken inside the event, with the scene layout drawn and the road
users that triggered the rule outlined in red. Output: assets/examples/*.jpg and
assets/examples/index.json (used by the website's Results page).

    python scripts/replay_rules.py cache samples/*.MP4 --cache-dir cache   # once (perception)
    python scripts/make_examples.py samples/*.MP4 --cache-dir cache
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.annotate import draw_scene  # noqa: E402
from src.events import open_scene  # noqa: E402
from src.rules import RuleEngine  # noqa: E402
from src.traffic_light import SignalState  # noqa: E402

OUT = Path("assets/examples")
PER_CLASS = 2


def involved_ids(label: str, key) -> set[int]:
    if isinstance(key, tuple):
        return {int(key[0])} if label == "failure_to_yield" else {int(k) for k in key}
    return {int(key)} if isinstance(key, (int, np.integer)) else set()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--cache-dir", default="cache")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    candidates = []                          # (label, duration, video, start, end, key, obs_list, scene, signal)
    for video in args.videos:
        data = pickle.load(open(Path(args.cache_dir) / f"{Path(video).stem}.pkl", "rb"))
        scene, _ = open_scene(video)
        signal = SignalState()
        engine = RuleEngine(scene, signal)
        obs_list = []
        for obs, raw in data["obs"]:
            signal.update(obs.t, raw)
            engine.update(obs)
            obs_list.append(obs)
        cap = cv2.VideoCapture(video)
        duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 25.0)
        cap.release()
        final = engine.finalize(duration)
        for s0, e0, label, key in engine.raw:
            if any(lbl == label and abs(s - s0) < 0.6 for s, e, lbl in final):
                candidates.append((label, e0 - s0, video, s0, e0, key, obs_list, scene, signal))

    index = []
    for label in sorted({c[0] for c in candidates}):
        picks = sorted((c for c in candidates if c[0] == label), key=lambda c: -c[1])
        seen_videos: set[str] = set()
        chosen = []
        for c in picks:                          # prefer examples from different videos
            if c[2] not in seen_videos:
                chosen.append(c)
                seen_videos.add(c[2])
            if len(chosen) == PER_CLASS:
                break
        for k, (lbl, _, video, s0, e0, key, obs_list, scene, signal) in enumerate(chosen):
            t = s0 + 0.4 * (e0 - s0)
            obs = min(obs_list, key=lambda o: abs(o.t - t))
            cap = cv2.VideoCapture(video)
            cap.set(cv2.CAP_PROP_POS_MSEC, obs.t * 1000.0)
            ok, frame = cap.read()
            cap.release()
            if not ok:
                continue
            ids = involved_ids(lbl, key)
            img = draw_scene(frame.copy(), scene)
            hi = []
            for tid, b, name in zip(obs.tids, obs.boxes, obs.names):
                on = int(tid) in ids
                col = (0, 0, 255) if on else ((0, 200, 255) if name == "pedestrian" else (255, 180, 0))
                cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), col, 8 if on else 2)
                if on:
                    hi.append(b)
            h, w = img.shape[:2]
            if hi:
                cx = float(np.mean([(b[0] + b[2]) / 2 for b in hi]))
                cy = float(np.mean([(b[1] + b[3]) / 2 for b in hi]))
                cw, ch = int(w * 0.40), int(h * 0.45)
                x0 = int(np.clip(cx - cw / 2, 0, w - cw))
                y0 = int(np.clip(cy - ch / 2, 0, h - ch))
                img = img[y0:y0 + ch, x0:x0 + cw]
            img = cv2.resize(img, (800, int(800 * img.shape[0] / img.shape[1])), interpolation=cv2.INTER_AREA)
            name = f"{lbl}_{k}.jpg"
            cv2.imwrite(str(OUT / name), img, [cv2.IMWRITE_JPEG_QUALITY, 82])
            index.append({"label": lbl, "file": f"{OUT.as_posix()}/{name}", "video": Path(video).name,
                          "start": round(s0, 1), "end": round(e0, 1), "signal": signal.state_at(obs.t)})
            print(f"{name}: {Path(video).name} {s0:.1f}-{e0:.1f}s")
    (OUT / "index.json").write_text(json.dumps(index, indent=1))


if __name__ == "__main__":
    main()
