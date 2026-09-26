"""Website live demo: the submission pipeline in a CPU-friendly configuration.

Same registration, signal read-out, tracking, rules and post-processing as
`detect_events`, with a lighter perception setting so that a free CPU host can
answer in minutes:

    detector  YOLO11-S @768 (submission: YOLO11-L @960)
    rate      every 6th frame (submission: every 3rd)
    anomaly   model off (submission: YOLOv8x at 1 Hz)

Part B's risk is computed from the same causal detections (RiskCore._score), so
the video is decoded once. Annotated clips are drawn from the stored tracks
instead of running the detector again.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from src.annotate import SIGNAL_BGR, _active_labels, _transcode_h264, draw_scene
from src.events import run_part_a
from src.risk import RiskCore

DEMO_SETTINGS = {
    "detector": "yolo11s.pt",
    "imgsz": 768,
    "stride": 6,
    "use_anomaly": False,
    "budget_factor": 1e9,      # no time guard in the demo; progress is shown instead
    "max_width": 1280,         # frames are downscaled right after decoding (RAM on small servers)
}
DEMO_MAX_SEC = 120.0
DEMO_MAX_MB = 300
MAX_CLIPS = 5
CLIP_MAX_SEC = 12.0
CLIP_WIDTH = 960


@dataclass
class DemoResult:
    events: list[list]
    risk: list[tuple[float, float]]
    clips: list[tuple[str, str]] = field(default_factory=list)   # (path, caption)
    duration: float = 0.0
    elapsed: float = 0.0
    info: dict = field(default_factory=dict)


def _clip_windows(events: list[list], duration: float) -> list[tuple[float, float, list[str]]]:
    """One clip per event (merged with neighbours only while it stays <= CLIP_MAX_SEC),
    starting 1.5 s before the event so its onset is visible."""
    wins: list[list] = []
    for s, e, lbl in sorted(events, key=lambda x: x[0]):
        cs, ce = max(0.0, s - 1.5), min(duration, e + 1.5, s - 1.5 + CLIP_MAX_SEC)
        if wins and cs - wins[-1][1] < 1.0 and max(ce, wins[-1][1]) - wins[-1][0] <= CLIP_MAX_SEC:
            wins[-1][1] = max(wins[-1][1], ce)
            wins[-1][2].add(lbl)
        else:
            wins.append([cs, ce, {lbl}])
    return [(cs, ce, sorted(lbls)) for cs, ce, lbls in wins[:MAX_CLIPS]]


def render_clips(video_path: str, events: list[list], obs: list, scene: dict, signal, duration: float,
                 out_dir: Path, tag: str, progress=None, frame_scale: float = 1.0) -> list[tuple[str, str]]:
    """Annotated clips around the events, drawn from the stored tracks (no detector)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    times = np.array([o.t for o, _ in obs]) if obs else np.zeros(0)
    cap = cv2.VideoCapture(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    clips = []
    wins = _clip_windows(events, duration)
    for k, (cs, ce, labels) in enumerate(wins):
        path = out_dir / f"{tag}_clip{k}.mp4"
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(cs * fps))
        writer = None
        idx = int(cs * fps)
        while idx / fps <= ce:
            ok, frame = cap.read()
            if not ok:
                break
            t = idx / fps
            idx += 1
            if idx % 2:                       # 12-15 fps output is enough for review
                continue
            if frame_scale < 1.0:             # same pixel grid as the scene and the stored boxes
                frame = cv2.resize(frame, None, fx=frame_scale, fy=frame_scale, interpolation=cv2.INTER_AREA)
            img = draw_scene(frame, scene)
            if len(times):
                o = obs[int(np.clip(np.searchsorted(times, t, side="right") - 1, 0, len(obs) - 1))][0]
                for tid, box, name in zip(o.tids, o.boxes, o.names):
                    x1, y1, x2, y2 = (int(v) for v in box)
                    col = (0, 200, 255) if name == "pedestrian" else (255, 180, 0)
                    cv2.rectangle(img, (x1, y1), (x2, y2), col, max(2, int(3 * scene["px_scale"])))
            h, w = img.shape[:2]
            out_h = int(round(h * CLIP_WIDTH / w)) // 2 * 2
            disp = cv2.resize(img, (CLIP_WIDTH, out_h), interpolation=cv2.INTER_AREA)
            sig = signal.state_at(t)
            cv2.rectangle(disp, (8, 8), (300, 56), (12, 12, 12), -1)
            cv2.putText(disp, f"t={t:6.1f}s", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(disp, f"signal: {sig}", (16, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, SIGNAL_BGR[sig], 1, cv2.LINE_AA)
            active = _active_labels(events, t)
            if active:
                text = "EVENT: " + ", ".join(active)
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(disp, (8, out_h - th - 24), (16 + tw, out_h - 8), (0, 0, 200), -1)
                cv2.putText(disp, text, (13, out_h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            if writer is None:
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps / 2, (CLIP_WIDTH, out_h))
            writer.write(disp)
        if writer is not None:
            writer.release()
            _transcode_h264(str(path))
            clips.append((str(path), f"{cs:.1f}-{ce:.1f} s: {', '.join(labels)}"))
        if progress:
            progress((k + 1) / max(1, len(wins)))
    cap.release()
    return clips


def run_demo(video_path: str, clip_dir: Path, tag: str, progress=None) -> DemoResult:
    """Run Part A + Part B on one uploaded clip. `progress(fraction, message)` is optional."""
    t0 = time.perf_counter()
    say = progress or (lambda f, m: None)
    obs: list = []
    say(0.02, "Registering the scene layout onto this video ...")
    events, diag = run_part_a(
        video_path,
        progress_callback=lambda i, n: say(0.02 + 0.78 * i / max(1, n), f"Part A: frame {i} / {n}"),
        obs_sink=obs,
        settings=DEMO_SETTINGS,
    )
    say(0.82, "Part B: risk curve from the same causal tracks ...")
    core = RiskCore()
    core.reset({"fps": 25.0}, with_model=False)
    core.scene = diag["scene"]
    risk = [(round(o.t, 2), round(core._score(o.t, o.tids, o.boxes, o.names), 4)) for o, _ in obs]
    clips = []
    if events:
        clips = render_clips(video_path, events, obs, diag["scene"], diag["signal"], diag["duration"], clip_dir, tag,
                             progress=lambda f: say(0.85 + 0.14 * f, "Rendering annotated event clips ..."),
                             frame_scale=diag.get("frame_scale", 1.0))
    say(1.0, "Done")
    info = {"registration": diag.get("registration"), "signal_phases": diag.get("signal_phases"),
            "frames_processed": len(obs)}
    return DemoResult(events, risk, clips, diag["duration"], time.perf_counter() - t0, info)
