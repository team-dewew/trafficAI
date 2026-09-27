#!/usr/bin/env python3
"""
annotate.py — annotated-video renderer for the Traffic AI pipeline.

Draws the registered scene layout, tracked road users, the signal state read
from the lamps, a time HUD and an active-event banner, and writes an .mp4.
It uses the same registration / perception / signal code as Part A, so what
the video shows is what the rules saw. Used for:

  1. samples/previews/*_preview.mp4 — annotated sample videos (website Results)
  2. live-demo event clips          — short clips around each detected event

CLI:
    python -m src.annotate --video samples/C3896.MP4 --out samples/previews/C3896_preview.mp4 \
        --events predictions_samples.json [--stride 2] [--width 960]
    python -m src.annotate --video upload.mp4 --out clip.mp4 --start 10 --end 20
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.events import open_scene  # noqa: E402
from src.perception import Perception  # noqa: E402
from src.traffic_light import SignalState, classify, lamp_scores  # noqa: E402

ZONE_STYLE = [  # (scene key, BGR colour, thickness)
    ("lane_ltr", (120, 200, 120), 2), ("lane_rtl", (200, 160, 120), 2),
    ("intersection_core", (170, 170, 170), 2), ("right_turn_zone", (170, 170, 170), 2),
    ("lower_core", (170, 170, 170), 2),
]
MULTI_STYLE = [("crosswalks", (0, 220, 0), 3), ("ped_refuge", (200, 0, 200), 2),
               ("barriers", (60, 60, 230), 2), ("sidewalks", (230, 230, 0), 2)]
LINE_STYLE = [("stop_line_strict", (0, 0, 255)), ("stop_line_tolerance", (0, 165, 255)),
              ("yield_ped_line", (0, 255, 255))]
SIGNAL_BGR = {"RED": (0, 0, 255), "GREEN": (0, 220, 0), "YELLOW": (0, 200, 255), "UNKNOWN": (160, 160, 160)}


def _active_labels(events: list[list] | None, t_sec: float) -> list[str]:
    """Labels of events covering timestamp t_sec."""
    if not events:
        return []
    return sorted({str(lbl) for s, e, lbl in events if float(s) <= t_sec <= float(e)})


def draw_scene(img: np.ndarray, scene: dict) -> np.ndarray:
    for key, col, th in ZONE_STYLE:
        cv2.polylines(img, [scene[key].reshape(-1, 1, 2)], True, col, th, cv2.LINE_AA)
    for key, col, th in MULTI_STYLE:
        for poly in scene[key]:
            cv2.polylines(img, [poly.reshape(-1, 1, 2)], True, col, th, cv2.LINE_AA)
    for key, col in LINE_STYLE:
        p = scene[key]
        cv2.line(img, tuple(int(v) for v in p[0]), tuple(int(v) for v in p[1]), col, 5, cv2.LINE_AA)
    return img


def render_annotated(
    video_path: str,
    out_path: str,
    events: list[list] | None = None,
    width: int = 960,
    stride: int = 1,
    start_sec: float = 0.0,
    end_sec: float | None = None,
    progress_callback=None,
) -> str:
    """Render an annotated copy of `video_path` (optionally only [start_sec, end_sec])."""
    scene, _ = open_scene(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start_f = max(0, int(start_sec * fps))
    stop_f = total - 1 if end_sec is None else min(total - 1, int(end_sec * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

    out_w = int(width)
    out_h = int(round(src_h * out_w / src_w))
    out_h -= out_h % 2
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), max(1.0, fps / stride), (out_w, out_h))

    perception = Perception(fps, stride, use_anomaly=False)
    signal = SignalState()
    idx, written = start_f, 0
    try:
        while idx <= stop_f:
            if (idx - start_f) % stride != 0:
                if not cap.grab():
                    break
                idx += 1
                continue
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            t = idx / fps
            sig = signal.update(t, classify(lamp_scores(frame, scene["main_signal_lamps"], scene["px_scale"])))
            obs = perception(frame, t)

            ann = draw_scene(frame.copy(), scene)
            for (lx, ly) in scene["main_signal_lamps"].values():
                cv2.circle(ann, (int(lx), int(ly)), int(14 * scene["px_scale"]), SIGNAL_BGR[sig], 3)
            for tid, box, name in zip(obs.tids, obs.boxes, obs.names):
                x1, y1, x2, y2 = (int(v) for v in box)
                col = (0, 200, 255) if name == "pedestrian" else (255, 180, 0)
                cv2.rectangle(ann, (x1, y1), (x2, y2), col, 3)
                cv2.putText(ann, f"#{tid} {name}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2, cv2.LINE_AA)

            disp = cv2.resize(ann, (out_w, out_h), interpolation=cv2.INTER_AREA)
            cv2.rectangle(disp, (8, 8), (330, 58), (12, 12, 12), -1)
            cv2.putText(disp, f"t={t:6.1f}s  tracks={len(obs.tids)}", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(disp, f"signal: {sig}", (16, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, SIGNAL_BGR[sig], 1, cv2.LINE_AA)
            active = _active_labels(events, t)
            if active:
                text = "EVENT: " + ", ".join(active)
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                by = out_h - 14
                cv2.rectangle(disp, (8, by - th - 10), (16 + tw, by + 6), (0, 0, 200), -1)
                cv2.putText(disp, text, (13, by), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(disp)
            written += 1
            idx += 1
            if progress_callback and written % 30 == 0:
                progress_callback(min(1.0, (idx - start_f) / max(1, stop_f - start_f)))
    finally:
        writer.release()
        cap.release()
    _transcode_h264(out_path)
    return out_path


def _transcode_h264(path: str, crf: int = 28) -> None:
    """Re-encode the mp4v render to H.264 (yuv420p) for browser playback; keeps
    the original if the bundled ffmpeg (imageio-ffmpeg) is unavailable."""
    try:
        import imageio_ffmpeg

        src = Path(path)
        tmp = src.with_suffix(".h264.tmp.mp4")
        subprocess.run(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(src),
             "-c:v", "libx264", "-crf", str(crf), "-preset", "veryfast",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp)],
            check=True,
        )
        for attempt in range(5):             # Windows may briefly hold the file open
            try:
                tmp.replace(src)
                return
            except PermissionError:
                time.sleep(1.0 + attempt)
        print(f"warning: could not replace {src} with its H.264 version ({tmp} kept)")
    except Exception as exc:
        print(f"warning: H.264 transcode skipped for {path}: {exc}")


def load_events_for_video(predictions_path: str, video_name: str) -> list[list]:
    """Event list of one video from a predictions JSON file."""
    data = json.loads(Path(predictions_path).read_text())
    return data.get("videos", {}).get(video_name, {}).get("events", [])


def main() -> None:
    ap = argparse.ArgumentParser(description="Render an annotated copy of a traffic video.")
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--events", default=None, help="predictions JSON to read event banners from")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=None)
    args = ap.parse_args()

    events = load_events_for_video(args.events, Path(args.video).name) if args.events else None
    render_annotated(
        args.video, args.out, events=events, width=args.width, stride=max(1, args.stride),
        start_sec=args.start, end_sec=args.end,
        progress_callback=lambda p: print(f"\r{p * 100:5.1f}%", end="", flush=True),
    )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
