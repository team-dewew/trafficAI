#!/usr/bin/env python3
"""
annotate.py — offline annotated-video renderer for the Traffic AI pipeline.

Draws the AI-aligned 21-zone scene layout, tracked road-user boxes with track
ids, live traffic-light state, a time HUD and an active-event banner onto every
frame, and writes the result as an .mp4. Used for:

  1. samples/previews/*_preview.mp4  — fully annotated sample videos (website
     "Results" section, regenerated offline via --events predictions_samples.json)
  2. Live-demo event clips           — short annotated clips around each
     detected event (website "Live demo" section)

CLI:
    python src/annotate.py --video samples/C3896.MP4 --out samples/previews/C3896_preview.mp4 \
        --events predictions_samples.json [--stride 2] [--width 960]
    python src/annotate.py --video upload.mp4 --out clip.mp4 --start 10 --end 20
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

# Allow running both as `python src/annotate.py` and `python -m src.annotate`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solution import (  # noqa: E402
    COCO_ROAD_USERS,
    SCENE_CONFIG,
    _load_yolo,
    get_ai_offset,
    get_traffic_light_state,
    shift_scene_config,
)
from visualizer import build_scene_zones  # noqa: E402


def _active_labels(events: list[list] | None, t_sec: float) -> list[str]:
    """Labels of events covering timestamp t_sec."""
    if not events:
        return []
    return sorted({str(lbl) for s, e, lbl in events if float(s) <= t_sec <= float(e)})


def render_annotated(
    video_path: str,
    out_path: str,
    events: list[list] | None = None,
    width: int = 960,
    stride: int = 1,
    start_sec: float = 0.0,
    end_sec: float | None = None,
    progress_callback=None,
    frame_sink=None,
) -> str:
    """Render an annotated copy of `video_path` to `out_path`.

    stride=2 keeps every 2nd frame (halves render time and file size with no
    visible smoothness loss for previews).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ret, first_frame = cap.read()
    if not ret or first_frame is None:
        cap.release()
        raise RuntimeError(f"cannot read first frame of {video_path}")

    det_path = Path("weights/yolo11l.pt")
    model = _load_yolo(str(det_path) if det_path.exists() else "yolo11l.pt")
    dx, dy = get_ai_offset(first_frame, model)
    aligned = shift_scene_config(SCENE_CONFIG, dx, dy)
    tl_bbox = aligned["traffic_light_main_bbox"]
    zones = build_scene_zones(aligned)

    start_frame = max(0, int(start_sec * fps))
    stop_frame = total_frames - 1 if end_sec is None else min(total_frames - 1, int(end_sec * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    out_w = int(width)
    out_h = int(round(src_h * (out_w / src_w)))
    out_h -= out_h % 2  # codec-friendly even height
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*"mp4v"), max(1.0, fps / max(1, stride)), (out_w, out_h)
    )

    tracker = sv.ByteTrack()
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.6, text_thickness=2)

    frame_idx = start_frame
    written = 0
    while cap.isOpened() and frame_idx <= stop_frame:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frame_idx += 1
        if stride > 1 and (frame_idx % stride) != 0:
            continue

        t_sec = frame_idx / fps
        tl_state = get_traffic_light_state(frame, tl_bbox)

        results = model(frame, verbose=False, classes=list(COCO_ROAD_USERS.keys()), imgsz=640)[0]
        detections = sv.Detections.from_ultralytics(results)
        tracked = tracker.update_with_detections(detections)

        if frame_sink is not None:
            frame_sink(frame_idx, t_sec, frame, tracked)

        zones["stop_line_red"].trigger(tracked)
        zones["stop_line_jam"].trigger(tracked)
        zones["yield_ped_line"].trigger(tracked)

        ann = frame.copy()
        for r_annotator, r_label in zones["road_annotators"]:
            ann = r_annotator.annotate(scene=ann, label=r_label)
        for cw_annotator in zones["crosswalk_annotators"]:
            ann = cw_annotator.annotate(scene=ann, label="Crosswalk")
        for isl_annotator in zones["island_annotators"]:
            ann = isl_annotator.annotate(scene=ann, label="Island")
        for sw_annotator in zones["sidewalk_annotators"]:
            ann = sw_annotator.annotate(scene=ann, label="Sidewalk")
        ann = zones["line_annotator_red"].annotate(frame=ann, line_counter=zones["stop_line_red"])
        ann = zones["line_annotator_jam"].annotate(frame=ann, line_counter=zones["stop_line_jam"])
        ann = zones["line_annotator_yield"].annotate(frame=ann, line_counter=zones["yield_ped_line"])

        x1, y1, x2, y2 = [int(v) for v in tl_bbox]
        tl_color = (0, 0, 255) if tl_state == "RED" else (0, 255, 0)
        cv2.rectangle(ann, (x1, y1), (x2, y2), tl_color, 3)
        cv2.putText(ann, f"TL: {tl_state}", (x1 - 8, max(40, y1 - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, tl_color, 2, cv2.LINE_AA)

        labels = []
        for i in range(len(tracked)):
            t_id = tracked.tracker_id[i] if tracked.tracker_id is not None else -1
            labels.append(f"#{t_id} {COCO_ROAD_USERS.get(int(tracked.class_id[i]), 'obj')}")
        ann = box_annotator.annotate(scene=ann, detections=tracked)
        ann = label_annotator.annotate(scene=ann, detections=tracked, labels=labels)

        disp = cv2.resize(ann, (out_w, out_h))

        # HUD panel (top-left)
        hud = disp.copy()
        cv2.rectangle(hud, (8, 8), (360, 64), (12, 12, 12), -1)
        cv2.addWeighted(hud, 0.75, disp, 0.25, 0, disp)
        cv2.rectangle(disp, (8, 8), (360, 64), (0, 255, 255), 1)
        cv2.putText(disp, f"t={t_sec:6.1f}s  align=({dx:+d},{dy:+d})  tracks={len(tracked)}",
                    (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(disp, f"TL: {tl_state}", (16, 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, tl_color, 1, cv2.LINE_AA)

        # Active-event banner (bottom-left)
        active = _active_labels(events, t_sec)
        if active:
            text = "EVENT: " + ", ".join(active)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            by = out_h - 14
            cv2.rectangle(disp, (8, by - th - 10), (16 + tw, by + 6), (0, 0, 200), -1)
            cv2.putText(disp, text, (13, by), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 2, cv2.LINE_AA)

        writer.write(disp)
        written += 1
        if progress_callback and written % 30 == 0:
            total_est = max(1, (stop_frame - start_frame) // max(1, stride))
            progress_callback(min(1.0, written / total_est))

    writer.release()
    cap.release()
    _transcode_h264(out_path)
    return out_path


def _transcode_h264(path: str, crf: int = 28) -> None:
    """Re-encode an mp4v render to compact H.264 (yuv420p) for web playback.

    Uses the ffmpeg binary bundled with imageio-ffmpeg; silently keeps the
    original file if no encoder is available.
    """
    try:
        import subprocess
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        src = Path(path)
        tmp = src.with_suffix(".h264.tmp.mp4")
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-i", str(src),
             "-c:v", "libx264", "-crf", str(crf), "-preset", "veryfast",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp)],
            check=True,
        )
        tmp.replace(src)
    except Exception:
        pass


def load_events_for_video(predictions_path: str, video_name: str) -> list[list]:
    """Pull the event list for one video out of a predictions JSON file."""
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
        args.video, args.out,
        events=events, width=args.width, stride=max(1, args.stride),
        start_sec=args.start, end_sec=args.end,
        progress_callback=lambda p: print(f"\r{p * 100:5.1f}%", end="", flush=True),
    )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
