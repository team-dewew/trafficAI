#!/usr/bin/env python3
"""
visualizer.py — Fast, interactive visual inspection tool with dynamic coordinate tuning.

Features:
  1. Downscales 4K input to 1080p (SCALE = 0.5) and runs YOLOv8n (Nano) for buttery smooth live FPS.
  2. Dynamically shifts all SCENE_CONFIG zones by (X_OFFSET, Y_OFFSET) without modifying solution.py.
  3. Real-time visual feedback for tuning zone positions on the road.
  4. Interactive playback: 'q'/ESC to quit, SPACE/'p' to pause/resume, 's' to step single frame.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

from solution import (
    SCENE_CONFIG,
    get_traffic_light_state,
    COCO_ROAD_USERS,
)

# ----------------------------------------------------------------------------
# Live Tuning Constants (Adjust these to shift and scale all zones in real-time)
# ----------------------------------------------------------------------------
X_OFFSET = -80  # Moves all zones left/right (negative = left, positive = right)
Y_OFFSET = -80  # Moves all zones up/down (negative = up, positive = down)
SCALE = 0.5     # Downscales processing from 4K to 1080p for high performance


def adjust_coordinates(
    arr: np.ndarray, x_off: int, y_off: int, scale: float
) -> np.ndarray:
    """Shift and scale a numpy coordinate array."""
    offset = np.array([x_off, y_off], dtype=np.float32)
    return ((arr.astype(np.float32) + offset) * scale).astype(np.int32)


def adjust_bbox(
    bbox: tuple[int, int, int, int], x_off: int, y_off: int, scale: float
) -> tuple[int, int, int, int]:
    """Shift and scale a bounding box (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = bbox
    return (
        int((x1 + x_off) * scale),
        int((y1 + y_off) * scale),
        int((x2 + x_off) * scale),
        int((y2 + y_off) * scale),
    )


def find_default_video() -> str:
    """Find the first available sample video or return a default path."""
    samples_dir = Path("samples")
    if samples_dir.exists():
        for ext in (".MP4", ".mp4", ".avi", ".mkv"):
            found = list(samples_dir.glob(f"*{ext}"))
            if found:
                return str(found[0])
    return "samples/C3896.MP4"


def main():
    parser = argparse.ArgumentParser(
        description="Traffic AI Realtime Visualizer (Fast 1080p Tuning Preview)"
    )
    parser.add_argument(
        "--video",
        type=str,
        default=find_default_video(),
        help="Path to .mp4 video file (default: first found in samples/)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="YOLO model (default: yolov8n.pt for maximum live FPS)",
    )
    parser.add_argument(
        "--x-offset",
        type=int,
        default=X_OFFSET,
        help=f"X coordinate offset (default: {X_OFFSET})",
    )
    parser.add_argument(
        "--y-offset",
        type=int,
        default=Y_OFFSET,
        help=f"Y coordinate offset (default: {Y_OFFSET})",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=SCALE,
        help=f"Processing scale factor (default: {SCALE})",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Process every N-th frame (default: 1)",
    )
    args = parser.parse_args()

    x_off = args.x_offset
    y_off = args.y_offset
    scale = args.scale

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"[ERROR] Video file not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Opening video: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}", file=sys.stderr)
        sys.exit(1)

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 29.97)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    proc_w = int(orig_w * scale)
    proc_h = int(orig_h * scale)

    print(f"[INFO] Original 4K: {orig_w}x{orig_h} | Scaled Display: {proc_w}x{proc_h}")
    print(f"[INFO] Offsets applied: X={x_off}, Y={y_off} | Scale={scale}")

    # 1. Initialize Lightweight YOLO Model (Nano) & ByteTrack
    print(f"[INFO] Loading YOLO model: {args.model}...")
    model = YOLO(args.model)
    tracker = sv.ByteTrack()

    # 2. Dynamically Adjust SCENE_CONFIG Coordinates
    adj_stop_bottom = adjust_coordinates(
        SCENE_CONFIG["stop_line_bottom"], x_off, y_off, scale
    )
    adj_stop_top = adjust_coordinates(
        SCENE_CONFIG["stop_line_top"], x_off, y_off, scale
    )
    adj_road_area = adjust_coordinates(
        SCENE_CONFIG["road_area"], x_off, y_off, scale
    )
    adj_zebra_main = adjust_coordinates(
        SCENE_CONFIG["zebra_main"], x_off, y_off, scale
    )
    adj_zebra_left = adjust_coordinates(
        SCENE_CONFIG["zebra_left"], x_off, y_off, scale
    )
    adj_tl_bbox = adjust_bbox(
        SCENE_CONFIG["traffic_light_bbox"], x_off, y_off, scale
    )

    # 3. Setup Supervision Zones using Adjusted Coordinates
    stop_line_bottom = sv.LineZone(
        start=sv.Point(int(adj_stop_bottom[0][0]), int(adj_stop_bottom[0][1])),
        end=sv.Point(int(adj_stop_bottom[1][0]), int(adj_stop_bottom[1][1])),
    )
    line_annotator_bottom = sv.LineZoneAnnotator(
        thickness=2,
        color=sv.Color.RED,
        text_scale=0.5,
        custom_in_text="Stop Bottom",
        custom_out_text="",
    )

    stop_line_top = sv.LineZone(
        start=sv.Point(int(adj_stop_top[0][0]), int(adj_stop_top[0][1])),
        end=sv.Point(int(adj_stop_top[1][0]), int(adj_stop_top[1][1])),
    )
    line_annotator_top = sv.LineZoneAnnotator(
        thickness=2,
        color=sv.Color(r=255, g=140, b=0),
        text_scale=0.5,
        custom_in_text="Stop Top",
        custom_out_text="",
    )

    road_area_zone = sv.PolygonZone(polygon=adj_road_area)
    road_annotator = sv.PolygonZoneAnnotator(
        zone=road_area_zone,
        color=sv.Color(r=30, g=144, b=255),
        thickness=2,
        text_scale=0.5,
    )

    zebra_main_zone = sv.PolygonZone(polygon=adj_zebra_main)
    zebra_main_annotator = sv.PolygonZoneAnnotator(
        zone=zebra_main_zone,
        color=sv.Color.YELLOW,
        thickness=2,
        text_scale=0.5,
    )

    zebra_left_zone = sv.PolygonZone(polygon=adj_zebra_left)
    zebra_left_annotator = sv.PolygonZoneAnnotator(
        zone=zebra_left_zone,
        color=sv.Color.YELLOW,
        thickness=2,
        text_scale=0.5,
    )

    # 4. Setup Detection Annotators for 1080p
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(
        text_scale=0.45,
        text_thickness=1,
        text_padding=4,
    )

    window_name = "Traffic AI Realtime Visualizer"
    frame_idx = 0
    paused = False

    print("\n[CONTROLS]")
    print("  'q' or ESC: Quit")
    print("  SPACE or 'p': Pause / Resume playback")
    print("  's': Step single frame while paused\n")

    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("\n[INFO] End of video reached.")
                break

            frame_idx += 1
            if args.stride > 1 and (frame_idx % args.stride != 0):
                continue

            t_sec = frame_idx / fps

            # Immediately downscale 4K frame to 1080p for fast processing
            frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

            # a) Determine Traffic Light State on adjusted bbox
            tl_state = get_traffic_light_state(frame, adj_tl_bbox)

            # b) YOLO Detection & Tracking on scaled frame
            results = model(
                frame,
                verbose=False,
                classes=list(COCO_ROAD_USERS.keys()),
            )[0]
            detections = sv.Detections.from_ultralytics(results)
            tracked_detections = tracker.update_with_detections(detections)

            # Update Line Zones counters
            stop_line_bottom.trigger(tracked_detections)
            stop_line_top.trigger(tracked_detections)

            # c) Annotate Adjusted Zones onto Scaled Frame
            frame = road_annotator.annotate(scene=frame, label="Road Area")
            frame = zebra_main_annotator.annotate(scene=frame, label="Zebra Main")
            frame = zebra_left_annotator.annotate(scene=frame, label="Zebra Left")
            frame = line_annotator_bottom.annotate(
                frame=frame, line_counter=stop_line_bottom
            )
            frame = line_annotator_top.annotate(frame=frame, line_counter=stop_line_top)

            # d) Annotate Traffic Light ROI box
            x1, y1, x2, y2 = adj_tl_bbox
            tl_color = (0, 0, 255) if tl_state == "RED" else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), tl_color, 2)
            cv2.putText(
                frame,
                f"TL: {tl_state}",
                (x1 - 5, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                tl_color,
                1,
                cv2.LINE_AA,
            )

            # e) Annotate Tracked Objects
            labels = []
            for i in range(len(tracked_detections)):
                t_id = (
                    tracked_detections.tracker_id[i]
                    if tracked_detections.tracker_id is not None
                    else -1
                )
                c_id = int(tracked_detections.class_id[i])
                c_name = COCO_ROAD_USERS.get(c_id, "obj")
                labels.append(f"#{t_id} {c_name}")

            frame = box_annotator.annotate(scene=frame, detections=tracked_detections)
            frame = label_annotator.annotate(
                scene=frame, detections=tracked_detections, labels=labels
            )

            # f) Draw Top-Left HUD Dashboard
            hud_bg = frame.copy()
            cv2.rectangle(hud_bg, (15, 15), (380, 115), (20, 20, 20), -1)
            cv2.addWeighted(hud_bg, 0.75, frame, 0.25, 0, frame)
            cv2.rectangle(frame, (15, 15), (380, 115), (100, 100, 100), 1)

            # Status Texts
            cv2.putText(
                frame,
                f"TRAFFIC LIGHT: {tl_state}",
                (25, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255) if tl_state == "RED" else (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Time: {t_sec:5.1f}s | Frame: {frame_idx}/{total_frames}",
                (25, 68),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Tracks: {len(tracked_detections)} | Tuning: X={x_off}, Y={y_off}",
                (25, 94),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 215, 255),
                1,
                cv2.LINE_AA,
            )

        # Show frame directly (native 1080p display)
        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1 if not paused else 30) & 0xFF
        if key == ord("q") or key == 27:  # 'q' or ESC
            print("\n[INFO] Exiting visualizer...")
            break
        elif key == ord(" ") or key == ord("p"):  # Space or 'p' to toggle pause
            paused = not paused
            print(f"[INFO] {'PAUSED' if paused else 'RESUMED'}")
        elif key == ord("s") and paused:
            # Step forward one frame
            paused = False
            ret, frame = cap.read()
            paused = True

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
