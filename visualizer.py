#!/usr/bin/env python3
"""
visualizer.py — Interactive Live Calibrator for Traffic AI.

Live Calibration Controls:
  - W / Up Arrow:    Shift zones UP (Y_OFFSET -= 5)
  - S / Down Arrow:  Shift zones DOWN (Y_OFFSET += 5)
  - A / Left Arrow:  Shift zones LEFT (X_OFFSET -= 5)
  - D / Right Arrow: Shift zones RIGHT (X_OFFSET += 5)
  - SPACE or 'p':    Pause / Resume playback
  - 'q' or ESC:      Quit and print final tuned offsets to terminal
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
# Calibration Defaults
# ----------------------------------------------------------------------------
X_OFFSET = 0    # Dynamic X shift on original 4K scale
Y_OFFSET = 0    # Dynamic Y shift on original 4K scale
SCALE = 0.5     # Downscale 4K to 1080p for buttery smooth real-time performance


def adjust_coordinates(
    arr: np.ndarray, x_off: int, y_off: int, scale: float
) -> np.ndarray:
    """Shift 4K coordinates by offset and scale down to display resolution."""
    offset = np.array([x_off, y_off], dtype=np.float32)
    return ((arr.astype(np.float32) + offset) * scale).astype(np.int32)


def adjust_bbox(
    bbox: tuple[int, int, int, int], x_off: int, y_off: int, scale: float
) -> tuple[int, int, int, int]:
    """Shift 4K bounding box by offset and scale down to display resolution."""
    x1, y1, x2, y2 = bbox
    return (
        int((x1 + x_off) * scale),
        int((y1 + y_off) * scale),
        int((x2 + x_off) * scale),
        int((y2 + y_off) * scale),
    )


def build_zones_and_annotators(x_off: int, y_off: int, scale: float):
    """Rebuild all supervision zones and annotators with the current offsets."""
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

    return {
        "stop_line_bottom": stop_line_bottom,
        "line_annotator_bottom": line_annotator_bottom,
        "stop_line_top": stop_line_top,
        "line_annotator_top": line_annotator_top,
        "road_area_zone": road_area_zone,
        "road_annotator": road_annotator,
        "zebra_main_zone": zebra_main_zone,
        "zebra_main_annotator": zebra_main_annotator,
        "zebra_left_zone": zebra_left_zone,
        "zebra_left_annotator": zebra_left_annotator,
        "tl_bbox": adj_tl_bbox,
    }


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
    global X_OFFSET, Y_OFFSET

    parser = argparse.ArgumentParser(
        description="Traffic AI Realtime Interactive Calibrator"
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
        help="YOLO model (default: yolov8n.pt for fast live preview)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Process every N-th frame (default: 1)",
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"[ERROR] Video file not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    print(f"\n========================================================")
    print(f"       TRAFFIC AI — INTERACTIVE LIVE CALIBRATOR         ")
    print(f"========================================================")
    print(f"[INFO] Video: {video_path.name}")
    print(f"[INFO] Controls:")
    print(f"   [W] or [UP ARROW]    : Shift zones UP   (Y -= 5)")
    print(f"   [S] or [DOWN ARROW]  : Shift zones DOWN (Y += 5)")
    print(f"   [A] or [LEFT ARROW]  : Shift zones LEFT (X -= 5)")
    print(f"   [D] or [RIGHT ARROW] : Shift zones RIGHT (X += 5)")
    print(f"   [SPACE] or [P]       : Pause / Resume playback")
    print(f"   [Q] or [ESC]         : Exit & Print Final Offsets")
    print(f"========================================================\n")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}", file=sys.stderr)
        sys.exit(1)

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 29.97)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # 1. Initialize YOLO Model & ByteTrack
    print(f"[INFO] Loading {args.model} for real-time tracking...")
    model = YOLO(args.model)
    tracker = sv.ByteTrack()

    # 2. Setup Detection Annotators
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(
        text_scale=0.45,
        text_thickness=1,
        text_padding=4,
    )

    # 3. Build initial zones with (X_OFFSET, Y_OFFSET)
    zones = build_zones_and_annotators(X_OFFSET, Y_OFFSET, SCALE)
    last_offset = (X_OFFSET, Y_OFFSET)

    window_name = "Traffic AI Realtime Calibrator"
    frame_idx = 0
    paused = False
    display_frame = None

    while cap.isOpened():
        if not paused:
            ret, raw_frame = cap.read()
            if not ret or raw_frame is None:
                print("\n[INFO] End of video reached. Looping video...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_idx = 0
                continue

            frame_idx += 1
            if args.stride > 1 and (frame_idx % args.stride != 0):
                continue

            t_sec = frame_idx / fps

            # Immediately downscale 4K frame to 1080p for buttery smooth FPS
            frame = cv2.resize(raw_frame, (0, 0), fx=SCALE, fy=SCALE)

            # Rebuild zones if user changed offsets
            if (X_OFFSET, Y_OFFSET) != last_offset:
                zones = build_zones_and_annotators(X_OFFSET, Y_OFFSET, SCALE)
                last_offset = (X_OFFSET, Y_OFFSET)

            # a) Traffic light state on shifted bbox
            tl_state = get_traffic_light_state(frame, zones["tl_bbox"])

            # b) YOLO Detection & Tracking
            results = model(
                frame,
                verbose=False,
                classes=list(COCO_ROAD_USERS.keys()),
            )[0]
            detections = sv.Detections.from_ultralytics(results)
            tracked_detections = tracker.update_with_detections(detections)

            # Update Line Zones
            zones["stop_line_bottom"].trigger(tracked_detections)
            zones["stop_line_top"].trigger(tracked_detections)

            # c) Annotate shifted zones
            frame = zones["road_annotator"].annotate(scene=frame, label="Road Area")
            frame = zones["zebra_main_annotator"].annotate(scene=frame, label="Zebra Main")
            frame = zones["zebra_left_annotator"].annotate(scene=frame, label="Zebra Left")
            frame = zones["line_annotator_bottom"].annotate(
                frame=frame, line_counter=zones["stop_line_bottom"]
            )
            frame = zones["line_annotator_top"].annotate(
                frame=frame, line_counter=zones["stop_line_top"]
            )

            # d) Annotate Traffic Light ROI
            x1, y1, x2, y2 = zones["tl_bbox"]
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

            # e) Annotate detections
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

            # f) Draw Top-Left Large Calibration HUD
            hud_overlay = frame.copy()
            cv2.rectangle(hud_overlay, (20, 20), (520, 150), (15, 15, 15), -1)
            cv2.addWeighted(hud_overlay, 0.8, frame, 0.2, 0, frame)
            cv2.rectangle(frame, (20, 20), (520, 150), (0, 255, 255), 2)

            # Prominent Offset Readout
            cv2.putText(
                frame,
                f"OFFSET: X={X_OFFSET:+d}, Y={Y_OFFSET:+d}",
                (35, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Keys: [W/A/S/D] or [Arrows] to nudge (+/- 5px)",
                (35, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Light: {tl_state} | Tracks: {len(tracked_detections)} | [Q] to Save & Exit",
                (35, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (0, 220, 100),
                1,
                cv2.LINE_AA,
            )

            display_frame = frame

        # Show frame
        if display_frame is not None:
            cv2.imshow(window_name, display_frame)

        # Listen for key events including extended keys (Arrow keys)
        key = cv2.waitKeyEx(1 if not paused else 30)

        if key in (ord("q"), ord("Q"), 27):  # 'q' or ESC -> Exit
            break

        # Up: W or Up Arrow
        elif key in (ord("w"), ord("W"), 2490368, 65362, 0x260000):
            Y_OFFSET -= 5
            print(f"[CALIBRATE] X_OFFSET={X_OFFSET:+d}, Y_OFFSET={Y_OFFSET:+d}")

        # Down: S or Down Arrow
        elif key in (ord("s"), ord("S"), 2621440, 65364, 0x280000):
            Y_OFFSET += 5
            print(f"[CALIBRATE] X_OFFSET={X_OFFSET:+d}, Y_OFFSET={Y_OFFSET:+d}")

        # Left: A or Left Arrow
        elif key in (ord("a"), ord("A"), 2424832, 65361, 0x250000):
            X_OFFSET -= 5
            print(f"[CALIBRATE] X_OFFSET={X_OFFSET:+d}, Y_OFFSET={Y_OFFSET:+d}")

        # Right: D or Right Arrow
        elif key in (ord("d"), ord("D"), 2555904, 65363, 0x270000):
            X_OFFSET += 5
            print(f"[CALIBRATE] X_OFFSET={X_OFFSET:+d}, Y_OFFSET={Y_OFFSET:+d}")

        # Pause / Resume: Space or 'p'
        elif key in (ord(" "), ord("p"), ord("P")):
            paused = not paused
            print(f"[INFO] {'PAUSED' if paused else 'RESUMED'}")

    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print("           CALIBRATION COMPLETED!           ")
    print("=" * 60)
    print(f"Final Calibrated Offsets:")
    print(f"  X_OFFSET = {X_OFFSET:+d}")
    print(f"  Y_OFFSET = {Y_OFFSET:+d}")
    print("\nReady-to-use apply code for solution.py SCENE_CONFIG:")
    print(f"  OFFSET = np.array([{X_OFFSET}, {Y_OFFSET}])")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
