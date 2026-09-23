#!/usr/bin/env python3
"""
visualizer.py — High-definition, real-time visual inspection tool for Traffic AI.

Key Design:
  1. Live preview only (no VideoWriter / disk saving).
  2. Runs YOLO detection, ByteTrack tracking, and supervision annotations on the
     native 4K frame (3840x2160) to preserve crisp text, sharp lines, and exact coordinates.
  3. Uses OpenCV's native WINDOW_NORMAL property to downscale the display window
     to HD (1920x1080) on GPU/OS level without pixelating or lagging through manual cv2.resize().
  4. Interactive controls: 'q'/ESC to quit, SPACE/'p' to pause/resume, 's' to step single frame.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import supervision as sv
from ultralytics import YOLO

from solution import (
    SCENE_CONFIG,
    get_traffic_light_state,
    COCO_ROAD_USERS,
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
        description="Traffic AI Realtime Visualizer (High-Def Live Preview)"
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
        default="weights/yolov8s.pt" if Path("weights/yolov8s.pt").exists() else "yolov8s.pt",
        help="Path to YOLO weights",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Process every N-th frame for faster playback (default: 1)",
    )
    args = parser.parse_args()

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
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Native Resolution: {width}x{height} | FPS: {fps:.2f} | Total Frames: {total_frames}")

    # 1. Initialize YOLO Model & ByteTrack
    print(f"[INFO] Loading YOLO model: {args.model}...")
    model = YOLO(args.model)
    tracker = sv.ByteTrack()

    # 2. Setup Zones from SCENE_CONFIG (native 4K coordinates)
    # Line Zones
    stop_bottom_arr = SCENE_CONFIG["stop_line_bottom"]
    stop_line_bottom = sv.LineZone(
        start=sv.Point(int(stop_bottom_arr[0][0]), int(stop_bottom_arr[0][1])),
        end=sv.Point(int(stop_bottom_arr[1][0]), int(stop_bottom_arr[1][1])),
    )
    line_annotator_bottom = sv.LineZoneAnnotator(
        thickness=4,
        color=sv.Color.RED,
        text_scale=1.0,
        text_thickness=2,
        custom_in_text="Stop Bottom",
        custom_out_text="",
    )

    stop_top_arr = SCENE_CONFIG["stop_line_top"]
    stop_line_top = sv.LineZone(
        start=sv.Point(int(stop_top_arr[0][0]), int(stop_top_arr[0][1])),
        end=sv.Point(int(stop_top_arr[1][0]), int(stop_top_arr[1][1])),
    )
    line_annotator_top = sv.LineZoneAnnotator(
        thickness=4,
        color=sv.Color(r=255, g=140, b=0),
        text_scale=1.0,
        text_thickness=2,
        custom_in_text="Stop Top",
        custom_out_text="",
    )

    # Polygon Zones
    road_area_zone = sv.PolygonZone(polygon=SCENE_CONFIG["road_area"])
    road_annotator = sv.PolygonZoneAnnotator(
        zone=road_area_zone,
        color=sv.Color(r=30, g=144, b=255),
        thickness=3,
        text_scale=1.0,
        text_thickness=2,
    )

    zebra_main_zone = sv.PolygonZone(polygon=SCENE_CONFIG["zebra_main"])
    zebra_main_annotator = sv.PolygonZoneAnnotator(
        zone=zebra_main_zone,
        color=sv.Color.YELLOW,
        thickness=4,
        text_scale=1.0,
        text_thickness=2,
    )

    zebra_left_zone = sv.PolygonZone(polygon=SCENE_CONFIG["zebra_left"])
    zebra_left_annotator = sv.PolygonZoneAnnotator(
        zone=zebra_left_zone,
        color=sv.Color.YELLOW,
        thickness=4,
        text_scale=1.0,
        text_thickness=2,
    )

    # 3. Setup Detection Annotators for 4K Resolution
    box_annotator = sv.BoxAnnotator(thickness=3)
    label_annotator = sv.LabelAnnotator(
        text_scale=0.8,
        text_thickness=2,
        text_padding=8,
    )

    # 4. Configure OpenCV Window Properties for Native Window Scaling
    window_name = "Traffic AI Realtime Visualizer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1920, 1080)

    tl_bbox = SCENE_CONFIG["traffic_light_bbox"]
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

            # a) Determine Traffic Light State on the 4K frame
            tl_state = get_traffic_light_state(frame, tl_bbox)

            # b) YOLO Detection & Tracking on the 4K frame
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

            # c) Annotate Zones directly on 4K Frame
            frame = road_annotator.annotate(scene=frame, label="Road Area")
            frame = zebra_main_annotator.annotate(scene=frame, label="Zebra Main")
            frame = zebra_left_annotator.annotate(scene=frame, label="Zebra Left")
            frame = line_annotator_bottom.annotate(
                frame=frame, line_counter=stop_line_bottom
            )
            frame = line_annotator_top.annotate(frame=frame, line_counter=stop_line_top)

            # d) Annotate Traffic Light ROI box
            x1, y1, x2, y2 = tl_bbox
            tl_color = (0, 0, 255) if tl_state == "RED" else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), tl_color, 4)
            cv2.putText(
                frame,
                f"TL: {tl_state}",
                (x1 - 10, max(40, y1 - 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                tl_color,
                3,
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

            # f) Draw Top-Left HUD Dashboard on 4K Frame
            hud_bg = frame.copy()
            cv2.rectangle(hud_bg, (30, 30), (700, 210), (20, 20, 20), -1)
            cv2.addWeighted(hud_bg, 0.75, frame, 0.25, 0, frame)
            cv2.rectangle(frame, (30, 30), (700, 210), (120, 120, 120), 3)

            # Status Texts (scaled for 4K crispness)
            cv2.putText(
                frame,
                f"TRAFFIC LIGHT: {tl_state}",
                (60, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.3,
                (0, 0, 255) if tl_state == "RED" else (0, 255, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Time: {t_sec:6.2f}s | Frame: {frame_idx}/{total_frames}",
                (60, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (220, 220, 220),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Active Tracks: {len(tracked_detections)} objects",
                (60, 185),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 215, 255),
                2,
                cv2.LINE_AA,
            )

        # Show native annotated 4K frame directly in the resized window
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
