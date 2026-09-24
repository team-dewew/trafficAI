#!/usr/bin/env python3
"""
visualizer.py — High-Precision 21-Zone Live Visualizer for Traffic AI Challenge.

Features:
- Pure static 21-zone layout built directly from SCENE_CONFIG (zero alignment drift).
- Processes frames natively in 4K resolution, downscaling to 1080p for smooth display.
- Real-time YOLO detection & ByteTrack vehicle/pedestrian tracking.
- Picture-in-Picture (PiP) Debug view in the top-right corner displaying:
    1) High-resolution zoomed traffic light crop from SCENE_CONFIG["traffic_light_main_bbox"].
    2) Binary HSV Red Mask with exact pixel counts and detection state.

Controls:
    [SPACE] or [P]       : Pause / Resume playback
    [Q] or [ESC]         : Quit visualizer
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
    COCO_ROAD_USERS,
    get_traffic_light_state,
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


def build_scene_zones(config: dict):
    """Build all supervision zones and line counters directly in 4K coordinate space from SCENE_CONFIG."""
    # 1. Stop Lines & Yield Line
    stop_red_pts = config["stop_line_red"]
    stop_line_red = sv.LineZone(
        start=sv.Point(x=int(stop_red_pts[0][0]), y=int(stop_red_pts[0][1])),
        end=sv.Point(x=int(stop_red_pts[1][0]), y=int(stop_red_pts[1][1])),
        triggering_anchors=[sv.Position.BOTTOM_CENTER],
    )
    line_annotator_red = sv.LineZoneAnnotator(
        thickness=4,
        color=sv.Color(r=255, g=0, b=0),
        display_text_box=False,
        display_in_count=False,
        display_out_count=False,
    )

    stop_jam_pts = config["stop_line_jam"]
    stop_line_jam = sv.LineZone(
        start=sv.Point(x=int(stop_jam_pts[0][0]), y=int(stop_jam_pts[0][1])),
        end=sv.Point(x=int(stop_jam_pts[1][0]), y=int(stop_jam_pts[1][1])),
        triggering_anchors=[sv.Position.BOTTOM_CENTER],
    )
    line_annotator_jam = sv.LineZoneAnnotator(
        thickness=4,
        color=sv.Color(r=255, g=165, b=0),
        display_text_box=False,
        display_in_count=False,
        display_out_count=False,
    )

    yield_pts = config["yield_ped_line"]
    yield_ped_line = sv.LineZone(
        start=sv.Point(x=int(yield_pts[0][0]), y=int(yield_pts[0][1])),
        end=sv.Point(x=int(yield_pts[1][0]), y=int(yield_pts[1][1])),
        triggering_anchors=[sv.Position.BOTTOM_CENTER],
    )
    line_annotator_yield = sv.LineZoneAnnotator(
        thickness=4,
        color=sv.Color(r=255, g=255, b=0),
        display_text_box=False,
        display_in_count=False,
        display_out_count=False,
    )

    # 2. Crosswalks
    crosswalk_annotators = []
    crosswalk_zones = []
    for cw in config["crosswalks"]:
        zone = sv.PolygonZone(polygon=cw)
        annotator = sv.PolygonZoneAnnotator(
            zone=zone,
            color=sv.Color(r=0, g=255, b=0),
            thickness=3,
            text_scale=0.8,
        )
        crosswalk_zones.append(zone)
        crosswalk_annotators.append(annotator)

    # 3. Traffic Islands
    islands = config.get("forbidden_islands", config.get("islands", []))
    island_annotators = []
    island_zones = []
    for isl in islands:
        zone = sv.PolygonZone(polygon=isl)
        annotator = sv.PolygonZoneAnnotator(
            zone=zone,
            color=sv.Color(r=128, g=0, b=128),
            thickness=3,
            text_scale=0.8,
        )
        island_zones.append(zone)
        island_annotators.append(annotator)

    # 4. Safe Sidewalks
    sidewalk_annotators = []
    sidewalk_zones = []
    for sw in config["sidewalks"]:
        zone = sv.PolygonZone(polygon=sw)
        annotator = sv.PolygonZoneAnnotator(
            zone=zone,
            color=sv.Color(r=0, g=255, b=255),
            thickness=2,
            text_scale=0.7,
        )
        sidewalk_zones.append(zone)
        sidewalk_annotators.append(annotator)

    # 5. Road Polygons
    road_annotators = []
    road_zones = []
    road_keys = ["lane_ltr", "lane_rtl", "intersection_core", "right_turn_zone", "lower_core"]
    road_colors = [
        sv.Color(r=30, g=144, b=255),
        sv.Color(r=0, g=191, b=255),
        sv.Color(r=138, g=43, b=226),
        sv.Color(r=72, g=209, b=204),
        sv.Color(r=100, g=149, b=237),
    ]
    for r_key, color in zip(road_keys, road_colors):
        if r_key in config:
            zone = sv.PolygonZone(polygon=config[r_key])
            annotator = sv.PolygonZoneAnnotator(
                zone=zone,
                color=color,
                thickness=3,
                text_scale=0.7,
            )
            road_zones.append(zone)
            road_annotators.append((annotator, r_key))

    return {
        "stop_line_red": stop_line_red,
        "line_annotator_red": line_annotator_red,
        "stop_line_jam": stop_line_jam,
        "line_annotator_jam": line_annotator_jam,
        "yield_ped_line": yield_ped_line,
        "line_annotator_yield": line_annotator_yield,
        "crosswalk_zones": crosswalk_zones,
        "crosswalk_annotators": crosswalk_annotators,
        "island_zones": island_zones,
        "island_annotators": island_annotators,
        "sidewalk_zones": sidewalk_zones,
        "sidewalk_annotators": sidewalk_annotators,
        "road_zones": road_zones,
        "road_annotators": road_annotators,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Static 21-Zone Live Visualizer for Traffic AI Challenge"
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

    print("\n========================================================")
    print("      TRAFFIC AI — STATIC 21-ZONE LIVE VISUALIZER       ")
    print("========================================================")
    print(f"[INFO] Video: {video_path.name}")
    print("[INFO] Controls:")
    print("   [SPACE] or [P]       : Pause / Resume playback")
    print("   [Q] or [ESC]         : Quit visualizer")
    print("========================================================\n")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}", file=sys.stderr)
        sys.exit(1)

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 29.97)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # 1. Traffic light bbox directly from static SCENE_CONFIG
    tl_main_bbox = SCENE_CONFIG["traffic_light_main_bbox"]

    # 2. Initialize YOLO Model & ByteTrack
    print(f"[INFO] Loading {args.model} for real-time tracking...")
    model = YOLO(args.model)
    tracker = sv.ByteTrack()

    # Setup Annotators
    box_annotator = sv.BoxAnnotator(thickness=3)
    label_annotator = sv.LabelAnnotator(text_scale=0.8, text_thickness=2)

    # 3. Build supervision zones directly in 4K coordinate space from static SCENE_CONFIG
    zones = build_scene_zones(SCENE_CONFIG)

    window_name = "Traffic AI Visualizer"
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

            # a) Evaluate traffic light state on high-resolution 4K frame using SCENE_CONFIG
            tl_state = get_traffic_light_state(raw_frame, tl_main_bbox)

            # PiP Debug: Extract crop from original 4K frame directly using SCENE_CONFIG
            x1, y1, x2, y2 = tl_main_bbox
            raw_h, raw_w = raw_frame.shape[:2]
            csx1, csx2 = max(0, min(x1, raw_w)), max(0, min(x2, raw_w))
            csy1, csy2 = max(0, min(y1, raw_h)), max(0, min(y2, raw_h))

            if csx2 > csx1 and csy2 > csy1:
                tl_crop = raw_frame[csy1:csy2, csx1:csx2]
                tl_hsv = cv2.cvtColor(tl_crop, cv2.COLOR_BGR2HSV)
                lower_red1 = np.array([0, 40, 40], dtype=np.uint8)
                upper_red1 = np.array([10, 255, 255], dtype=np.uint8)
                lower_red2 = np.array([160, 40, 40], dtype=np.uint8)
                upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

                m1 = cv2.inRange(tl_hsv, lower_red1, upper_red1)
                m2 = cv2.inRange(tl_hsv, lower_red2, upper_red2)
                tl_red_mask = m1 | m2
                tl_red_pixel_count = cv2.countNonZero(tl_red_mask)
            else:
                tl_crop = np.zeros((250, 100, 3), dtype=np.uint8)
                tl_red_mask = np.zeros((250, 100), dtype=np.uint8)
                tl_red_pixel_count = 0

            # b) YOLO Detection & Tracking on 4K frame (imgsz=640 for fast real-time preview)
            results = model(
                raw_frame,
                verbose=False,
                classes=list(COCO_ROAD_USERS.keys()),
                imgsz=640,
            )[0]
            detections = sv.Detections.from_ultralytics(results)
            tracked_detections = tracker.update_with_detections(detections)

            # Trigger Line Zones in native 4K coordinate space
            zones["stop_line_red"].trigger(tracked_detections)
            zones["stop_line_jam"].trigger(tracked_detections)
            zones["yield_ped_line"].trigger(tracked_detections)

            # c) Draw annotations on 4K frame
            annotated_frame = raw_frame.copy()
            for r_annotator, r_label in zones["road_annotators"]:
                annotated_frame = r_annotator.annotate(scene=annotated_frame, label=r_label)

            for cw_annotator in zones["crosswalk_annotators"]:
                annotated_frame = cw_annotator.annotate(scene=annotated_frame, label="Crosswalk")

            for isl_annotator in zones["island_annotators"]:
                annotated_frame = isl_annotator.annotate(scene=annotated_frame, label="Island")

            for sw_annotator in zones["sidewalk_annotators"]:
                annotated_frame = sw_annotator.annotate(scene=annotated_frame, label="Sidewalk")

            annotated_frame = zones["line_annotator_red"].annotate(
                frame=annotated_frame, line_counter=zones["stop_line_red"]
            )
            annotated_frame = zones["line_annotator_jam"].annotate(
                frame=annotated_frame, line_counter=zones["stop_line_jam"]
            )
            annotated_frame = zones["line_annotator_yield"].annotate(
                frame=annotated_frame, line_counter=zones["yield_ped_line"]
            )

            # d) Annotate Traffic Light ROI on 4K frame
            tl_color = (0, 0, 255) if tl_state == "RED" else (0, 255, 0)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), tl_color, 4)
            cv2.putText(
                annotated_frame,
                f"TL: {tl_state}",
                (x1 - 10, max(40, y1 - 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                tl_color,
                2,
                cv2.LINE_AA,
            )

            # e) Annotate detections on 4K frame
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

            annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=tracked_detections)
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame, detections=tracked_detections, labels=labels
            )

            # f) Downscale annotated 4K frame to 1080p for smooth display
            display_frame = cv2.resize(annotated_frame, (1920, 1080))

            # g) Draw Top-Left HUD on display frame
            hud_overlay = display_frame.copy()
            cv2.rectangle(hud_overlay, (20, 20), (520, 140), (15, 15, 15), -1)
            cv2.addWeighted(hud_overlay, 0.8, display_frame, 0.2, 0, display_frame)
            cv2.rectangle(display_frame, (20, 20), (520, 140), (0, 255, 255), 2)

            cv2.putText(
                display_frame,
                "TRAFFIC AI — 21-ZONE MONITOR",
                (35, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                display_frame,
                f"Time: {t_sec:.1f}s | Frame: {frame_idx}/{total_frames} | {'PAUSED' if paused else 'LIVE'}",
                (35, 88),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                display_frame,
                f"Active Tracks: {len(tracked_detections)} | Light: {tl_state}",
                (35, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 128),
                1,
                cv2.LINE_AA,
            )

            # h) Top-Right Picture-in-Picture (PiP) Traffic Light Debug View
            pip_w, pip_h = 100, 250
            pip_crop = cv2.resize(tl_crop, (pip_w, pip_h))
            pip_mask = cv2.resize(tl_red_mask, (pip_w, pip_h))
            pip_mask_bgr = cv2.cvtColor(pip_mask, cv2.COLOR_GRAY2BGR)

            fw = display_frame.shape[1]
            panel_w, panel_h = 245, 365
            panel_x1 = fw - panel_w - 20
            panel_y1 = 20
            panel_x2 = panel_x1 + panel_w
            panel_y2 = panel_y1 + panel_h

            pip_overlay = display_frame.copy()
            cv2.rectangle(pip_overlay, (panel_x1, panel_y1), (panel_x2, panel_y2), (15, 15, 15), -1)
            cv2.addWeighted(pip_overlay, 0.85, display_frame, 0.15, 0, display_frame)
            status_color = (0, 0, 255) if tl_state == "RED" else (0, 255, 0)
            cv2.rectangle(display_frame, (panel_x1, panel_y1), (panel_x2, panel_y2), status_color, 2)

            cv2.putText(
                display_frame,
                "TL DEBUG PiP",
                (panel_x1 + 65, panel_y1 + 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            crop_x1 = panel_x1 + 15
            crop_x2 = crop_x1 + pip_w
            mask_x1 = crop_x2 + 15
            mask_x2 = mask_x1 + pip_w
            img_y1 = panel_y1 + 55
            img_y2 = img_y1 + pip_h

            cv2.putText(
                display_frame,
                "AI CROP",
                (crop_x1 + 15, img_y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                display_frame,
                "RED MASK",
                (mask_x1 + 8, img_y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

            display_frame[img_y1:img_y2, crop_x1:crop_x2] = pip_crop
            display_frame[img_y1:img_y2, mask_x1:mask_x2] = pip_mask_bgr

            cv2.rectangle(display_frame, (crop_x1, img_y1), (crop_x2, img_y2), (120, 120, 120), 1)
            cv2.rectangle(display_frame, (mask_x1, img_y1), (mask_x2, img_y2), (120, 120, 120), 1)

            cv2.putText(
                display_frame,
                f"STATE: {tl_state}",
                (crop_x1, img_y2 + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                status_color,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                display_frame,
                f"Red: {tl_red_pixel_count} px (th=5)",
                (crop_x1, img_y2 + 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

        # Show display frame
        if display_frame is not None:
            cv2.imshow(window_name, display_frame)

        # Listen for pause/resume or exit
        key = cv2.waitKey(1 if not paused else 30) & 0xFF

        if key in (ord("q"), ord("Q"), 27):  # 'q' or ESC
            print("\n[INFO] Exiting visualizer...")
            break
        elif key in (ord(" "), ord("p"), ord("P")):  # Space or 'p'
            paused = not paused
            print(f"[INFO] {'PAUSED' if paused else 'RESUMED'}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
