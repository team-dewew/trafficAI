#!/usr/bin/env python3
"""
visualizer.py — High-precision 21-Zone Interactive Live Calibrator for Traffic AI.

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
    # 1. Line Zones
    stop_red_pts = adjust_coordinates(SCENE_CONFIG["stop_line_red"], x_off, y_off, scale)
    stop_line_red = sv.LineZone(
        start=sv.Point(int(stop_red_pts[0][0]), int(stop_red_pts[0][1])),
        end=sv.Point(int(stop_red_pts[1][0]), int(stop_red_pts[1][1])),
    )
    line_annotator_red = sv.LineZoneAnnotator(
        thickness=2,
        color=sv.Color.RED,
        text_scale=0.5,
        custom_in_text="Stop Red",
        custom_out_text="",
    )

    stop_jam_pts = adjust_coordinates(SCENE_CONFIG["stop_line_jam"], x_off, y_off, scale)
    stop_line_jam = sv.LineZone(
        start=sv.Point(int(stop_jam_pts[0][0]), int(stop_jam_pts[0][1])),
        end=sv.Point(int(stop_jam_pts[1][0]), int(stop_jam_pts[1][1])),
    )
    line_annotator_jam = sv.LineZoneAnnotator(
        thickness=2,
        color=sv.Color(r=255, g=140, b=0),
        text_scale=0.5,
        custom_in_text="Stop Jam",
        custom_out_text="",
    )

    yield_pts = adjust_coordinates(SCENE_CONFIG["yield_ped_line"], x_off, y_off, scale)
    yield_ped_line = sv.LineZone(
        start=sv.Point(int(yield_pts[0][0]), int(yield_pts[0][1])),
        end=sv.Point(int(yield_pts[1][0]), int(yield_pts[1][1])),
    )
    line_annotator_yield = sv.LineZoneAnnotator(
        thickness=2,
        color=sv.Color(r=255, g=0, b=255),
        text_scale=0.5,
        custom_in_text="Yield Line",
        custom_out_text="",
    )

    # 2. Crosswalks
    crosswalk_annotators = []
    crosswalk_zones = []
    for cw in SCENE_CONFIG["crosswalks"]:
        adj_cw = adjust_coordinates(cw, x_off, y_off, scale)
        zone = sv.PolygonZone(polygon=adj_cw)
        annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.YELLOW, thickness=2, text_scale=0.45)
        crosswalk_zones.append(zone)
        crosswalk_annotators.append(annotator)

    # 3. Forbidden Concrete Islands
    island_annotators = []
    island_zones = []
    for isl in SCENE_CONFIG["forbidden_islands"]:
        adj_isl = adjust_coordinates(isl, x_off, y_off, scale)
        zone = sv.PolygonZone(polygon=adj_isl)
        annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color(r=255, g=20, b=147), thickness=2, text_scale=0.4)
        island_zones.append(zone)
        island_annotators.append(annotator)

    # 4. Safe Sidewalks
    sidewalk_annotators = []
    sidewalk_zones = []
    for sw in SCENE_CONFIG["sidewalks"]:
        adj_sw = adjust_coordinates(sw, x_off, y_off, scale)
        zone = sv.PolygonZone(polygon=adj_sw)
        annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color(r=0, g=255, b=255), thickness=1, text_scale=0.4)
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
        adj_r = adjust_coordinates(SCENE_CONFIG[r_key], x_off, y_off, scale)
        zone = sv.PolygonZone(polygon=adj_r)
        annotator = sv.PolygonZoneAnnotator(zone=zone, color=color, thickness=2, text_scale=0.45)
        road_zones.append(zone)
        road_annotators.append((annotator, r_key))

    # 6. Traffic light bbox
    adj_tl_bbox = adjust_bbox(SCENE_CONFIG["traffic_light_main_bbox"], x_off, y_off, scale)

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
        description="Traffic AI Realtime Interactive Calibrator (21-Zone Setup)"
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
    print(f"   TRAFFIC AI — 21-ZONE INTERACTIVE LIVE CALIBRATOR     ")
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

            # a) Evaluate traffic light state on high-resolution 4K frame BEFORE resizing
            orig_tl = SCENE_CONFIG["traffic_light_main_bbox"]
            tl_4k_bbox = (
                orig_tl[0] + X_OFFSET,
                orig_tl[1] + Y_OFFSET,
                orig_tl[2] + X_OFFSET,
                orig_tl[3] + Y_OFFSET,
            )
            tl_state = get_traffic_light_state(raw_frame, tl_4k_bbox)

            # PiP Debug: Extract crop from original 4K frame and compute HSV red mask
            x1, y1, x2, y2 = SCENE_CONFIG["traffic_light_main_bbox"]
            sx1, sy1 = x1 + X_OFFSET, y1 + Y_OFFSET
            sx2, sy2 = x2 + X_OFFSET, y2 + Y_OFFSET
            raw_h, raw_w = raw_frame.shape[:2]
            csx1, csx2 = max(0, min(sx1, raw_w)), max(0, min(sx2, raw_w))
            csy1, csy2 = max(0, min(sy1, raw_h)), max(0, min(sy2, raw_h))

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

            # Immediately downscale 4K frame to 1080p for smooth FPS
            frame = cv2.resize(raw_frame, (0, 0), fx=SCALE, fy=SCALE)

            # Rebuild zones if user changed offsets
            if (X_OFFSET, Y_OFFSET) != last_offset:
                zones = build_zones_and_annotators(X_OFFSET, Y_OFFSET, SCALE)
                last_offset = (X_OFFSET, Y_OFFSET)

            # b) YOLO Detection & Tracking
            results = model(
                frame,
                verbose=False,
                classes=list(COCO_ROAD_USERS.keys()),
            )[0]
            detections = sv.Detections.from_ultralytics(results)
            tracked_detections = tracker.update_with_detections(detections)

            # Update Line Zones
            zones["stop_line_red"].trigger(tracked_detections)
            zones["stop_line_jam"].trigger(tracked_detections)
            zones["yield_ped_line"].trigger(tracked_detections)

            # c) Annotate shifted zones
            for r_annotator, r_label in zones["road_annotators"]:
                frame = r_annotator.annotate(scene=frame, label=r_label)

            for cw_annotator in zones["crosswalk_annotators"]:
                frame = cw_annotator.annotate(scene=frame, label="Crosswalk")

            for isl_annotator in zones["island_annotators"]:
                frame = isl_annotator.annotate(scene=frame, label="Island")

            for sw_annotator in zones["sidewalk_annotators"]:
                frame = sw_annotator.annotate(scene=frame, label="Sidewalk")

            frame = zones["line_annotator_red"].annotate(
                frame=frame, line_counter=zones["stop_line_red"]
            )
            frame = zones["line_annotator_jam"].annotate(
                frame=frame, line_counter=zones["stop_line_jam"]
            )
            frame = zones["line_annotator_yield"].annotate(
                frame=frame, line_counter=zones["yield_ped_line"]
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

            # g) Top-Right Picture-in-Picture (PiP) Traffic Light Debug View
            pip_w, pip_h = 100, 250
            pip_crop = cv2.resize(tl_crop, (pip_w, pip_h))
            pip_mask = cv2.resize(tl_red_mask, (pip_w, pip_h))
            pip_mask_bgr = cv2.cvtColor(pip_mask, cv2.COLOR_GRAY2BGR)

            fw = frame.shape[1]
            panel_w, panel_h = 245, 365
            panel_x1 = fw - panel_w - 20
            panel_y1 = 20
            panel_x2 = panel_x1 + panel_w
            panel_y2 = panel_y1 + panel_h

            # Translucent dark HUD card
            pip_overlay = frame.copy()
            cv2.rectangle(pip_overlay, (panel_x1, panel_y1), (panel_x2, panel_y2), (15, 15, 15), -1)
            cv2.addWeighted(pip_overlay, 0.85, frame, 0.15, 0, frame)
            status_color = (0, 0, 255) if tl_state == "RED" else (0, 255, 0)
            cv2.rectangle(frame, (panel_x1, panel_y1), (panel_x2, panel_y2), status_color, 2)

            # Panel Title
            cv2.putText(
                frame,
                "TRAFFIC LIGHT DEBUG",
                (panel_x1 + 18, panel_y1 + 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # Image positions
            crop_x1 = panel_x1 + 15
            crop_x2 = crop_x1 + pip_w
            mask_x1 = crop_x2 + 15
            mask_x2 = mask_x1 + pip_w
            img_y1 = panel_y1 + 55
            img_y2 = img_y1 + pip_h

            # Text labels above images
            cv2.putText(
                frame,
                "AI CROP",
                (crop_x1 + 15, img_y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                "RED MASK",
                (mask_x1 + 8, img_y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # Embed resized crop and mask
            frame[img_y1:img_y2, crop_x1:crop_x2] = pip_crop
            frame[img_y1:img_y2, mask_x1:mask_x2] = pip_mask_bgr

            # Thin frames around the image crops
            cv2.rectangle(frame, (crop_x1, img_y1), (crop_x2, img_y2), (120, 120, 120), 1)
            cv2.rectangle(frame, (mask_x1, img_y1), (mask_x2, img_y2), (120, 120, 120), 1)

            # State and pixel statistics footer
            cv2.putText(
                frame,
                f"STATE: {tl_state}",
                (crop_x1, img_y2 + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                status_color,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Red: {tl_red_pixel_count} px (th=5)",
                (crop_x1, img_y2 + 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (200, 200, 200),
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
