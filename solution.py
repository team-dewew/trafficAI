"""
solution.py — the ONLY file a team has to implement.

The organizers' harness (run_submission.py) imports this module and calls:

    detect_events(video_path)  -> [[start_sec, end_sec, label], ...]    # Part A
    RiskEstimator().reset(meta); .step(frame, t_sec) -> float           # Part B (optional)

Keep the names and signatures exactly as they are. Everything else — models,
tracking, rules, helper modules under src/ — is up to you.

Labels must come from CLASSES. You may REMOVE classes you never predict;
do not add new ids.
"""
from __future__ import annotations

import copy
from pathlib import Path
import random
import warnings

# Suppress library deprecation warnings (e.g. ByteTrack FutureWarning in supervision) to maintain clean console
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import numpy as np
import supervision as sv
import torch
from ultralytics import YOLO

# Enforce deterministic execution (seed = 42)
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Official class ids (14). See the task description for definitions and
# start/end conventions. Remove entries you never predict; never add.
CLASSES: list[str] = [
    "accident",            # collision between road users / with a fixed object
    "near_miss",           # sharp braking or swerving to avoid a collision, no contact
    "red_light",           # crossing the stop line on red
    "wrong_way",           # driving against the traffic direction / in the oncoming lane
    "illegal_u_turn",      # U-turn where prohibited
    "stopped_vehicle",     # stationary on the carriageway >= 10 s, not queued at a signal
    "jaywalking",          # pedestrian on the carriageway outside a crossing
    "failure_to_yield",    # driving through a crossing while a pedestrian is on it
    "illegal_turn",        # turn from the wrong lane or in a prohibited direction
    "solid_line_crossing", # lane change / manoeuvre across a solid marking
    "stop_line",           # stopped past the stop line on red
    "congestion",          # standstill / crawling traffic across all lanes of a direction
    "road_obstacle",       # debris, animal or fallen object on the carriageway
    "fire_smoke",          # visible fire or smoke from a vehicle or on the road
]

# Anticipation horizon used by the metric (seconds). step() should return
# P(an `accident` starts within the next RISK_HORIZON_SEC seconds).
RISK_HORIZON_SEC = 5.0



# Relevant COCO class IDs for detection & tracking
COCO_ROAD_USERS = {
    0: "pedestrian",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

COCO_OBSTACLES = {
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    24: "backpack",
    25: "umbrella",
    28: "suitcase",
}

TARGET_COCO_CLASSES = {**COCO_ROAD_USERS, **COCO_OBSTACLES}
VEHICLE_CLASSES = {"car", "bus", "truck", "motorcycle"}
OBSTACLE_CLASSES = set(COCO_OBSTACLES.values())

# Minimum durations (seconds) before an event of each class may be emitted
MIN_EVENT_DURATION: dict[str, float] = {
    "stopped_vehicle": 10.0,
    "road_obstacle": 1.0,
}

from src.models import _load_yolo


def _open_event(active_events: dict, track_id: int, label: str, t_sec: float) -> None:
    """Idempotently mark an event as active for (track_id, label)."""
    key = (track_id, label)
    if key not in active_events:
        active_events[key] = {"label": label, "start_sec": t_sec}


def _close_event(active_events: dict, events: list, track_id: int, label: str, end_sec: float) -> None:
    """Close an active per-track event and emit it if it meets its min duration."""
    info = active_events.pop((track_id, label), None)
    if info is None:
        return
    start = info["start_sec"]
    if end_sec > start and (end_sec - start) >= MIN_EVENT_DURATION.get(label, 0.0):
        events.append([round(start, 3), round(end_sec, 3), label])


from src.traffic_light import TrafficLightDetector





def get_ai_offset(
    first_frame: np.ndarray,
    model: YOLO,
) -> tuple[int, int]:
    """Auto-alignment disabled as camera is fixed (Stage 3.2)."""
    print("[AI ALIGNMENT] Disabled as camera is fixed (dx=0, dy=0).")
    return 0, 0


def get_direction(
    history: list[tuple[float, float, float]], dt: float = 1.0
) -> tuple[float, float]:
    """Calculate displacement (dx, dy) over approximately `dt` seconds from movement history.

    history: [(t_sec, x_center, y_center), ...]
    Returns (dx, dy) = (curr_x - prev_x, curr_y - prev_y).
    """
    if len(history) < 2:
        return 0.0, 0.0

    curr_t, curr_x, curr_y = history[-1]
    target_t = curr_t - dt

    # Search backwards for the point closest to ~dt seconds ago
    prev_x, prev_y = history[0][1], history[0][2]
    for t, x, y in reversed(history[:-1]):
        if t <= target_t:
            prev_x, prev_y = x, y
            break

    return float(curr_x - prev_x), float(curr_y - prev_y)


def get_speed(history: list[tuple[float, float, float]], dt: float = 1.0) -> float:
    """Calculate displacement speed in pixels over approximately `dt` seconds."""
    if len(history) < 2:
        return float("inf")

    curr_t = history[-1][0]
    first_t = history[0][0]
    # Require at least ~0.8s of tracking history before judging speed
    if (curr_t - first_t) < 0.8:
        return float("inf")

    dx, dy = get_direction(history, dt)
    return float(np.hypot(dx, dy))


def _boxes_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Intersection over Union of two [x1, y1, x2, y2] boxes."""
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    if inter <= 0.0:
        return 0.0
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return float(inter / union) if union > 0 else 0.0


def get_speed_between(
    history: list[tuple[float, float, float]], dt_newer: float, dt_older: float
) -> float | None:
    """Speed (px/s) measured between two points in the past.

    dt_newer / dt_older: offsets in seconds before the latest history point,
    with dt_older > dt_newer. Returns None if history doesn't reach that far back.
    """
    p_new = get_history_point(history, dt_newer)
    p_old = get_history_point(history, dt_older)
    if p_new is None or p_old is None or dt_older <= dt_newer:
        return None
    dist = float(np.hypot(p_new[0] - p_old[0], p_new[1] - p_old[1]))
    return dist / (dt_older - dt_newer)


def get_history_point(
    history: list[tuple[float, float, float]], dt_ago: float
) -> tuple[float, float] | None:
    """Find (x, y) point closest to `dt_ago` seconds in the past from the last tracked point."""
    if not history:
        return None
    curr_t = history[-1][0]
    target_t = curr_t - dt_ago
    # Only return point if history extends back at least ~70% of dt_ago
    if history[0][0] > target_t + (dt_ago * 0.3):
        return None

    best_p = history[0]
    best_diff = abs(history[0][0] - target_t)
    for p in history:
        diff = abs(p[0] - target_t)
        if diff < best_diff:
            best_diff = diff
            best_p = p
    return (best_p[1], best_p[2])


def merge_same_class_segments(events: list[list]) -> list[list]:
    """
    Temporal smoothing and segment merger conforming to hackathon rules:
    1. Groups events by class label.
    2. Sorts segments chronologically by start_sec.
    3. Merges segments of the same class if next_start - current_end <= 2.0 seconds.
    4. Drops micro-fragments/blips where (end_sec - start_sec) < 0.5 seconds.
    """
    if not events:
        return []

    # 1. Group events by class label
    by_class: dict[str, list[list[float]]] = {}
    for item in events:
        s, e, label = float(item[0]), float(item[1]), str(item[2])
        if e > s:
            by_class.setdefault(label, []).append([s, e])

    merged_events: list[list] = []
    for label, intervals in by_class.items():
        # 2. Sort by start_sec
        intervals.sort(key=lambda x: x[0])
        merged: list[list[float]] = [intervals[0]]

        for cur in intervals[1:]:
            prev = merged[-1]
            # 3. Merge Gap: If next_start - current_end <= 2.0 seconds, merge them
            if cur[0] - prev[1] <= 2.0:
                prev[1] = max(prev[1], cur[1])
            else:
                merged.append(cur)

        for s, e in merged:
            merged_events.append([round(s, 3), round(e, 3), label])

    # 4. BRUTAL PART A FILTERING: STRICTLY drop anything under 0.5 seconds
    final_events = []
    for e in merged_events:
        duration = e[1] - e[0]
        if duration >= 0.5:  # STRICTLY drop anything under 0.5 seconds
            final_events.append(e)

    # Final sort across all merged events by start_sec, then end_sec
    final_events.sort(key=lambda x: (x[0], x[1]))
    return final_events



def detect_events(video_path: str, progress_callback=None) -> list[list]:
    """Part A — traffic event detection.

    Args:
        video_path: path to one .mp4 file.

    Returns:
        A list of events, each [start_sec, end_sec, label] with
        0 <= start_sec < end_sec <= duration and label in CLASSES.
    """
    # 1. Open video stream and read first frame for AI auto-alignment
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    ret, first_frame = cap.read()
    if not ret or first_frame is None:
        cap.release()
        return []

    # 2. Initialize YOLO detector (YOLO11 Large) — cached across videos
    model = _load_yolo("yolo11l.pt")

    # 3. AI Auto-Alignment: detect traffic light displacement using YOLO
    dx, dy = get_ai_offset(first_frame, model)
    print(f"[AI ALIGNMENT] Shifted by dx={dx:+d}, dy={dy:+d} using YOLO Traffic Light Detection")

    # Rewind video capture back to frame 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if cap.get(cv2.CAP_PROP_POS_FRAMES) != 0:
        cap.release()
        cap = cv2.VideoCapture(video_path)

    # 4. External Anomaly Detection Model (accident, crash, fire, smoke) — cached
    anomaly_model = _load_yolo("accident_model.pt")

    # 5. Initialize Zones directly using src.scene.build_scene with W, H scaling
    from src.scene import build_scene
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    zones = build_scene(W, H, dx, dy)
    
    stop_line_red = zones["stop_line_strict"]
    stop_line_jam = zones["stop_line_tolerance"]
    yield_ped_line = zones["yield_ped_line"]

    crosswalk_zones = zones["crosswalks"]
    island_zones = zones["ped_refuge"] + zones["barriers"]
    sidewalk_zones = zones["sidewalks"]

    road_zones = zones["road_zones"]
    lane_ltr_zone = zones["lane_ltr"]
    lane_rtl_zone = zones["lane_rtl"]
    intersection_core_zone = zones["intersection_core"]
    right_turn_zone = zones["right_turn_zone"]

    # 6. Initialize Multi-Object Tracker (ByteTrack)
    tracker = sv.ByteTrack()

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 29.97)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_from_meta = (total_frames / fps) if (fps > 0 and total_frames > 0) else 0.0

    frame_idx = 0
    events: list[list] = []
    anomaly_recent: list[str | None] = []  # temporal gate for learned classes (last 3 checks)

    # 7. State tracking structures
    track_history: dict[int, list[tuple[float, float, float]]] = {}
    last_seen_time: dict[int, float] = {}

    # active_events[(track_id, label)] = {"label": str, "start_sec": float}
    active_events: dict[tuple[int, str], dict] = {}
    # active_nm[(id_a, id_b)] = {"start": float, "last": float} — near-miss pairs
    active_nm: dict[tuple[int, int], dict] = {}
    global_events: dict[str, dict] = {}  # for zone-wide congestion tracking
    crossed_red_light_set: set[int] = set()
    u_turn_set: set[int] = set()
    illegal_turn_set: set[int] = set()

    # Line crossing states per vehicle track
    crossed_red_line_map: dict[int, bool] = {}
    crossed_jam_line_map: dict[int, bool] = {}

    tl_main_bbox = zones["raw"]["main_signal"]
    tl_detector = TrafficLightDetector()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        t_sec = frame_idx / fps

        if progress_callback and total_frames > 0 and frame_idx % 15 == 0:
            progress_callback(frame_idx, total_frames)

        # a) Determine current main traffic light status
        tl_main_state = tl_detector.get_state(frame, tl_main_bbox)

        # b) Detect road users & obstacles and update tracker
        results = model(
            frame,
            verbose=False,
            classes=list(TARGET_COCO_CLASSES.keys()),
            imgsz=640,
        )[0]
        detections = sv.Detections.from_ultralytics(results)
        tracked_detections = tracker.update_with_detections(detections)
        num_dets = len(tracked_detections)

        # Secondary Anomaly Model for Part A: accident & fire_smoke (run every 5 frames).
        # Temporal consistency gate: a learned event is emitted only when the
        # anomaly fires in >= 2 of the last 3 checks (~0.5 s), which removes
        # isolated single-frame false positives while keeping sustained crashes.
        if anomaly_model is not None and frame_idx % 5 == 0:
            anom_results = anomaly_model(frame, verbose=False, imgsz=640)[0]
            hit: str | None = None
            for box in anom_results.boxes:
                c_name = anomaly_model.names[int(box.cls[0])].lower()
                conf = float(box.conf[0])
                if conf > 0.45:
                    if "acc" in c_name or "crash" in c_name or "colli" in c_name or c_name in {"high", "medium", "low", "detected-injury"}:
                        hit = "accident"
                    elif ("fire" in c_name or "smoke" in c_name) and hit is None:
                        hit = "fire_smoke"
            anomaly_recent.append(hit)
            if len(anomaly_recent) > 3:
                anomaly_recent.pop(0)
            for lbl in ("accident", "fire_smoke"):
                if anomaly_recent.count(lbl) >= 2:
                    events.append([round(t_sec, 3), round(t_sec + 2.0, 3), lbl])

        # c) Evaluate all zones on tracked detections using vectorized triggers
        in_any_road = np.zeros(num_dets, dtype=bool)
        for rz in road_zones:
            in_any_road |= rz.trigger(tracked_detections)

        in_any_crosswalk = np.zeros(num_dets, dtype=bool)
        for cz in crosswalk_zones:
            in_any_crosswalk |= cz.trigger(tracked_detections)

        in_any_sidewalk = np.zeros(num_dets, dtype=bool)
        for sz in sidewalk_zones:
            in_any_sidewalk |= sz.trigger(tracked_detections)

        in_any_island = np.zeros(num_dets, dtype=bool)
        for iz in island_zones:
            in_any_island |= iz.trigger(tracked_detections)

        in_lane_ltr = lane_ltr_zone.trigger(tracked_detections)
        in_lane_rtl = lane_rtl_zone.trigger(tracked_detections)
        in_intersection_core = intersection_core_zone.trigger(tracked_detections)
        in_right_turn = right_turn_zone.trigger(tracked_detections)

        # Trigger Line Zones: returns (crossed_in, crossed_out)
        cin_red, cout_red = stop_line_red.trigger(tracked_detections)
        crossed_red_line = cin_red | cout_red

        cin_jam, cout_jam = stop_line_jam.trigger(tracked_detections)
        crossed_jam_line = cin_jam | cout_jam

        cin_yield, cout_yield = yield_ped_line.trigger(tracked_detections)
        crossed_yield_line = cin_yield | cout_yield

        # Speed cache for congestion calculation
        det_speeds: dict[int, float] = {}

        # Check if ANY pedestrian is currently inside ANY crosswalk in this frame
        pedestrian_on_crosswalk = False
        for j in range(num_dets):
            if tracked_detections.class_id[j] == 0 and in_any_crosswalk[j]:
                pedestrian_on_crosswalk = True
                break

        current_frame_track_ids: set[int] = set()
        vehicles_frame: list[dict] = []  # vehicle geometry + braking state for near-miss

        # d) Loop over tracked detections
        for i in range(num_dets):
            if tracked_detections.tracker_id is None:
                continue

            track_id = int(tracked_detections.tracker_id[i])
            if track_id < 0:
                continue

            current_frame_track_ids.add(track_id)
            last_seen_time[track_id] = t_sec

            class_id = int(tracked_detections.class_id[i])
            class_name = TARGET_COCO_CLASSES.get(class_id, "unknown")

            # Calculate centroid (cx, cy)
            x1, y1, x2, y2 = tracked_detections.xyxy[i]
            cx = float((x1 + x2) / 2.0)
            cy = float((y1 + y2) / 2.0)

            # Update movement trajectory history (keep up to ~4 seconds)
            if track_id not in track_history:
                track_history[track_id] = []
            track_history[track_id].append((t_sec, cx, cy))

            max_hist = max(150, int(4.0 * fps))
            if len(track_history[track_id]) > max_hist:
                track_history[track_id] = track_history[track_id][-max_hist:]

            # Calculate motion metrics over ~1 second
            mv_dx, mv_dy = get_direction(track_history[track_id], dt=1.0)
            speed = get_speed(track_history[track_id], dt=1.0)
            det_speeds[i] = speed

            # Near-miss sensing: hard-braking evidence + vehicle geometry
            if class_name in VEHICLE_CLASSES:
                spd_recent = get_speed_between(track_history[track_id], 0.0, 0.5)
                spd_prior = get_speed_between(track_history[track_id], 0.5, 1.5)
                hard_brake = (
                    spd_recent is not None
                    and spd_prior is not None
                    and spd_prior > 60.0
                    and spd_recent < 0.45 * spd_prior
                )
                vehicles_frame.append({
                    "id": track_id,
                    "i": i,
                    "cx": cx,
                    "cy": cy,
                    "diag": float(np.hypot(x2 - x1, y2 - y1)),
                    "box": tracked_detections.xyxy[i],
                    "hard_brake": hard_brake,
                })

            # Update line crossing memory for this vehicle
            if crossed_red_line[i]:
                crossed_red_line_map[track_id] = True
            if crossed_jam_line[i]:
                crossed_jam_line_map[track_id] = True

            # -------------------------------------------------------------
            # 1. Logic for JAYWALKING:
            # Pedestrian on ANY road section, but NOT on crosswalks or sidewalks
            # -------------------------------------------------------------
            if class_name == "pedestrian":
                is_jaywalking = in_any_road[i] and not in_any_crosswalk[i] and not in_any_sidewalk[i]
                if is_jaywalking:
                    _open_event(active_events, track_id, "jaywalking", t_sec)
                else:
                    _close_event(active_events, events, track_id, "jaywalking", t_sec)

            # -------------------------------------------------------------
            # 2. Logic for RED LIGHT & STOP LINE:
            # -------------------------------------------------------------
            if class_name in VEHICLE_CLASSES:
                # Red light running: crossing strict stop line on RED
                if crossed_red_line[i] and tl_main_state == "RED":
                    if track_id not in crossed_red_light_set:
                        crossed_red_light_set.add(track_id)
                        events.append([round(t_sec, 3), round(t_sec + 2.0, 3), "red_light"])

                # Stop line violation: stopped past stop_line_red but before jam line / core on RED
                is_past_red_line = crossed_red_line_map.get(track_id, False)
                is_in_intersection = in_intersection_core[i] or crossed_jam_line_map.get(track_id, False)
                is_stopped_on_red = (
                    is_past_red_line
                    and not is_in_intersection
                    and speed < 10.0
                    and tl_main_state == "RED"
                )

                if is_stopped_on_red:
                    _open_event(active_events, track_id, "stop_line", t_sec)
                else:
                    _close_event(active_events, events, track_id, "stop_line", t_sec)

            # -------------------------------------------------------------
            # 3. Logic for FAILURE TO YIELD:
            # Vehicle in crosswalk OR crossing yield line WHILE pedestrian is on crosswalk
            # -------------------------------------------------------------
            if class_name in VEHICLE_CLASSES:
                vehicle_in_conflict_zone = in_any_crosswalk[i] or crossed_yield_line[i]
                is_failing_yield = vehicle_in_conflict_zone and pedestrian_on_crosswalk

                if is_failing_yield:
                    _open_event(active_events, track_id, "failure_to_yield", t_sec)
                else:
                    _close_event(active_events, events, track_id, "failure_to_yield", t_sec)

            # -------------------------------------------------------------
            # 4. Logic for SOLID LINE CROSSING (Concrete islands / dividers):
            # Vehicle enters any forbidden concrete divider / island
            # -------------------------------------------------------------
            if class_name in VEHICLE_CLASSES:
                if in_any_island[i]:
                    _open_event(active_events, track_id, "solid_line_crossing", t_sec)
                else:
                    _close_event(active_events, events, track_id, "solid_line_crossing", t_sec)

            # -------------------------------------------------------------
            # 5. Logic for WRONG WAY:
            # - lane_ltr (expected left-to-right), moving right-to-left (dx < -30)
            # - lane_rtl (expected right-to-left), moving left-to-right (dx > 30)
            # -------------------------------------------------------------
            if class_name in VEHICLE_CLASSES:
                is_wrong_way = False
                if in_lane_ltr[i] and mv_dx < -30.0:
                    is_wrong_way = True
                elif in_lane_rtl[i] and mv_dx > 30.0:
                    is_wrong_way = True

                if is_wrong_way:
                    _open_event(active_events, track_id, "wrong_way", t_sec)
                else:
                    _close_event(active_events, events, track_id, "wrong_way", t_sec)

            # -------------------------------------------------------------
            # 6. Logic for STOPPED VEHICLE:
            # Stationary (speed < 10 px/s) on carriageway for >= 10.0 seconds
            # -------------------------------------------------------------
            if class_name in VEHICLE_CLASSES:
                is_stopped = in_any_road[i] and (speed < 10.0)

                if is_stopped:
                    _open_event(active_events, track_id, "stopped_vehicle", t_sec)
                else:
                    _close_event(active_events, events, track_id, "stopped_vehicle", t_sec)

            # -------------------------------------------------------------
            # 7. Logic for ROAD OBSTACLE:
            # Animal or debris on carriageway stationary or crawling (speed < 5 px/s)
            # -------------------------------------------------------------
            if class_name in OBSTACLE_CLASSES:
                obs_speed = speed if speed != float("inf") else 0.0
                is_obstacle = in_any_road[i] and (obs_speed < 5.0)

                if is_obstacle:
                    _open_event(active_events, track_id, "road_obstacle", t_sec)
                else:
                    _close_event(active_events, events, track_id, "road_obstacle", t_sec)

            # -------------------------------------------------------------
            # 8. Logic for ILLEGAL U-TURN:
            # Vehicle reverses heading inside intersection_core (e.g. dx > 30 then dx < -30)
            # -------------------------------------------------------------
            if class_name in VEHICLE_CLASSES and in_intersection_core[i]:
                if track_id not in u_turn_set:
                    p_3s = get_history_point(track_history[track_id], dt_ago=3.0)
                    p_1_5s = get_history_point(track_history[track_id], dt_ago=1.5)
                    if p_3s is not None and p_1_5s is not None:
                        dx_prev = p_1_5s[0] - p_3s[0]
                        dx_curr = cx - p_1_5s[0]
                        if (dx_prev > 30.0 and dx_curr < -30.0) or (dx_prev < -30.0 and dx_curr > 30.0):
                            u_turn_set.add(track_id)
                            events.append([round(max(0.0, t_sec - 1.5), 3), round(t_sec + 1.5, 3), "illegal_u_turn"])

            # -------------------------------------------------------------
            # 9. Logic for ILLEGAL TURN:
            # Sharp 90-degree turn inside intersection_core but strictly outside right_turn_zone
            # -------------------------------------------------------------
            if class_name in VEHICLE_CLASSES and in_intersection_core[i] and not in_right_turn[i]:
                if track_id not in illegal_turn_set and track_id not in u_turn_set:
                    p_2s = get_history_point(track_history[track_id], dt_ago=2.0)
                    p_1s = get_history_point(track_history[track_id], dt_ago=1.0)
                    if p_2s is not None and p_1s is not None:
                        dx_p = p_1s[0] - p_2s[0]
                        dy_p = p_1s[1] - p_2s[1]
                        dx_c = cx - p_1s[0]
                        dy_c = cy - p_1s[1]
                        if np.hypot(dx_p, dy_p) > 15.0 and np.hypot(dx_c, dy_c) > 15.0:
                            h_to_v = (abs(dx_p) > abs(dy_p) * 2.0) and (abs(dy_c) > abs(dx_c) * 2.0)
                            v_to_h = (abs(dy_p) > abs(dx_p) * 2.0) and (abs(dx_c) > abs(dy_c) * 2.0)
                            if h_to_v or v_to_h:
                                illegal_turn_set.add(track_id)
                                events.append([round(max(0.0, t_sec - 1.5), 3), round(t_sec + 1.5, 3), "illegal_turn"])

        # -----------------------------------------------------------------
        # 10. Logic for CONGESTION (zone-wide lane crawl / standstill):
        # -----------------------------------------------------------------
        # Lane LTR
        veh_ltr = [
            i for i in range(num_dets)
            if in_lane_ltr[i] and TARGET_COCO_CLASSES.get(int(tracked_detections.class_id[i]), "") in VEHICLE_CLASSES
        ]
        if len(veh_ltr) >= 4:
            valid_speeds_ltr = [det_speeds[i] for i in veh_ltr if det_speeds.get(i, float("inf")) != float("inf")]
            avg_speed_ltr = (sum(valid_speeds_ltr) / len(valid_speeds_ltr)) if valid_speeds_ltr else 0.0
            is_cong_ltr = avg_speed_ltr < 5.0
        else:
            is_cong_ltr = False

        if is_cong_ltr:
            if "congestion_ltr" not in global_events:
                global_events["congestion_ltr"] = {"start_sec": t_sec}
        else:
            if "congestion_ltr" in global_events:
                s_sec = global_events["congestion_ltr"]["start_sec"]
                if t_sec > s_sec:
                    events.append([round(s_sec, 3), round(t_sec, 3), "congestion"])
                del global_events["congestion_ltr"]

        # Lane RTL
        veh_rtl = [
            i for i in range(num_dets)
            if in_lane_rtl[i] and TARGET_COCO_CLASSES.get(int(tracked_detections.class_id[i]), "") in VEHICLE_CLASSES
        ]
        if len(veh_rtl) >= 4:
            valid_speeds_rtl = [det_speeds[i] for i in veh_rtl if det_speeds.get(i, float("inf")) != float("inf")]
            avg_speed_rtl = (sum(valid_speeds_rtl) / len(valid_speeds_rtl)) if valid_speeds_rtl else 0.0
            is_cong_rtl = avg_speed_rtl < 5.0
        else:
            is_cong_rtl = False

        if is_cong_rtl:
            if "congestion_rtl" not in global_events:
                global_events["congestion_rtl"] = {"start_sec": t_sec}
        else:
            if "congestion_rtl" in global_events:
                s_sec = global_events["congestion_rtl"]["start_sec"]
                if t_sec > s_sec:
                    events.append([round(s_sec, 3), round(t_sec, 3), "congestion"])
                del global_events["congestion_rtl"]

        # -----------------------------------------------------------------
        # 10b. Logic for NEAR MISS (pairwise, no contact):
        # Two vehicles come abnormally close while at least one brakes hard.
        # Start = onset of the evasive braking; end = they are clear of each
        # other (hysteresis so the segment covers the whole evasive action).
        # -----------------------------------------------------------------
        for pa in range(len(vehicles_frame)):
            for pb in range(pa + 1, len(vehicles_frame)):
                va, vb = vehicles_frame[pa], vehicles_frame[pb]
                if not (in_any_road[va["i"]] and in_any_road[vb["i"]]):
                    continue
                pair_key = (min(va["id"], vb["id"]), max(va["id"], vb["id"]))
                dist = float(np.hypot(va["cx"] - vb["cx"], va["cy"] - vb["cy"]))
                close_thresh = 0.85 * (va["diag"] + vb["diag"]) / 2.0
                clear_thresh = 1.4 * close_thresh
                boxes_touch = _boxes_iou(va["box"], vb["box"]) >= 0.03
                if (
                    dist < close_thresh
                    and not boxes_touch
                    and (va["hard_brake"] or vb["hard_brake"])
                ):
                    if pair_key not in active_nm:
                        active_nm[pair_key] = {"start": t_sec, "last": t_sec}
                    active_nm[pair_key]["last"] = t_sec
                elif pair_key in active_nm and (dist >= clear_thresh or boxes_touch):
                    nm = active_nm.pop(pair_key)
                    if nm["last"] > nm["start"]:
                        events.append([round(nm["start"], 3), round(nm["last"], 3), "near_miss"])

        # Close active near-miss pairs whose tracks left the scene
        for pair_key, nm in list(active_nm.items()):
            if any(
                tid not in current_frame_track_ids
                and (t_sec - last_seen_time.get(tid, t_sec)) >= 1.5
                for tid in pair_key
            ):
                if nm["last"] > nm["start"]:
                    events.append([round(nm["start"], 3), round(nm["last"], 3), "near_miss"])
                del active_nm[pair_key]

        # Close active events for tracks that disappeared from the camera view
        for (t_id, ev_label), ev_info in list(active_events.items()):
            if t_id not in current_frame_track_ids:
                last_t = last_seen_time.get(t_id, t_sec)
                # If track has been unseen for more than 1.5 seconds, close it
                if (t_sec - last_t) >= 1.5:
                    _close_event(active_events, events, t_id, ev_label, last_t)

        frame_idx += 1

    cap.release()

    # Determine final video duration
    final_duration = duration_from_meta if duration_from_meta > 0 else (frame_idx / fps)

    # Close any remaining global events (e.g. congestion)
    for g_key, g_info in list(global_events.items()):
        s_sec = g_info["start_sec"]
        e_sec = final_duration
        if e_sec > s_sec:
            events.append([round(s_sec, 3), round(e_sec, 3), "congestion"])
    global_events.clear()

    # Close any remaining active events at the end of the video
    for (t_id, ev_label), ev_info in list(active_events.items()):
        _close_event(active_events, events, t_id, ev_label, final_duration)
    active_events.clear()

    # Close any remaining near-miss pairs at the end of the video
    for pair_key, nm in list(active_nm.items()):
        if nm["last"] > nm["start"]:
            events.append([round(nm["start"], 3), round(nm["last"], 3), "near_miss"])
    active_nm.clear()

    # Merge any overlapping segments of the same class to ensure format compliance
    events = merge_same_class_segments(events)

    return events


class RiskEstimator:
    """Part B — causal accident anticipation (optional, bonus).

    The harness calls ``reset(meta)`` once per video and then ``step`` for
    EVERY frame, in order. ``step`` must use only the frames it has seen so
    far: do not open the video file inside this class, and do not reuse
    Part A results that were computed with access to future frames.
    """

    def reset(self, meta: dict) -> None:
        """Called once before the first frame of each video.

        meta = {"video_id": str, "fps": float, "width": int, "height": int,
                "n_frames": int}
        """
        self.meta = meta
        self.model = _load_yolo("yolov8n.pt")
        self.tracker = sv.ByteTrack()
        self.frame_count = 0
        self.last_risk = 0.0
        self.track_history = {}
        self.track_frames = {}
        self.pair_last_dist: dict[tuple[int, int], float] = {}
        self._road_polys: list[np.ndarray] | None = None
        self._safe_polys: list[np.ndarray] | None = None

    def _ensure_scene_polys(self, frame: np.ndarray) -> None:
        """Build AI-aligned roadway / safe-area polygons once (first stepped frame).

        Reuses the same 21-zone scene calibration as Part A; the primary
        detector is already cached in-process by Part A, so the alignment
        probe costs one cheap traffic-light inference.
        """
        h, w = frame.shape[:2]
        sx, sy = w / 3840.0, h / 2160.0
        cfg = shift_scene_config(SCENE_CONFIG, 0, 0)

        def _scale(poly: np.ndarray) -> np.ndarray:
            return (poly.astype(np.float64) * np.array([sx, sy])).astype(np.int32)

        self._road_polys = [
            _scale(cfg[k])
            for k in ("lane_ltr", "lane_rtl", "intersection_core", "right_turn_zone", "lower_core")
        ]
        self._safe_polys = [_scale(p) for p in cfg["crosswalks"]] + [
            _scale(p) for p in cfg["sidewalks"]
        ]

    def _point_in(self, polys: list[np.ndarray], x: float, y: float) -> bool:
        return any(
            cv2.pointPolygonTest(p, (float(x), float(y)), False) >= 0 for p in polys
        )

    def _compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute Intersection over Union between two [x1, y1, x2, y2] boxes."""
        return _boxes_iou(box1, box2)

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        """Return P(accident starts within the next RISK_HORIZON_SEC s).

        Args:
            frame: BGR uint8 array of shape (H, W, 3) — OpenCV convention.
            t_sec: timestamp of this frame in seconds.

        Returns:
            A float in [0, 1]. Skipping frames internally and returning the
            previous score is fine; the harness still expects a value for
            every call.
        """
        self.frame_count += 1

        # Optimization: run YOLO tracking only every 3rd frame
        if self.frame_count % 3 != 0:
            return float(self.last_risk)

        if self._road_polys is None:
            self._ensure_scene_polys(frame)

        height, width = frame.shape[:2]
        # COCO road user classes: pedestrian (0), bicycle (1), car (2), motorcycle (3), bus (5), truck (7)
        target_classes = [0, 1, 2, 3, 5, 7]
        vehicle_classes = {1, 2, 3, 5, 7}

        results = self.model(
            frame,
            verbose=False,
            imgsz=640,
            classes=target_classes,
        )[0]
        detections = sv.Detections.from_ultralytics(results)
        tracked_detections = self.tracker.update_with_detections(detections)
        num_dets = len(tracked_detections)

        current_risk = 0.0
        scale_to_640 = 640.0 / width if width > 0 else 1.0

        if num_dets > 0:
            xyxy = tracked_detections.xyxy
            class_ids = tracked_detections.class_id
            tracker_ids = tracked_detections.tracker_id

            # Calculate speed (displacement in pixels per frame) for each track_id over the last 3-5 frames
            speeds: dict[int, float] = {}
            for k in range(num_dets):
                t_id = int(tracker_ids[k]) if tracker_ids is not None and tracker_ids[k] is not None else k
                box = xyxy[k]
                cx = float((box[0] + box[2]) / 2.0) * scale_to_640
                cy = float((box[1] + box[3]) / 2.0) * scale_to_640

                # Reset history if track was lost for > 10 frames
                if t_id in self.track_frames and self.track_frames[t_id]:
                    if (self.frame_count - self.track_frames[t_id][-1]) > 10:
                        self.track_history[t_id] = []
                        self.track_frames[t_id] = []

                if t_id not in self.track_history:
                    self.track_history[t_id] = []
                    self.track_frames[t_id] = []

                self.track_history[t_id].append((cx, cy))
                self.track_frames[t_id].append(self.frame_count)
                if len(self.track_history[t_id]) > 5:
                    self.track_history[t_id].pop(0)
                    self.track_frames[t_id].pop(0)

                hist = self.track_history[t_id]
                frames = self.track_frames[t_id]
                if len(hist) >= 2:
                    p0 = hist[0]
                    p1 = hist[-1]
                    df = max(1, frames[-1] - frames[0])
                    spd = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]) / df)
                else:
                    spd = 0.0
                speeds[t_id] = spd

            # Pedestrian Hazard: pedestrians on the carriageway (outside
            # crosswalks/sidewalks). In this busy scene pedestrians are on the
            # roadway almost constantly, so a bare-presence signal would be a
            # constant floor (chance-level AP). It only counts when a moving
            # vehicle is actually closing in (see conflict check below).
            peds_on_road: list[dict] = []
            for i in range(num_dets):
                if class_ids is not None and int(class_ids[i]) == 0:
                    box = xyxy[i]
                    px = (box[0] + box[2]) / 2.0
                    py = (box[1] + box[3]) / 2.0
                    if self._point_in(self._road_polys, px, py) and not self._point_in(
                        self._safe_polys, px, py
                    ):
                        peds_on_road.append({
                            "cx": px,
                            "cy": py,
                            "diag": float(np.hypot(box[2] - box[0], box[3] - box[1])),
                        })

            # Collision Course (TTC Proxy): Iterate through all pairs of tracked vehicles
            vehicle_indices = [
                i for i in range(num_dets)
                if class_ids is not None and int(class_ids[i]) in vehicle_classes
            ]
            num_vehicles = len(vehicle_indices)

            for idx_a in range(num_vehicles):
                i = vehicle_indices[idx_a]
                box_i = xyxy[i]
                cx_i = float((box_i[0] + box_i[2]) / 2.0)
                cy_i = float((box_i[1] + box_i[3]) / 2.0)
                diag_i = float(np.hypot(box_i[2] - box_i[0], box_i[3] - box_i[1]))
                t_id_i = int(tracker_ids[i]) if tracker_ids is not None and tracker_ids[i] is not None else i
                spd_i = speeds.get(t_id_i, 0.0)

                for idx_b in range(idx_a + 1, num_vehicles):
                    j = vehicle_indices[idx_b]
                    box_j = xyxy[j]
                    cx_j = float((box_j[0] + box_j[2]) / 2.0)
                    cy_j = float((box_j[1] + box_j[3]) / 2.0)
                    diag_j = float(np.hypot(box_j[2] - box_j[0], box_j[3] - box_j[1]))
                    t_id_j = int(tracker_ids[j]) if tracker_ids is not None and tracker_ids[j] is not None else j
                    spd_j = speeds.get(t_id_j, 0.0)

                    # Scale-aware proximity: thresholds follow the vehicles' own
                    # size, so near-camera and far-field pairs are judged fairly.
                    dist = float(np.hypot(cx_i - cx_j, cy_i - cy_j))
                    avg_diag = 0.5 * (diag_i + diag_j)
                    pair_key = (min(t_id_i, t_id_j), max(t_id_i, t_id_j))
                    prev_dist = self.pair_last_dist.get(pair_key)
                    self.pair_last_dist[pair_key] = dist
                    closing = prev_dist is not None and dist < prev_dist - (2.0 / scale_to_640)
                    approach_speed = max(spd_i, spd_j)

                    if self._compute_iou(box_i, box_j) >= 0.03 or not closing:
                        continue  # already in contact (accident territory) or separating

                    if dist < 0.75 * avg_diag and approach_speed > 2.5:
                        current_risk = max(current_risk, 0.80)
                    elif dist < 0.95 * avg_diag and approach_speed > 1.5:
                        current_risk = max(current_risk, 0.45)

            # Pedestrian–vehicle conflict: a moving vehicle bearing down on a
            # pedestrian who is on the carriageway is a strong pre-crash signal.
            for ped in peds_on_road:
                for i in vehicle_indices:
                    box_i = xyxy[i]
                    vx = float((box_i[0] + box_i[2]) / 2.0)
                    vy = float((box_i[1] + box_i[3]) / 2.0)
                    vdiag = float(np.hypot(box_i[2] - box_i[0], box_i[3] - box_i[1]))
                    t_id = int(tracker_ids[i]) if tracker_ids is not None and tracker_ids[i] is not None else i
                    if speeds.get(t_id, 0.0) <= 2.0:
                        continue  # parked / crawling vehicles are not an imminent threat
                    d = float(np.hypot(ped["cx"] - vx, ped["cy"] - vy))
                    if d < 0.8 * (ped["diag"] + vdiag):
                        current_risk = max(current_risk, 0.50)
                        break

        # Periodic cleanup of stale tracks (> 60 frames inactive)
        if self.frame_count % 90 == 0:
            stale_ids = [tid for tid, f_list in self.track_frames.items() if (self.frame_count - f_list[-1]) > 60]
            stale = set(stale_ids)
            for tid in stale_ids:
                self.track_history.pop(tid, None)
                self.track_frames.pop(tid, None)
            for pair_key in [k for k in self.pair_last_dist if k[0] in stale or k[1] in stale]:
                del self.pair_last_dist[pair_key]

        # Exponential smoothing (alpha = 0.45): two consecutive strong hits
        # cross the 0.5 alarm threshold; the curve decays in ~1 s once clear.
        self.last_risk = (self.last_risk * 0.55) + (current_risk * 0.45)
        return float(np.clip(self.last_risk, 0.0, 1.0))
