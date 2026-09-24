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
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

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

# ----------------------------------------------------------------------------
# High-Precision 21-Zone Scene Configuration (calibrated for 4K 3840x2160)
# ----------------------------------------------------------------------------
SCENE_CONFIG: dict[str, list[np.ndarray] | np.ndarray | tuple[int, int, int, int]] = {
    # LINES
    "stop_line_red": np.array([[1878, 925], [481, 1111]], dtype=np.int32),      # strict red light stop
    "stop_line_jam": np.array([[2164, 1048], [645, 1267]], dtype=np.int32),     # allowed to wait here in jam
    "yield_ped_line": np.array([[2625, 1115], [3717, 977]], dtype=np.int32),    # Right side yield line

    # CROSSWALKS (Zebras)
    "crosswalks": [
        np.array([[2150, 1033], [2410, 1137], [2413, 1181], [615, 1464], [671, 1378], [652, 1285], [619, 1245]], dtype=np.int32),
        np.array([[3766, 988], [2633, 1126], [2603, 1092], [2558, 1092], [2491, 1092], [2380, 1037], [2376, 1025], [2373, 1014], [3439, 895]], dtype=np.int32),
        np.array([[344, 1575], [656, 1404], [1005, 1590], [1355, 1757], [1440, 1828], [1496, 1835], [1930, 2155], [1132, 2159], [853, 1880], [664, 1750], [214, 1605]], dtype=np.int32),
    ],

    # ROAD SECTIONS
    "lane_ltr": np.array([[1707, 847], [2198, 1055], [630, 1259], [441, 1096], [363, 1029], [43, 717], [21, 487], [62, 219], [166, 260], [192, 301], [463, 390], [1028, 591]], dtype=np.int32), # Left to Right
    "lane_rtl": np.array([[2005, 880], [1459, 698], [1184, 598], [671, 427], [374, 331], [188, 260], [147, 178], [32, 115], [36, 78], [117, 78], [273, 115], [489, 193], [727, 260], [972, 308], [1184, 320], [1336, 360], [1670, 427], [2417, 624], [2711, 750], [3439, 929], [2399, 1029], [2176, 933]], dtype=np.int32), # Right to Left
    "intersection_core": np.array([[2428, 1189], [2629, 1141], [3781, 996], [3836, 1018], [3836, 2028], [3836, 2133], [3810, 2155], [3714, 2155], [1949, 2155], [1511, 1835], [1949, 1791], [1945, 1772], [1745, 1642], [1392, 1709], [1362, 1742], [1031, 1594], [1533, 1508], [1317, 1363]], dtype=np.int32),
    "right_turn_zone": np.array([[678, 1367], [1139, 1326], [1217, 1341], [1217, 1382], [1132, 1479], [1002, 1568], [641, 1746], [337, 1858], [155, 1936], [6, 2002], [10, 1750], [203, 1642], [511, 1505]], dtype=np.int32),
    "lower_core": np.array([[6, 1947], [259, 1950], [853, 1898], [1106, 2147], [43, 2147], [6, 2129]], dtype=np.int32),

    # CONCRETE ISLANDS & DIVIDERS (Cars entering here = violation / divider hit)
    "forbidden_islands": [
        np.array([[236, 1913], [645, 1754], [842, 1880], [608, 1917], [244, 1936]], dtype=np.int32),
        np.array([[1016, 1568], [1225, 1393], [1288, 1367], [1511, 1497]], dtype=np.int32),
        np.array([[1381, 1750], [1403, 1716], [1760, 1657], [1927, 1772], [1897, 1780], [1492, 1817], [1444, 1820]], dtype=np.int32),
        np.array([[530, 379], [920, 513], [1410, 684], [1871, 836], [2205, 951], [2410, 1040], [2272, 1063], [2205, 1059], [1864, 899], [1358, 710], [838, 524], [526, 409], [229, 308], [184, 275], [229, 275]], dtype=np.int32),
        np.array([[2543, 1063], [2603, 1066], [2647, 1122], [2618, 1152], [2517, 1163], [2443, 1148], [2369, 1092]], dtype=np.int32),
    ],

    # SAFE SIDEWALKS (Pedestrians here = safe)
    "sidewalks": [
        np.array([[114, 854], [470, 1148], [615, 1274], [656, 1349], [634, 1397], [500, 1471], [288, 1568], [6, 1702], [10, 1092], [10, 747]], dtype=np.int32),
        np.array([[1076, 149], [1641, 275], [2194, 364], [2688, 457], [3357, 583], [3829, 717], [3825, 914], [3773, 944], [3721, 973], [3350, 884], [2792, 732], [2621, 687], [2387, 591], [1852, 457], [1425, 368], [1269, 316], [1087, 305], [935, 290], [808, 271], [615, 189], [604, 123], [719, 82], [987, 141]], dtype=np.int32),
        np.array([[2216, 1051], [2410, 1029], [2532, 1070], [2347, 1111]], dtype=np.int32), # Ped island
    ],

    # TRAFFIC LIGHTS
    "traffic_light_main_bbox": (2290, 720, 2360, 860),
    "traffic_light_ped_bbox": (500, 1000, 560, 1120),
}

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


def get_traffic_light_state(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    red_threshold: int = 5,
) -> str:
    """Determine traffic light state (RED or GREEN) inside the specified bbox using HSV color masking."""
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]

    # Clip coordinates to frame boundary
    x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
    y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))

    if x2 <= x1 or y2 <= y1:
        return "GREEN"

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return "GREEN"

    # Convert to HSV color space
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # In HSV, red hue wraps around 0 and 180 (broadened bounds to capture washed out/overexposed red LEDs)
    lower_red1 = np.array([0, 40, 40], dtype=np.uint8)
    upper_red1 = np.array([10, 255, 255], dtype=np.uint8)
    lower_red2 = np.array([160, 40, 40], dtype=np.uint8)
    upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 | mask2

    red_pixel_count = cv2.countNonZero(red_mask)
    return "RED" if red_pixel_count > red_threshold else "GREEN"


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
    """Merge overlapping or adjacent segments of the same class to conform to hackathon rules."""
    if not events:
        return []

    by_class: dict[str, list[list[float]]] = {}
    for item in events:
        s, e, label = float(item[0]), float(item[1]), str(item[2])
        if e > s:
            by_class.setdefault(label, []).append([s, e])

    merged_out: list[list] = []
    for label, intervals in by_class.items():
        intervals.sort(key=lambda x: x[0])
        merged = [intervals[0]]
        for cur in intervals[1:]:
            prev = merged[-1]
            if cur[0] <= prev[1]:  # overlap or contiguous
                prev[1] = max(prev[1], cur[1])
            else:
                merged.append(cur)
        for s, e in merged:
            merged_out.append([round(s, 3), round(e, 3), label])

    merged_out.sort(key=lambda x: (x[0], x[1]))
    return merged_out


def detect_events(video_path: str) -> list[list]:
    """Part A — traffic event detection.

    Args:
        video_path: path to one .mp4 file.

    Returns:
        A list of events, each [start_sec, end_sec, label] with
        0 <= start_sec < end_sec <= duration and label in CLASSES.
    """
    # 1. Open video stream
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    # 2. External Kaggle Anomaly Model Interface
    # Target complex classes: accident, near_miss, fire_smoke
    # TODO: Load Kaggle Anomaly Model (e.g. trained on DoTA/CCD dataset)
    # anomaly_model = ...
    anomaly_model = None

    # 3. Initialize YOLO detector
    local_weights = Path("weights/yolov8s.pt")
    model_path = str(local_weights) if local_weights.exists() else "yolov8s.pt"
    model = YOLO(model_path)

    # 4. Initialize Line Zones directly using static SCENE_CONFIG
    stop_red_pts = SCENE_CONFIG["stop_line_red"]
    stop_line_red = sv.LineZone(
        start=sv.Point(int(stop_red_pts[0][0]), int(stop_red_pts[0][1])),
        end=sv.Point(int(stop_red_pts[1][0]), int(stop_red_pts[1][1])),
    )

    stop_jam_pts = SCENE_CONFIG["stop_line_jam"]
    stop_line_jam = sv.LineZone(
        start=sv.Point(int(stop_jam_pts[0][0]), int(stop_jam_pts[0][1])),
        end=sv.Point(int(stop_jam_pts[1][0]), int(stop_jam_pts[1][1])),
    )

    yield_pts = SCENE_CONFIG["yield_ped_line"]
    yield_ped_line = sv.LineZone(
        start=sv.Point(int(yield_pts[0][0]), int(yield_pts[0][1])),
        end=sv.Point(int(yield_pts[1][0]), int(yield_pts[1][1])),
    )

    # 5. Initialize Grouped Polygon Zones directly using static SCENE_CONFIG
    crosswalk_zones = [sv.PolygonZone(polygon=p) for p in SCENE_CONFIG["crosswalks"]]
    island_zones = [sv.PolygonZone(polygon=p) for p in SCENE_CONFIG["forbidden_islands"]]
    sidewalk_zones = [sv.PolygonZone(polygon=p) for p in SCENE_CONFIG["sidewalks"]]

    road_polygons = [
        SCENE_CONFIG["lane_ltr"],
        SCENE_CONFIG["lane_rtl"],
        SCENE_CONFIG["intersection_core"],
        SCENE_CONFIG["right_turn_zone"],
        SCENE_CONFIG["lower_core"],
    ]
    road_zones = [sv.PolygonZone(polygon=p) for p in road_polygons]

    lane_ltr_zone = sv.PolygonZone(polygon=SCENE_CONFIG["lane_ltr"])
    lane_rtl_zone = sv.PolygonZone(polygon=SCENE_CONFIG["lane_rtl"])
    intersection_core_zone = sv.PolygonZone(polygon=SCENE_CONFIG["intersection_core"])
    right_turn_zone = sv.PolygonZone(polygon=SCENE_CONFIG["right_turn_zone"])

    # 6. Initialize Multi-Object Tracker (ByteTrack)
    tracker = sv.ByteTrack()

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 29.97)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_from_meta = (total_frames / fps) if (fps > 0 and total_frames > 0) else 0.0

    frame_idx = 0
    events: list[list] = []

    # 7. State tracking structures
    track_history: dict[int, list[tuple[float, float, float]]] = {}
    last_seen_time: dict[int, float] = {}

    # active_events[(track_id, label)] = {"label": str, "start_sec": float}
    active_events: dict[tuple[int, str], dict] = {}
    global_events: dict[str, dict] = {}  # for zone-wide congestion tracking
    crossed_red_light_set: set[int] = set()
    u_turn_set: set[int] = set()
    illegal_turn_set: set[int] = set()

    # Line crossing states per vehicle track
    crossed_red_line_map: dict[int, bool] = {}
    crossed_jam_line_map: dict[int, bool] = {}

    tl_main_bbox = SCENE_CONFIG["traffic_light_main_bbox"]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        t_sec = frame_idx / fps

        # a) Determine current main traffic light status
        tl_main_state = get_traffic_light_state(frame, tl_main_bbox)

        # b) Detect road users & obstacles and update tracker
        results = model(frame, verbose=False, classes=list(TARGET_COCO_CLASSES.keys()))[0]
        detections = sv.Detections.from_ultralytics(results)
        tracked_detections = tracker.update_with_detections(detections)
        num_dets = len(tracked_detections)

        # Placeholder: Run Kaggle anomaly model (accident, near_miss, fire_smoke)
        if anomaly_model is not None:
            # anomaly_preds = anomaly_model.predict(frame, t_sec)
            # for pred in anomaly_preds:
            #     events.append([pred["start_sec"], pred["end_sec"], pred["label"]])
            pass

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
            dx, dy = get_direction(track_history[track_id], dt=1.0)
            speed = get_speed(track_history[track_id], dt=1.0)
            det_speeds[i] = speed

            # Update line crossing memory for this vehicle
            if crossed_red_line[i]:
                crossed_red_line_map[track_id] = True
            if crossed_jam_line[i]:
                crossed_jam_line_map[track_id] = True

            # -------------------------------------------------------------
            # 1. Logic for JAYWALKING:
            # Pedestrian on ANY road section, but NOT on crosswalks or sidewalks
            # -------------------------------------------------------------
            jw_key = (track_id, "jaywalking")
            if class_name == "pedestrian":
                is_jaywalking = in_any_road[i] and not in_any_crosswalk[i] and not in_any_sidewalk[i]
                if is_jaywalking:
                    if jw_key not in active_events:
                        active_events[jw_key] = {"label": "jaywalking", "start_sec": t_sec}
                else:
                    if jw_key in active_events:
                        start_sec = active_events[jw_key]["start_sec"]
                        if t_sec > start_sec:
                            events.append([round(start_sec, 3), round(t_sec, 3), "jaywalking"])
                        del active_events[jw_key]

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
                sl_key = (track_id, "stop_line")
                is_past_red_line = crossed_red_line_map.get(track_id, False)
                is_in_intersection = in_intersection_core[i] or crossed_jam_line_map.get(track_id, False)
                is_stopped_on_red = (
                    is_past_red_line
                    and not is_in_intersection
                    and speed < 10.0
                    and tl_main_state == "RED"
                )

                if is_stopped_on_red:
                    if sl_key not in active_events:
                        active_events[sl_key] = {"label": "stop_line", "start_sec": t_sec}
                else:
                    if sl_key in active_events:
                        start_sec = active_events[sl_key]["start_sec"]
                        if t_sec > start_sec:
                            events.append([round(start_sec, 3), round(t_sec, 3), "stop_line"])
                        del active_events[sl_key]

            # -------------------------------------------------------------
            # 3. Logic for FAILURE TO YIELD:
            # Vehicle in crosswalk OR crossing yield line WHILE pedestrian is on crosswalk
            # -------------------------------------------------------------
            fty_key = (track_id, "failure_to_yield")
            if class_name in VEHICLE_CLASSES:
                vehicle_in_conflict_zone = in_any_crosswalk[i] or crossed_yield_line[i]
                is_failing_yield = vehicle_in_conflict_zone and pedestrian_on_crosswalk

                if is_failing_yield:
                    if fty_key not in active_events:
                        active_events[fty_key] = {"label": "failure_to_yield", "start_sec": t_sec}
                else:
                    if fty_key in active_events:
                        start_sec = active_events[fty_key]["start_sec"]
                        if t_sec > start_sec:
                            events.append([round(start_sec, 3), round(t_sec, 3), "failure_to_yield"])
                        del active_events[fty_key]

            # -------------------------------------------------------------
            # 4. Logic for SOLID LINE CROSSING (Concrete islands / dividers):
            # Vehicle enters any forbidden concrete divider / island
            # -------------------------------------------------------------
            slc_key = (track_id, "solid_line_crossing")
            if class_name in VEHICLE_CLASSES:
                if in_any_island[i]:
                    if slc_key not in active_events:
                        active_events[slc_key] = {"label": "solid_line_crossing", "start_sec": t_sec}
                else:
                    if slc_key in active_events:
                        start_sec = active_events[slc_key]["start_sec"]
                        if t_sec > start_sec:
                            events.append([round(start_sec, 3), round(t_sec, 3), "solid_line_crossing"])
                        del active_events[slc_key]

            # -------------------------------------------------------------
            # 5. Logic for WRONG WAY:
            # - lane_ltr (expected left-to-right), moving right-to-left (dx < -30)
            # - lane_rtl (expected right-to-left), moving left-to-right (dx > 30)
            # -------------------------------------------------------------
            ww_key = (track_id, "wrong_way")
            if class_name in VEHICLE_CLASSES:
                is_wrong_way = False
                if in_lane_ltr[i] and dx < -30.0:
                    is_wrong_way = True
                elif in_lane_rtl[i] and dx > 30.0:
                    is_wrong_way = True

                if is_wrong_way:
                    if ww_key not in active_events:
                        active_events[ww_key] = {"label": "wrong_way", "start_sec": t_sec}
                else:
                    if ww_key in active_events:
                        start_sec = active_events[ww_key]["start_sec"]
                        if t_sec > start_sec:
                            events.append([round(start_sec, 3), round(t_sec, 3), "wrong_way"])
                        del active_events[ww_key]

            # -------------------------------------------------------------
            # 6. Logic for STOPPED VEHICLE:
            # Stationary (speed < 10 px/s) on carriageway for >= 10.0 seconds
            # -------------------------------------------------------------
            stop_key = (track_id, "stopped_vehicle")
            if class_name in VEHICLE_CLASSES:
                is_stopped = in_any_road[i] and (speed < 10.0)

                if is_stopped:
                    if stop_key not in active_events:
                        active_events[stop_key] = {"label": "stopped_vehicle", "start_sec": t_sec}
                else:
                    if stop_key in active_events:
                        start_sec = active_events[stop_key]["start_sec"]
                        duration = t_sec - start_sec
                        if duration >= 10.0:
                            events.append([round(start_sec, 3), round(t_sec, 3), "stopped_vehicle"])
                        del active_events[stop_key]

            # -------------------------------------------------------------
            # 7. Logic for ROAD OBSTACLE:
            # Animal or debris on carriageway stationary or crawling (speed < 5 px/s)
            # -------------------------------------------------------------
            ro_key = (track_id, "road_obstacle")
            if class_name in OBSTACLE_CLASSES:
                obs_speed = speed if speed != float("inf") else 0.0
                is_obstacle = in_any_road[i] and (obs_speed < 5.0)

                if is_obstacle:
                    if ro_key not in active_events:
                        active_events[ro_key] = {"label": "road_obstacle", "start_sec": t_sec}
                else:
                    if ro_key in active_events:
                        start_sec = active_events[ro_key]["start_sec"]
                        if (t_sec - start_sec) >= 1.0:
                            events.append([round(start_sec, 3), round(t_sec, 3), "road_obstacle"])
                        del active_events[ro_key]

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

        # Close active events for tracks that disappeared from the camera view
        for (t_id, ev_label), ev_info in list(active_events.items()):
            if t_id not in current_frame_track_ids:
                last_t = last_seen_time.get(t_id, t_sec)
                # If track has been unseen for more than 1.5 seconds, close it
                if (t_sec - last_t) >= 1.5:
                    s = ev_info["start_sec"]
                    e = last_t
                    if ev_label == "stopped_vehicle":
                        if (e - s) >= 10.0:
                            events.append([round(s, 3), round(e, 3), ev_label])
                    elif ev_label == "road_obstacle":
                        if (e - s) >= 1.0:
                            events.append([round(s, 3), round(e, 3), ev_label])
                    else:
                        if e > s:
                            events.append([round(s, 3), round(e, 3), ev_label])
                    del active_events[(t_id, ev_label)]

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
        start_sec = ev_info["start_sec"]
        end_sec = final_duration
        if ev_label == "stopped_vehicle":
            if (end_sec - start_sec) >= 10.0:
                events.append([round(start_sec, 3), round(end_sec, 3), ev_label])
        elif ev_label == "road_obstacle":
            if (end_sec - start_sec) >= 1.0:
                events.append([round(start_sec, 3), round(end_sec, 3), ev_label])
        else:
            if end_sec > start_sec:
                events.append([round(start_sec, 3), round(end_sec, 3), ev_label])
    active_events.clear()

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
        self.last_score = 0.0

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
        # TODO: replace this stub. A simple strong baseline: track vehicles,
        # estimate time-to-collision between pairs, map min TTC -> risk.
        return self.last_score
