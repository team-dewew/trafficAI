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
# Global Scene Configuration (calibrated for 4K 3840x2160 resolution)
# ----------------------------------------------------------------------------
SCENE_CONFIG: dict[str, np.ndarray | tuple[int, int, int, int]] = {
    # Stop lines (LineZone coordinates: 2 points [x, y])
    "stop_line_bottom": np.array([[562, 1101], [1862, 933]], dtype=np.int32),
    "stop_line_top": np.array([[2180, 947], [2396, 1018]], dtype=np.int32),

    # Pedestrian Crossings (PolygonZone coordinates)
    "zebra_main": np.array([
        [646, 1271], [648, 1128], [1963, 955], [2435, 922], [3383, 865],
        [3692, 985], [3761, 1023], [2750, 1176], [2652, 1206], [2551, 1255],
        [2401, 1247], [1712, 1330], [653, 1371]
    ], dtype=np.int32),
    "zebra_left": np.array([
        [682, 1375], [1618, 1768], [1858, 2158], [1015, 2158], [839, 2027],
        [667, 1862], [465, 1761], [232, 1656], [337, 1557]
    ], dtype=np.int32),

    # Full Intersection Road Area (PolygonZone coordinates)
    "road_area": np.array([
        [1240, 1293], [2370, 1129], [2459, 1167], [2555, 1167], [2607, 1155],
        [2637, 1141], [2633, 1103], [3183, 1048], [3714, 985], [3830, 1011],
        [3833, 2125], [3818, 2151], [1820, 2151], [1757, 2099], [1508, 1832],
        [1786, 1813], [1942, 1798], [1942, 1768], [1820, 1683], [1753, 1650],
        [1608, 1676], [1444, 1694], [1381, 1728], [1359, 1765], [1017, 1590],
        [1043, 1586], [1270, 1560], [1444, 1527], [1526, 1508], [1482, 1475],
        [1344, 1386], [1270, 1367], [1218, 1382], [1129, 1479], [1032, 1546],
        [985, 1570], [627, 1419], [672, 1378]
    ], dtype=np.int32),

    # Additional contextual regions
    "sidewalk_left": np.array([
        [300, 1572], [378, 1638], [642, 1750], [542, 1798], [337, 1869],
        [244, 1910], [230, 1932], [270, 1947], [854, 1898], [1173, 2154],
        [48, 2155], [7, 2136], [7, 1716]
    ], dtype=np.int32),
    "approach_bottom": np.array([
        [363, 1029], [159, 836], [22, 661], [-1, 316], [140, 264],
        [404, 368], [1013, 580], [1671, 828], [1854, 918], [802, 1045],
        [495, 1094]
    ], dtype=np.int32),
    "road_top": np.array([
        [2180, 947], [1664, 765], [1084, 568], [516, 372], [204, 267],
        [62, 193], [66, 160], [92, 119], [170, 85], [289, 89],
        [404, 152], [549, 212], [753, 245], [995, 286], [1199, 301],
        [1296, 338], [1716, 431], [2076, 535], [2392, 598], [2581, 654],
        [2670, 706], [2722, 754], [2838, 769], [3417, 918], [2396, 1018]
    ], dtype=np.int32),

    # Traffic light region of interest (x1, y1, x2, y2)
    "traffic_light_bbox": (2300, 730, 2345, 845),
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

VEHICLE_CLASSES = {"car", "bus", "truck", "motorcycle"}


def get_traffic_light_state(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    red_threshold: int = 25,
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

    # In HSV, red hue wraps around 0 and 180
    lower_red1 = np.array([0, 70, 70], dtype=np.uint8)
    upper_red1 = np.array([10, 255, 255], dtype=np.uint8)
    lower_red2 = np.array([170, 70, 70], dtype=np.uint8)
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
    # 1. Initialize YOLO detector
    local_weights = Path("weights/yolov8s.pt")
    model_path = str(local_weights) if local_weights.exists() else "yolov8s.pt"
    model = YOLO(model_path)

    # 2. Initialize Line Zones
    stop_bottom_arr = SCENE_CONFIG["stop_line_bottom"]
    stop_line_bottom = sv.LineZone(
        start=sv.Point(int(stop_bottom_arr[0][0]), int(stop_bottom_arr[0][1])),
        end=sv.Point(int(stop_bottom_arr[1][0]), int(stop_bottom_arr[1][1])),
    )

    stop_top_arr = SCENE_CONFIG["stop_line_top"]
    stop_line_top = sv.LineZone(
        start=sv.Point(int(stop_top_arr[0][0]), int(stop_top_arr[0][1])),
        end=sv.Point(int(stop_top_arr[1][0]), int(stop_top_arr[1][1])),
    )

    # 3. Initialize Polygon Zones
    zebra_main_zone = sv.PolygonZone(polygon=SCENE_CONFIG["zebra_main"])
    zebra_left_zone = sv.PolygonZone(polygon=SCENE_CONFIG["zebra_left"])
    road_area_zone = sv.PolygonZone(polygon=SCENE_CONFIG["road_area"])

    # 4. Initialize Multi-Object Tracker (ByteTrack)
    tracker = sv.ByteTrack()

    # 5. Open video stream
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 29.97)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_from_meta = (total_frames / fps) if (fps > 0 and total_frames > 0) else 0.0

    frame_idx = 0
    events: list[list] = []

    # 6. State tracking structures
    # track_history[track_id] = [(t_sec, cx, cy), ...]
    track_history: dict[int, list[tuple[float, float, float]]] = {}
    last_seen_time: dict[int, float] = {}

    # active_events[(track_id, label)] = {"label": str, "start_sec": float}
    active_events: dict[tuple[int, str], dict] = {}
    crossed_red_light: set[int] = set()

    tl_bbox = SCENE_CONFIG["traffic_light_bbox"]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        t_sec = frame_idx / fps

        # a) Determine current traffic light status
        tl_state = get_traffic_light_state(frame, tl_bbox)

        # b) Detect road users and update tracker
        results = model(frame, verbose=False, classes=list(COCO_ROAD_USERS.keys()))[0]
        detections = sv.Detections.from_ultralytics(results)
        tracked_detections = tracker.update_with_detections(detections)

        # Evaluate all zones at once on tracked detections to get boolean masks
        in_road = road_area_zone.trigger(tracked_detections)
        in_zebra_main = zebra_main_zone.trigger(tracked_detections)
        in_zebra_left = zebra_left_zone.trigger(tracked_detections)

        # LineZone trigger returns a tuple: (crossed_in, crossed_out)
        crossed_in, crossed_out = stop_line_bottom.trigger(tracked_detections)
        crossed_bottom = crossed_in | crossed_out

        # Track which IDs are seen in this frame
        current_frame_track_ids: set[int] = set()

        # c) Loop over tracked detections
        for i in range(len(tracked_detections)):
            if tracked_detections.tracker_id is None:
                continue

            track_id = int(tracked_detections.tracker_id[i])
            if track_id < 0:
                continue

            current_frame_track_ids.add(track_id)
            last_seen_time[track_id] = t_sec

            class_id = int(tracked_detections.class_id[i])
            class_name = COCO_ROAD_USERS.get(class_id, "unknown")

            # Calculate centroid (cx, cy)
            x1, y1, x2, y2 = tracked_detections.xyxy[i]
            cx = float((x1 + x2) / 2.0)
            cy = float((y1 + y2) / 2.0)

            # Update movement trajectory history
            if track_id not in track_history:
                track_history[track_id] = []
            track_history[track_id].append((t_sec, cx, cy))

            # Keep only the last ~100 entries (approx. 3-5 seconds of history)
            if len(track_history[track_id]) > 100:
                track_history[track_id] = track_history[track_id][-100:]

            # Calculate motion metrics over ~1 second
            dx, dy = get_direction(track_history[track_id], dt=1.0)
            speed = get_speed(track_history[track_id], dt=1.0)

            # -------------------------------------------------------------
            # 1. Logic for jaywalking:
            # -------------------------------------------------------------
            jw_key = (track_id, "jaywalking")
            if class_name == "pedestrian":
                # Inside road area, but outside all authorized crosswalks
                if in_road[i] and not in_zebra_main[i] and not in_zebra_left[i]:
                    if jw_key not in active_events:
                        active_events[jw_key] = {
                            "label": "jaywalking",
                            "start_sec": t_sec,
                        }
                else:
                    if jw_key in active_events:
                        start_sec = active_events[jw_key]["start_sec"]
                        if t_sec > start_sec:
                            events.append([round(start_sec, 3), round(t_sec, 3), "jaywalking"])
                        del active_events[jw_key]

            # -------------------------------------------------------------
            # 2. Logic for red_light:
            # -------------------------------------------------------------
            if class_name in VEHICLE_CLASSES:
                if crossed_bottom[i] and tl_state == "RED":
                    if track_id not in crossed_red_light:
                        crossed_red_light.add(track_id)
                        events.append([round(t_sec, 3), round(t_sec + 2.0, 3), "red_light"])

            # -------------------------------------------------------------
            # 3. Logic for wrong_way:
            # Top lanes (y < 900): expected right-to-left, wrong if moving left-to-right (dx > 50)
            # Bottom lanes (y > 1000): expected left-to-right, wrong if moving right-to-left (dx < -50)
            # -------------------------------------------------------------
            ww_key = (track_id, "wrong_way")
            if class_name in VEHICLE_CLASSES:
                is_wrong_way = False
                if cy < 900 and dx > 50:
                    is_wrong_way = True
                elif cy > 1000 and dx < -50:
                    is_wrong_way = True

                if is_wrong_way:
                    if ww_key not in active_events:
                        active_events[ww_key] = {
                            "label": "wrong_way",
                            "start_sec": t_sec,
                        }
                else:
                    if ww_key in active_events:
                        start_sec = active_events[ww_key]["start_sec"]
                        if t_sec > start_sec:
                            events.append([round(start_sec, 3), round(t_sec, 3), "wrong_way"])
                        del active_events[ww_key]

            # -------------------------------------------------------------
            # 4. Logic for stopped_vehicle:
            # Stationary (speed < 10 px in 1s) for >= 10.0 seconds
            # -------------------------------------------------------------
            stop_key = (track_id, "stopped_vehicle")
            if class_name in VEHICLE_CLASSES:
                is_stopped = speed < 10.0

                if is_stopped:
                    if stop_key not in active_events:
                        active_events[stop_key] = {
                            "label": "stopped_vehicle",
                            "start_sec": t_sec,
                        }
                else:
                    if stop_key in active_events:
                        start_sec = active_events[stop_key]["start_sec"]
                        duration = t_sec - start_sec
                        # Only log as violation if stationary for at least 10 seconds
                        if duration >= 10.0:
                            events.append([round(start_sec, 3), round(t_sec, 3), "stopped_vehicle"])
                        del active_events[stop_key]

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
                    else:
                        if e > s:
                            events.append([round(s, 3), round(e, 3), ev_label])
                    del active_events[(t_id, ev_label)]

        frame_idx += 1

    cap.release()

    # Determine final video duration
    final_duration = duration_from_meta if duration_from_meta > 0 else (frame_idx / fps)

    # Close any remaining active events at the end of the video
    for (t_id, ev_label), ev_info in list(active_events.items()):
        start_sec = ev_info["start_sec"]
        end_sec = final_duration
        if ev_label == "stopped_vehicle":
            if (end_sec - start_sec) >= 10.0:
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
