"""Part B: causal accident-risk estimator.

Only frames passed to ``step`` are used; the estimator never opens the video.
Every ``DETECT_EVERY`` calls it runs a light detector + tracker; between runs it
returns the previous score.

Risk signal: time-to-collision between pairs of road users on a collision
course, i.e. their closest approach is smaller than their size. It is computed
from ground-point tracks in pixels per second of *video time* (t_sec), so the
curve does not depend on how often step() is called. A pair has to stay on a
collision course for PERSIST consecutive detector updates before it counts,
which removes one-update velocity noise.
"""
from __future__ import annotations

import math

import numpy as np
import supervision as sv

from src.geometry import in_any
from src.models import load_yolo, predict
from src.perception import COCO_NAMES, MOTOR_VEHICLES, TWO_WHEELERS, suppress_duplicate_vehicles
from src.registration import estimate_scene_transform
from src.scene import build_scene
from src.tracking import make_tracker
from src.tracks import TrackStore

DETECT_EVERY = 3
ROAD_USER_IDS = [0, 1, 2, 3, 5, 7]
TTC_MID = 1.0            # seconds; the pair score crosses 0.5 here
TTC_SLOPE = 3.0
TTC_MAX = 6.0            # seconds; beyond this a pair contributes no risk
EMA_ALPHA = 0.35
MIN_MOVING = 0.5         # diag/s: at least one of the pair must be moving (~10 km/h; junction crashes are often slow)
MIN_REL_SPEED = 0.6      # relative speed, in joint sizes per second
PERSIST = 2              # consecutive detector updates on a collision course
COURSE_DMIN = 0.3        # closest approach (in joint sizes) that counts as a collision course
MIN_DIAG_4K = 150        # ignore far-field objects smaller than this (reference 4K px): image-space motion is unreliable there
SAME_DIR_COS = 0.6       # heading cosine above which a pair is "moving the same way"
SAME_LANE_LATERAL = 0.35 # same-way pairs count only in the same lane (lateral offset, in joint sizes) ...
REAR_END_CLOSING = 1.5   # ... and when closing fast (joint sizes per second)


def pair_risk(pa, pb, va, vb, size: float) -> float:
    """Risk in [0, 1] for two ground points with velocities (px, px/s) and a joint size (px)."""
    p = np.array([pb[0] - pa[0], pb[1] - pa[1]], dtype=np.float64)
    v = np.array([vb[0] - va[0], vb[1] - va[1]], dtype=np.float64)
    dist = float(np.hypot(*p))
    if dist < 0.5 * size:
        return 0.0                       # already overlapping: duplicate box or occlusion, not a forecast
    vv = float(v @ v)
    if vv < (MIN_REL_SPEED * size) ** 2:
        return 0.0                       # moving together (queue, platoon)
    closing = -float(p @ v) / dist
    if closing <= 0:
        return 0.0
    t_star = -float(p @ v) / vv
    d_min = float(np.hypot(*(p + v * t_star)))
    if d_min > COURSE_DMIN * size:
        return 0.0                       # they pass each other
    ttc = (dist - 0.5 * size) / closing
    if ttc > TTC_MAX:
        return 0.0                       # far in the future (also avoids exp overflow)
    base = 1.0 / (1.0 + math.exp(TTC_SLOPE * (ttc - TTC_MID)))
    return base * (1.0 if d_min < 0.5 * COURSE_DMIN * size else 0.6)


def _crossing_or_rear_end(pa, pb, va, vb, size: float) -> bool:
    """Both moving: keep crossing / head-on pairs; keep same-way pairs only for a fast
    closing in the same lane. In perspective, vehicles in adjacent lanes moving away
    from the camera converge in image space, which would otherwise look like a collision course."""
    na, nb = float(np.hypot(*va)), float(np.hypot(*vb))
    cos = (va[0] * vb[0] + va[1] * vb[1]) / (na * nb)
    if cos <= SAME_DIR_COS:
        return True
    ux, uy = (va[0] / na + vb[0] / nb), (va[1] / na + vb[1] / nb)
    n = math.hypot(ux, uy)
    ux, uy = ux / n, uy / n
    dx, dy = pb[0] - pa[0], pb[1] - pa[1]
    lateral = abs(-dx * uy + dy * ux) / size
    closing = abs((vb[0] - va[0]) * ux + (vb[1] - va[1]) * uy) / size
    return lateral < SAME_LANE_LATERAL and closing >= REAR_END_CLOSING


class RiskCore:
    def reset(self, meta: dict, with_model: bool = True) -> None:
        self.meta = meta
        fps = float(meta.get("fps") or 25.0)
        self.model = load_yolo("yolov8n.pt") if with_model else None
        self.tracker = make_tracker(fps, DETECT_EVERY)
        self.store = TrackStore(forget_after=3.0)
        self.calls = 0
        self.ema = 0.0
        self.scene: dict | None = None
        self.streak: dict[tuple[int, int], int] = {}

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        self.calls += 1
        if (self.calls - 1) % DETECT_EVERY != 0:
            return float(self.ema)
        if self.scene is None:
            A, _ = estimate_scene_transform([frame])   # uses only the current frame
            self.scene = build_scene(A)
        tids, boxes, names = self._detect(frame)
        return self._score(t_sec, tids, boxes, names)

    def _detect(self, frame: np.ndarray):
        res = predict(self.model, frame, imgsz=640, conf=0.25, classes=ROAD_USER_IDS)
        det = suppress_duplicate_vehicles(sv.Detections.from_ultralytics(res), 0.6)
        det = self.tracker.update_with_detections(det)
        if det.tracker_id is None or len(det) == 0:
            return [], np.zeros((0, 4)), []
        return det.tracker_id.astype(int), det.xyxy.astype(np.float64), [COCO_NAMES.get(int(c), "unknown") for c in det.class_id]

    def _score(self, t_sec: float, tids, boxes, names) -> float:
        tracks = self.store.update(t_sec, tids, boxes, names)
        road = self.scene["road"]
        rtl = self.scene["lane_rtl"]
        safe = self.scene["crosswalks"] + self.scene["sidewalks"] + self.scene["ped_refuge"]
        carriers = [tr.box for tr in tracks if tr.cls in TWO_WHEELERS or tr.cls in {"car", "bus", "truck"}]
        users = []
        for tr in tracks:
            vel = tr.velocity(0.6)
            if vel is None or not in_any(road, *tr.pos):
                continue
            if tr.diag < MIN_DIAG_4K * self.scene["px_scale"]:
                continue
            if tr.cls == "pedestrian":
                x, y = tr.pos
                if any(b[0] <= x <= b[2] and b[1] <= y <= b[3] + 0.15 * (b[3] - b[1]) for b in carriers):
                    continue                      # rider or passenger
                if in_any(safe, x, y):
                    continue                      # on a crossing / sidewalk
            elif tr.cls not in MOTOR_VEHICLES and tr.cls not in TWO_WHEELERS:
                continue
            users.append((tr, vel))

        risk = 0.0
        streak: dict[tuple[int, int], int] = {}
        for i in range(len(users)):
            a, va = users[i]
            for j in range(i + 1, len(users)):
                b, vb = users[j]
                if a.cls == "pedestrian" and b.cls == "pedestrian":
                    continue
                if in_any([rtl], *a.pos) and in_any([rtl], *b.pos):
                    continue          # far carriageway: image-space geometry too compressed to judge
                sa = float(np.hypot(*va)) / max(1.0, a.diag)
                sb = float(np.hypot(*vb)) / max(1.0, b.diag)
                if max(sa, sb) < MIN_MOVING:
                    continue
                size = 0.5 * (a.diag + b.diag)
                if min(sa, sb) >= 0.3 and not _crossing_or_rear_end(a.pos, b.pos, va, vb, size):
                    continue
                r = pair_risk(a.pos, b.pos, va, vb, size)
                if r <= 0.0:
                    continue
                key = (min(a.tid, b.tid), max(a.tid, b.tid))
                streak[key] = self.streak.get(key, 0) + 1
                if streak[key] >= PERSIST:
                    risk = max(risk, r)
        self.streak = streak

        self.ema = (1.0 - EMA_ALPHA) * self.ema + EMA_ALPHA * risk
        return float(min(1.0, max(0.0, self.ema)))
