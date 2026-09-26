"""Perception front-end: detector + duplicate suppression + ByteTrack + anomaly model.

Produces one ``Observation`` per processed frame. Everything downstream (rules,
renderer) consumes observations only, so the rule engine can also be replayed
from cached observations during development.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import supervision as sv

from src.config import PERCEPTION
from src.geometry import box_iou
from src.models import load_yolo, predict
from src.tracking import make_tracker

COCO_NAMES = {
    0: "pedestrian", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck",
    15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
    24: "backpack", 25: "umbrella", 28: "suitcase",
}
VEHICLES = {"car", "bus", "truck", "motorcycle", "bicycle"}
MOTOR_VEHICLES = {"car", "bus", "truck", "motorcycle"}
TWO_WHEELERS = {"motorcycle", "bicycle"}
OBSTACLES = {"cat", "dog", "horse", "sheep", "cow", "backpack", "umbrella", "suitcase"}


@dataclass
class Observation:
    t: float
    tids: np.ndarray                 # (N,) tracker ids
    boxes: np.ndarray                # (N, 4) xyxy in video pixels
    names: list[str]                 # (N,) class names
    conf: np.ndarray                 # (N,)
    anomaly: list[tuple[str, float, np.ndarray]] | None = field(default=None)  # None when not evaluated


def suppress_duplicate_vehicles(det: sv.Detections, iou_thr: float) -> sv.Detections:
    """Class-agnostic NMS among car/bus/truck: one vehicle detected as two classes
    would otherwise become two tracks (fake near-misses, doubled events)."""
    if len(det) < 2:
        return det
    names = [COCO_NAMES.get(int(c), "") for c in det.class_id]
    cand = [i for i, n in enumerate(names) if n in {"car", "bus", "truck"}]
    cand.sort(key=lambda i: -float(det.confidence[i]))
    drop: set[int] = set()
    for a_pos, a in enumerate(cand):
        if a in drop:
            continue
        for b in cand[a_pos + 1:]:
            if b not in drop and box_iou(det.xyxy[a], det.xyxy[b]) > iou_thr:
                drop.add(b)
    if not drop:
        return det
    keep = np.array([i not in drop for i in range(len(det))])
    return det[keep]


class Perception:
    def __init__(self, fps: float, stride: int, use_anomaly: bool = True) -> None:
        cfg = PERCEPTION
        self.detector = load_yolo(cfg["detector"])
        self.anomaly_model = load_yolo(cfg["anomaly_model"]) if use_anomaly else None
        self.tracker = make_tracker(fps, stride)
        self.imgsz = cfg["imgsz"]
        self.conf = cfg["conf"]
        self.anomaly_every_sec = cfg["anomaly_every_sec"]
        self._next_anomaly_t = 0.0
        self._classes = list(COCO_NAMES)

    def __call__(self, frame: np.ndarray, t: float) -> Observation:
        res = predict(self.detector, frame, imgsz=self.imgsz, conf=self.conf, classes=self._classes)
        det = sv.Detections.from_ultralytics(res)
        det = suppress_duplicate_vehicles(det, PERCEPTION["dup_iou"])
        det = self.tracker.update_with_detections(det)
        if det.tracker_id is None or len(det) == 0:
            obs = Observation(t, np.zeros(0, int), np.zeros((0, 4)), [], np.zeros(0))
        else:
            obs = Observation(
                t,
                det.tracker_id.astype(int),
                det.xyxy.astype(np.float64),
                [COCO_NAMES.get(int(c), "unknown") for c in det.class_id],
                det.confidence.astype(np.float64),
            )
        if self.anomaly_model is not None and t >= self._next_anomaly_t:
            self._next_anomaly_t = t + self.anomaly_every_sec
            ares = predict(self.anomaly_model, frame, imgsz=640, conf=0.25)
            names = self.anomaly_model.names
            obs.anomaly = [
                (str(names[int(c)]).lower(), float(p), b.astype(np.float64))
                for c, p, b in zip(ares.boxes.cls.cpu().numpy(), ares.boxes.conf.cpu().numpy(), ares.boxes.xyxy.cpu().numpy())
            ]
        return obs
