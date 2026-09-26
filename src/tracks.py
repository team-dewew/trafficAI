"""Per-track motion state built from tracker output (ground point history, size, class votes)."""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np

from src.geometry import bottom_center, box_diag

HISTORY_SEC = 8.0


@dataclass
class TrackState:
    tid: int
    first_t: float
    last_t: float = 0.0
    box: np.ndarray = field(default_factory=lambda: np.zeros(4))
    hist: deque = field(default_factory=deque)       # (t, x, y) ground point
    diags: deque = field(default_factory=lambda: deque(maxlen=15))
    votes: Counter = field(default_factory=Counter)
    rider_votes: deque = field(default_factory=lambda: deque(maxlen=15))

    def update(self, t: float, box: np.ndarray, cls_name: str) -> None:
        self.last_t = t
        self.box = np.asarray(box, dtype=np.float64)
        x, y = bottom_center(self.box)
        self.hist.append((t, x, y))
        while self.hist and t - self.hist[0][0] > HISTORY_SEC:
            self.hist.popleft()
        self.diags.append(box_diag(self.box))
        self.votes[cls_name] += 1

    @property
    def cls(self) -> str:
        return self.votes.most_common(1)[0][0] if self.votes else "unknown"

    @property
    def pos(self) -> tuple[float, float]:
        return self.hist[-1][1], self.hist[-1][2]

    @property
    def diag(self) -> float:
        return float(np.median(self.diags)) if self.diags else 1.0

    @property
    def age(self) -> float:
        return self.last_t - self.first_t

    @property
    def is_rider(self) -> bool:
        return len(self.rider_votes) >= 3 and sum(self.rider_votes) >= 0.4 * len(self.rider_votes)

    def point_ago(self, dt: float) -> tuple[float, float, float] | None:
        """History sample closest to `dt` seconds before the latest, if history reaches back far enough."""
        if not self.hist:
            return None
        target = self.hist[-1][0] - dt
        if self.hist[0][0] > target + 0.3 * dt:
            return None
        best = min(self.hist, key=lambda p: abs(p[0] - target))
        return best

    def velocity(self, window: float = 1.0) -> tuple[float, float] | None:
        """Ground-point velocity (px/s) over the last `window` seconds, or None if too little history."""
        p0 = self.point_ago(window)
        if p0 is None:
            return None
        t1, x1, y1 = self.hist[-1]
        dt = t1 - p0[0]
        if dt < 0.5 * window:
            return None
        return (x1 - p0[1]) / dt, (y1 - p0[2]) / dt

    def speed_rel(self, window: float = 1.0) -> float | None:
        """Speed in body-diagonals per second (scale-free, works for near and far objects)."""
        v = self.velocity(window)
        if v is None:
            return None
        return float(np.hypot(*v)) / max(1.0, self.diag)

    def speed_between(self, newer: float, older: float) -> float | None:
        """Relative speed between two past offsets (seconds before now)."""
        a, b = self.point_ago(newer) if newer > 0 else self.hist[-1], self.point_ago(older)
        if a is None or b is None or a[0] <= b[0]:
            return None
        return float(np.hypot(a[1] - b[1], a[2] - b[2])) / (a[0] - b[0]) / max(1.0, self.diag)


class TrackStore:
    """All tracks seen in a video, pruned when they have been gone for a while."""

    def __init__(self, forget_after: float = 10.0) -> None:
        self.tracks: dict[int, TrackState] = {}
        self.forget_after = forget_after

    def update(self, t: float, tids, boxes, cls_names) -> list[TrackState]:
        current = []
        for tid, box, name in zip(tids, boxes, cls_names):
            tid = int(tid)
            tr = self.tracks.get(tid)
            if tr is None:
                tr = self.tracks[tid] = TrackState(tid=tid, first_t=t)
            tr.update(t, box, name)
            current.append(tr)
        for tid in [k for k, tr in self.tracks.items() if t - tr.last_t > self.forget_after]:
            del self.tracks[tid]
        return current
