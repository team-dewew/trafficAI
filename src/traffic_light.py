"""Traffic-signal state from the lamps of the lane_ltr signal head.

Each lit lamp is only a few pixels tall at 4K, so the signal is read from small
windows centred on the calibrated lamp positions (mapped through the per-video
registration), not from the area of a colour mask over the whole housing:

  red lamp    top-k mean of  R - max(G, B)
  green lamp  top-k mean of  G - R          (the LEDs are cyan-green)
  yellow lamp top-k mean of  min(R, G) - B

States: RED, YELLOW, GREEN, UNKNOWN. A new state is accepted only after it has
persisted for ``hold_sec`` of video time; UNKNOWN never counts as green or red,
so the red-light rules simply stay silent when the signal cannot be read.
"""
from __future__ import annotations

import numpy as np

from src.config import SIGNAL

TOP_K = 8


def lamp_scores(frame: np.ndarray, lamps: dict[str, tuple[float, float]], px_scale: float) -> dict[str, float]:
    """Colour evidence for each lamp (larger = more lit in that lamp's colour)."""
    h, w = frame.shape[:2]
    r = max(3, int(round(SIGNAL["lamp_radius_4k"] * px_scale)))
    out = {}
    for name, (x, y) in lamps.items():
        xi, yi = int(round(x)), int(round(y))
        x0, x1, y0, y1 = max(0, xi - r), min(w, xi + r + 1), max(0, yi - r), min(h, yi + r + 1)
        if x1 <= x0 or y1 <= y0:
            out[name] = 0.0
            continue
        win = frame[y0:y1, x0:x1].astype(np.int32)
        b, g, rr = win[..., 0], win[..., 1], win[..., 2]
        if name == "red":
            ness = rr - np.maximum(g, b)
        elif name == "green":
            ness = g - rr
        else:
            ness = np.minimum(rr, g) - b
        flat = np.sort(ness.ravel())
        out[name] = float(flat[-TOP_K:].mean())
    return out


def classify(scores: dict[str, float]) -> str:
    """Instantaneous state from lamp scores."""
    red = scores.get("red", 0.0) >= SIGNAL["red_on"]
    green = scores.get("green", 0.0) >= SIGNAL["green_on"]
    yellow = scores.get("yellow", 0.0) >= SIGNAL["yellow_on"]
    if red and not green:
        return "YELLOW" if yellow and scores["yellow"] > scores["red"] else "RED"
    if green and not red:
        return "GREEN"
    if yellow and not (red or green):
        return "YELLOW"
    return "UNKNOWN"


class SignalState:
    """Debounced signal state over time, with the time of the last change."""

    def __init__(self) -> None:
        self.state = "UNKNOWN"
        self.since = 0.0                  # time the current state was accepted
        self._cand = "UNKNOWN"
        self._cand_since = 0.0
        self._last_known_t = -1e9
        self.phases: list[tuple[str, float]] = []   # (state, start time) history

    def update(self, t: float, raw: str) -> str:
        if raw != "UNKNOWN":
            self._last_known_t = t
            if raw != self._cand:
                self._cand, self._cand_since = raw, t
            if self._cand != self.state and t - self._cand_since >= SIGNAL["hold_sec"]:
                self._set(self._cand, self._cand_since)
        elif self.state != "UNKNOWN" and t - self._last_known_t >= SIGNAL["unknown_after_sec"]:
            self._set("UNKNOWN", t)
        return self.state

    def _set(self, state: str, t: float) -> None:
        self.state, self.since = state, t
        self.phases.append((state, t))

    def age(self, t: float) -> float:
        return t - self.since

    def time_in_state(self, state: str, t0: float, t1: float) -> float:
        """Seconds of [t0, t1] spent in `state` according to the phase history."""
        total = 0.0
        for k, (s, start) in enumerate(self.phases):
            end = self.phases[k + 1][1] if k + 1 < len(self.phases) else t1
            if s == state:
                total += max(0.0, min(end, t1) - max(start, t0))
        return total

    def state_at(self, t: float) -> str:
        cur = "UNKNOWN"
        for s, start in self.phases:
            if start <= t:
                cur = s
            else:
                break
        return cur
