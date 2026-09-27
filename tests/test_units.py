"""Fast unit tests (no video, no GPU needed except where a model is loaded)."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.config import ENABLED_CLASSES
from src.perception import Observation
from src.postprocess import finalize_events
from src.registration import REF_IMAGE_PATH, estimate_scene_transform
from src.risk import pair_risk
from src.rules import Hysteresis, RuleEngine
from src.scene import SCENE_CONFIG, build_scene
from src.traffic_light import SignalState, classify, lamp_scores

IDENTITY = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


# ---------------------------------------------------------------- post-processing
def test_finalize_merges_clips_and_drops_blips():
    ev = [[1.0, 3.0, "jaywalking"], [3.5, 5.0, "jaywalking"], [10.0, 10.2, "jaywalking"],
          [8.0, 20.0, "congestion"], [0.0, 1.0, "illegal_turn"]]
    out = finalize_events(ev, duration=15.0)
    assert [1.0, 5.0, "jaywalking"] in out                     # merged across a 0.5 s gap
    assert all(not (e[2] == "jaywalking" and e[0] == 10.0) for e in out)   # 0.2 s blip dropped
    assert [8.0, 15.0, "congestion"] in out                    # clipped to the duration
    assert all(e[2] in ENABLED_CLASSES for e in out)
    for label in {e[2] for e in out}:                          # no same-class overlap
        segs = sorted(e[:2] for e in out if e[2] == label)
        assert all(a[1] < b[0] for a, b in zip(segs, segs[1:]))


def test_hysteresis_min_duration_and_gap():
    sink: list = []
    h = Hysteresis("x", min_duration=1.0, gap=0.5, sink=sink)
    for t in np.arange(0, 2.0, 0.1):
        h.update("k", t, True)
        h.sweep(t)
    h.sweep(3.0)
    assert len(sink) == 1 and sink[0][2] == "x" and sink[0][1] - sink[0][0] >= 1.0
    h.update("k", 5.0, True)
    h.sweep(6.0)                                               # a single sample is too short
    assert len(sink) == 1


# ---------------------------------------------------------------- traffic light
def _lamp_frame(color_bgr=None, lamp="red"):
    frame = np.full((2160, 3840, 3), 40, np.uint8)
    if color_bgr is not None:
        x, y = SCENE_CONFIG["main_signal_lamps"][lamp]
        cv2.circle(frame, (x, y), 4, color_bgr, -1)
    return frame


@pytest.mark.parametrize("color,lamp,expected", [
    ((30, 30, 230), "red", "RED"),
    ((170, 230, 40), "green", "GREEN"),
    (None, "red", "UNKNOWN"),
])
def test_signal_classification(color, lamp, expected):
    scene = build_scene(IDENTITY)
    scores = lamp_scores(_lamp_frame(color, lamp), scene["main_signal_lamps"], scene["px_scale"])
    assert classify(scores) == expected


def test_signal_state_debounce():
    s = SignalState()
    for t in np.arange(0, 1.0, 0.1):
        s.update(t, "RED")
    assert s.state == "RED"
    s.update(1.0, "GREEN")                                     # a single green sample is ignored
    assert s.state == "RED"
    for t in np.arange(1.1, 2.0, 0.1):
        s.update(t, "GREEN")
    assert s.state == "GREEN"
    assert s.time_in_state("RED", 0.0, 1.0) > 0.5


# ---------------------------------------------------------------- registration
def test_registration_of_reference_is_identity():
    ref = cv2.imread(str(REF_IMAGE_PATH))
    A, info = estimate_scene_transform([cv2.resize(ref, (3840, 2160))])
    assert info["status"] == "ok"
    assert np.allclose(A, IDENTITY, atol=0.02 * 3840) and abs(info["rot_deg"]) < 0.2


def test_registration_recovers_a_shift():
    ref = cv2.resize(cv2.imread(str(REF_IMAGE_PATH)), (3840, 2160))
    shifted = cv2.warpAffine(ref, np.float32([[1, 0, 60], [0, 1, -30]]), (3840, 2160))
    _, info = estimate_scene_transform([shifted])
    assert abs(info["dx_4k"] - 60) < 6 and abs(info["dy_4k"] + 30) < 6


# ---------------------------------------------------------------- rules
def _obs(t, items):
    tids = np.array([i[0] for i in items], dtype=int)
    boxes = np.array([i[1] for i in items], dtype=float).reshape(-1, 4)
    names = [i[2] for i in items]
    return Observation(t, tids, boxes, names, np.ones(len(items)))


def _engine():
    return RuleEngine(build_scene(IDENTITY), SignalState())


def test_congestion_ignores_brand_new_tracks():
    """Regression: vehicles without a speed estimate used to count as 'stopped'."""
    eng = _engine()
    boxes = [(900, 500, 1100, 620), (600, 400, 800, 520), (300, 300, 500, 420), (1300, 650, 1500, 770)]
    for k, t in enumerate(np.arange(0, 0.8, 0.1)):
        eng.update(_obs(t, [(k * 10 + i, b, "car") for i, b in enumerate(boxes)]))   # new ids each frame
    assert not [e for e in eng.finalize(10.0) if e[2] == "congestion"]


def test_rider_is_not_a_jaywalker():
    eng = _engine()
    moto = (1000, 560, 1120, 700)            # on lane_ltr
    rider = (1020, 470, 1100, 660)           # person whose feet are on the motorcycle
    for t in np.arange(0, 4.0, 0.1):
        dx = 40 * t
        eng.update(_obs(t, [(1, (moto[0] + dx, moto[1], moto[2] + dx, moto[3]), "motorcycle"),
                            (2, (rider[0] + dx, rider[1], rider[2] + dx, rider[3]), "pedestrian")]))
    assert not [e for e in eng.finalize(4.0) if e[2] == "jaywalking"]


def test_pedestrian_on_carriageway_is_jaywalking():
    eng = _engine()
    for t in np.arange(0, 4.0, 0.1):
        eng.update(_obs(t, [(7, (870, 740, 930, 900), "pedestrian")]))   # standing mid lane_ltr
    assert [e for e in eng.finalize(4.0) if e[2] == "jaywalking"]


# ---------------------------------------------------------------- Part B
def test_pair_risk_head_on_vs_diverging():
    head_on = pair_risk((0, 0), (300, 0), (150, 0), (-150, 0), size=100)
    diverging = pair_risk((0, 0), (300, 0), (-150, 0), (150, 0), size=100)
    passing = pair_risk((0, 0), (300, 400), (150, 0), (-150, 0), size=100)
    assert head_on > 0.5 and diverging == 0.0 and passing == 0.0


def test_pair_risk_slow_approach_does_not_overflow():
    """Regression: a very slow closing speed gave a huge TTC and math.exp overflowed."""
    assert pair_risk((0, 0), (50000, 0), (0.001, 0), (-0.001, 0), size=100) == 0.0


# ---------------------------------------------------------------- accident candidates + verifier
class _StubVerifier:
    def __init__(self, p):
        self.p, self.calls = p, []

    def __call__(self, kind, t_from, t_to, box, n_frames):
        self.calls.append((kind, round(t_from, 2), round(t_to, 2)))
        return self.p


def _crash_run(stub, drive_on=False):
    eng = RuleEngine(build_scene(IDENTITY), SignalState(), verifier=stub)
    for t in np.arange(0, 6.0, 0.1):
        x = 500 + 300 * t                                  # car 1 drives at ~1.3 diag/s ...
        if not drive_on:
            x = min(x, 900)                                # ... and stops on contact with car 2
        eng.update(_obs(t, [(1, (x, 560, x + 200, 680), "car"), (2, (1000, 560, 1200, 680), "car")]))
    return [e for e in eng.finalize(6.0) if e[2] == "accident"]


def test_collision_candidate_is_verified_and_reported():
    stub = _StubVerifier(0.95)
    events = _crash_run(stub)
    assert [c[0] for c in stub.calls].count("accident") == 4          # four verifier windows; later calls: smoke checks
    assert len(events) == 1 and 1.0 <= events[0][0] <= 1.6


def test_rejected_candidate_gives_no_accident():
    stub = _StubVerifier(0.1)
    assert not _crash_run(stub) and [c[0] for c in stub.calls] == ["accident"]   # no smoke watch without a crash


def test_vehicles_that_drive_on_are_not_asked_about():
    stub = _StubVerifier(0.95)
    assert not _crash_run(stub, drive_on=True) and not stub.calls


def test_weak_fire_hit_is_verified():
    stub = _StubVerifier(0.9)
    eng = RuleEngine(build_scene(IDENTITY), SignalState(), verifier=stub)
    for t in np.arange(0, 3.0, 0.1):
        obs = _obs(t, [])
        obs.anomaly = [("fire", 0.4, np.array([1000.0, 560.0, 1200.0, 680.0]))]
        eng.update(obs)
    events = [e for e in eng.finalize(10.0) if e[2] == "fire_smoke"]
    assert stub.calls and stub.calls[0][0] == "fire_smoke" and len(events) == 1


class _SmokeAfter:
    """Accepts the crash; reports smoke only from `t_smoke` on."""

    def __init__(self, t_smoke):
        self.t_smoke, self.fire_calls = t_smoke, 0

    def __call__(self, kind, t_from, t_to, box, n_frames):
        if kind == "accident":
            return 0.95
        self.fire_calls += 1
        return 0.9 if t_from >= self.t_smoke else 0.05


def test_smoke_after_a_crash_is_found_by_the_site_watch():
    stub = _SmokeAfter(t_smoke=4.5)
    eng = RuleEngine(build_scene(IDENTITY), SignalState(), verifier=stub)
    for t in np.arange(0, 10.0, 0.1):
        x = min(500 + 300 * t, 900)
        eng.update(_obs(t, [(1, (x, 560, x + 200, 680), "car"), (2, (1000, 560, 1200, 680), "car")]))
    events = eng.finalize(10.0)
    fire = [e for e in events if e[2] == "fire_smoke"]
    assert [e for e in events if e[2] == "accident"]
    assert len(fire) == 1 and 3.5 <= fire[0][0] <= 5.5 and fire[0][1] == 10.0
    assert stub.fire_calls >= 3


def test_head_on_hit_with_barely_touching_boxes_is_a_candidate():
    """Regression: two cars meeting nose to nose leave side-by-side boxes with ~zero IoU."""
    stub = _StubVerifier(0.95)
    eng = RuleEngine(build_scene(IDENTITY), SignalState(), verifier=stub)
    for t in np.arange(0, 6.0, 0.1):
        x1 = min(500 + 200 * t, 800)                       # right edge stops at 1000 ...
        x2 = max(1300 - 200 * t, 1003)                     # ... left edge stops at 1003: 3 px apart
        eng.update(_obs(t, [(1, (x1, 560, x1 + 200, 680), "car"), (2, (x2, 560, x2 + 200, 680), "car")]))
    assert [c[0] for c in stub.calls].count("accident") == 4
    assert [e for e in eng.finalize(6.0) if e[2] == "accident"]


def test_accident_needs_a_high_mean_over_the_verifier_windows():
    answers = iter([0.9, 0.5, 0.5, 0.5])                     # passes the screen, mean 0.6 < accept_p
    calls = []

    def verifier(kind, t_from, t_to, box, n_frames):
        calls.append(kind)
        return next(answers) if kind == "accident" else 0.0

    assert not _crash_run(verifier) and calls.count("accident") == 4
