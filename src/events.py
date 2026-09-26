"""Part A pipeline: registration -> perception -> signal -> rules -> post-processing."""
from __future__ import annotations

import time

import cv2

from src.config import PERCEPTION
from src.perception import Perception
from src.postprocess import finalize_events
from src.registration import estimate_scene_transform, sample_frames
from src.rules import RuleEngine
from src.scene import build_scene
from src.traffic_light import SignalState, classify, lamp_scores


def open_scene(video_path: str) -> tuple[dict, dict]:
    """Register the video against the reference frame and build its scene layout."""
    A, info = estimate_scene_transform(sample_frames(video_path))
    return build_scene(A), info


def run_part_a(video_path: str, progress_callback=None, obs_sink: list | None = None,
               settings: dict | None = None) -> tuple[list[list], dict]:
    """Return (events, diagnostics). `obs_sink`, if given, receives (observation, raw_signal)
    for every processed frame so the rules can be replayed offline. `settings` overrides
    keys of config.PERCEPTION (used by the website's CPU demo; the submission uses defaults)."""
    cfg = {**PERCEPTION, **(settings or {})}
    t_start = time.perf_counter()
    scene, reg_info = open_scene(video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], {"error": "cannot open video"}
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = n_frames / fps if n_frames > 0 else 0.0

    stride = int(cfg["stride"])
    perception = Perception(fps, stride, settings=cfg)
    signal = SignalState()
    engine = RuleEngine(scene, signal)

    idx = 0
    t_loop = time.perf_counter()        # budget guard measures the frame loop only
    try:
        while True:
            if idx % stride != 0:
                if not cap.grab():
                    break
                idx += 1
                continue
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            t = idx / fps
            raw = classify(lamp_scores(frame, scene["main_signal_lamps"], scene["px_scale"]))
            signal.update(t, raw)
            obs = perception(frame, t)
            engine.update(obs)
            if obs_sink is not None:
                obs_sink.append((obs, raw))
            if progress_callback is not None and n_frames > 0 and (idx // stride) % 8 == 0:
                progress_callback(idx, n_frames)
            idx += 1
            # Time guard: the harness budget is 3x duration for Part A + Part B together
            # (Part B needs ~0.7x). If perception runs slower than `budget_factor` x
            # real time, halve its rate for the rest of the video instead of failing.
            if stride == cfg["stride"] and idx > fps * 30 and (idx // stride) % 30 == 0:
                if time.perf_counter() - t_loop > cfg["budget_factor"] * (idx / fps):
                    stride *= 2
    finally:
        cap.release()

    if duration <= 0:
        duration = idx / fps
    raw_events = engine.finalize(duration)
    events = finalize_events(raw_events, duration)
    diag = {
        "registration": reg_info,
        "stride": stride,
        "frames": idx,
        "signal_phases": [(s, round(t, 2)) for s, t in signal.phases],
        "part_a_sec": round(time.perf_counter() - t_start, 1),
        "duration": duration,
        "scene": scene,
        "signal": signal,
    }
    return events, diag


def replay_rules(video_path: str, cached: list, duration: float) -> list[list]:
    """Re-run only the rule engine on cached (observation, raw_signal) pairs."""
    scene, _ = open_scene(video_path)
    signal = SignalState()
    engine = RuleEngine(scene, signal)
    for obs, raw in cached:
        signal.update(obs.t, raw)
        engine.update(obs)
    return finalize_events(engine.finalize(duration), duration)
