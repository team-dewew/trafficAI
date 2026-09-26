"""ByteTrack factory (supervision >= 0.26 argument names)."""
from __future__ import annotations

import warnings

import supervision as sv


def make_tracker(fps: float, stride: int = 1) -> sv.ByteTrack:
    """Tracker for detections arriving every `stride` frames of a `fps` video.

    Lost tracks are kept for ~4 s of video time regardless of stride.
    """
    update_rate = max(1.0, fps / max(1, stride))
    with warnings.catch_warnings():
        # supervision 0.30 marks ByteTrack deprecated (removal in 0.31); requirements pin 0.30.x.
        warnings.simplefilter("ignore", FutureWarning)
        return sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=120,            # frames at 30 Hz, rescaled by frame_rate -> 4 s
            minimum_matching_threshold=0.8,
            frame_rate=int(round(update_rate)),
        )
