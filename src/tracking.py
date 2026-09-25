import supervision as sv

def make_tracker(fps: float, stride: int = 1) -> sv.ByteTrack:
    """Create a configured ByteTrack instance for the given fps and stride."""
    effective_fps = max(1.0, fps / max(1, stride))
    return sv.ByteTrack(
        track_activation_threshold=0.25,
        lost_track_buffer=int(fps * 4),  # Use raw fps for buffer size if we want 4 seconds
        frame_rate=int(round(effective_fps))
    )
