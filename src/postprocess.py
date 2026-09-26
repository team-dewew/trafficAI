"""Final clean-up of event segments so they satisfy the task's format rules."""
from __future__ import annotations

from src.config import DEFAULT_MERGE_GAP, ENABLED_CLASSES, MERGE_GAP, MIN_SEGMENT


def finalize_events(events: list[list], duration: float, enabled: list[str] | None = None) -> list[list]:
    """Clip to [0, duration], keep enabled classes, merge same-class segments that
    overlap or nearly touch, drop sub-`MIN_SEGMENT` blips. Output is sorted."""
    enabled_set = set(ENABLED_CLASSES if enabled is None else enabled)
    by_class: dict[str, list[list[float]]] = {}
    for item in events:
        s, e, label = float(item[0]), float(item[1]), str(item[2])
        if label not in enabled_set:
            continue
        s, e = max(0.0, s), (min(e, duration) if duration > 0 else e)
        if e > s:
            by_class.setdefault(label, []).append([s, e])

    out: list[list] = []
    for label, segs in by_class.items():
        segs.sort()
        gap = MERGE_GAP.get(label, DEFAULT_MERGE_GAP)
        merged = [segs[0]]
        for s, e in segs[1:]:
            if s - merged[-1][1] <= gap:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        out.extend([round(s, 3), round(e, 3), label] for s, e in merged if e - s >= MIN_SEGMENT)
    out.sort(key=lambda x: (x[0], x[1], x[2]))
    return out
