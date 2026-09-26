"""Small geometric helpers shared by the rules, the risk estimator and the renderer."""
from __future__ import annotations

import cv2
import numpy as np


def poly_dist(poly: np.ndarray, x: float, y: float) -> float:
    """Signed distance to a polygon edge: > 0 inside, < 0 outside (pixels)."""
    return float(cv2.pointPolygonTest(poly.reshape(-1, 1, 2), (float(x), float(y)), True))


def in_any(polys: list[np.ndarray], x: float, y: float, margin: float = 0.0) -> bool:
    """True if (x, y) lies at least `margin` px inside any of the polygons."""
    return any(poly_dist(p, x, y) >= margin for p in polys)


def which_poly(polys: list[np.ndarray], x: float, y: float, margin: float = 0.0) -> int:
    """Index of the first polygon containing (x, y), or -1."""
    for k, p in enumerate(polys):
        if poly_dist(p, x, y) >= margin:
            return k
    return -1


def line_side(line: np.ndarray, x: float, y: float) -> float:
    """Signed side of point (x, y) relative to the directed segment line[0] -> line[1]."""
    (x1, y1), (x2, y2) = np.asarray(line, dtype=np.float64)
    return float((x2 - x1) * (y - y1) - (y2 - y1) * (x - x1))


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two [x1, y1, x2, y2] boxes."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0.0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0


def box_diag(b: np.ndarray) -> float:
    return float(np.hypot(b[2] - b[0], b[3] - b[1]))


def bottom_center(b: np.ndarray) -> tuple[float, float]:
    """Ground contact point of a box (image-space proxy for the object's road position)."""
    return float((b[0] + b[2]) / 2.0), float(b[3])
