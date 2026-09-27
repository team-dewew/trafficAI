"""Scene layout of the fixed road camera, hand-calibrated on the reference frame.

All coordinates are in the reference 4K frame (3840x2160, ``assets/scene_ref.jpg``
downscaled). ``build_scene`` maps them onto a specific video with the similarity
transform from ``src.registration`` (which also absorbs any resolution change).

Zone semantics (see docs/scene.md):
  stop_line_strict      stop line vehicles in lane_ltr must not cross on red
  stop_line_tolerance   queue tolerance line: stopping before it on red is allowed
  yield_ped_line        right-side line where drivers must yield to pedestrians
  crosswalks            zebra crossings (pedestrians allowed)
  lane_ltr / lane_rtl   the two carriageways of the main road and their directions
  intersection_core     central junction box
  right_turn_zone       right-turn slip from lane_ltr (must yield at the zebra)
  lower_core            lower part of the junction
  ped_refuge            islands pedestrians may stand on (vehicles may not)
  barriers              physical road dividers (a vehicle hitting them is a crash)
  sidewalks             off-carriageway pedestrian areas
  main_signal lamps     red / yellow / green lamp centres of the lane_ltr signal
"""
from __future__ import annotations

import numpy as np

from src.registration import transform_points

SCENE_CONFIG: dict = {
    # LINES (start, end)
    "stop_line_strict": np.array([[1878, 925], [481, 1111]], dtype=np.int32),
    "stop_line_tolerance": np.array([[2164, 1048], [645, 1267]], dtype=np.int32),
    "yield_ped_line": np.array([[2625, 1115], [3717, 977]], dtype=np.int32),

    # CROSSWALKS (zebras)
    "crosswalks": [
        np.array([[2150, 1033], [2410, 1137], [2413, 1181], [615, 1464], [671, 1378], [652, 1285], [619, 1245]], dtype=np.int32),
        np.array([[3766, 988], [2633, 1126], [2603, 1092], [2558, 1092], [2491, 1092], [2380, 1037], [2376, 1025], [2373, 1014], [3439, 895]], dtype=np.int32),
        np.array([[344, 1575], [656, 1404], [1005, 1590], [1355, 1757], [1440, 1828], [1496, 1835], [1930, 2155], [1132, 2159], [853, 1880], [664, 1750], [214, 1605]], dtype=np.int32),
    ],

    # ROAD SECTIONS
    "lane_ltr": np.array([[1707, 847], [2198, 1055], [630, 1259], [441, 1096], [363, 1029], [43, 717], [21, 487], [62, 219], [166, 260], [192, 301], [463, 390], [1028, 591]], dtype=np.int32),
    "lane_rtl": np.array([[2005, 880], [1459, 698], [1184, 598], [671, 427], [374, 331], [188, 260], [147, 178], [32, 115], [36, 78], [117, 78], [273, 115], [489, 193], [727, 260], [972, 308], [1184, 320], [1336, 360], [1670, 427], [2417, 624], [2711, 750], [3439, 929], [2399, 1029], [2176, 933]], dtype=np.int32),
    "intersection_core": np.array([[2428, 1189], [2629, 1141], [3781, 996], [3836, 1018], [3836, 2028], [3836, 2133], [3810, 2155], [3714, 2155], [1949, 2155], [1511, 1835], [1949, 1791], [1945, 1772], [1745, 1642], [1392, 1709], [1362, 1742], [1031, 1594], [1533, 1508], [1317, 1363]], dtype=np.int32),
    "right_turn_zone": np.array([[678, 1367], [1139, 1326], [1217, 1341], [1217, 1382], [1132, 1479], [1002, 1568], [641, 1746], [337, 1858], [155, 1936], [6, 2002], [10, 1750], [203, 1642], [511, 1505]], dtype=np.int32),
    "lower_core": np.array([[6, 1947], [259, 1950], [853, 1898], [1106, 2147], [43, 2147], [6, 2129]], dtype=np.int32),

    # ISLANDS & DIVIDERS
    "ped_refuge": [
        np.array([[2216, 1051], [2410, 1029], [2532, 1070], [2347, 1111]], dtype=np.int32),  # zone 18
        np.array([[236, 1913], [645, 1754], [842, 1880], [608, 1917], [244, 1936]], dtype=np.int32),  # zone 11
        np.array([[1016, 1568], [1225, 1393], [1288, 1367], [1511, 1497]], dtype=np.int32),  # zone 12
        np.array([[1381, 1750], [1403, 1716], [1760, 1657], [1927, 1772], [1897, 1780], [1492, 1817], [1444, 1820]], dtype=np.int32),  # zone 13
    ],
    "barriers": [
        np.array([[530, 379], [920, 513], [1410, 684], [1871, 836], [2205, 951], [2410, 1040], [2272, 1063], [2205, 1059], [1864, 899], [1358, 710], [838, 524], [526, 409], [229, 308], [184, 275], [229, 275]], dtype=np.int32),  # zone 16
        np.array([[2543, 1063], [2603, 1066], [2647, 1122], [2618, 1152], [2517, 1163], [2443, 1148], [2369, 1092]], dtype=np.int32),  # zone 17
    ],

    # OFF-CARRIAGEWAY PEDESTRIAN AREAS
    "sidewalks": [
        np.array([[114, 854], [470, 1148], [615, 1274], [656, 1349], [634, 1397], [500, 1471], [288, 1568], [6, 1702], [10, 1092], [10, 747]], dtype=np.int32),
        np.array([[1076, 149], [1641, 275], [2194, 364], [2688, 457], [3357, 583], [3829, 717], [3825, 914], [3773, 944], [3721, 973], [3350, 884], [2792, 732], [2621, 687], [2387, 591], [1852, 457], [1425, 368], [1269, 316], [1087, 305], [935, 290], [808, 271], [615, 189], [604, 123], [719, 82], [987, 141]], dtype=np.int32),
    ],

    # TRAFFIC SIGNAL of lane_ltr: lamp centres measured on the reference frame
    "main_signal_lamps": {"red": (2328, 751), "yellow": (2326, 783), "green": (2323, 816)},
    "main_signal": (2290, 720, 2360, 860),  # housing box, used for drawing only
    "ped_signal": (500, 1000, 560, 1120),
}

# Direction of travel in lane_ltr (unit vector, reference frame): along the
# median towards the stop line / junction. lane_rtl flows the opposite way.
FLOW_LTR = np.array([0.935, 0.355])

# A point well upstream of the stop lines inside lane_ltr (tells which side of
# each stop line is "before the line").
LTR_UPSTREAM_POINT = (700, 700)


def _xf(arr: np.ndarray, A: np.ndarray) -> np.ndarray:
    return np.round(transform_points(arr, A)).astype(np.int32).reshape(np.asarray(arr).shape)


def build_scene(A: np.ndarray) -> dict:
    """Map SCENE_CONFIG through the 2x3 transform ``A`` (reference 4K -> video px)."""
    out: dict = {}
    for key, val in SCENE_CONFIG.items():
        if isinstance(val, np.ndarray):
            out[key] = _xf(val, A)
        elif isinstance(val, list):
            out[key] = [_xf(p, A) for p in val]
        elif key == "main_signal_lamps":
            out[key] = {k: tuple(float(c) for c in transform_points(np.array([v]), A)[0]) for k, v in val.items()}
        elif isinstance(val, tuple) and len(val) == 4:
            corners = transform_points(np.array([[val[0], val[1]], [val[2], val[3]]]), A)
            out[key] = tuple(int(round(c)) for c in corners.ravel())
    rot = A[:, :2] / max(1e-9, float(np.sqrt(abs(np.linalg.det(A[:, :2])))))
    out["flow_ltr"] = rot @ FLOW_LTR
    out["flow_rtl"] = -out["flow_ltr"]
    out["ltr_upstream"] = tuple(transform_points(np.array([LTR_UPSTREAM_POINT]), A)[0])
    # pixels of the video per reference 4K pixel (for size-dependent constants)
    out["px_scale"] = float(np.sqrt(abs(np.linalg.det(A[:, :2]))))
    out["road"] = [out[k] for k in ("lane_ltr", "lane_rtl", "intersection_core", "right_turn_zone", "lower_core")]
    return out
