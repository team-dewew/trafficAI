import numpy as np
import copy
import supervision as sv

# High-Precision 21-Zone Scene Configuration (calibrated for 4K 3840x2160)
SCENE_CONFIG: dict[str, list[np.ndarray] | np.ndarray | tuple[int, int, int, int]] = {
    # LINES
    "stop_line_strict": np.array([[1878, 925], [481, 1111]], dtype=np.int32),      # strict red light stop (was stop_line_red)
    "stop_line_tolerance": np.array([[2164, 1048], [645, 1267]], dtype=np.int32),  # allowed to wait here in jam (was stop_line_jam)
    "yield_ped_line": np.array([[2625, 1115], [3717, 977]], dtype=np.int32),    # Right side yield line

    # CROSSWALKS (Zebras)
    "crosswalks": [
        np.array([[2150, 1033], [2410, 1137], [2413, 1181], [615, 1464], [671, 1378], [652, 1285], [619, 1245]], dtype=np.int32),
        np.array([[3766, 988], [2633, 1126], [2603, 1092], [2558, 1092], [2491, 1092], [2380, 1037], [2376, 1025], [2373, 1014], [3439, 895]], dtype=np.int32),
        np.array([[344, 1575], [656, 1404], [1005, 1590], [1355, 1757], [1440, 1828], [1496, 1835], [1930, 2155], [1132, 2159], [853, 1880], [664, 1750], [214, 1605]], dtype=np.int32),
    ],

    # ROAD SECTIONS
    "lane_ltr": np.array([[1707, 847], [2198, 1055], [630, 1259], [441, 1096], [363, 1029], [43, 717], [21, 487], [62, 219], [166, 260], [192, 301], [463, 390], [1028, 591]], dtype=np.int32), # Left to Right
    "lane_rtl": np.array([[2005, 880], [1459, 698], [1184, 598], [671, 427], [374, 331], [188, 260], [147, 178], [32, 115], [36, 78], [117, 78], [273, 115], [489, 193], [727, 260], [972, 308], [1184, 320], [1336, 360], [1670, 427], [2417, 624], [2711, 750], [3439, 929], [2399, 1029], [2176, 933]], dtype=np.int32), # Right to Left
    "intersection_core": np.array([[2428, 1189], [2629, 1141], [3781, 996], [3836, 1018], [3836, 2028], [3836, 2133], [3810, 2155], [3714, 2155], [1949, 2155], [1511, 1835], [1949, 1791], [1945, 1772], [1745, 1642], [1392, 1709], [1362, 1742], [1031, 1594], [1533, 1508], [1317, 1363]], dtype=np.int32),
    "right_turn_zone": np.array([[678, 1367], [1139, 1326], [1217, 1341], [1217, 1382], [1132, 1479], [1002, 1568], [641, 1746], [337, 1858], [155, 1936], [6, 2002], [10, 1750], [203, 1642], [511, 1505]], dtype=np.int32),
    "lower_core": np.array([[6, 1947], [259, 1950], [853, 1898], [1106, 2147], [43, 2147], [6, 2129]], dtype=np.int32),

    # ISLANDS & DIVIDERS
    "ped_refuge": [
        np.array([[2216, 1051], [2410, 1029], [2532, 1070], [2347, 1111]], dtype=np.int32), # Ped island (18)
        np.array([[236, 1913], [645, 1754], [842, 1880], [608, 1917], [244, 1936]], dtype=np.int32), # (11)
        np.array([[1016, 1568], [1225, 1393], [1288, 1367], [1511, 1497]], dtype=np.int32), # (12)
        np.array([[1381, 1750], [1403, 1716], [1760, 1657], [1927, 1772], [1897, 1780], [1492, 1817], [1444, 1820]], dtype=np.int32), # (13)
    ],
    
    "barriers": [
        np.array([[530, 379], [920, 513], [1410, 684], [1871, 836], [2205, 951], [2410, 1040], [2272, 1063], [2205, 1059], [1864, 899], [1358, 710], [838, 524], [526, 409], [229, 308], [184, 275], [229, 275]], dtype=np.int32), # (16)
        np.array([[2543, 1063], [2603, 1066], [2647, 1122], [2618, 1152], [2517, 1163], [2443, 1148], [2369, 1092]], dtype=np.int32), # (17)
    ],

    # SAFE SIDEWALKS (Pedestrians here = safe)
    "sidewalks": [
        np.array([[114, 854], [470, 1148], [615, 1274], [656, 1349], [634, 1397], [500, 1471], [288, 1568], [6, 1702], [10, 1092], [10, 747]], dtype=np.int32),
        np.array([[1076, 149], [1641, 275], [2194, 364], [2688, 457], [3357, 583], [3829, 717], [3825, 914], [3773, 944], [3721, 973], [3350, 884], [2792, 732], [2621, 687], [2387, 591], [1852, 457], [1425, 368], [1269, 316], [1087, 305], [935, 290], [808, 271], [615, 189], [604, 123], [719, 82], [987, 141]], dtype=np.int32),
    ],

    # TRAFFIC LIGHTS
    "main_signal": (2290, 720, 2360, 860),
    "ped_signal": (500, 1000, 560, 1120),
}


def build_scene(W: int, H: int, dx: int = 0, dy: int = 0) -> dict:
    """Returns a scaled and shifted scene configuration with supervision zones.
    All coordinates are scaled from 4K (3840x2160) to (W, H).
    """
    sx = W / 3840.0
    sy = H / 2160.0
    
    offset = np.array([dx, dy], dtype=np.int32)
    shifted: dict = copy.deepcopy(SCENE_CONFIG)
    
    for key, val in shifted.items():
        if isinstance(val, np.ndarray):
            shifted[key] = val + offset
        elif isinstance(val, list):
            shifted[key] = [
                (item + offset) if isinstance(item, np.ndarray) else copy.deepcopy(item)
                for item in val
            ]
        elif isinstance(val, tuple) and len(val) == 4:
            x1, y1, x2, y2 = val
            shifted[key] = (x1 + dx, y1 + dy, x2 + dx, y2 + dy)
            
    # Now scale everything
    def _scale_poly(poly: np.ndarray) -> np.ndarray:
        return (poly.astype(np.float64) * np.array([sx, sy])).astype(np.int32)

    scaled: dict = {}
    for key, val in shifted.items():
        if isinstance(val, np.ndarray):
            scaled[key] = _scale_poly(val)
        elif isinstance(val, list):
            scaled[key] = [_scale_poly(item) for item in val]
        elif isinstance(val, tuple) and len(val) == 4:
            x1, y1, x2, y2 = val
            scaled[key] = (int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy))

    # Build supervision zones
    zones = {}
    
    stop_strict_pts = scaled["stop_line_strict"]
    zones["stop_line_strict"] = sv.LineZone(
        start=sv.Point(int(stop_strict_pts[0][0]), int(stop_strict_pts[0][1])),
        end=sv.Point(int(stop_strict_pts[1][0]), int(stop_strict_pts[1][1])),
    )
    
    stop_tol_pts = scaled["stop_line_tolerance"]
    zones["stop_line_tolerance"] = sv.LineZone(
        start=sv.Point(int(stop_tol_pts[0][0]), int(stop_tol_pts[0][1])),
        end=sv.Point(int(stop_tol_pts[1][0]), int(stop_tol_pts[1][1])),
    )
    
    yield_pts = scaled["yield_ped_line"]
    zones["yield_ped_line"] = sv.LineZone(
        start=sv.Point(int(yield_pts[0][0]), int(yield_pts[0][1])),
        end=sv.Point(int(yield_pts[1][0]), int(yield_pts[1][1])),
    )
    
    zones["crosswalks"] = [sv.PolygonZone(polygon=p) for p in scaled["crosswalks"]]
    zones["ped_refuge"] = [sv.PolygonZone(polygon=p) for p in scaled["ped_refuge"]]
    zones["barriers"] = [sv.PolygonZone(polygon=p) for p in scaled["barriers"]]
    zones["sidewalks"] = [sv.PolygonZone(polygon=p) for p in scaled["sidewalks"]]
    
    road_polygons = [
        scaled["lane_ltr"],
        scaled["lane_rtl"],
        scaled["intersection_core"],
        scaled["right_turn_zone"],
        scaled["lower_core"],
    ]
    zones["road_zones"] = [sv.PolygonZone(polygon=p) for p in road_polygons]
    
    zones["lane_ltr"] = sv.PolygonZone(polygon=scaled["lane_ltr"])
    zones["lane_rtl"] = sv.PolygonZone(polygon=scaled["lane_rtl"])
    zones["intersection_core"] = sv.PolygonZone(polygon=scaled["intersection_core"])
    zones["right_turn_zone"] = sv.PolygonZone(polygon=scaled["right_turn_zone"])
    
    # Store the raw scaled coords as well
    zones["raw"] = scaled
    
    return zones
