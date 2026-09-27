"""Rule engine for Part A: turns observations + scene layout + signal state into events.

Each rule follows the start/end conventions of the task's class table. The
engine is purely a function of the observation stream, so it can be replayed
from cached perception output (scripts/replay_rules.py) while tuning.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from src.config import RULES
from src.geometry import box_iou, in_any, line_side, poly_dist, which_poly
from src.perception import MOTOR_VEHICLES, OBSTACLES, TWO_WHEELERS, VEHICLES, Observation
from src.tracks import TrackState, TrackStore
from src.traffic_light import SignalState

ACCIDENT_WORDS = ("acc", "crash", "colli")
ACCIDENT_NAMES = {"high", "medium", "low", "detected-injury"}   # severity classes of accident_model.pt


class Hysteresis:
    """Per-key on/off intervals: a segment opens on the first true sample and
    closes once the condition has been false for longer than `gap`."""

    def __init__(self, label: str, min_duration: float, gap: float, sink: list) -> None:
        self.label, self.min_duration, self.gap, self.sink = label, min_duration, gap, sink
        self.active: dict = {}           # key -> [start, last_true]

    def update(self, key, t: float, cond: bool) -> None:
        if cond:
            seg = self.active.get(key)
            if seg is None:
                self.active[key] = [t, t]
            else:
                seg[1] = t

    def sweep(self, t: float) -> None:
        for key in [k for k, (_, last) in self.active.items() if t - last > self.gap]:
            self.close(key)

    def close(self, key, end: float | None = None) -> None:
        start, last = self.active.pop(key)
        end = last if end is None else end
        if end - start >= self.min_duration:
            self.sink.append([start, end, self.label, key])


def _cos(v: tuple[float, float], u: np.ndarray) -> float:
    n = float(np.hypot(*v))
    return float((v[0] * u[0] + v[1] * u[1]) / n) if n > 1e-9 else 0.0


class RuleEngine:
    def __init__(self, scene: dict, signal: SignalState, verifier=None) -> None:
        """`verifier(kind, t_from, t_to, box, n_frames) -> p(yes) | None` answers a yes/no question
        about recent frames (src/vlm.py). Without one, accident/fire candidates are only logged."""
        self.scene = scene
        self.signal = signal
        self.verifier = verifier
        self.store = TrackStore()
        self.raw: list = []                  # [start, end, label, key] candidates
        self.events: list[list] = []         # final [start, end, label]
        R = RULES
        self.h_jay = Hysteresis("jaywalking", R["jaywalking"]["min_duration"], R["jaywalking"]["gap"], self.raw)
        self.h_stopline = Hysteresis("stop_line", R["stop_line"]["min_duration"], R["stop_line"]["gap"], self.raw)
        self.h_stopped = Hysteresis("stopped_vehicle", R["stopped_vehicle"]["min_duration"], R["stopped_vehicle"]["gap"], self.raw)
        self.h_wrong = Hysteresis("wrong_way", R["wrong_way"]["min_duration"], R["wrong_way"]["gap"], self.raw)
        self.h_cong = Hysteresis("congestion", R["congestion"]["min_duration"], R["congestion"]["gap"], self.raw)
        self.h_obst = Hysteresis("road_obstacle", R["road_obstacle"]["min_duration"], R["road_obstacle"]["gap"], self.raw)
        self.h_acc = Hysteresis("accident", R["accident"]["min_duration"], R["accident"]["gap"], self.raw)
        self.h_fire = Hysteresis("fire_smoke", R["fire_smoke"]["min_duration"], R["fire_smoke"]["gap"], self.raw)
        self._hyst = [self.h_jay, self.h_stopline, self.h_stopped, self.h_wrong, self.h_cong,
                      self.h_obst, self.h_acc, self.h_fire]
        # per-track memory
        self.side_strict: dict[int, float] = {}
        self.was_ltr: dict[int, float] = {}          # last time the track was in lane_ltr
        self.red_runs: dict[int, dict] = {}          # tid -> {"start", "confirmed", "last_in"}
        self.red_done: set[int] = set()
        self.fty: dict[tuple[int, int], dict] = {}   # (tid, crosswalk) -> {"enter", "last_in", "flag"}
        self.ped_cw_since: dict[int, tuple[int, float]] = {}
        self.nm_active: dict[tuple[int, int], dict] = {}
        self.stop_meta: dict[int, dict] = {}         # stopped_vehicle context per track
        self.moved: set[int] = set()                 # tracks seen driving at some point (not parked)
        self.anom_hist = {"accident": deque(maxlen=RULES["accident"]["window"]),
                          "fire_smoke": deque(maxlen=RULES["fire_smoke"]["window"])}
        # collision candidates -> verifier -> accident events
        self.acc_pairs: set[tuple[int, int]] = set()
        self.acc_pending: list[dict] = []
        self.acc_seen: list[tuple[float, np.ndarray]] = []
        self.acc_active: list[dict] = []
        self.fire_pending: list[dict] = []
        self.fire_seen: list[float] = []
        self.fire_watch: list[dict] = []             # places of verified accidents, checked for smoke
        self.last_t = 0.0
        self.vlm_log: list[dict] = []                # every question asked (or skipped), for diagnostics
        up = scene["ltr_upstream"]
        self.up_sign_strict = np.sign(line_side(scene["stop_line_strict"], *up))

    # ------------------------------------------------------------------ helpers
    def _on_segment(self, line: np.ndarray, x: float, y: float, pad: float = 0.08) -> bool:
        p0, p1 = np.asarray(line, dtype=np.float64)
        d = p1 - p0
        u = float(np.dot(np.array([x, y]) - p0, d) / max(1e-9, np.dot(d, d)))
        return -pad <= u <= 1.0 + pad

    # ------------------------------------------------------------------ main
    def update(self, obs: Observation) -> None:
        t = obs.t
        self.last_t = t
        s = self.scene
        tracks = self.store.update(t, obs.tids, obs.boxes, obs.names)
        sig = self.signal.state
        sig_age = self.signal.age(t)

        vehicles = [tr for tr in tracks if tr.cls in VEHICLES]
        motor = [tr for tr in tracks if tr.cls in MOTOR_VEHICLES]
        two_wheel_boxes = [tr.box for tr in tracks if tr.cls in TWO_WHEELERS]
        big_vehicle_boxes = [tr.box for tr in tracks if tr.cls in {"car", "bus", "truck"}]

        # riders and passengers are not pedestrians
        peds = []
        for tr in tracks:
            if tr.cls != "pedestrian":
                continue
            x, y = tr.pos
            riding = any(
                b[0] - 0.15 * (b[2] - b[0]) <= x <= b[2] + 0.15 * (b[2] - b[0]) and b[1] <= y <= b[3] + 0.15 * (b[3] - b[1])
                for b in two_wheel_boxes
            ) or any(b[0] <= x <= b[2] and b[1] <= y <= b[3] - 0.1 * (b[3] - b[1]) for b in big_vehicle_boxes)
            tr.rider_votes.append(1 if riding else 0)
            if not tr.is_rider:
                peds.append(tr)

        self._jaywalking(t, peds)
        self._failure_to_yield(t, peds, motor)
        for tr in vehicles:
            x, y = tr.pos
            if poly_dist(s["lane_ltr"], x, y) >= 0:
                self.was_ltr[tr.tid] = t
            spd = tr.speed_rel(1.0)
            if spd is not None and spd >= RULES["stopped_vehicle"]["moving_speed"]:
                self.moved.add(tr.tid)
        self._red_light_and_stop_line(t, vehicles, sig, sig_age)
        self._stopped_vehicle(t, motor, sig)
        self._wrong_way(t, motor)
        self._congestion(t, motor)
        self._near_miss(t, motor, peds)
        self._collisions(t, motor, peds)
        self._obstacles(t, tracks)
        self._anomaly(t, obs, motor)
        self._verify_pending(t)
        self._track_accidents(t)

        for h in self._hyst:
            h.sweep(t)
        self._sweep_lost(t)

    # ------------------------------------------------------------------ rules
    def _jaywalking(self, t: float, peds: list[TrackState]) -> None:
        cfg, s = RULES["jaywalking"], self.scene
        safe = s["crosswalks"] + s["sidewalks"] + s["ped_refuge"] + s["barriers"]
        for p in peds:
            x, y = p.pos
            d = p.diag
            on_road = in_any(s["road"], x, y, margin=cfg["road_margin_diag"] * d)
            is_safe = in_any(safe, x, y, margin=-cfg["safe_margin_diag"] * d)
            self.h_jay.update(p.tid, t, on_road and not is_safe and p.age >= 0.3)

    def _failure_to_yield(self, t: float, peds: list[TrackState], motor: list[TrackState]) -> None:
        cfg, s = RULES["failure_to_yield"], self.scene
        peds_on: dict[int, list[TrackState]] = {}
        for p in peds:
            # well inside the zebra: people waiting at the kerb are not "on the crossing"
            k = which_poly(s["crosswalks"], *p.pos, margin=cfg["ped_inside_diag"] * p.diag)
            if k < 0:
                self.ped_cw_since.pop(p.tid, None)
                continue
            prev = self.ped_cw_since.get(p.tid)
            if prev is None or prev[0] != k:
                self.ped_cw_since[p.tid] = (k, t)
            elif t - prev[1] >= cfg["ped_min_age"]:
                peds_on.setdefault(k, []).append(p)
        for v in motor:
            x, y = v.pos
            k = which_poly(s["crosswalks"], x, y)
            if k < 0:
                continue
            key = (v.tid, k)
            st = self.fty.get(key)
            if st is None:
                st = self.fty[key] = {"enter": t, "last_in": t, "flag_t": None}
            st["last_in"] = t
            vel = v.velocity(0.5)
            spd = v.speed_rel(0.5)
            if vel is None or spd is None or spd < cfg["vehicle_min_speed"]:
                continue
            speed_px = float(np.hypot(*vel))
            ux, uy = vel[0] / speed_px, vel[1] / speed_px
            b = v.box
            for p in peds_on.get(k, []):
                px, py = p.pos
                if b[0] <= px <= b[2] and b[1] <= py <= b[3]:
                    continue                      # rider / occluded by this very vehicle
                dx, dy = px - x, py - y
                ahead = dx * ux + dy * uy         # distance along the vehicle's path
                lateral = abs(-dx * uy + dy * ux)
                if 0.0 < ahead <= cfg["ahead_diag"] * v.diag and lateral <= cfg["lateral_diag"] * v.diag:
                    st["flag_t"] = st["flag_t"] or t
                    break

    def _red_light_and_stop_line(self, t: float, vehicles: list[TrackState], sig: str, sig_age: float) -> None:
        cfg, cfg_sl, s = RULES["red_light"], RULES["stop_line"], self.scene
        for v in vehicles:
            x, y = v.pos
            from_ltr = t - self.was_ltr.get(v.tid, -1e9) <= 4.0
            # strict stop line crossing (upstream -> downstream)
            side = np.sign(line_side(s["stop_line_strict"], x, y))
            prev = self.side_strict.get(v.tid)
            self.side_strict[v.tid] = side
            if (prev is not None and prev == self.up_sign_strict and side == -self.up_sign_strict
                    and from_ltr and self._on_segment(s["stop_line_strict"], x, y)
                    and v.tid not in self.red_done):
                vel = v.velocity(0.5)
                spd = v.speed_rel(0.5)
                forward = vel is not None and _cos(vel, s["flow_ltr"]) > 0.3
                if sig == "RED" and sig_age >= cfg["min_red_age"] and forward and spd is not None and spd >= cfg["min_speed"]:
                    self.red_runs[v.tid] = {"start": t, "confirmed": False, "last_in": t}
                    self.red_done.add(v.tid)
            run = self.red_runs.get(v.tid)
            if run is not None:
                in_box = in_any([s["intersection_core"], s["right_turn_zone"], s["lower_core"]], x, y)
                if in_box and t - run["start"] <= cfg["confirm_sec"]:
                    run["confirmed"] = True          # it really drove into the junction
                if in_box or poly_dist(s["crosswalks"][0], x, y) >= 0:
                    run["last_in"] = t
                if t - run["start"] > cfg["max_duration"] or (not run["confirmed"] and t - run["start"] > cfg["confirm_sec"]):
                    self._close_red_run(v.tid)

            if v.cls in MOTOR_VEHICLES:
                # Past the tolerance line but not yet in the junction = standing on
                # the first zebra (crosswalks[0]) of lane_ltr.
                spd1 = v.speed_rel(1.0)
                cond = (
                    sig == "RED" and v.tid in self.was_ltr
                    and poly_dist(s["crosswalks"][0], x, y) >= 0.1 * v.diag
                    and spd1 is not None and spd1 < cfg_sl["stopped_speed"]
                )
                self.h_stopline.update(v.tid, t, cond)

    def _close_red_run(self, tid: int) -> None:
        run = self.red_runs.pop(tid, None)
        if run and run["confirmed"] and run["last_in"] > run["start"]:
            self.raw.append([run["start"], run["last_in"], "red_light", tid])

    def _stopped_vehicle(self, t: float, motor: list[TrackState], sig: str) -> None:
        cfg, s = RULES["stopped_vehicle"], self.scene
        stationary = []
        for v in motor:
            spd = v.speed_rel(2.0)
            x, y = v.pos
            if spd is not None and spd < cfg["stopped_speed"] and in_any(s["road"], x, y):
                stationary.append(v)
        for v in stationary:
            x, y = v.pos
            if v.tid not in self.moved:
                continue                  # never seen moving: parked since before the clip
            if in_any([s["intersection_core"], s["lower_core"]], x, y):
                continue                  # waiting to turn inside the junction
            if in_any(s["crosswalks"], x, y, margin=-cfg["crossing_clear_diag"] * v.diag):
                continue                  # at a zebra: yielding / queueing, not "stopped"
            meta = self.stop_meta.setdefault(v.tid, {"queue": False, "in_ltr": False})
            x, y = v.pos
            in_ltr = poly_dist(s["lane_ltr"], x, y) >= 0 or poly_dist(s["crosswalks"][0], x, y) >= 0
            meta["in_ltr"] |= in_ltr
            flow = s["flow_ltr"] if in_ltr else s["flow_rtl"]
            # another stationary vehicle just ahead in the flow direction -> queue
            for o in stationary:
                if o.tid == v.tid:
                    continue
                dx, dy = o.pos[0] - x, o.pos[1] - y
                dist = float(np.hypot(dx, dy))
                if dist < cfg["queue_ahead_diag"] * max(v.diag, o.diag) and (dx * flow[0] + dy * flow[1]) > 0.3 * dist:
                    meta["queue"] = True
                    break
            self.h_stopped.update(v.tid, t, True)

    def _wrong_way(self, t: float, motor: list[TrackState]) -> None:
        cfg, s = RULES["wrong_way"], self.scene
        for v in motor:
            x, y = v.pos
            if in_any([s["intersection_core"], s["right_turn_zone"], s["lower_core"]], x, y):
                continue
            vel = v.velocity(1.5)
            spd = v.speed_rel(1.5)
            if vel is None or spd is None or spd < cfg["min_speed"]:
                continue
            margin = 0.2 * v.diag
            wrong = (
                (poly_dist(s["lane_ltr"], x, y) >= margin and _cos(vel, s["flow_ltr"]) < cfg["max_cos"])
                or (poly_dist(s["lane_rtl"], x, y) >= margin and _cos(vel, s["flow_rtl"]) < cfg["max_cos"])
            )
            self.h_wrong.update(v.tid, t, wrong)

    def _congestion(self, t: float, motor: list[TrackState]) -> None:
        cfg, s = RULES["congestion"], self.scene
        for lane in ("lane_ltr", "lane_rtl"):
            speeds = []
            for v in motor:
                if v.tid in self.moved and poly_dist(s[lane], *v.pos) >= 0:   # parked cars do not count
                    spd = v.speed_rel(1.0)
                    if spd is not None:          # only vehicles with a real speed estimate
                        speeds.append(spd)
            crawl = len(speeds) >= cfg["min_vehicles"] and float(np.median(speeds)) < cfg["crawl_speed"]
            self.h_cong.update(lane, t, crawl)

    def _near_miss(self, t: float, motor: list[TrackState], peds: list[TrackState]) -> None:
        cfg, s = RULES["near_miss"], self.scene
        users = [u for u in motor + peds if in_any(s["road"], *u.pos)]
        for i in range(len(users)):
            for j in range(i + 1, len(users)):
                a, b = users[i], users[j]
                if a.cls == "pedestrian" and b.cls == "pedestrian":
                    continue
                key = (min(a.tid, b.tid), max(a.tid, b.tid))
                big = max(a.diag, b.diag)
                gap = float(np.hypot(a.pos[0] - b.pos[0], a.pos[1] - b.pos[1])) / big
                st = self.nm_active.get(key)
                if st is None:
                    if gap >= cfg["close_diag"] or box_iou(a.box, b.box) > 0.3:
                        continue
                    braking = any(self._hard_brake(u) for u in (a, b) if u.cls != "pedestrian")
                    if not braking:
                        continue
                    # they must have been approaching each other over the last second
                    pa, pb = a.point_ago(1.0), b.point_ago(1.0)
                    if pa is None or pb is None:
                        continue
                    gap_before = float(np.hypot(pa[1] - pb[1], pa[2] - pb[2])) / big
                    if gap_before - gap < 0.3:
                        continue
                    self.nm_active[key] = {"start": max(0.0, t - 0.8), "last": t}
                elif gap < cfg["clear_diag"]:
                    st["last"] = t
                else:
                    self._close_nm(key, t)

    def _hard_brake(self, v: TrackState) -> bool:
        cfg = RULES["near_miss"]
        recent = v.speed_between(0.0, 0.5)
        before = v.speed_between(0.5, 1.5)
        return (recent is not None and before is not None and before >= cfg["brake_min_speed"]
                and recent < cfg["brake_ratio"] * before)

    def _close_nm(self, key, end: float) -> None:
        st = self.nm_active.pop(key)
        if end - st["start"] >= RULES["near_miss"]["min_duration"]:
            self.raw.append([st["start"], end, "near_miss", key])

    def _collisions(self, t: float, motor: list[TrackState], peds: list[TrackState]) -> None:
        """Propose a collision candidate the first time two road users touch while closing in."""
        cfg, s = RULES["accident"], self.scene
        users = [u for u in motor + peds if in_any(s["road"], *u.pos)]
        for i in range(len(users)):
            for j in range(i + 1, len(users)):
                a, b = users[i], users[j]
                if a.cls == "pedestrian" and b.cls == "pedestrian":
                    continue
                key = (min(a.tid, b.tid), max(a.tid, b.tid))
                if key in self.acc_pairs or not self._touching(a, b, cfg["touch_pad"]):
                    continue
                big = max(a.diag, b.diag)
                gap = float(np.hypot(a.pos[0] - b.pos[0], a.pos[1] - b.pos[1])) / big
                if gap >= cfg["contact_diag"]:
                    continue
                pa, pb = a.point_ago(1.0), b.point_ago(1.0)
                if pa is None or pb is None:
                    continue
                gap_before = float(np.hypot(pa[1] - pb[1], pa[2] - pb[2])) / big
                if gap_before - gap < cfg["approach_diag"]:
                    continue
                speeds = [u.speed_rel(1.0) for u in (a, b)]
                v_pre = max((v for v in speeds if v is not None), default=0.0)
                if v_pre < cfg["min_speed"]:
                    continue
                self.acc_pairs.add(key)
                union = np.array([min(a.box[0], b.box[0]), min(a.box[1], b.box[1]),
                                  max(a.box[2], b.box[2]), max(a.box[3], b.box[3])])
                self._propose_accident(t, key, union, "tracks",
                                       {"v_pre": v_pre, "closing": gap_before - gap, "gap": gap})

    @staticmethod
    def _touching(a: TrackState, b: TrackState, pad: float) -> bool:
        """Boxes overlap or nearly touch. A head-on hit leaves two boxes side by side with almost no overlap."""
        pa, pb = pad * a.diag, pad * b.diag
        return (a.box[0] - pa <= b.box[2] + pb and b.box[0] - pb <= a.box[2] + pa
                and a.box[1] - pa <= b.box[3] + pb and b.box[1] - pb <= a.box[3] + pa)

    def _propose_accident(self, t: float, tids: tuple, box: np.ndarray, source: str,
                          feats: dict | None = None) -> None:
        cfg = RULES["accident"]
        for t_s, b_s in self.acc_seen:
            if abs(t - t_s) < cfg["dedup_sec"] and box_iou(box, b_s) > 0.1:
                return
        self.acc_seen.append((t, box))
        self.acc_pending.append({"t0": t, "tids": tids, "box": box, "source": source, "feats": feats or {}})

    def _verify_pending(self, t: float) -> None:
        cfg = RULES["accident"]
        for c in [c for c in self.acc_pending if t >= c["t0"] + cfg["post_sec"]]:
            self.acc_pending.remove(c)
            after = self._after(c)
            if c["source"] == "tracks" and (after.get("v_post", 0.0) >= cfg["post_max_speed"]
                                            or after.get("gap_post", 0.0) >= cfg["post_max_gap"]):
                continue          # they drove on / apart: not a crash
            # the first window screens; the rest are asked only for a plausible crash, then averaged
            ps: list[float] = []
            for pre, post, n in cfg["windows"] if self.verifier is not None else []:
                q = self.verifier("accident", c["t0"] - pre, c["t0"] + post, c["box"], n)
                if q is None:
                    ps = []
                    break
                ps.append(q)
                if q < cfg["screen_p"]:
                    break
            p = float(np.mean(ps)) if ps else None
            self.vlm_log.append({"kind": "accident", "t": round(c["t0"], 2), "source": c["source"],
                                 "p": None if p is None else round(p, 4), "ps": [round(q, 4) for q in ps],
                                 **{k: round(float(v), 3) for k, v in {**c["feats"], **after}.items()}})
            if p is not None and p >= cfg["accept_p"] and len(ps) == len(cfg["windows"]):
                self.acc_active.append({"start": c["t0"], "tids": c["tids"], "key": ("vlm", c["tids"])})
                fw = RULES["fire_smoke"]
                self.fire_watch.append({"box": c["box"], "next": c["t0"] + fw["watch_step"],
                                        "until": c["t0"] + fw["watch_sec"], "start": None, "last_pos": None, "neg": 0})
        self._watch_fire(t)
        fcfg = RULES["fire_smoke"]
        for c in list(self.fire_pending):
            self.fire_pending.remove(c)
            p = None
            if self.verifier is not None:
                p = self.verifier("fire_smoke", c["t0"], c["t0"], c["box"], 1)
            self.vlm_log.append({"kind": "fire_smoke", "t": round(c["t0"], 2), "source": "anomaly",
                                 "p": None if p is None else round(p, 4)})
            if p is not None and p >= fcfg["accept_p"]:
                self.raw.append([c["t0"], c["t0"] + fcfg["event_sec"], "fire_smoke", "vlm"])

    def _watch_fire(self, t: float) -> None:
        """Smoke often starts seconds after a crash: re-check a verified crash site every `watch_step` s.
        The event runs from the first positive check (minus half a step) until two checks in a row are negative."""
        fw = RULES["fire_smoke"]
        step = fw["watch_step"]
        for w in list(self.fire_watch):
            while w["next"] <= min(t, w["until"]):
                tc = w["next"]
                w["next"] += step
                p = None if self.verifier is None else self.verifier("fire_smoke", tc, tc, w["box"], 1)
                self.vlm_log.append({"kind": "fire_smoke", "t": round(tc, 2), "source": "crash_site",
                                     "p": None if p is None else round(p, 4)})
                if p is None:
                    continue
                if p >= fw["accept_p"]:
                    if w["start"] is None:
                        w["start"] = tc - step / 2
                    w["last_pos"], w["neg"] = tc, 0
                    w["until"] = max(w["until"], tc + 2 * step)     # keep watching while it burns
                elif w["start"] is not None:
                    w["neg"] += 1
                    if w["neg"] >= 2:
                        self.raw.append([w["start"], w["last_pos"] + step / 2, "fire_smoke", "vlm_watch"])
                        w["start"] = None
            if w["next"] > w["until"] and w["start"] is None:
                self.fire_watch.remove(w)

    def _after(self, c: dict) -> dict:
        """How the candidate's road users behave `post_sec` after contact."""
        pts, speeds = [], []
        for tid in c["tids"]:
            tr = self.store.tracks.get(tid)
            if tr is None or not tr.hist:
                continue
            pts.append((tr.pos, tr.diag))
            spd = tr.speed_rel(1.0)
            if spd is not None:
                speeds.append(spd)
        out = {"n_seen": len(pts)}
        if speeds:
            out["v_post"] = max(speeds)
        if len(pts) == 2:
            (p1, d1), (p2, d2) = pts
            out["gap_post"] = float(np.hypot(p1[0] - p2[0], p1[1] - p2[1])) / max(d1, d2)
        return out

    def _track_accidents(self, t: float) -> None:
        """A verified accident ends once every involved road user has stopped or left the frame."""
        cfg = RULES["accident"]
        for a in list(self.acc_active):
            moving = False
            for tid in a["tids"]:
                tr = self.store.tracks.get(tid)
                if tr is not None and t - tr.last_t < 1.0:
                    spd = tr.speed_rel(0.5)
                    moving = moving or (spd is not None and spd >= cfg["end_stop_speed"])
            elapsed = t - a["start"]
            if (not moving and elapsed >= cfg["post_sec"]) or elapsed >= cfg["max_duration"]:
                self.acc_active.remove(a)
                self.raw.append([a["start"], t, "accident", a["key"]])

    def _obstacles(self, t: float, tracks: list[TrackState]) -> None:
        cfg, s = RULES["road_obstacle"], self.scene
        people = [tr.box for tr in tracks if tr.cls == "pedestrian"]
        for o in tracks:
            if o.cls not in OBSTACLES:
                continue
            x, y = o.pos
            carried = any(
                box_iou(o.box, pb) > 0.0 or (pb[0] <= x <= pb[2] and pb[1] <= y <= pb[3] + 0.2 * (pb[3] - pb[1]))
                for pb in people
            )
            spd = o.speed_rel(1.0)
            cond = (not carried and spd is not None and spd <= cfg["max_speed"]
                    and in_any(s["road"], x, y) and not in_any(s["sidewalks"], x, y))
            self.h_obst.update(o.tid, t, cond)

    def _anomaly(self, t: float, obs: Observation, motor: list[TrackState]) -> None:
        if obs.anomaly is None:
            return
        s = self.scene
        hits = {"accident": False, "fire_smoke": False}
        for name, conf, box in obs.anomaly:
            cx, cy = float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)
            if not in_any(s["road"], cx, box[3]) and not in_any(s["road"], cx, cy):
                continue
            is_acc = name in ACCIDENT_NAMES or any(w in name for w in ACCIDENT_WORDS)
            is_fire = "fire" in name or "smoke" in name
            if is_acc and conf >= RULES["accident"]["cand_anomaly_conf"]:
                covered = tuple(sorted(v.tid for v in motor if box_iou(box, v.box) > 0.1))
                if covered:
                    self._propose_accident(t, covered, box, "anomaly")
            elif is_fire and conf >= RULES["fire_smoke"]["cand_anomaly_conf"]:
                if not any(abs(t - t_s) < RULES["accident"]["dedup_sec"] for t_s in self.fire_seen):
                    self.fire_seen.append(t)
                    self.fire_pending.append({"t0": t, "box": box})
            if is_acc and conf >= RULES["accident"]["conf"]:
                # a crash involves road users: the box must cover at least one vehicle
                if any(box_iou(box, v.box) > 0.1 for v in motor):
                    hits["accident"] = True
            elif is_fire and conf >= RULES["fire_smoke"]["conf"]:
                hits["fire_smoke"] = True
        for label, h in (("accident", self.h_acc), ("fire_smoke", self.h_fire)):
            win = self.anom_hist[label]
            win.append(hits[label])
            h.update(label, t, sum(win) >= RULES[label]["hits"])

    # ------------------------------------------------------------------ bookkeeping
    def _sweep_lost(self, t: float) -> None:
        present = {tid for tid, tr in self.store.tracks.items() if tr.last_t == t}
        for tid in [k for k in self.red_runs if k not in present and t - self.store.tracks.get(k, TrackState(k, 0.0)).last_t > 1.5]:
            self._close_red_run(tid)
        for tid, run in list(self.red_runs.items()):
            if t - run["last_in"] > 1.5:
                self._close_red_run(tid)
        for key, st in list(self.fty.items()):
            if t - st["last_in"] > 0.5:
                self.fty.pop(key)
                if st["flag_t"] is not None and st["last_in"] - st["enter"] >= RULES["failure_to_yield"]["min_duration"]:
                    self.raw.append([st["enter"], st["last_in"], "failure_to_yield", key])
        for key, st in list(self.nm_active.items()):
            if t - st["last"] > 1.0:
                self._close_nm(key, st["last"])

    def finalize(self, duration: float) -> list[list]:
        for tid in list(self.red_runs):
            self._close_red_run(tid)
        for key, st in list(self.fty.items()):
            if st["flag_t"] is not None and st["last_in"] - st["enter"] >= RULES["failure_to_yield"]["min_duration"]:
                self.raw.append([st["enter"], st["last_in"], "failure_to_yield", key])
        self.fty.clear()
        for key in list(self.nm_active):
            self._close_nm(key, self.nm_active[key]["last"])
        # candidates still waiting for their post-contact frames are asked with what the video has
        self._verify_pending(self.last_t + RULES["accident"]["post_sec"])
        for w in self.fire_watch:
            if w["start"] is not None:
                self.raw.append([w["start"], duration, "fire_smoke", "vlm_watch"])
        self.fire_watch.clear()
        for a in self.acc_active:
            self.raw.append([a["start"], min(duration, a["start"] + RULES["accident"]["max_duration"]),
                             "accident", a["key"]])
        self.acc_active.clear()
        # events still running at the end of the clip end at the clip end
        for h in self._hyst:
            for key in list(h.active):
                h.close(key, end=duration)
        return self._post_filter(duration)

    def _signal_queue(self, s0: float, e0: float) -> bool:
        """A lane_ltr stop is a signal queue if the vehicle moves off soon after a green onset
        (or, when the signal could not be read, if the stop is no longer than a red phase)."""
        greens = [start for st, start in self.signal.phases if st == "GREEN" and s0 - 1.0 <= start <= e0 + 2.0]
        if greens:
            return any(0.0 <= e0 - g <= 15.0 for g in greens)
        if not self.signal.phases:
            return e0 - s0 < 90.0
        return self.signal.time_in_state("RED", s0, e0) > 0.5 * (e0 - s0)

    def _post_filter(self, duration: float) -> list[list]:
        out = []
        cong = [(s0, e0) for s0, e0, lbl, _ in self.raw if lbl == "congestion"]
        for s0, e0, label, key in self.raw:
            if label == "stopped_vehicle":
                meta = self.stop_meta.get(key, {})
                if meta.get("queue"):
                    continue          # waiting behind another stopped vehicle
                if meta.get("in_ltr") and self._signal_queue(s0, e0):
                    continue          # waiting at the signal of lane_ltr
                if any(cs <= s0 and e0 <= ce for cs, ce in cong):
                    continue
            if label == "red_light" and any(st == "GREEN" and s0 <= g <= s0 + RULES["red_light"]["early_start_sec"]
                                            for st, g in self.signal.phases):
                continue              # moved off just before green (anticipation), not running the red
            if label == "congestion" and key == "lane_ltr" and self.signal.phases:
                known = self.signal.time_in_state("GREEN", s0, e0) + self.signal.time_in_state("RED", s0, e0)
                if known > 0 and self.signal.time_in_state("GREEN", s0, e0) < RULES["congestion"]["min_green_sec"]:
                    continue          # an ordinary red-light queue
            out.append([max(0.0, s0), min(duration, e0), label])
        return out
