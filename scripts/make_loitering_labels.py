#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import os
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import cos, sin, pi, isfinite
from typing import List, Tuple, Optional

try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

try:
    import yaml
except Exception:
    yaml = None

from PIL import Image, ImageDraw, ImageFont

DEFAULT_CONFIG = {
    "labels": {
        "lock_gate": "Lock_gate",
        "ships": ["Fully_loaded_cargo_ship", "Fully_loaded_container_ship", "Unladen_cargo_ship"],
    },
    "geometry": {
        "theta_occ": 0.0,
        "chamber_x_range": [39.7, 62.7],
        "chamber_y_range": [0.0, 102.4],
        "gate_y_range": [19.078, 23.002],
    },
    "thresholds": {
        "gate_opening_min_width_m": 22.8,
        "gate_closing_min_max_width_m": 0.2,
        "gate_trend_window": 3,
        "track_max_match_dist_m": 10,
        "track_max_size_ratio": 4.0,
        "track_max_time_gap_s": 5.0,
        #"track_max_time_gap_s": 10.0,
        "track_confirm_hits": 2,
        "track_max_missed": 8,
        "loiter_window_s": 60.0,
        "momentum_switch_thresh": 3,
        "momentum_clip": 6,
        "gate_trend_delta_min_m": 0.1,
        "gate_closed_width_m": 22.8
    },
    "data": {"images_dir": "./images"},
    "output": {"root": "./outputs"},
    "debug": {
        "visualize_non_start_reasons": True,
        "always_overlay_debug": False,
        "print_bbox_on_image": True,
        "max_ships_overlay": 6
    }
}

FRAME_BASENAME_RE = re.compile(r'^(\d+)\.txt$')

@dataclass
class Detection:
    cls: str
    size_hwl: Tuple[float, float, float]
    center_xy: Tuple[float, float]
    yaw: Optional[float]

@dataclass
class FrameInfo:
    frame_id: str
    timestamp: datetime
    timestamp_str: str
    seg_id: int
    gate_width_m: Optional[float]
    gate_present: bool
    gate_state: str
    ships: List[Detection]

@dataclass
class Track:
    tid: int
    last_xy: Tuple[float, float]
    last_size_wl: Tuple[float, float]
    last_yaw: Optional[float]
    last_time: datetime
    hits: int
    misses: int
    confirmed: bool
    inside_chamber: bool
    exited_time: Optional[datetime]
    behavior_momentum: int = 0
    behavior_state: str = "UNKNOWN"

def load_config(path: str) -> dict:
    import json as _json
    cfg = _json.loads(_json.dumps(DEFAULT_CONFIG))
    if os.path.exists(path) and yaml is not None:
        with open(path, 'r', encoding='utf-8') as f:
            user_cfg = yaml.safe_load(f) or {}
        def merge(a, b):
            for k, v in b.items():
                if isinstance(v, dict) and k in a and isinstance(a[k], dict):
                    merge(a[k], v)
                else:
                    a[k] = v
        merge(cfg, user_cfg)
    return cfg

def load_add_info(path: str) -> dict:
    mp = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            cols = s.split()
            if len(cols) < 4:
                continue
            fid = cols[0]
            ts_str = cols[1]
            seg = int(cols[3])
            ts = datetime.strptime(ts_str, "%Y_%m_%d_%H_%M_%S_%f")
            mp[fid] = (ts, ts_str, seg)
    return mp

def parse_label_file(path: str, cfg: dict) -> Tuple[List[Detection], Optional[float]]:
    ships_names = set(cfg["labels"]["ships"])
    lock_name = cfg["labels"]["lock_gate"]
    ships = []
    gate_widths = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            p = line.strip().split()
            if not p:
                continue
            name = p[0]
            try:
                h, w, l = map(float, p[8:11])
                cx, cy = map(float, p[11:13])
                yaw = float(p[-1]) if len(p) >= 15 else (float(p[13]) if len(p) >= 14 else None)
            except Exception:
                continue
            if name == lock_name:
                gate_widths.append(w)
            if name in ships_names:
                ships.append(Detection(name, (h, w, l), (cx, cy), yaw))
    gate_width = None
    if gate_widths:
        gate_widths.sort()
        gate_width = gate_widths[len(gate_widths)//2]
    return ships, gate_width

def euclid(a, b):
    dx = a[0]-b[0]; dy = a[1]-b[1]
    return (dx*dx+dy*dy)**0.5

def size_ratio_ok(wl1, wl2, max_ratio):
    w1, l1 = wl1; w2, l2 = wl2
    def ok(a,b):
        r = max(a,b)/max(1e-6, min(a,b))
        return r <= max_ratio
    return ok(w1,w2) and ok(l1,l2)

def oriented_rect_intersects_aabb(cx, cy, w, l, yaw, xr, yr):
    if yaw is None:
        yaw = 0.0
    ux, uy = cos(yaw), sin(yaw)
    vx, vy = -uy, ux
    el, ew = l*0.5, w*0.5
    rcx, rcy = (xr[0]+xr[1])*0.5, (yr[0]+yr[1])*0.5
    ex, ey = (xr[1]-xr[0])*0.5, (yr[1]-yr[0])*0.5
    tx, ty = rcx-cx, rcy-cy
    def overlap_on(Lx, Ly):
        Ra = el*abs(ux*Lx + uy*Ly) + ew*abs(vx*Lx + vy*Ly)
        Rb = ex*abs(Lx) + ey*abs(Ly)
        T = abs(tx*Lx + ty*Ly)
        return T <= (Ra + Rb) + 1e-6
    if not overlap_on(ux, uy): return False
    if not overlap_on(vx, vy): return False
    if not overlap_on(1.0, 0.0): return False
    if not overlap_on(0.0, 1.0): return False
    return True

# Exact rotated-box clipping; no enclosing-AABB approximation.

def _valid(cx,cy,w,l,yaw,xr,yr):
    try:
        return (all(isfinite(float(v)) for v in (cx,cy,w,l,yaw,*xr,*yr))
                and w>0 and l>0 and len(xr)==len(yr)==2
                and xr[1]>xr[0] and yr[1]>yr[0])
    except (TypeError,ValueError,OverflowError):return False

def vessel_corners(cx,cy,w,l,yaw):
    """CCW corners; length axis matches the original track geometry."""
    ux,uy=cos(yaw),sin(yaw);vx,vy=-uy,ux
    return [(cx+a*l*.5*ux+b*w*.5*vx,cy+a*l*.5*uy+b*w*.5*vy)
            for a,b in [(-1,-1),(1,-1),(1,1),(-1,1)]]

def polygon_area(poly):
    if len(poly)<3:return 0.0
    ox,oy=poly[0]
    p=[(x-ox,y-oy) for x,y in poly]
    return abs(sum(a[0]*b[1]-b[0]*a[1] for a,b in zip(p,p[1:]+p[:1])))*.5

def _clip(poly,axis,bound,lower):
    if not poly:return []
    result=[]
    inside=lambda p:p[axis]>=bound if lower else p[axis]<=bound
    before=poly[-1];before_inside=inside(before)
    for after in poly:
        after_inside=inside(after)
        if before_inside!=after_inside:
            fraction=(bound-before[axis])/(after[axis]-before[axis])
            q=[before[k]+fraction*(after[k]-before[k]) for k in [0,1]]
            q[axis]=bound
            result.append(tuple(q))
        if after_inside:result.append(after)
        before,before_inside=after,after_inside
    return result

def intersection_area(cx,cy,w,l,yaw,xr,yr):
    yaw=0.0 if yaw is None else yaw
    if not _valid(cx,cy,w,l,yaw,xr,yr):return 0.0
    poly=vessel_corners(cx,cy,w,l,yaw)
    for axis,bound,lower in [(0,xr[0],True),(0,xr[1],False),(1,yr[0],True),(1,yr[1],False)]:
        poly=_clip(poly,axis,bound,lower)
    return min(w*l,max(0.0,polygon_area(poly)))

def occupancy_ratio(cx,cy,w,l,yaw,xr,yr):
    yaw=0.0 if yaw is None else yaw
    if not _valid(cx,cy,w,l,yaw,xr,yr):return 0.0
    return intersection_area(cx,cy,w,l,yaw,xr,yr)/(w*l)

def vessel_inside_chamber(cx,cy,w,l,yaw,xr,yr,theta_occ=0.0):
    if not isfinite(theta_occ) or not 0<=theta_occ<=1:
        raise ValueError('theta_occ must be finite and within [0,1]')
    yaw=0.0 if yaw is None else yaw
    if not _valid(cx,cy,w,l,yaw,xr,yr):return False
    rho = occupancy_ratio(cx,cy,w,l,yaw,xr,yr)
    # Manuscript Eq. (9): a boundary-only contact has rho=0 and is outside.
    return rho > 0.0 and rho >= theta_occ

def compute_gate_states(frames: List[FrameInfo], cfg: dict) -> None:
    W = int(cfg["thresholds"]["gate_trend_window"])   
    open_min = float(cfg["thresholds"]["gate_opening_min_width_m"])
    closing_max = float(cfg["thresholds"]["gate_closing_min_max_width_m"])
    closed_w = float(cfg["thresholds"].get("gate_closed_width_m", 22.8))
    trend_eps = float(cfg["thresholds"].get("gate_trend_delta_min_m", 0.1))

    def motion_from_window(win: List[float]) -> str:

        if len(win) < 2:
            return "STATIC"

        span = max(win) - min(win)   
        if span < trend_eps:
            return "STATIC"

        net = win[-1] - win[0]      
        if net > 0 and max(win) > closing_max:
            return "CLOSING"
        if net < 0 and min(win) < open_min:
            return "OPENING"
        return "STATIC"

    recent = deque(maxlen=W)

    for fi in frames:
        curr_w = 0.0 if fi.gate_width_m is None else float(fi.gate_width_m)
        recent.append(curr_w)

        if not fi.gate_present:
            fi.gate_state = "OPEN"
            continue

        if curr_w <= closing_max:
            fi.gate_state = "OPEN"
            continue


        if curr_w >= closed_w:
            fi.gate_state = "CLOSED"
            continue

        motion = motion_from_window(list(recent))
        fi.gate_state = motion if motion != "STATIC" else "CLOSED"

    head_n = min(W, len(frames))
    if head_n >= 2:
        head_win = [
            0.0 if frames[i].gate_width_m is None else float(frames[i].gate_width_m)
            for i in range(head_n)
        ]
        head_motion = motion_from_window(head_win)

        if head_motion in ("OPENING", "CLOSING"):
            for i in range(head_n):
                fi = frames[i]
                curr_w = 0.0 if fi.gate_width_m is None else float(fi.gate_width_m)

                if not fi.gate_present:
                    fi.gate_state = "OPEN"
                elif curr_w >= closed_w:
                    fi.gate_state = "CLOSED"
                else:
                    fi.gate_state = head_motion

def track_segment(frames: List[FrameInfo], cfg: dict):
    chamber_x = tuple(cfg["geometry"]["chamber_x_range"])
    y_min = float(cfg["geometry"]["gate_y_range"][0])
    y_max = float(cfg["geometry"]["chamber_y_range"][1])
    chamber_y_rect = (y_min, y_max)

    theta_occ = float(cfg["geometry"].get("theta_occ", 0.0))

    def inside_chamber_fn(xy, w, l, yaw):
        return vessel_inside_chamber(xy[0], xy[1], w, l, yaw,
                                     chamber_x, chamber_y_rect, theta_occ)

    max_gap = timedelta(seconds=float(cfg["thresholds"]["track_max_time_gap_s"]))
    confirm_hits = int(cfg["thresholds"]["track_confirm_hits"])
    max_missed = int(cfg["thresholds"]["track_max_missed"])
    max_d = float(cfg["thresholds"]["track_max_match_dist_m"])
    max_ratio = float(cfg["thresholds"]["track_max_size_ratio"])

    momentum_switch = int(cfg["thresholds"].get("momentum_switch_thresh", 3))
    momentum_clip = int(cfg["thresholds"].get("momentum_clip", 6))

    tracks: List[Track] = []
    next_tid = 1
    per_frame_ship_tids = []
    per_frame_inside_counts = []
    per_frame_bboxinfo = []
    long_traj_rows = []
    entry_events = []
    exit_events = []
    exit_behaviors = []

    def build_cost(trs, dets):
        if not trs or not dets:
            return []
        M = [[1e6 for _ in range(len(dets))] for __ in range(len(trs))]
        for i, t in enumerate(trs):
            for j, d in enumerate(dets):
                dxy = euclid(t.last_xy, d.center_xy)
                if dxy > max_d:
                    continue
                if not size_ratio_ok(t.last_size_wl, (d.size_hwl[1], d.size_hwl[2]), max_ratio):
                    continue
                size_pen = abs(t.last_size_wl[0]-d.size_hwl[1]) + abs(t.last_size_wl[1]-d.size_hwl[2])
                M[i][j] = dxy + 0.1 * size_pen
        return M

    def assign(cost):
        if not cost:
            return [], set(), set()
        nT, nD = len(cost), len(cost[0])
        if SCIPY_OK:
            import numpy as np
            C = np.array(cost, dtype=float)
            row_ind, col_ind = linear_sum_assignment(C)
            pairs = []
            usedT = set(); usedD = set()
            for r, c in zip(row_ind, col_ind):
                if C[r, c] >= 1e6:
                    continue
                pairs.append((r, c)); usedT.add(r); usedD.add(c)
            return pairs, usedT, usedD
        flat = sorted((cost[i][j], i, j) for i in range(nT) for j in range(nD))
        pairs = []; usedT = set(); usedD = set()
        for val, i, j in flat:
            if val >= 1e6:
                break
            if i in usedT or j in usedD:
                continue
            pairs.append((i, j)); usedT.add(i); usedD.add(j)
        return pairs, usedT, usedD

    def yaw_downstream(yaw: Optional[float]) -> bool:
        return yaw is not None and (-pi <= yaw <= 0.0)

    def yaw_upstream(yaw: Optional[float]) -> bool:
        return yaw is not None and (0.0 <= yaw <= pi)

    for fi in frames:
        now = fi.timestamp
        dets = fi.ships

        tracks = [t for t in tracks if (now - t.last_time) <= max_gap]

        cost = build_cost(tracks, dets)
        pairs, usedT, usedD = assign(cost)

        prev_inside_map = {idx: t.inside_chamber for idx, t in enumerate(tracks)}
        this_frame_bbox = []

        for (i, j) in pairs:
            t = tracks[i]; d = dets[j]
            prev_inside = prev_inside_map[i]

            inside_now = inside_chamber_fn(d.center_xy, d.size_hwl[1], d.size_hwl[2], d.yaw)

            vote = 0
            if yaw_upstream(d.yaw): vote += 1
            elif yaw_downstream(d.yaw): vote -= 1

            t.behavior_momentum = max(-momentum_clip, min(momentum_clip, t.behavior_momentum + vote))
            if t.behavior_momentum >= momentum_switch:
                t.behavior_state = "UP"
            elif t.behavior_momentum <= -momentum_switch:
                t.behavior_state = "DOWN"

            if (not prev_inside) and inside_now and (t.behavior_state == "UP" or yaw_upstream(d.yaw)):
                entry_events.append((now.isoformat(), t.tid))

            if prev_inside and (not inside_now) and (t.behavior_state == "DOWN" or yaw_downstream(d.yaw)):
                if t.exited_time is None:
                    t.exited_time = now
                exit_events.append((now.isoformat(), t.tid))

            if inside_now and (yaw_downstream(d.yaw) or t.behavior_state == "DOWN"):
                exit_behaviors.append((now.isoformat(), t.tid))

            t.last_xy = d.center_xy
            t.last_size_wl = (d.size_hwl[1], d.size_hwl[2])
            t.last_yaw = d.yaw
            t.last_time = now
            t.hits += 1; t.misses = 0
            t.confirmed = t.confirmed or (t.hits >= confirm_hits)
            t.inside_chamber = inside_now

            this_frame_bbox.append((t.tid, inside_now, d.center_xy[0], d.center_xy[1], d.size_hwl[1], d.size_hwl[2], d.yaw))

        for idx, t in enumerate(tracks):
            if idx not in {a for a, _ in pairs}:
                t.misses += 1

        for j, d in enumerate(dets):
            if j in {b for _, b in pairs}:
                continue
            inside_flag = inside_chamber_fn(d.center_xy, d.size_hwl[1], d.size_hwl[2], d.yaw)
            new_tid = next_tid
            tracks.append(Track(
                tid=new_tid,
                last_xy=d.center_xy,
                last_size_wl=(d.size_hwl[1], d.size_hwl[2]),
                last_yaw=d.yaw,
                last_time=now,
                hits=1, misses=0, confirmed=False,
                inside_chamber=inside_flag,
                exited_time=None
            ))
            this_frame_bbox.append((new_tid, inside_flag, d.center_xy[0], d.center_xy[1], d.size_hwl[1], d.size_hwl[2], d.yaw))
            next_tid += 1

        tracks = [t for t in tracks if t.misses <= max_missed]

        present_tids = [t.tid for t in tracks if t.last_time == now]
        inside_cnt = sum(1 for t in tracks if (t.last_time == now and t.inside_chamber))
        per_frame_ship_tids.append((fi, present_tids))
        per_frame_inside_counts.append(inside_cnt)
        per_frame_bboxinfo.append(this_frame_bbox)

        for t in tracks:
            if t.last_time == now:
                long_traj_rows.append((fi.seg_id, t.tid, fi.frame_id, fi.timestamp.isoformat(), t.last_xy[0], t.last_xy[1]))

    return per_frame_ship_tids, long_traj_rows, entry_events, exit_events, exit_behaviors, float(y_min), per_frame_inside_counts, per_frame_bboxinfo

def build_loiter_timeline(frames, per_frame_ship_tids, exit_events, exit_behaviors, inside_counts, cfg):
    LOITER_WINDOW = timedelta(seconds=float(cfg["thresholds"]["loiter_window_s"]))
    status_list = []

    exit_times = [datetime.fromisoformat(ts) for (ts, _) in exit_events]
    exit_times.sort()

    behavior_times = [datetime.fromisoformat(ts) for (ts, _) in exit_behaviors]
    behavior_times.sort()

    open_indices = []
    for i in range(1, len(frames)):
        if frames[i-1].gate_state == "OPENING" and frames[i].gate_state == "OPEN":
            open_indices.append(i)

    window_active = False
    deadline: Optional[datetime] = None
    next_open_ptr = 0
    ex_ptr = 0
    beh_ptr = 0
    loiter_due_to_late_exit_behavior_this_frame = False

    def start_window_at(t: datetime):
        nonlocal window_active, deadline, loiter_due_to_late_exit_behavior_this_frame
        window_active = True
        deadline = t + LOITER_WINDOW
        loiter_due_to_late_exit_behavior_this_frame = False

    for k in range(len(frames)):
        fi, _ = per_frame_ship_tids[k]

        if next_open_ptr < len(open_indices) and k == open_indices[next_open_ptr]:
            if fi.gate_state == "OPEN":
                start_window_at(fi.timestamp)
            next_open_ptr += 1

        while ex_ptr < len(exit_times) and exit_times[ex_ptr] <= fi.timestamp:
            ex_time = exit_times[ex_ptr]
            start_window_at(ex_time)
            ex_ptr += 1

        if fi.gate_state != "OPEN":
            window_active = False; deadline = None
            status_list.append((fi, "OK", "Gate not open", "—"))
            continue

        if not window_active:
            status_list.append((fi, "OK", "No active window", "—"))
            continue

        expired = (deadline is not None and fi.timestamp >= deadline)

        loiter_due_to_late_exit_behavior_this_frame = False
        while beh_ptr < len(behavior_times) and behavior_times[beh_ptr] <= fi.timestamp:
            beh_time = behavior_times[beh_ptr]
            if deadline is not None and beh_time >= deadline:
                loiter_due_to_late_exit_behavior_this_frame = True
            beh_ptr += 1

        if loiter_due_to_late_exit_behavior_this_frame:
            status_list.append((fi, "LOITERING", "Late exit behavior after deadline (yaw only)", "—"))
            continue

        if not expired:
            rem = int((deadline - fi.timestamp).total_seconds())
            status_list.append((fi, f"PENDING ({rem} s)", "Window running", "—"))
        else:
            status_list.append((fi, "EXPIRED", "Sticky window (waiting for late exit behavior)", "—"))

    return status_list

def overlay_frame(image_path: str, save_path: str, lines: List[str], status: str = "OK") -> None:
    bg_color = (30, 30, 30)
    if status.startswith("PENDING"):
        bg_color = (20, 30, 60)
    elif status == "LOITERING":
        bg_color = (60, 20, 20)
    elif status == "EXPIRED":
        bg_color = (50, 50, 20)

    try:
        im = Image.open(image_path).convert("RGB")
    except Exception:
        im = Image.new("RGB", (1280, 720), (30, 30, 30))
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    pad = 8; line_h = 16
    box_w = max([len(s) for s in lines]+[1]) * 8 + pad*2
    box_h = line_h*len(lines) + pad*2
    draw.rectangle([0, 0, box_w, box_h], fill=bg_color)
    y = pad
    for s in lines:
        draw.text((pad, y), s, fill=(255, 255, 255), font=font)
        y += line_h
    os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
    im.save(save_path)

def compute_gate_states_for_segments(frames_by_segment, cfg):
    for frames in frames_by_segment.values():
        frames.sort(key=lambda x: x.timestamp)
        compute_gate_states(frames, cfg)

def run(add_info_path: str, labels_dir: str, config_path: str, visualize: bool = False):
    cfg = load_config(config_path)
    add_map = load_add_info(add_info_path)

    out_root = cfg.get("output", {}).get("root", "./outputs")
    os.makedirs(out_root, exist_ok=True)
    out_dir = os.path.join(out_root, "loitering_labels")
    os.makedirs(out_dir, exist_ok=True)

    vis_dir = None
    if visualize:
        vis_dir = os.path.join(out_dir, "vis")
        os.makedirs(vis_dir, exist_ok=True)

    frames_by_segment: dict[int, List[FrameInfo]] = defaultdict(list)
    for fname in os.listdir(labels_dir):
        m = FRAME_BASENAME_RE.match(fname)
        if not m:
            continue
        fid = m.group(1)
        if fid not in add_map:
            continue
        ts, ts_str, seg = add_map[fid]
        ships, gate_w = parse_label_file(os.path.join(labels_dir, fname), cfg)
        frames_by_segment[seg].append(FrameInfo(
            frame_id=fid,
            timestamp=ts,
            timestamp_str=ts_str,
            seg_id=seg,
            gate_width_m=gate_w,
            gate_present=(gate_w is not None),
            gate_state="UNKNOWN",
            ships=ships
        ))

    compute_gate_states_for_segments(frames_by_segment, cfg)

    # Stable segment order: by first timestamp in segment, then seg id
    seg_order = sorted(
        frames_by_segment.keys(),
        key=lambda s: (min(fr.timestamp for fr in frames_by_segment[s]), s)
    )

    per_frame_rows = []
    traj_rows = []
    entry_events_all = []
    exit_events_all = []
    exit_behaviors_all = []
    loiter_events = []

    images_dir = cfg.get("data", {}).get("images_dir", "./images")
    show_bbox = cfg.get("debug", {}).get("print_bbox_on_image", True)
    max_ships_overlay = int(cfg.get("debug", {}).get("max_ships_overlay", 6))

    chamber_x = tuple(cfg["geometry"]["chamber_x_range"])
    y_min = float(cfg["geometry"]["gate_y_range"][0])
    y_max = float(cfg["geometry"]["chamber_y_range"][1])
    chamber_y_rect = (y_min, y_max)

    for seg in seg_order:
        frames = frames_by_segment[seg]
        (per_frame_ship_tids,
         long_traj,
         entry_events,
         exit_events,
         exit_behaviors,
         _,
         inside_counts,
         per_frame_bboxinfo) = track_segment(frames, cfg)

        traj_rows.extend(long_traj)
        entry_events_all.extend([{"segment": seg, "time": t, "tid": tid} for (t, tid) in entry_events])
        exit_events_all.extend([{"segment": seg, "time": t, "tid": tid} for (t, tid) in exit_events])
        exit_behaviors_all.extend([{"segment": seg, "time": t, "tid": tid} for (t, tid) in exit_behaviors])

        timeline = build_loiter_timeline(frames, per_frame_ship_tids, exit_events, exit_behaviors, inside_counts, cfg)

        active = False
        for (fi, st, reason, _) in timeline:
            if st == "LOITERING" and not active:
                loiter_events.append({
                    "segment": seg,
                    "start_frame": fi.frame_id,
                    "start_time": fi.timestamp.isoformat(),
                    "reason": reason
                })
                active = True
            if st != "LOITERING" and active:
                active = False

        events_by_ts = defaultdict(lambda: {"entry": [], "exit": []})
        for e in entry_events_all:
            if e["segment"] == seg:
                events_by_ts[e["time"]]["entry"].append(e["tid"])
        for e in exit_events_all:
            if e["segment"] == seg:
                events_by_ts[e["time"]]["exit"].append(e["tid"])

        for idx, (fi, tids) in enumerate(per_frame_ship_tids):
            _, st, reason, _ = timeline[idx]
            key = fi.timestamp.isoformat()
            ev = events_by_ts.get(key, {"entry": [], "exit": []})
            ev_str = ""
            if ev["entry"]:
                ev_str += f"entry:{','.join(map(str, sorted(ev['entry'])))}"
            if ev["exit"]:
                if ev_str:
                    ev_str += " | "
                ev_str += f"exit:{','.join(map(str, sorted(ev['exit'])))}"

            lines = [
                f"ts: {fi.timestamp_str}  seg: {fi.seg_id}",
                f"gate: {fi.gate_state}  present: {int(fi.gate_present)}  width: {fi.gate_width_m if fi.gate_width_m is not None else 'NA'} m",
                f"inside_cnt: {inside_counts[idx]}  ships: {len(tids)}  ids: {','.join(map(str, sorted(tids))) if tids else '-'}",
                f"status: {st}",
                f"reason: {reason}",
                f"events: {ev_str if ev_str else '-'}",
                f"rect: x[{chamber_x[0]}, {chamber_x[1]}] y[{chamber_y_rect[0]}, {chamber_y_rect[1]}] (bbox intersect)"
            ]

            if show_bbox and idx < len(per_frame_bboxinfo):
                lst = per_frame_bboxinfo[idx]
                if lst:
                    lines.append("bbox/intersect:")
                    for tid, inside_flag, cx, cy, w, l, yaw in lst[:max_ships_overlay]:
                        lines.append(f"  T{tid} inside={int(inside_flag)} cx={cx:.2f} cy={cy:.2f} w={w:.2f} l={l:.2f} yaw={yaw:.3f}")
                    if len(lst) > max_ships_overlay:
                        lines.append(f"  ... (+{len(lst)-max_ships_overlay} more)")

            if visualize:
                inp_img = os.path.join(images_dir, f"{fi.frame_id}.png")
                out_img = os.path.join(vis_dir, f"{fi.timestamp_str}.png")
                overlay_frame(inp_img, out_img, lines, status=st)

            per_frame_rows.append({
                "frame_id": fi.frame_id,
                "timestamp": fi.timestamp.isoformat(),
                "segment": fi.seg_id,
                "gate_present": int(fi.gate_present),
                "gate_width_m": f"{fi.gate_width_m:.3f}" if fi.gate_width_m is not None else "",
                "gate_state": fi.gate_state,
                "ship_tids": ";".join(map(str, sorted(tids))) if tids else "",
                "num_ships": len(tids),
                "status": st,
                "reason": reason,
                "why_not_start": "—",
                "events": ev_str
            })

    # Global, stable ordering before writing
    per_frame_rows.sort(key=lambda r: (r["timestamp"], r["segment"], r["frame_id"]))
    traj_rows.sort(key=lambda t: (t[3], t[0], t[1]))  # (timestamp, segment, tid)
    entry_events_all.sort(key=lambda e: (e["time"], e["segment"], e["tid"]))
    exit_events_all.sort(key=lambda e: (e["time"], e["segment"], e["tid"]))
    exit_behaviors_all.sort(key=lambda e: (e["time"], e["segment"], e["tid"]))
    loiter_events.sort(key=lambda e: (e["start_time"], e["segment"], e["start_frame"]))

    csv_path = os.path.join(out_dir, "labels_per_frame.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["frame_id","timestamp","segment","gate_present","gate_width_m","gate_state",
                      "ship_tids","num_ships","status","reason","why_not_start","events"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in per_frame_rows:
            writer.writerow(r)

    traj_csv = os.path.join(out_dir, "labels_tracks.csv")
    with open(traj_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["segment","tid","frame_id","timestamp","x","y"])
        for seg, tid, fid, t, x, y in traj_rows:
            w.writerow([seg, tid, fid, t, f"{x:.3f}", f"{y:.3f}"])

    json_path = os.path.join(out_dir, "labels_events_and_loiter.json")
    with open(json_path, "w", encoding="utf-8") as fj:
        json.dump({
            "entry_events": entry_events_all,
            "exit_events": exit_events_all,
            "exit_behaviors": exit_behaviors_all,
            "loitering_events": loiter_events
        }, fj, ensure_ascii=False, indent=2)

    print("[OK] Output directory:", out_dir)
    print("[OK] Wrote:", csv_path)
    print("[OK] Wrote:", traj_csv)
    print("[OK] Wrote:", json_path)
    if visualize:
        print("[OK] Visualizations dir:", vis_dir)
    else:
        print("[OK] Visualization skipped (use --viz to enable)")

def main():
    ap = argparse.ArgumentParser(description="Yaw-only behavior-based loitering labels (with momentum smoothing).")
    ap.add_argument("--add-info", default="add_info_testing/add_info_testing_by_timestamp.txt")
    ap.add_argument("--labels-dir", default="trout_test_label")
    ap.add_argument("--cfg", default="configs/config.yaml")
    ap.add_argument("--viz", action="store_true", help="Enable visualization overlays")
    args = ap.parse_args()

    run(args.add_info, args.labels_dir, args.cfg, visualize=args.viz)

if __name__ == "__main__":
    main()

