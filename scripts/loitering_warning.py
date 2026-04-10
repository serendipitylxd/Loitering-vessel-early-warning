#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
loitering_warning.py 
"""

import os
import json
import math
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Optional, List, Tuple, Union
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import csv

from make_loitering_labels import (
    load_config,
    Detection,
    FrameInfo,
    compute_gate_states_for_segments,  
    track_segment,
    #build_loiter_timeline as _orig_build_loiter_timeline  
)

# --------------------------- utils ---------------------------

def rot_corners_xy(x, y, dx, dy, yaw):
    hx, hy = dx / 2.0, dy / 2.0
    base = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]], dtype=np.float32)
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    return base @ R.T + np.array([x, y], dtype=np.float32)


def load_points(points_path, max_points=200000):
    if not os.path.exists(points_path):
        return None
    try:
        a = np.fromfile(points_path, dtype=np.float32)
        pts = a.reshape(-1, 4)[:, :3]
        if max_points > 0 and pts.shape[0] > max_points:
            idx = np.random.choice(pts.shape[0], size=max_points, replace=False)
            pts = pts[idx, :]
        return pts
    except Exception:
        return None


def draw_2d_with_boxes(img_path, det2d, cfg):
    """Load 2D image, draw det2d boxes with category colors, rotate CCW 90°, resize 320x320."""
    if not os.path.exists(img_path):
        return None
    try:
        im = Image.open(img_path).convert('RGB')
    except Exception:
        return None

    draw = ImageDraw.Draw(im)
    font = ImageFont.load_default()
    ship_labels = set(cfg['labels']['ships'])
    gate_label = cfg['labels']['lock_gate']

    col_ship = (31, 119, 180)
    col_gate = (255, 127, 14)
    col_other = (127, 127, 127)

    for d in det2d:
        bb = d.get('bbox', [])
        if len(bb) != 4:
            continue
        x1, y1, x2, y2 = [float(t) for t in bb]
        lbl = d.get('label', '')
        if lbl in ship_labels:
            color = col_ship
        elif lbl == gate_label:
            color = col_gate
        else:
            color = col_other
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        txt = f"{lbl}"
        # Pillow >= 10 compatibility: Prefer getbbox
        try:
            bbox = font.getbbox(txt)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            try:
                tw, th = font.getsize(txt)
            except Exception:
                tw, th = len(txt) * font.size, font.size
        draw.rectangle([x1, y1 - th, x1 + tw + 4, y1], fill=(255, 255, 255))
        draw.text((x1 + 2, y1 - th), txt, fill=color, font=font)

    # Clockwise rotation by 90 degrees
    im = im.rotate(90, expand=True)
    im = im.resize((320, 320))
    return im

# ---------------------- Gate state from 2D/width/bev_area ----------------------

def _select_gate_area(det2d, gate_label: str) -> Optional[float]:
    """Select the Lock_gate box from det2d (select the one with the largest area), and return the area; if none exists, return None"""
    best = None
    for d in det2d:
        if d.get('label') != gate_label:
            continue
        bb = d.get('bbox', [])
        if len(bb) != 4:
            continue
        x1, y1, x2, y2 = map(float, bb)
        area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
        if best is None or area > best:
            best = area
    return best

def _select_gate_bev_area(det3d, gate_label: str) -> Optional[float]:
    "Select the Lock_gate box from det3d (choose the one with the largest BEV area), and return the top-down area; if none exists, return None."
    best = None
    for d in det3d:
        if d.get('label') != gate_label:
            continue
        box = d.get('box7d', [])
        if len(box) < 7:
            continue
        _, _, _, dx, dy, _, _ = map(float, box[:7])
        area = max(0.0, dx) * max(0.0, dy)
        if best is None or area > best:
            best = area
    return best

def apply_gate_state_from_2d(frames, fused_data, cfg):
    """Using 2D Lock_gate to infer gate_state based on the continuous frame area trend: 
    OPENING/OPEN/CLOSING/CLOSED.Only consider the overall trend; default N = 5 frame window.
    """
    gate_label = cfg['labels']['lock_gate']
    trend_N = int(cfg.get('gate_2d', {}).get('trend_frames', 5))

    areas = []
    for fi in frames:
        det2d = fused_data.get(fi.frame_id, {}).get('det2d', [])
        a = _select_gate_area(det2d, gate_label)
        areas.append(a)

    for i, fi in enumerate(frames):
        a_now = areas[i]
        if a_now is None:
            fi.gate_state = 'OPEN'
            continue
        if i >= trend_N - 1 and all(areas[j] is not None for j in range(i - trend_N + 1, i + 1)):
            a_first = areas[i - trend_N + 1]
            a_last  = areas[i]
            if a_last < a_first:
                fi.gate_state = 'OPENING'
            elif a_last > a_first:
                fi.gate_state = 'CLOSING'
            else:
                fi.gate_state = 'CLOSED'
        else:
            fi.gate_state = 'CLOSED'

def apply_gate_state_from_bev_area(frames, fused_data, cfg):
    """Infer the gate_state based on the 3D Lock_gate box7d's top-down area trend"""
    gate_label = cfg['labels']['lock_gate']
    trend_N = int(cfg.get('gate_bev', {}).get('trend_frames', 5))

    areas = []
    for fi in frames:
        det3d = fused_data.get(fi.frame_id, {}).get('det3d', [])
        a = _select_gate_bev_area(det3d, gate_label)
        areas.append(a)

    for i, fi in enumerate(frames):
        a_now = areas[i]
        if a_now is None:
            fi.gate_state = 'OPEN'
            continue
        if i >= trend_N - 1 and all(areas[j] is not None for j in range(i - trend_N + 1, i + 1)):
            a_first = areas[i - trend_N + 1]
            a_last  = areas[i]
            if a_last < a_first:
                fi.gate_state = 'OPENING'
            elif a_last > a_first:
                fi.gate_state = 'CLOSING'
            else:
                fi.gate_state = 'CLOSED'
        else:
            fi.gate_state = 'CLOSED'

def apply_gate_state_by_mode(frames, fused_data, cfg):
    """Select the method for determining the gate state based on gate_state.mode." 
    gate_state.mode:
    - "width":  Trend of the width of 3D Lock_gate box7d
    - "2d":      Area trend of the image 2D Lock_gate bbox
    - "bev_area": Area trend of the 3D Lock_gate box7d from the top view
    """
    mode = str(cfg.get("gate_state", {}).get("mode", "width")).strip().lower()

    if mode == "width":
        apply_gate_state_from_width(frames, cfg, fused_data=fused_data)
    elif mode == "2d":
        if fused_data is None:
            raise ValueError("gate_state.mode='2d' requires fused_data")
        apply_gate_state_from_2d(frames, fused_data, cfg)
    elif mode == "bev_area":
        if fused_data is None:
            raise ValueError("gate_state.mode='bev_area' requires fused_data")
        apply_gate_state_from_bev_area(frames, fused_data, cfg)
    else:
        raise ValueError(
            f"Unsupported gate_state.mode={mode!r}, expected 'width', '2d' or 'bev_area'"
        )

def apply_gate_state_from_width(frames, cfg, fused_data=None):
    p = cfg.get("gate_width", {})

    closed_width = float(p.get("closed_width_m", 23.0))
    closed_tol   = float(p.get("closed_tol_m", 0.5))

    N       = int(p.get("trend_frames", 7))
    med_win = int(p.get("median_window", 5))
    slope_eps = float(p.get("slope_eps_m", 0.25))
    confirm_k = int(p.get("confirm_frames", 2))

    min_valid = int(p.get("min_valid_frames", 3))
    fallback_to_2d = bool(p.get("fallback_to_2d", False))
    open_min_width = float(p.get("open_min_width_m", 0.5))  # Less than this value indicates OPEN.

    # 1) Original width sequence: Directly recorded as 0 without the presence of a gate.
    w_raw = []
    for fi in frames:
        v = getattr(fi, "gate_width_m", None)
        try:
            w_raw.append(0.0 if v is None else float(v))
        except Exception:
            w_raw.append(0.0)

    valid = np.array([v for v in w_raw if v > 0], dtype=float)
    if valid.size < min_valid:
        if fused_data is not None and fallback_to_2d:
            apply_gate_state_from_2d(frames, fused_data, cfg)
            return
        for fi in frames:
            fi.gate_state = "OPEN"
        return

    # 2) Causal median smoothing: Only consider the past med_win frames
    if med_win < 1:
        med_win = 1

    w_smooth = []
    for i in range(len(w_raw)):
        lo = max(0, i - med_win + 1)
        hi = i + 1
        win = w_raw[lo:hi]
        w_smooth.append(float(np.median(win)) if len(win) > 0 else 0.0)

    # 3) State machine: Only believe in multi-frame trends
    state = "CLOSED"
    closed_run = 0

    for i, fi in enumerate(frames):
        w = w_smooth[i]

        # 3.1 If the width is close to 0, directly consider it as OPEN
        if w <= open_min_width:
            state = "OPEN"
            fi.gate_state = "OPEN"
            continue

        # 3.2 The width is close to the closed width. After continuous confirmation, it is considered as "CLOSED".
        if abs(w - closed_width) <= closed_tol:
            closed_run += 1
        else:
            closed_run = 0

        if closed_run >= confirm_k:
            state = "CLOSED"
            fi.gate_state = "CLOSED"
            continue

        # 3.3 Using Only Multi-frame Trends
        trend = None
        j = i - (N - 1)
        if j >= 0:
            dw = w - w_smooth[j]
            if dw >= slope_eps:
                trend = "CLOSING"
            elif dw <= -slope_eps:
                trend = "OPENING"

        # 3.4 Combination Rule: Only follow the trend
        if trend is not None:
            cand = trend
        else:
            # When there is no clear trend, hold the position at the current level.
            if state == "OPEN":
                cand = "CLOSING"
            elif w < (closed_width - closed_tol):
                cand = "OPENING"
            else:
                cand = state

        state = cand
        fi.gate_state = cand



# ---------------- build_loiter_timeline (robust edge) -----------

def build_loiter_timeline(frames, per_frame_ship_tids, exit_events, exit_behaviors, inside_counts, cfg):
    """Identify: When a non-OPEN signal transitions to an OPEN state and the first frame is OPEN with inside > 0, trigger."""
    win_s = None
    try:
        win_s = float(cfg['thresholds']['loiter_window_s'])
    except Exception:
        try:
            win_s = float(cfg['loiter']['window_seconds'])
        except Exception:
            win_s = 60.0
    LOITER_WINDOW = timedelta(seconds=win_s)

    status_list = []
    exit_times = [datetime.fromisoformat(ts) for (ts, _) in exit_events]; exit_times.sort()
    behavior_times = [datetime.fromisoformat(ts) for (ts, _) in exit_behaviors]; behavior_times.sort()

    open_indices = []
    if len(frames) > 0 and getattr(frames[0], 'gate_state', 'UNKNOWN') == 'OPEN' and (inside_counts[0] if inside_counts else 0) > 0:
        open_indices.append(0)
    for i in range(1, len(frames)):
        prev = getattr(frames[i-1], 'gate_state', 'UNKNOWN')
        now  = getattr(frames[i],   'gate_state', 'UNKNOWN')
        if prev != 'OPEN' and now == 'OPEN' and (inside_counts[i] if inside_counts else 0) > 0:
            open_indices.append(i)

    window_active = False
    deadline: Optional[datetime] = None
    next_open_ptr = 0
    ex_ptr = 0
    beh_ptr = 0

    def start_window_at(t: datetime):
        nonlocal window_active, deadline
        window_active = True
        deadline = t + LOITER_WINDOW

    for k in range(len(frames)):
        fi, _ = per_frame_ship_tids[k]

        # 1) gate open 
        if next_open_ptr < len(open_indices) and k == open_indices[next_open_ptr]:
            if fi.gate_state == 'OPEN':
                start_window_at(fi.timestamp)
            next_open_ptr += 1

        # 2) ship exit 
        while ex_ptr < len(exit_times) and exit_times[ex_ptr] <= fi.timestamp:
            ex_time = exit_times[ex_ptr]
            start_window_at(ex_time)
            ex_ptr += 1

        # 3) If not "OPEN", then clear the window.
        if fi.gate_state != 'OPEN':
            window_active = False; deadline = None
            reason = 'Gate ' + fi.gate_state if fi.gate_state in ('OPENING','CLOSING','CLOSED') else 'Gate not open'
            status_list.append((fi, 'OK', reason, '—'))
            continue

        # 4) OPEN but the window is not activated
        if not window_active:
            status_list.append((fi, 'OK', 'Open without active window', '—'))
            continue

        # 5) Handle the late-exit behavior
        while beh_ptr < len(behavior_times) and behavior_times[beh_ptr] < fi.timestamp:
            beh_ptr += 1
        late_exit = False
        if beh_ptr < len(behavior_times):
            if behavior_times[beh_ptr] == fi.timestamp and deadline is not None and fi.timestamp >= deadline:
                late_exit = True

        if late_exit:
            status_list.append((fi, 'LOITERING', 'Late exit behavior after deadline', '—'))
            continue

        if deadline is not None and fi.timestamp < deadline:
            rem = int((deadline - fi.timestamp).total_seconds())
            status_list.append((fi, f'PENDING ({rem} s)', 'Window running', '—'))
        else:
            status_list.append((fi, 'EXPIRED', 'Sticky window (waiting for late exit behavior)', '—'))

    return status_list

# --------------------------- Kalman Filtering (robust) ---------------------------

def _kf_predict(x: np.ndarray, P: np.ndarray, F: np.ndarray, Q: np.ndarray):
    x = F @ x
    P = F @ P @ F.T + Q
    return x, P

def _kf_update(x: np.ndarray, P: np.ndarray, z: np.ndarray, H: np.ndarray, R: np.ndarray):
    y = z - (H @ x)
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    x = x + K @ y
    I = np.eye(P.shape[0])
    P = (I - K @ H) @ P
    return x, P

def _smooth_xy_sequence(seq):
    """
    Perform KF smoothing on the trajectory sequence.
    Compatible with each element as: - (x,y)
    - (k,x,y) / (t,x,y)
    - None (No measurement)
    Return: [(x, y), ...] (with the same length as the input)
    """
    if not seq or len(seq) < 2:
        return seq

    # Find the first frame of valid observation for initialization
    first = None
    for pt in seq:
        xy = _extract_xy(pt)
        if xy is not None:
            first = xy
            break
    if first is None:
        return seq

    x0, y0 = float(first[0]), float(first[1])
    x = np.array([x0, y0, 0., 0.], dtype=float)  # [x,y,vx,vy]
    P = np.eye(4) * 10.0
    F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], dtype=float)
    H = np.array([[1,0,0,0],[0,1,0,0]], dtype=float)
    R = np.eye(2) * 0.5
    Q = np.eye(4) * 0.01

    out = []
    for pt in seq:
        # predict
        x = F @ x
        P = F @ P @ F.T + Q

        xy = _extract_xy(pt)
        if xy is not None:
            z = np.array([float(xy[0]), float(xy[1])], dtype=float)
            y = z - (H @ x)
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ y
            I = np.eye(P.shape[0])
            P = (I - K @ H) @ P

        out.append((float(x[0]), float(x[1])))

    return out

def apply_kalman_to_tracks(long_traj: Union[dict, list]):
    """
    Perform KF smoothing on the trajectory. Compatibility: - dict: {tid: [(x,y), ...] }
    - list: [ [(x,y), ...], ... ] The indexed value is simply referred to as "tid".
    If the structure is unable to recognize it (for example, the element is not in the form of (x, y)), it will be returned as is.
    """
    if isinstance(long_traj, dict):
        out = {}
        for tid, seq in long_traj.items():
            if isinstance(seq, (list, tuple)) and len(seq) > 0 and isinstance(seq[0], (list, tuple)) and len(seq[0]) >= 2:
                out[tid] = _smooth_xy_sequence(seq)
            else:
                out[tid] = seq
        return out
    if isinstance(long_traj, (list, tuple)):
        if len(long_traj) > 0 and isinstance(long_traj[0], (list, tuple)):
            
            if len(long_traj[0]) > 0 and isinstance(long_traj[0][0], (list, tuple)) and len(long_traj[0][0]) >= 2:
                return [_smooth_xy_sequence(seq) for seq in long_traj]
        return long_traj
    return long_traj

def kalman_smooth_counts(counts: List[float]) -> List[float]:
    """Perform 1D KF smoothing on the "inside_counts" (the number of ships within the gate in each frame)."""
    if not counts:
        return counts
    x = np.array([float(counts[0]), 0.], dtype=float)  # [c, dc]
    P = np.eye(2) * 10.0
    F = np.array([[1,1],[0,1]], dtype=float)
    H = np.array([[1,0]], dtype=float)
    R = np.array([[1.0]], dtype=float)
    Q = np.eye(2) * 0.01
    out = []
    for c in counts:
        x, P = _kf_predict(x, P, F, Q)
        z = np.array([float(c)], dtype=float)
        x, P = _kf_update(x, P, z, H, R)
        out.append(float(x[0]))
    return out



def build_heading_sequences(per_frame_ship_tids, fused_data, cfg):
    """
    Build an occurrence-aligned heading sequence for each tid.
    Priority is given to reading the vessel heading from fused_data[frame_id]["det3d"] via box7d[6];
    if unavailable, fall back to fused_data[frame_id]["detections"].
    In the current scene, there is usually only one vessel per frame,
    so the same yaw is reused for all tids within that frame.

    """
    ship_labels = set(cfg.get("labels", {}).get("ships", []))
    heading_map = defaultdict(list)

    for fi, tids in per_frame_ship_tids:
        info = fused_data.get(fi.frame_id, {}) or {}
        yaw = None

        dets3 = info.get("det3d", []) or []
        for d in dets3:
            lbl = d.get("label", "")
            box = d.get("box7d", []) or []
            if lbl in ship_labels and len(box) >= 7:
                try:
                    yaw = float(box[6])
                    break
                except Exception:
                    yaw = None

        if yaw is None:
            dets = info.get("detections", []) or []
            for d in dets:
                lbl = d.get("label", "")
                box = d.get("box7d", []) or []
                if (not ship_labels or lbl in ship_labels) and len(box) >= 7:
                    try:
                        yaw = float(box[6])
                        break
                    except Exception:
                        yaw = None

        for tid in tids:
            heading_map[int(tid)].append(yaw)

    return dict(heading_map)

def wrap_to_pi(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi

def angle_diff(a: float, b: float) -> float:
    return wrap_to_pi(a - b)


def smooth_heading_by_history(headings, stable_len=3, stable_tol_deg=20.0, jump_deg=120.0):
    """
    Strictly causal heading despiking:
    - Only historical frames are used
    - If the current frame shows a large jump relative to the previous frame and the historical sequence is stable, correct the current frame to match the previous one
    - If the previous frame has just been corrected, do not continue spike detection on the current frame, so as to avoid chained misjudgment
    """
    if not headings:
        return headings, set()

    stable_tol = np.deg2rad(stable_tol_deg)
    jump_rad = np.deg2rad(jump_deg)

    out = list(headings)
    fixed_idx = set()

    for i in range(1, len(out)):
        cur = out[i]
        prev = out[i - 1]

        if cur is None or prev is None:
            continue


        if (i - 1) in fixed_idx:
            continue

        hist = []
        j0 = max(0, i - int(stable_len))
        for j in range(j0, i):
            if out[j] is not None:
                hist.append(out[j])

        if len(hist) < 1:
            continue

        hist_stable = all(abs(angle_diff(h, prev)) <= stable_tol for h in hist)
        cur_jump = abs(angle_diff(cur, prev)) >= jump_rad

        if hist_stable and cur_jump:
            out[i] = prev
            fixed_idx.add(i)

    return out, fixed_idx


def filter_exit_behaviors_by_heading_spikes(
    per_frame_ship_tids,
    exit_behaviors,
    heading_fix_map,
    timeline_pre,
    prev_status_frames=3
):
    """
    Only apply a minor correction on top of the original decision:
    - The current event must fall on a frame whose heading was corrected
    - The current frame's pre label must be LOITERING
    - The preceding few frames must not have already been continuously labeled as LOITERING
    Only when these conditions are satisfied should the exit_behavior be removed
    """
    tid_occ = defaultdict(int)
    ts_tid_to_occ = {}
    ts_to_frame_idx = {}

    for k, (fi, tids) in enumerate(per_frame_ship_tids):
        ts_iso = fi.timestamp.isoformat()
        ts_to_frame_idx[ts_iso] = k
        for tid in tids:
            tid = int(tid)
            occ = tid_occ[tid]
            ts_tid_to_occ[(tid, ts_iso)] = occ
            tid_occ[tid] += 1

    pre_status_list = []
    for item in timeline_pre:
        _fi, status, _reason, _extra = item
        pre_status_list.append(status)

    filtered = []

    for ts_iso, tid in exit_behaviors:
        tid = int(tid)
        occ = ts_tid_to_occ.get((tid, ts_iso), None)
        fixed_set = heading_fix_map.get(tid, set())
        frame_idx = ts_to_frame_idx.get(ts_iso, None)

        # Condition 1: The "corrected course frame" must be hit.”
        if occ is None or occ not in fixed_set or frame_idx is None:
            filtered.append((ts_iso, tid))
            continue

        # Condition 2: The current frame's pre label must be LOITERING
        cur_status = pre_status_list[frame_idx] if 0 <= frame_idx < len(pre_status_list) else ""
        if cur_status != "LOITERING":
            filtered.append((ts_iso, tid))
            continue

        # Condition 3: Only inspect the statuses of the preceding few frames
        j0 = max(0, frame_idx - int(prev_status_frames))
        prev_stats = pre_status_list[j0:frame_idx]

        # If the preceding frames have already been continuously labeled as LOITERING, do not remove it
        if any(s == "LOITERING" for s in prev_stats):
            filtered.append((ts_iso, tid))
            continue

        # Remove it only when the preceding frames are not LOITERING and the current frame suddenly becomes LOITERING
        continue

    return filtered

# --------------------------- Writers (TXT) ---------------------------

def _extract_xy(pt):
    """Compatible with multiple structures such as (x,y), (k,x,y), (t,x,y), and dict, and returns (x,y) or None."""
    if pt is None:
        return None

    # dict
    if isinstance(pt, dict):
        for kx, ky in (("x", "y"), ("cx", "cy"), ("X", "Y")):
            if kx in pt and ky in pt:
                try:
                    return float(pt[kx]), float(pt[ky])
                except Exception:
                    return None
        return None

    # list/tuple/np
    if isinstance(pt, (list, tuple, np.ndarray)):
        if len(pt) < 2:
            return None

        # Try in order: (x,y) / (k,x,y) / (...,x,y)
        cand_pairs = [(0, 1)]
        if len(pt) >= 3:
            cand_pairs.append((1, 2))
        cand_pairs.append((-2, -1))

        for i, j in cand_pairs:
            try:
                x = float(pt[i])
                y = float(pt[j])
                return x, y
            except Exception:
                continue
        return None

    return None

def _is_flat_traj_rows(obj):
    """
    Determine whether long_traj is already in flat row-record format:
    [(seg, tid, frame_id, ts_iso, x, y), ...]
    """
    if not isinstance(obj, (list, tuple)) or len(obj) == 0:
        return False
    first = obj[0]
    if not isinstance(first, (list, tuple)) or len(first) < 6:
        return False
    # The first element of a flat row is usually seg (int), not a nested structure like (x,y)
    return not isinstance(first[0], (list, tuple, dict, np.ndarray))


def _traj_to_rows(seg, per_frame_ship_tids, long_traj):
    """
    Convert trajectories into rows in a unified format:
    (seg, tid, frame_idx_or_frame_id, ts_iso, x, y)
    """
    if _is_flat_traj_rows(long_traj):
        return list(long_traj)
    return _collect_tracks_rows(seg, per_frame_ship_tids, long_traj)

def _collect_tracks_rows(seg, per_frame_ship_tids, long_traj):
    """
    More robust trajectory export:
      - If the sequence length is approximately equal to the number of frames, prioritize seq[k]
      - Otherwise, fall back to seq[occ] (original logic)
      - Write empty values if point extraction fails
    """
    rows = []
    is_dict = isinstance(long_traj, dict)
    tid_occ_idx = defaultdict(int)
    n_frames = len(per_frame_ship_tids)

    for k, (fi, tids) in enumerate(per_frame_ship_tids):
        ts_iso = fi.timestamp.isoformat()
        for tid in tids:
            occ = tid_occ_idx[tid]

            # Get the sequence for this tid
            seq = None
            if is_dict:
                if tid in long_traj:
                    seq = long_traj[tid]
                elif str(tid) in long_traj:   # Compatible with string keys
                    seq = long_traj[str(tid)]
            else:
                if isinstance(long_traj, (list, tuple)):
                    # Compatible with list mode: tid may start from 1
                    if 0 <= tid < len(long_traj):
                        seq = long_traj[tid]
                    elif 0 <= tid - 1 < len(long_traj):
                        seq = long_traj[tid - 1]

            xys = None

            if isinstance(seq, (list, tuple)):
                # 1) Prefer frame index k first (many implementations are frame-aligned)
                if 0 <= k < len(seq):
                    xys = _extract_xy(seq[k])
                # 2) Fall back to occurrence count occ (your original logic)
                if xys is None and 0 <= occ < len(seq):
                    xys = _extract_xy(seq[occ])

            if xys is None:
                rows.append((seg, tid, k, ts_iso, "", ""))
            else:
                x, y = xys
                rows.append((seg, tid, k, ts_iso, f"{x:.3f}", f"{y:.3f}"))

            tid_occ_idx[tid] += 1

    return rows

def _collect_labels_per_frame_rows(seg, per_frame_ship_tids, timeline,
                                   entry_events, exit_events, exit_behaviors):
    ev_map = defaultdict(list)
    for t, tid in entry_events:
        ev_map[t].append(f"entry:T{tid}")
    for t, tid in exit_events:
        ev_map[t].append(f"exit:T{tid}")
    for t, tid in exit_behaviors:
        ev_map[t].append(f"exit_behavior:T{tid}")
    rows = []
    for k, (fi, tids) in enumerate(per_frame_ship_tids):
        _fi2, status, reason, _extra = timeline[k]
        ship_tids_str = ";".join(str(t) for t in tids)
        evs = ev_map.get(fi.timestamp.isoformat(), [])
        evs_str = "|".join(evs) if evs else "—"
        why_not_start = "—"
        gate_width_val = "" if fi.gate_width_m is None else f"{fi.gate_width_m:.3f}"
        rows.append(dict(
            frame_id=fi.frame_id,
            timestamp=fi.timestamp.isoformat(),
            segment=seg,
            gate_present=int(fi.gate_present),
            gate_width_m=gate_width_val,
            gate_state=fi.gate_state,
            ship_tids=ship_tids_str,
            num_ships=len(tids),
            status=status,
            reason=reason,
            why_not_start=why_not_start,
            events=evs_str
        ))
    return rows

def _collect_events_and_timeline(seg, timeline, entry_events, exit_events, exit_behaviors):
    tl = [dict(
        frame_id=fi.frame_id,
        timestamp=fi.timestamp.isoformat(),
        gate_state=fi.gate_state,
        status=status,
        reason=reason,
        segment=seg
    ) for (fi, status, reason, _extra) in timeline]
    ee = [dict(time=t, tid=tid, segment=seg) for (t, tid) in entry_events]
    xe = [dict(time=t, tid=tid, segment=seg) for (t, tid) in exit_events]
    xb = [dict(time=t, tid=tid, segment=seg) for (t, tid) in exit_behaviors]
    return tl, ee, xe, xb
# --------------------------- drawing ---------------------------

def _is_virtual_ship_det(d):
    if not isinstance(d, dict):
        return False
    if bool(d.get('is_virtual', False)):
        return True
    tag = str(d.get('virtual_tag', '')).strip().lower()
    src = str(d.get('virtual_source', '')).strip().lower()
    if 'virtual' in tag or 'virtual' in src:
        return True
    return False


def draw_loitering_planview(fi, pts, dets3, det2d, img_path, status, reason, num_ships, cfg, out_path, events_text=""):
    colors_status = {
        'OK': '#2ca02c',
        'PENDING': '#1f77b4',
        'LOITERING': '#d62728',
        'EXPIRED': '#ff7f0e'
    }

    col_ship = '#1f77b4'
    col_gate = '#ff7f0e'
    col_other = '#7f7f7f'

    chamber_x = tuple(cfg['geometry']['chamber_x_range'])
    gate_y_range = tuple(cfg['geometry']['gate_y_range'])
    x_min_ch, x_max_ch = chamber_x
    gate_y0, gate_y1 = gate_y_range

    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)

    # Point cloud (point size x2)
    if pts is not None and pts.size > 0:
        ax.scatter(pts[:, 0], pts[:, 1], s=0.5, c='black', alpha=0.35, linewidths=0)

    # Orange gate band
    ax.add_patch(Rectangle((x_min_ch, gate_y0), x_max_ch - x_min_ch, gate_y1 - gate_y0,
                           facecolor='#ff7f0e', alpha=0.08, edgecolor='none', zorder=1))

    # 3D detection boxes (virtual vessels use dashed lines)
    for d in dets3:
        box = d.get('box7d', [])
        if len(box) != 7:
            continue
        x, y, z, dx, dy, dz, yaw = map(float, box)
        poly = rot_corners_xy(x, y, dx, dy, yaw)
        lbl = d.get('label', '')
        is_virtual_ship = (lbl in cfg['labels']['ships']) and _is_virtual_ship_det(d)

        if lbl in cfg['labels']['ships']:
            ec = col_ship
        elif lbl == cfg['labels']['lock_gate']:
            ec = col_gate
        else:
            ec = col_other

        patch = Polygon(
            poly,
            closed=True,
            fill=False,
            edgecolor=ec,
            linewidth=2.0,
            linestyle='--' if is_virtual_ship else '-'
        )
        ax.add_patch(patch)

        text_lbl = f"{lbl} (V)" if is_virtual_ship else lbl
        ax.text(x, y, text_lbl, fontsize=10, color=ec)

    # Boundaries / coordinates
    ax.vlines([x_min_ch, x_max_ch], 0, 102.4, colors='gray', linestyles=':', linewidth=1.5)
    ax.set_xlim(0, 102.4)
    ax.set_ylim(0, 102.4)
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.4)

    # 2D inset in the upper-right corner
    im2d = draw_2d_with_boxes(img_path, det2d, cfg)
    if im2d is not None:
        extent = [75, 102.4, 75, 102.4]
        ax.imshow(im2d, extent=extent, origin='upper', zorder=20)
        ax.add_patch(Rectangle((75, 75), 27.4, 27.4, fill=False, edgecolor='black', linewidth=1.2, zorder=21))

    # Multi-line information panel in the upper-left corner
    color = colors_status.get(str(status).split()[0], 'black')
    info_text = (
        f"Status: {status}\n"
        f"Reason: {reason}\n"
        f"Ships: {num_ships}\n"
        f"Gate: {fi.gate_state}\n"
        f"Segment: {fi.seg_id}"
        + (f"\nEvents: {events_text}" if events_text else "")
    )
    ax.text(0.02, 0.97, info_text, ha='left', va='top', transform=ax.transAxes,
            fontsize=11, color=color, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    ax.set_title(f"Loitering Warning | frame {fi.frame_id}", fontsize=12, color=color)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)

# --------------------------- main flow ---------------------------

def run(fused_json_path, config_path, debug_events=False, viz=False):
    cfg = load_config(config_path)

    out_root = cfg.get('output', {}).get('root', './outputs')
    out_dir = os.path.join(out_root, 'loitering_warning')
    os.makedirs(out_dir, exist_ok=True)
    vis_dir = os.path.join(out_dir, 'vis')
    if viz:
        os.makedirs(vis_dir, exist_ok=True)


    # === Aggregators for combined outputs across all segments ===
    # Pre-KF combined outputs
    combined_pre_per_frame_rows = []
    combined_pre_traj_rows = []
    combined_pre_entry_events = []
    combined_pre_exit_events = []
    combined_pre_exit_behaviors = []
    combined_pre_timeline = []

    combined_per_frame_rows = []  # list of dict rows
    combined_traj_rows = []       # list of tuples (segment, tid, frame_idx, timestamp_iso, x_str, y_str)
    combined_entry_events = []
    combined_exit_events = []
    combined_exit_behaviors = []
    combined_timeline = []        # list of dicts
    # Read fused results
    with open(fused_json_path, 'r', encoding='utf-8') as f:
        fused_data = json.load(f)

    # Build frames_by_segment (used only for gate state / tracking)
    frames_by_segment = defaultdict(list)
    order_counter = 0
    for fid, info in fused_data.items():
        seg = int(info.get('segment', 0))
        ts_str = info.get('timestamp')
        ts = datetime.strptime(ts_str, '%Y_%m_%d_%H_%M_%S_%f')
        det3d = info.get('det3d', [])

        # Collect 3D vessels and gate width (keep yaw unchanged to remain consistent with make_loitering_labels)
        ships, gate_widths = [], []
        for d in det3d:
            label = d.get('label')
            box = d.get('box7d', [])
            if len(box) < 7:
                continue
            x, y, z, l, w, h, yaw = box
            if label == cfg['labels']['lock_gate']:
                gate_widths.append(w)
                #print("gate box7d:", box)
            if label in cfg['labels']['ships']:
                ships.append(Detection(label, (h, w, l), (x, y), float(yaw)))

        gate_width = float(np.median(gate_widths)) if gate_widths else None

        fi = FrameInfo(
            frame_id=fid,
            timestamp=ts,
            timestamp_str=ts_str,
            seg_id=seg,
            gate_width_m=gate_width,
            gate_present=(gate_width is not None),
            gate_state='UNKNOWN',  # Initialize as UNKNOWN, then override according to gate_state.mode
            ships=ships
        )
        # Preserve the original traversal order of fused_2d3d.json
        fi.order_idx = order_counter
        order_counter += 1
        frames_by_segment[seg].append(fi)

    # For compatibility, run the original algorithm once first to keep other dependencies stable, then override gate_state according to gate_state.mode
    compute_gate_states_for_segments(frames_by_segment, cfg)

    # Process segment by segment
    seg_order = sorted(frames_by_segment.keys(), key=lambda s: (min(fr.order_idx for fr in frames_by_segment[s]), s))

    for seg in tqdm(seg_order, desc="Processing segments"):
        # Follow the original JSON order
        frames = list(frames_by_segment[seg])

        # Select the gate_state inference mode according to the configuration: width / 2d
        apply_gate_state_by_mode(frames, fused_data, cfg)

        # Obtain events (entry/exit/behaviors) and the timeline from tracking (get the original version first here)
        (per_frame_ship_tids,
         long_traj,
         entry_events,
         exit_events,
         exit_behaviors,
         _dummy,
         inside_counts,
         per_frame_bboxinfo) = track_segment(frames, cfg)

        
        # ===== First keep an unfiltered copy (make as few assumptions about the internal structure as possible) =====
        def _deepcopy_traj(obj):
            if isinstance(obj, dict):
                return {k: (v[:] if isinstance(v, list) else list(v) if isinstance(v, tuple) else v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [ (vi[:] if isinstance(vi, list) else list(vi) if isinstance(vi, tuple) else vi) for vi in obj ]
            return obj
        orig_long_traj = _deepcopy_traj(long_traj)
        orig_inside_counts = inside_counts[:] if isinstance(inside_counts, list) else list(inside_counts)

        # ===== Build the pre-KF timeline using the original inside_counts =====
        timeline_pre = build_loiter_timeline(
            frames, per_frame_ship_tids, exit_events, exit_behaviors, orig_inside_counts, cfg
        )

        # ===== Apply KF smoothing (trajectories + vessel counts inside the chamber) =====
        long_traj_kf = apply_kalman_to_tracks(long_traj)
        inside_counts_kf = kalman_smooth_counts(list(inside_counts))

        # ===== Historical heading despiking: only correct single-frame 180° flips like frame 2561 =====
        heading_map_raw = build_heading_sequences(per_frame_ship_tids, fused_data, cfg)
        heading_fix_map = {}
        for tid, seq in heading_map_raw.items():
            _seq_smooth, fixed_idx = smooth_heading_by_history(
                seq,
                stable_len=int(cfg.get("kf", {}).get("heading_history_frames", 3)),
                stable_tol_deg=float(cfg.get("kf", {}).get("heading_stable_tol_deg", 20.0)),
                jump_deg=float(cfg.get("kf", {}).get("heading_jump_deg", 120.0)),
            )
            heading_fix_map[int(tid)] = fixed_idx



        # Only in the KF branch, remove original exit_behavior events that fall on heading-spike frames corrected above
        exit_behaviors_kf = filter_exit_behaviors_by_heading_spikes(
            per_frame_ship_tids,
            exit_behaviors,
            heading_fix_map,
            timeline_pre,
            prev_status_frames=int(cfg.get("kf", {}).get("prev_status_frames", 3)),
        )
        



        # ===== Post-KF timeline (used as the original set of three outputs) =====
        timeline = build_loiter_timeline(
            frames, per_frame_ship_tids, exit_events, exit_behaviors_kf, inside_counts_kf, cfg
        )

        # ====== Export six files: three pre-KF files + three post-KF files (with the original names) ======
        # 1) Post-KF (keep the original filenames)
        # Collect KF combined outputs
        #combined_traj_rows.extend(_collect_tracks_rows(seg, per_frame_ship_tids, long_traj_kf))
        combined_traj_rows.extend(_traj_to_rows(seg, per_frame_ship_tids, long_traj_kf))
        combined_per_frame_rows.extend(
            _collect_labels_per_frame_rows(seg, per_frame_ship_tids, timeline,
                                           entry_events, exit_events, exit_behaviors_kf))
        tl, ee, xe, xb = _collect_events_and_timeline(
            seg, timeline, entry_events, exit_events, exit_behaviors_kf)
        combined_timeline.extend(tl)
        combined_entry_events.extend(ee)
        combined_exit_events.extend(xe)
        combined_exit_behaviors.extend(xb)

        # Collect Pre-KF combined outputs
        tl_pre, ee_pre, xe_pre, xb_pre = _collect_events_and_timeline(
            seg, timeline_pre, entry_events, exit_events, exit_behaviors)
        combined_pre_timeline.extend(tl_pre)
        combined_pre_entry_events.extend(ee_pre)
        combined_pre_exit_events.extend(xe_pre)
        combined_pre_exit_behaviors.extend(xb_pre)
        
        combined_pre_traj_rows.extend(_traj_to_rows(seg, per_frame_ship_tids, orig_long_traj))
        
        combined_pre_per_frame_rows.extend(
            _collect_labels_per_frame_rows(seg, per_frame_ship_tids, timeline_pre,
                                           entry_events, exit_events, exit_behaviors))




        # 2) Pre-KF (add three new files)
        # (pre-KF exports disabled in combined mode)
        # ====== Plotting (keep your original logic and display the post-KF timeline) ======
        if viz:
                for idx, (fi, tids) in enumerate(per_frame_ship_tids):
                    _fi2, status, reason, _extra = timeline[idx]
                    frame_info = fused_data.get(fi.frame_id, {})
                    pts_path = frame_info.get('points_path', '')
                    img_path = frame_info.get('image_path', '')
                    dets3 = frame_info.get('det3d', [])
                    dets2 = frame_info.get('det2d', [])
                    pts = load_points(pts_path) if pts_path else None

                    events_text = ""
                    if debug_events:
                        t_iso = fi.timestamp.isoformat()
                        evs = []
                        e_tids = [tid for (t, tid) in entry_events if t == t_iso]
                        x_tids = [tid for (t, tid) in exit_events if t == t_iso]
                        xb_tids = [tid for (t, tid) in exit_behaviors if t == t_iso]
                        if e_tids:
                            evs.append("entry:" + ",".join(f"T{tid}" for tid in e_tids))
                        if x_tids:
                            evs.append("exit:" + ",".join(f"T{tid}" for tid in x_tids))
                        if xb_tids:
                            evs.append("exit_behavior:" + ",".join(f"T{tid}" for tid in xb_tids))
                        events_text = " | ".join(evs)
        
                    out_img = os.path.join(vis_dir, f"{fi.timestamp_str}.png")
                    draw_loitering_planview(
                        fi, pts, dets3, dets2, img_path, status, reason, len(tids), cfg,
                        out_img, events_text=events_text
                    )

    
    # === Write ONE merged txt (TSV) ===

    # 1) Build the frame index (within each segment, sort by timestamp to obtain frame_idx)
    def _build_frame_idx_map(per_frame_rows):
        by_seg = defaultdict(list)
        for r in per_frame_rows:
            seg = int(r["segment"])
            by_seg[seg].append(r["timestamp"])
        idx_map = {}
        for seg, ts_list in by_seg.items():
            # Timestamps within one segment are generally increasing; sorting is used here to ensure stability
            for i, ts in enumerate(sorted(set(ts_list))):
                idx_map[(seg, ts)] = i
        return idx_map

    # 2) Frame-row mapping (KF / pre-KF)
    kf_frame_map = {(int(r["segment"]), r["timestamp"]): r for r in combined_per_frame_rows}
    pre_frame_map = {(int(r["segment"]), r["timestamp"]): r for r in combined_pre_per_frame_rows}

    # 3) Trajectory-point mapping (KF / pre-KF)
    def _build_pos_map(traj_rows):
        """
        Compatible with two formats:
          A) (seg, tid, frame_id, ts_iso, x, y)   # long_traj returned by track_segment
          B) (seg, tid, k, ts_iso, xs, ys)        # output from the old _collect_tracks_rows
        Returns: pos[(seg, ts_iso, tid)] = (x, y)
        """
        pos = {}
        for row in traj_rows:
            if not isinstance(row, (list, tuple)) or len(row) < 6:
                continue
            seg_v, tid_v, third, ts, x_raw, y_raw = row[:6]
            key = (int(seg_v), str(ts), int(tid_v))

            try:
                x = None if (x_raw is None or x_raw == "") else float(x_raw)
            except Exception:
                x = None
            try:
                y = None if (y_raw is None or y_raw == "") else float(y_raw)
            except Exception:
                y = None

            pos[key] = (x, y)
        return pos

    pos_kf  = _build_pos_map(combined_traj_rows)
    pos_pre = _build_pos_map(combined_pre_traj_rows)

    # 4) Event set (keep only one copy: pre-KF and post-KF share the same event set, so do not output duplicates)
    entry_set = {(int(e["segment"]), e["time"], int(e["tid"])) for e in combined_entry_events}
    exit_set  = {(int(e["segment"]), e["time"], int(e["tid"])) for e in combined_exit_events}
    xb_set    = {(int(e["segment"]), e["time"], int(e["tid"])) for e in combined_exit_behaviors}

    # 5) Use the KF per_frame rows as the full frame set (usually the most complete)
    frame_keys = sorted(kf_frame_map.keys(), key=lambda k: (k[1], k[0]))
    frame_idx_map = _build_frame_idx_map(combined_per_frame_rows)

    traj_csv  = os.path.join(out_dir, "pred_traj_xy.csv")
    frame_csv = os.path.join(out_dir, "pred_frame.csv")

    traj_cols = ["timestamp", "segment", "frame_idx", "frame_id", "tid",
                 "x_kf", "y_kf", "x_pre", "y_pre"]

    frame_cols = ["timestamp", "segment", "frame_idx", "frame_id",
                  "gate_present", "gate_width_m", "gate_state",
                  "num_ships", "ship_tids",
                  "status_kf", "reason_kf", "status_pre", "reason_pre",
                  "events", "is_entry", "is_exit", "is_exit_behavior"]


    def _safe(v):
        return "" if v is None else v

    # ====== Aggregate events by frame: is_entry/is_exit/is_exit_behavior indicate whether the event occurred in that frame ======
    entry_by_frame = defaultdict(set)
    exit_by_frame  = defaultdict(set)
    xb_by_frame    = defaultdict(set)

    for e in combined_entry_events:
        entry_by_frame[(int(e["segment"]), e["time"])].add(int(e["tid"]))
    for e in combined_exit_events:
        exit_by_frame[(int(e["segment"]), e["time"])].add(int(e["tid"]))
    for e in combined_exit_behaviors:
        xb_by_frame[(int(e["segment"]), e["time"])].add(int(e["tid"]))

    with open(traj_csv, "w", encoding="utf-8", newline="") as f_traj, \
         open(frame_csv, "w", encoding="utf-8", newline="") as f_frame:

        w_traj = csv.writer(f_traj)
        w_frame = csv.writer(f_frame)

        w_traj.writerow(traj_cols)
        w_frame.writerow(frame_cols)

        for (seg, ts) in frame_keys:
            rkf = kf_frame_map.get((seg, ts), {})
            rpre = pre_frame_map.get((seg, ts), {})

            frame_id  = rkf.get("frame_id", rpre.get("frame_id", ""))
            frame_idx = frame_idx_map.get((seg, ts), "")

            gate_present = rkf.get("gate_present", rpre.get("gate_present", ""))
            gate_width_m = rkf.get("gate_width_m", rpre.get("gate_width_m", ""))
            gate_state   = rkf.get("gate_state",   rpre.get("gate_state",   ""))

            num_ships = rkf.get("num_ships", rpre.get("num_ships", ""))
            ship_tids = rkf.get("ship_tids", rpre.get("ship_tids", ""))

            status_kf  = rkf.get("status", "")
            reason_kf  = rkf.get("reason", "")
            status_pre = rpre.get("status", "")
            reason_pre = rpre.get("reason", "")

            events = rkf.get("events", rpre.get("events", "—"))

            # --- Parse the tid list of the current frame ---
            tids = []
            if isinstance(ship_tids, str) and ship_tids.strip():
                tids = [int(x) for x in ship_tids.split(";") if x.strip().isdigit()]

            # ====== Output pred_traj_xy.csv for each vessel in each frame ======
            for tid in tids:
                xk = yk = xp = yp = ""

                pk = pos_kf.get((seg, ts, tid))
                if pk is not None:
                    xk = "" if pk[0] is None else f"{pk[0]:.3f}"
                    yk = "" if pk[1] is None else f"{pk[1]:.3f}"

                pp = pos_pre.get((seg, ts, tid))
                if pp is not None:
                    xp = "" if pp[0] is None else f"{pp[0]:.3f}"
                    yp = "" if pp[1] is None else f"{pp[1]:.3f}"

                w_traj.writerow([
                    ts, seg, frame_idx, frame_id, tid,
                    xk, yk, xp, yp
                ])

            # ====== Output pred_frame.csv frame by frame (aggregate events into 0/1 based on whether they occurred in that frame) ======
            ef = (seg, ts)
            is_entry = 1 if len(entry_by_frame.get(ef, set())) > 0 else 0
            is_exit  = 1 if len(exit_by_frame.get(ef, set()))  > 0 else 0
            is_xb    = 1 if len(xb_by_frame.get(ef, set()))    > 0 else 0

            w_frame.writerow([
                ts, seg, frame_idx, frame_id,
                gate_present, gate_width_m, gate_state,
                num_ships, ship_tids,
                status_kf, reason_kf, status_pre, reason_pre,
                events, is_entry, is_exit, is_xb
            ])

    print("[OK] Traj CSV saved to:", traj_csv)
    print("[OK] Frame CSV saved to:", frame_csv)

def main():
    import argparse
    ap = argparse.ArgumentParser(description='Loitering ship warning visualization v3.1 (2D gate trends + Kalman)')
    ap.add_argument('--fused-json', default='outputs/fused_2d3d/fused_2d3d_virtual.json')
    ap.add_argument('--cfg', default='config.yaml')
    ap.add_argument('--debug-events', action='store_true', help='The left upper information bar displays the "entry/exit/exit_behavior" events of this frame.')
    ap.add_argument('--viz', action='store_true', help='Enable visual rendering (default is off)')
    args = ap.parse_args()
    run(args.fused_json, args.cfg, debug_events=args.debug_events, viz=args.viz)


if __name__ == '__main__':
    main()
