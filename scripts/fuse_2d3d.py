# -*- coding: utf-8 -*-
"""
scripts/fuse_2d3d.py

Based on the user's uploaded fuse_2d3d.py, this version keeps the original
fusion outputs and additionally generates:
  1) fused_2d3d_virtual.json  - merged 2D/3D JSON with virtual 3D ship boxes added
  2) virtual_ship_frames.txt  - log of frames where virtual ships were added

Virtual ship triggering rule:
  - Initial trigger only when, within the same segment:
      current real-3D ship count < previous frame real-3D ship count
    and current 2D ship count > current real-3D ship count
  - After a track becomes ghost-active, it may continue to be filled for a few
    frames (max_gap_frames), as long as the current frame still has 2D evidence
    supporting more ships than the current real 3D detections.

Virtual 3D box generation:
  - x, y: extrapolated from the last two boxes of the track
  - z, dx, dy, dz, yaw: median over recent history
  - label / label_id: inherited from the source track
  - label stays as the original ship class so downstream code still recognizes it
    as a ship; extra flags are added to identify the detection as virtual.
"""

import argparse
import copy
import json
from pathlib import Path
from math import ceil
from typing import Dict, Any, List, Optional, Tuple, Callable
from collections import OrderedDict

import numpy as np
from tqdm import tqdm


def load_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        data: Dict[str, Any] = {}
        keys_stack: List[str] = []

        def set_nested(d: Dict[str, Any], ks: List[str], val):
            x = d
            for k in ks[:-1]:
                if k not in x or not isinstance(x[k], dict):
                    x[k] = {}
                x = x[k]
            x[ks[-1]] = val

        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                if not raw.strip() or raw.lstrip().startswith("#"):
                    continue
                s = raw.rstrip("\n")
                indent = len(s) - len(s.lstrip(" "))
                level = indent // 2
                s = s.strip()
                if ":" in s and not s.startswith("- "):
                    k, v = s.split(":", 1)
                    k = k.strip()
                    v = v.strip()
                    keys_stack = keys_stack[:level]
                    if v == "":
                        keys_stack.append(k)
                        set_nested(data, keys_stack, {})
                    else:
                        if v.startswith("[") and v.endswith("]"):
                            arr = [x.strip().strip('"').strip("'") for x in v[1:-1].split(",") if x.strip()]
                            set_nested(data, keys_stack + [k], arr)
                        else:
                            v = v.strip().strip('"').strip("'")
                            set_nested(data, keys_stack + [k], v)
                elif s.startswith("- "):
                    item = s[2:].strip().strip('"').strip("'")
                    x = data
                    for kk in keys_stack:
                        x = x.setdefault(kk, {})
                    if "ships" in x and isinstance(x["ships"], list):
                        x["ships"].append(item)
        return data


def load_jsons_as_models(json_dir: Optional[str]) -> List[Dict[str, Any]]:
    if not json_dir:
        return []
    p = Path(json_dir)
    if not p.exists() or not p.is_dir():
        return []
    paths = sorted(str(x) for x in p.glob("*.json"))
    models = []
    for fp in paths:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                models.append(json.load(f))
        except Exception as e:
            print(f"[WARN] Failed to read {fp}: {e}")
    return models


def union_frame_ids(models: List[Dict[str, Any]]) -> List[str]:
    fids = set()
    for m in models:
        fids.update(m.keys())
    return sorted(fids)


EPS = 1e-9


def _box7d_to_bev_corners(box7d: np.ndarray) -> np.ndarray:
    x, y, _, dx, dy, _, yaw = box7d[:7]
    dx = float(dx)
    dy = float(dy)
    if dx <= 0 or dy <= 0:
        return np.array([[x, y]] * 4, dtype=np.float32)
    hx = dx / 2.0
    hy = dy / 2.0
    local = np.array([[hx, hy], [hx, -hy], [-hx, -hy], [-hx, hy]], dtype=np.float32)
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    world = (local @ R.T) + np.array([x, y], dtype=np.float32)
    return world


def _poly_area(poly: np.ndarray) -> float:
    if poly.shape[0] < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _ensure_ccw(poly: np.ndarray) -> np.ndarray:
    if poly.shape[0] < 3:
        return poly
    x = poly[:, 0]
    y = poly[:, 1]
    signed = 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    if signed < 0:
        return poly[::-1].copy()
    return poly


def _inside(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> bool:
    return ((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])) >= -EPS


def _compute_intersection(a: np.ndarray, b: np.ndarray, p: np.ndarray, q: np.ndarray) -> np.ndarray:
    r = b - a
    s = q - p
    rxs = r[0] * s[1] - r[1] * s[0]
    if abs(rxs) < EPS:
        return a.copy()
    t = ((p[0] - a[0]) * s[1] - (p[1] - a[1]) * s[0]) / rxs
    return a + t * r


def _sutherland_hodgman(subject: np.ndarray, clipper: np.ndarray) -> np.ndarray:
    subject = np.asarray(subject, dtype=np.float32).reshape(-1, 2)
    clipper = np.asarray(clipper, dtype=np.float32).reshape(-1, 2)
    if subject.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float32)
    output = subject.copy()
    cp = clipper
    for i in range(cp.shape[0]):
        a = cp[i]
        b = cp[(i + 1) % cp.shape[0]]
        input_list = output
        if input_list.shape[0] == 0:
            return np.empty((0, 2), dtype=np.float32)
        new_out = []
        S = input_list[-1]
        for E in input_list:
            if _inside(E, a, b):
                if not _inside(S, a, b):
                    inter = _compute_intersection(S, E, a, b)
                    new_out.append(inter)
                new_out.append(E)
            elif _inside(S, a, b):
                inter = _compute_intersection(S, E, a, b)
                new_out.append(inter)
            S = E
        output = np.asarray(new_out, dtype=np.float32)
    if output.ndim == 1:
        output = output.reshape(-1, 2)
    return output


def _iou_poly(poly_a: np.ndarray, poly_b: np.ndarray) -> float:
    poly_a = np.asarray(poly_a, dtype=np.float32).reshape(-1, 2)
    poly_b = np.asarray(poly_b, dtype=np.float32).reshape(-1, 2)
    if poly_a.shape[0] < 3 or poly_b.shape[0] < 3:
        return 0.0
    A = _poly_area(poly_a)
    B = _poly_area(poly_b)
    if A <= EPS or B <= EPS:
        return 0.0
    inter_poly = _sutherland_hodgman(poly_a, poly_b)
    I = _poly_area(inter_poly) if inter_poly.shape[0] >= 3 else 0.0
    U = A + B - I
    if U <= EPS:
        return 0.0
    return float(I / U)


def bev_iou(a: np.ndarray, b: np.ndarray) -> float:
    return _iou_poly(_ensure_ccw(_box7d_to_bev_corners(a)), _ensure_ccw(_box7d_to_bev_corners(b)))


def iou2d_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def _extract_xyxy(det: Dict[str, Any]) -> Optional[List[float]]:
    for k in ("bbox_xyxy", "xyxy", "bbox", "box"):
        if k in det:
            v = det[k]
            if not isinstance(v, (list, tuple)) or len(v) < 4:
                return None
            x1, y1, x2, y2 = float(v[0]), float(v[1]), float(v[2]), float(v[3])
            if x2 <= x1 or y2 <= y1:
                x2 = x1 + float(v[2])
                y2 = y1 + float(v[3])
            return [x1, y1, x2, y2]
    for k in ("bbox_xywh", "xywh"):
        if k in det:
            v = det[k]
            if not isinstance(v, (list, tuple)) or len(v) < 4:
                return None
            x, y, w, h = float(v[0]), float(v[1]), float(v[2]), float(v[3])
            return [x, y, x + w, y + h]
    return None


def cluster_and_vote_generic(
    proposals: List[Tuple[np.ndarray, Tuple[float, str, int, int]]],
    iou_func: Callable[[np.ndarray, np.ndarray], float],
    iou_thr: float,
    vote_need: int,
) -> List[Dict[str, Any]]:
    if not proposals:
        return []
    boxes = np.stack([p[0] for p in proposals], axis=0)
    metas = [p[1] for p in proposals]
    order = np.argsort([-m[0] for m in metas])
    used = np.zeros((len(order),), dtype=bool)
    kept = []
    for oi in range(len(order)):
        if used[oi]:
            continue
        seed_idx = order[oi]
        seed_box = boxes[seed_idx]
        cluster_idx = [oi]
        remain_mask = ~used
        remain_mask[: oi + 1] = False
        remain_inds = np.where(remain_mask)[0]
        if remain_inds.size > 0:
            hits = []
            for r in remain_inds:
                if iou_func(seed_box, boxes[order[r]]) >= iou_thr:
                    hits.append(r)
            cluster_idx.extend(hits)
        support_models = set(metas[order[ci]][3] for ci in cluster_idx)
        if len(support_models) >= vote_need:
            best_ci = max(cluster_idx, key=lambda ci: metas[order[ci]][0])
            best_score, best_label, best_label_id, _ = metas[order[best_ci]]
            best_box = boxes[order[best_ci]]
            kept.append({
                "box": best_box.tolist(),
                "score": float(best_score),
                "label_id": int(best_label_id),
                "label": best_label,
            })
        for ci in cluster_idx:
            used[ci] = True
    return kept


def fuse_3d_models(models: List[Dict[str, Any]], lock_gate_label: str, ship_labels: List[str], iou_thr: float, vote_ratio: float) -> Dict[str, Any]:
    num_models = len(models)
    if num_models == 0:
        return {}
    vote_need = ceil(num_models * vote_ratio)
    ship_set = set(ship_labels)
    fids = union_frame_ids(models)
    fused: Dict[str, Any] = {}
    for fid in tqdm(fids, desc="[FUSE-3D]"):
        per_model = []
        points_path = None
        for m in models:
            entry = m.get(fid, {})
            if points_path is None:
                points_path = entry.get("points_path")
            per_model.append(entry.get("detections", []))
        fused_dets = []
        props = []
        for mi, dets in enumerate(per_model):
            for d in dets:
                if d.get("label") == lock_gate_label:
                    box7d = np.array(d["box7d"], dtype=np.float32)
                    props.append((box7d, (float(d.get("score", 0.0)), d["label"], int(d.get("label_id", -1)), mi)))
        kept = cluster_and_vote_generic(props, iou_func=bev_iou, iou_thr=iou_thr, vote_need=vote_need)
        for k in kept:
            fused_dets.append({
                "box7d": [float(t) for t in k["box"]],
                "score": k["score"],
                "label_id": k["label_id"],
                "label": k["label"],
            })
        props = []
        for mi, dets in enumerate(per_model):
            for d in dets:
                lab = str(d.get("label", ""))
                if lab in ship_set:
                    box7d = np.array(d["box7d"], dtype=np.float32)
                    props.append((box7d, (float(d.get("score", 0.0)), lab, int(d.get("label_id", -1)), mi)))
        kept = cluster_and_vote_generic(props, iou_func=bev_iou, iou_thr=iou_thr, vote_need=vote_need)
        for k in kept:
            fused_dets.append({
                "box7d": [float(t) for t in k["box"]],
                "score": k["score"],
                "label_id": k["label_id"],
                "label": k["label"],
            })
        fused[fid] = {
            "points_path": points_path,
            "detections": fused_dets,
            "lock_gate_present": any(d["label"] == lock_gate_label for d in fused_dets),
            "ship_present": any(d["label"] in ship_set for d in fused_dets),
        }
    return fused


def fuse_2d_models(models: List[Dict[str, Any]], iou_thr: float, vote_ratio: float) -> Dict[str, Any]:
    num_models = len(models)
    if num_models == 0:
        return {}
    vote_need = ceil(num_models * vote_ratio)
    fids = union_frame_ids(models)
    fused: Dict[str, Any] = {}
    for fid in tqdm(fids, desc="[FUSE-2D]"):
        per_model = []
        image_path = None
        for m in models:
            entry = m.get(fid, {})
            if image_path is None:
                image_path = entry.get("image_path")
            per_model.append(entry.get("detections", []))
        all_labels = set()
        for dets in per_model:
            for d in dets:
                all_labels.add(str(d.get("label", "")))
        fused_dets = []
        for lab in sorted(all_labels):
            props: List[Tuple[np.ndarray, Tuple[float, str, int, int]]] = []
            for mi, dets in enumerate(per_model):
                for d in dets:
                    if str(d.get("label", "")) != lab:
                        continue
                    xyxy = _extract_xyxy(d)
                    if xyxy is None:
                        continue
                    props.append((np.array(xyxy, dtype=np.float32), (float(d.get("score", 0.0)), lab, int(d.get("label_id", -1)), mi)))
            kept = cluster_and_vote_generic(proposals=props, iou_func=iou2d_xyxy, iou_thr=iou_thr, vote_need=vote_need)
            for k in kept:
                fused_dets.append({
                    "bbox": [float(t) for t in k["box"]],
                    "score": k["score"],
                    "label_id": k["label_id"],
                    "label": k["label"],
                })
        fused[fid] = {"image_path": image_path, "detections": fused_dets}
    return fused


def merge_2d3d_annotate_sort(fused2d: Dict[str, Any], fused3d: Dict[str, Any], add_info_path: Optional[str] = None) -> OrderedDict:
    ts_map: Dict[str, Tuple[str, Optional[float], Optional[int]]] = {}
    if add_info_path:
        p = Path(add_info_path)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                for ln in f:
                    s = ln.strip()
                    if not s:
                        continue
                    parts = s.split()
                    if len(parts) < 4:
                        if len(parts) == 3:
                            fid, ts, seg = parts[0], parts[1], parts[2]
                            try:
                                ts_map[fid] = (ts, None, int(seg))
                            except Exception:
                                ts_map[fid] = (ts, None, None)
                        continue
                    fid, ts, wl, seg = parts[0], parts[1], parts[2], parts[3]
                    try:
                        seg_i = int(seg)
                    except Exception:
                        seg_i = None
                    try:
                        wl_f = float(wl)
                    except Exception:
                        wl_f = None
                    ts_map[fid] = (ts, wl_f, seg_i)
        else:
            print(f"[WARN] add_info file not found: {add_info_path}")
    fids = sorted(set(fused2d.keys()) | set(fused3d.keys()))
    merged_unsorted: Dict[str, Any] = {}
    for fid in fids:
        e2 = fused2d.get(fid, {})
        e3 = fused3d.get(fid, {})
        ts, _wl, seg = ts_map.get(fid, (None, None, None))
        merged_unsorted[fid] = {
            "frame_id": fid,
            "timestamp": ts,
            "segment": seg,
            "image_path": e2.get("image_path"),
            "points_path": e3.get("points_path"),
            "det2d": e2.get("detections", []),
            "det3d": e3.get("detections", []),
            "lock_gate_present": e3.get("lock_gate_present", False),
            "ship_present": e3.get("ship_present", False),
        }

    def ts_key(fid_ts_item):
        _fid, item = fid_ts_item
        ts = item.get("timestamp")
        return (ts is None, "" if ts is None else ts)

    ordered = OrderedDict(sorted(merged_unsorted.items(), key=ts_key))
    return ordered


# ========================= Virtual ship augmentation =========================

def _get_virtual_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    rep = cfg.get("virtual_ship_repair", {}) or {}
    fallback = cfg.get("track_repair", {}) or {}

    def _pick(key: str, default):
        if key in rep:
            return rep.get(key)
        if key in fallback:
            return fallback.get(key)
        return default

    return {
        "enabled": bool(_pick("enabled", False)),
        "max_gap_frames": int(_pick("max_gap_frames", 2)),
        "min_prev_hist_frames": int(_pick("min_prev_hist_frames", 2)),
        "max_step_m": float(_pick("max_step_m", 10.0)),
        "stitch_dist_m": float(_pick("stitch_dist_m", 10.0)),  # reserved
        "max_match_dist_m": float(_pick("max_match_dist_m", 18.0)),
        "history_window": int(_pick("history_window", 3)),
        "virtual_score_base": float(_pick("virtual_score_base", 0.2)),
        "debug": bool(_pick("debug", False)),
    }


def _ship_dets_only(dets: List[Dict[str, Any]], ship_set: set, include_virtual: bool = True) -> List[Dict[str, Any]]:
    out = []
    for d in dets or []:
        if str(d.get("label", "")) not in ship_set:
            continue
        if not include_virtual and bool(d.get("is_virtual", False)):
            continue
        if "box7d" not in d or not isinstance(d["box7d"], (list, tuple)) or len(d["box7d"]) < 7:
            continue
        out.append(d)
    return out


def _count_ship_2d(det2d: List[Dict[str, Any]], ship_set: set) -> int:
    c = 0
    for d in det2d or []:
        if str(d.get("label", "")) in ship_set:
            c += 1
    return c


def _box_center(box7d: List[float]) -> Tuple[float, float]:
    return float(box7d[0]), float(box7d[1])


def _clip_vec(vx: float, vy: float, max_step_m: float) -> Tuple[float, float]:
    step = float(np.hypot(vx, vy))
    if step <= 1e-9 or max_step_m <= 0 or step <= max_step_m:
        return vx, vy
    scale = max_step_m / step
    return vx * scale, vy * scale


def _median_box_from_hist(hist: List[Dict[str, Any]], window: int) -> List[float]:
    tail = hist[-max(1, int(window)):]
    arr = np.array([h["box7d"][:7] for h in tail], dtype=np.float32)
    med = np.median(arr, axis=0)
    return [float(x) for x in med.tolist()]


def _make_virtual_det(track: Dict[str, Any], cur_frame_id: str, cur_timestamp: Optional[str], cur_segment: Optional[int], cfg_rep: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    hist = track.get("hist", [])
    if len(hist) < max(1, int(cfg_rep["min_prev_hist_frames"])):
        return None

    last_box = list(hist[-1]["box7d"][:7])
    prev_box = list(hist[-2]["box7d"][:7]) if len(hist) >= 2 else list(last_box)
    vx = float(last_box[0] - prev_box[0])
    vy = float(last_box[1] - prev_box[1])
    vx, vy = _clip_vec(vx, vy, float(cfg_rep["max_step_m"]))

    med = _median_box_from_hist(hist, int(cfg_rep["history_window"]))
    x = float(last_box[0] + vx)
    y = float(last_box[1] + vy)
    z = float(med[2])
    dx = max(1e-3, float(med[3]))
    dy = max(1e-3, float(med[4]))
    dz = max(1e-3, float(med[5]))
    yaw = float(med[6])

    gap = int(track.get("misses", 0)) + 1
    score = max(0.01, float(cfg_rep["virtual_score_base"]) / float(gap))

    return {
        "box7d": [x, y, z, dx, dy, dz, yaw],
        "score": score,
        "label_id": int(track.get("label_id", -1)),
        "label": str(track.get("label", "Unknown_ship")),
        "is_virtual": True,
        "virtual_tag": "VIRTUAL_SHIP_3D",
        "virtual_source": "temporal_extrapolation",
        "virtual_track_id": int(track["track_id"]),
        "virtual_gap": gap,
        "virtual_from_frame": str(track.get("last_frame_id", "")),
        "virtual_target_frame": str(cur_frame_id),
        "virtual_target_timestamp": cur_timestamp,
        "virtual_target_segment": cur_segment,
    }


def _distance_track_to_det(track: Dict[str, Any], det: Dict[str, Any]) -> float:
    tx, ty = _box_center(track["hist"][-1]["box7d"])
    dx, dy = _box_center(det["box7d"])
    return float(np.hypot(tx - dx, ty - dy))


def _associate_tracks_greedy(active_tracks: List[Dict[str, Any]], curr_real_dets: List[Dict[str, Any]], max_match_dist_m: float) -> Tuple[Dict[int, int], List[int], List[int]]:
    pairs: List[Tuple[float, int, int]] = []
    for ti, tr in enumerate(active_tracks):
        tr_label = str(tr.get("label", ""))
        for di, det in enumerate(curr_real_dets):
            if str(det.get("label", "")) != tr_label:
                continue
            dist = _distance_track_to_det(tr, det)
            if dist <= max_match_dist_m:
                pairs.append((dist, ti, di))
    pairs.sort(key=lambda x: x[0])

    matched_t = set()
    matched_d = set()
    match_map: Dict[int, int] = {}
    for dist, ti, di in pairs:
        if ti in matched_t or di in matched_d:
            continue
        matched_t.add(ti)
        matched_d.add(di)
        match_map[ti] = di

    unmatched_tracks = [i for i in range(len(active_tracks)) if i not in matched_t]
    unmatched_dets = [i for i in range(len(curr_real_dets)) if i not in matched_d]
    return match_map, unmatched_tracks, unmatched_dets


def augment_merged_with_virtual_ships(merged: OrderedDict, cfg: Dict[str, Any], ship_labels: List[str]) -> Tuple[OrderedDict, List[Dict[str, Any]]]:
    cfg_rep = _get_virtual_cfg(cfg)
    if not cfg_rep["enabled"]:
        return copy.deepcopy(merged), []

    ship_set = set(ship_labels)
    debug = bool(cfg_rep["debug"])

    augmented = copy.deepcopy(merged)
    frame_items = list(augmented.items())
    by_segment: Dict[Optional[int], List[Tuple[str, Dict[str, Any]]]] = OrderedDict()
    for fid, item in frame_items:
        by_segment.setdefault(item.get("segment"), []).append((fid, item))

    records: List[Dict[str, Any]] = []

    for seg, seg_items in by_segment.items():
        if seg is None:
            continue
        next_track_id = 1
        tracks: List[Dict[str, Any]] = []
        prev_real_count: Optional[int] = None

        for local_idx, (fid, item) in enumerate(seg_items):
            det2d = item.get("det2d", []) or []
            det3d = item.get("det3d", []) or []
            curr_real = _ship_dets_only(det3d, ship_set, include_virtual=False)
            curr_real_count = len(curr_real)
            curr_2d_count = _count_ship_2d(det2d, ship_set)
            ts = item.get("timestamp")

            active_tracks = [t for t in tracks if int(t.get("misses", 0)) <= int(cfg_rep["max_gap_frames"]) and bool(t.get("active", True))]
            match_map, unmatched_track_idxs, unmatched_det_idxs = _associate_tracks_greedy(active_tracks, curr_real, float(cfg_rep["max_match_dist_m"]))

            # matched real detections update existing tracks
            for ti, di in match_map.items():
                tr = active_tracks[ti]
                det = copy.deepcopy(curr_real[di])
                tr["hist"].append(det)
                tr["label"] = str(det.get("label", tr.get("label", "")))
                tr["label_id"] = int(det.get("label_id", tr.get("label_id", -1)))
                tr["last_frame_id"] = fid
                tr["last_timestamp"] = ts
                tr["last_segment"] = seg
                tr["misses"] = 0
                tr["ghost_active"] = False
                tr["active"] = True

            # new real detections spawn new tracks
            for di in unmatched_det_idxs:
                det = copy.deepcopy(curr_real[di])
                tracks.append({
                    "track_id": int(next_track_id),
                    "label": str(det.get("label", "")),
                    "label_id": int(det.get("label_id", -1)),
                    "hist": [det],
                    "last_frame_id": fid,
                    "last_timestamp": ts,
                    "last_segment": seg,
                    "misses": 0,
                    "ghost_active": False,
                    "active": True,
                })
                next_track_id += 1

            drop_trigger = (prev_real_count is not None and curr_real_count < prev_real_count and curr_2d_count > curr_real_count)
            need_now = max(0, curr_2d_count - curr_real_count)
            triggered_track_ids: List[int] = []

            if drop_trigger and need_now > 0:
                candidate_tracks: List[Dict[str, Any]] = []
                for ti in unmatched_track_idxs:
                    tr = active_tracks[ti]
                    # only activate ghosts for tracks that were visible very recently
                    if str(tr.get("last_frame_id", "")) == str(seg_items[local_idx - 1][0]) if local_idx > 0 else False:
                        candidate_tracks.append(tr)
                    elif int(tr.get("misses", 0)) == 0:
                        candidate_tracks.append(tr)
                # prioritize tracks with longer history
                candidate_tracks.sort(key=lambda t: len(t.get("hist", [])), reverse=True)
                for tr in candidate_tracks[:need_now]:
                    tr["ghost_active"] = True
                    tr["active"] = True
                    triggered_track_ids.append(int(tr["track_id"]))

            added_virtual = []
            support_slots = max(0, curr_2d_count - curr_real_count)
            if support_slots > 0:
                ghost_tracks = [t for t in tracks if bool(t.get("ghost_active", False)) and bool(t.get("active", True))]
                ghost_tracks.sort(key=lambda t: (int(t.get("misses", 0)), -len(t.get("hist", []))))
                added_count = 0
                for tr in ghost_tracks:
                    if added_count >= support_slots:
                        break
                    if int(tr.get("misses", 0)) >= int(cfg_rep["max_gap_frames"]):
                        tr["ghost_active"] = False
                        tr["active"] = False
                        continue
                    vdet = _make_virtual_det(tr, fid, ts, seg, cfg_rep)
                    if vdet is None:
                        continue
                    item["det3d"].append(copy.deepcopy(vdet))
                    tr["hist"].append(copy.deepcopy(vdet))
                    tr["last_frame_id"] = fid
                    tr["last_timestamp"] = ts
                    tr["last_segment"] = seg
                    tr["misses"] = int(tr.get("misses", 0)) + 1
                    tr["active"] = True
                    added_virtual.append(vdet)
                    added_count += 1

            # tracks that were unmatched and not ghost-filled age by one step
            ghost_ids_filled = set(int(v["virtual_track_id"]) for v in added_virtual)
            for ti in unmatched_track_idxs:
                tr = active_tracks[ti]
                if int(tr["track_id"]) in ghost_ids_filled:
                    continue
                if bool(tr.get("ghost_active", False)):
                    tr["misses"] = int(tr.get("misses", 0)) + 1
                    if int(tr["misses"]) > int(cfg_rep["max_gap_frames"]):
                        tr["ghost_active"] = False
                        tr["active"] = False
                else:
                    tr["active"] = False

            # update flags after augmentation
            item["ship_present"] = any(str(d.get("label", "")) in ship_set for d in item.get("det3d", []))
            item["lock_gate_present"] = any(str(d.get("label", "")) == str(cfg.get("labels", {}).get("lock_gate", "Lock_gate")) for d in item.get("det3d", []))

            if added_virtual:
                record = {
                    "frame_id": fid,
                    "segment": seg,
                    "timestamp": ts,
                    "prev_real_3d": prev_real_count,
                    "curr_real_3d": curr_real_count,
                    "curr_2d": curr_2d_count,
                    "num_virtual_added": len(added_virtual),
                    "triggered_track_ids": triggered_track_ids,
                    "virtual_track_ids": [int(v["virtual_track_id"]) for v in added_virtual],
                    "virtual_labels": [str(v.get("label", "")) for v in added_virtual],
                }
                records.append(record)
                if debug:
                    print(
                        f"[VIRTUAL-SHIP] seg={seg} frame={fid} prev3d={prev_real_count} curr3d={curr_real_count} "
                        f"curr2d={curr_2d_count} add={len(added_virtual)} tids={record['virtual_track_ids']}"
                    )

            prev_real_count = curr_real_count

    return augmented, records


def save_virtual_ship_records_txt(records: List[Dict[str, Any]], save_path: Path) -> None:
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("# Frames with virtual 3D ship boxes added\n")
        f.write("# format: frame_id segment timestamp prev_real_3d curr_real_3d curr_2d num_virtual_added virtual_track_ids virtual_labels\n\n")
        for r in records:
            tids = ",".join(str(x) for x in r.get("virtual_track_ids", [])) or "-"
            labs = ",".join(str(x) for x in r.get("virtual_labels", [])) or "-"
            f.write(
                f"{r.get('frame_id','')} {r.get('segment','')} {r.get('timestamp','')} "
                f"{r.get('prev_real_3d','')} {r.get('curr_real_3d','')} {r.get('curr_2d','')} "
                f"{r.get('num_virtual_added','')} {tids} {labs}\n"
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, required=True, help="path to configs/config.yaml")
    ap.add_argument(
        "--add_info",
        type=str,
        default="add_info_testing/add_info_testing_by_timestamp.txt",
        help="path to add_info file (frame_id timestamp water_level segment)",
    )
    args = ap.parse_args()

    cfg = load_yaml(args.cfg) or {}
    labels = cfg.get("labels", {})
    lock_gate_label = labels.get("lock_gate", "Lock_gate")
    ship_labels = labels.get("ships", ["Fully_loaded_cargo_ship", "Fully_loaded_container_ship", "Unladen_cargo_ship"])

    output = cfg.get("output", {})
    root = output.get("root", "./outputs")
    root_path = Path(root)
    det3d_dir = str(root_path / "det3d")
    det2d_dir = str(root_path / "det2d")
    save_dir = str(root_path / "fused_2d3d")
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    fusion = cfg.get("fusion", {})
    iou3d = float(fusion.get("iou3d", 0.4))
    vote3d = float(fusion.get("vote3d", 0.3))
    iou2d = float(fusion.get("iou2d", 0.5))
    vote2d = float(fusion.get("vote2d", 0.5))

    print("[CFG] lock_gate_label:", lock_gate_label)
    print("[CFG] ship_labels:", ship_labels)
    print("[CFG] det3d_dir:", det3d_dir)
    print("[CFG] det2d_dir:", det2d_dir)
    print("[CFG] save_dir:", save_dir)
    print("[CFG] thresholds:", dict(iou3d=iou3d, vote3d=vote3d, iou2d=iou2d, vote2d=vote2d))
    print("[CFG] add_info:", args.add_info)
    print("[CFG] virtual_ship_repair:", _get_virtual_cfg(cfg))

    models3d = load_jsons_as_models(det3d_dir)
    models2d = load_jsons_as_models(det2d_dir)

    fused3d = fuse_3d_models(models3d, lock_gate_label, ship_labels, iou3d, vote3d) if models3d else {}
    fused2d = fuse_2d_models(models2d, iou2d, vote2d) if models2d else {}

    if fused3d:
        with open(Path(save_dir) / "fused.det3d.json", "w", encoding="utf-8") as f:
            json.dump(fused3d, f, ensure_ascii=False, indent=2)
        print("[SAVE]", str(Path(save_dir) / "fused.det3d.json"))
    if fused2d:
        with open(Path(save_dir) / "fused.det2d.json", "w", encoding="utf-8") as f:
            json.dump(fused2d, f, ensure_ascii=False, indent=2)
        print("[SAVE]", str(Path(save_dir) / "fused.det2d.json"))

    merged = merge_2d3d_annotate_sort(fused2d, fused3d, add_info_path=args.add_info)
    with open(Path(save_dir) / "fused_2d3d.json", "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print("[SAVE]", str(Path(save_dir) / "fused_2d3d.json"))

    augmented, records = augment_merged_with_virtual_ships(merged, cfg, ship_labels)
    if _get_virtual_cfg(cfg)["enabled"]:
        virtual_json = Path(save_dir) / "fused_2d3d_virtual.json"
        with open(virtual_json, "w", encoding="utf-8") as f:
            json.dump(augmented, f, ensure_ascii=False, indent=2)
        print("[SAVE]", str(virtual_json))

        virtual_txt = Path(save_dir) / "virtual_ship_frames.txt"
        save_virtual_ship_records_txt(records, virtual_txt)
        print("[SAVE]", str(virtual_txt))
        print(f"[INFO] virtual frames: {len(records)}")
    else:
        print("[INFO] virtual_ship_repair.enabled = false, skip virtual outputs")


if __name__ == "__main__":
    main()
