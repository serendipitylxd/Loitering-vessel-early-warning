import os
import argparse
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# =========================
# data paths
# =========================
def pick_first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return paths[0]


parser = argparse.ArgumentParser(description='Frame/event evaluation with the original segment-wise matching protocol')
parser.add_argument('--cfg', help='Accepted for compatibility; warning settings are applied by loitering_warning.py')
parser.add_argument('--labels-per-frame')
parser.add_argument('--pred-frame')
parser.add_argument('--out-dir', default='outputs/eval/plots')
parser.add_argument('--metrics-only', action='store_true', help='Write metrics/CI CSVs without regenerating figures')
parser.add_argument('--bootstrap-replicates', type=int, default=50000)
parser.add_argument('--bootstrap-seed', type=int, default=20261012)
args = parser.parse_args()
if args.bootstrap_replicates < 0:
    parser.error('--bootstrap-replicates must be nonnegative')

labels_path = args.labels_per_frame or pick_first_existing([
    "outputs/loitering_labels/labels_per_frame.csv",
    "labels_per_frame.csv",
    "./labels_per_frame.csv",
    "/mnt/data/labels_per_frame.csv",
])
pred_path = args.pred_frame or pick_first_existing([
    "outputs/loitering_warning/pred_frame.csv",
    "pred_frame.csv",
    "./pred_frame.csv",
    "/mnt/data/pred_frame.csv",
])

lab = pd.read_csv(labels_path, dtype={'frame_id':str})
prd = pd.read_csv(pred_path, dtype={'frame_id':str})
lab.columns = [c.strip() for c in lab.columns]
prd.columns = [c.strip() for c in prd.columns]

if "timestamp" in lab.columns:
    lab["timestamp"] = pd.to_datetime(lab["timestamp"], errors="coerce")
if "timestamp" in prd.columns:
    prd["timestamp"] = pd.to_datetime(prd["timestamp"], errors="coerce")

# robust merge
use_full = all(c in lab.columns for c in ["segment", "frame_id", "timestamp"]) and \
           all(c in prd.columns for c in ["segment", "frame_id", "timestamp"])
if use_full:
    keys = ['segment','frame_id','timestamp']
    if lab['timestamp'].isna().any() or prd['timestamp'].isna().any():
        raise ValueError('Invalid evaluation timestamps')
    if lab.duplicated(keys).any() or prd.duplicated(keys).any():
        raise ValueError('Duplicate evaluation frame keys')
    if set(map(tuple,lab[keys].to_numpy())) != set(map(tuple,prd[keys].to_numpy())):
        raise ValueError('GT/prediction frame coverage differs; refusing silent inner-join row dropping')
    m = pd.merge(lab, prd, on=keys, how='inner', validate='one_to_one', suffixes=('_gt','_pred'))
else:
    m = pd.merge(lab, prd, on=["segment", "frame_id"], how="inner", suffixes=("_gt", "_pred"))
    if "timestamp_gt" in m.columns:
        m["timestamp"] = pd.to_datetime(m["timestamp_gt"], errors="coerce")

if args.metrics_only:
    # Load only the reusable evaluator functions; avoid module-level plotting.
    with open(__file__, encoding='utf-8') as source:
        tree = ast.parse(source.read())
    names = {'norm','normalize_series','bin_loiter','build_positive_episodes',
             'interval_iou','interval_overlap_ratio','match_events',
             'calc_frame_binary_metrics','collect_loiter_eval','bootstrap_event_intervals'}
    selected = [node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
    namespace = dict(np=np,pd=pd,POS='LOITERING')
    exec(compile(ast.Module(body=selected,type_ignores=[]),__file__,'exec'),namespace)
    os.makedirs(args.out_dir,exist_ok=True)
    outputs = [[] for _ in range(5)]
    gt_col = 'status_gt' if 'status_gt' in m.columns else 'status'
    for col in ['status_pre','status_kf']:
        if col not in m:
            continue
        values = namespace['collect_loiter_eval'](m,gt_col,col,col)
        for destination,value in zip(outputs,values):
            destination.append(value)
    if not outputs[0]:
        raise ValueError('Prediction CSV must contain status_pre or status_kf')
    for name,parts in zip(['frame_overall','event_overall','frame_segment','event_segment','event_matches'],outputs):
        pd.concat(parts,ignore_index=True).to_csv(os.path.join(args.out_dir,name+'.csv'),index=False)
    if args.bootstrap_replicates > 0:
        ci = namespace['bootstrap_event_intervals'](pd.concat(outputs[3],ignore_index=True),
            args.bootstrap_replicates,args.bootstrap_seed)
        ci.to_csv(os.path.join(args.out_dir,'event_confidence_intervals.csv'),index=False)
    print('[OK] Metrics only:',args.out_dir)
    raise SystemExit(0)

segs = sorted([s for s in m["segment"].dropna().unique().tolist()])[:18]
rows, cols = 6, 3


def norm(x):
    return "" if pd.isna(x) else str(x).strip()


gate_gt = "gate_state_gt" if "gate_state_gt" in m.columns else ("gate_state" if "gate_state" in m.columns else None)
gate_pr = "gate_state_pred" if "gate_state_pred" in m.columns else ("gate_state" if "gate_state" in m.columns else None)

status_gt = "status_gt" if "status_gt" in m.columns else ("status" if "status" in m.columns else None)
status_kf = "status_kf" if "status_kf" in m.columns else None
status_pre = "status_pre" if "status_pre" in m.columns else None

# gate map
cats = set()
for c in [gate_gt, gate_pr]:
    if c and c in m.columns:
        cats |= set(m.loc[m["segment"].isin(segs), c].map(norm).unique().tolist())
cats = [c for c in cats if c != ""]
cats = sorted(cats)
gate_to_int = {c: i for i, c in enumerate(cats)}


def gate_int(s):
    return s.map(lambda x: gate_to_int.get(norm(x), np.nan)).astype(float)


POS = "LOITERING"


def bin_loiter(s):
    return s.map(lambda x: 1 if norm(x) == POS else 0).astype(float)


out_dir = args.out_dir
os.makedirs(out_dir, exist_ok=True)

# =========================
# paper-like style
# =========================
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 14,
    "axes.linewidth": 0.8,
    "axes.unicode_minus": False,
})

PAPER = {
    "blue": "#1f77b4",   # GT
    "teal": "#ff7f0e",   # Pred / status_kf
    "gold": "#2ca02c",   # status_pre / F1
    "coral": "#d62728",  # highlight / error
    "gray": "#7f7f7f",
    "dark": "#2f2f2f",
    "light": "#f7f7f7",
}

CONF_CMAP = LinearSegmentedColormap.from_list(
    "paper_blue",
    ["#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#1f77b4"]
)


# =========================
# common helpers
# =========================
def normalize_series(s):
    return s.map(norm)


def build_episodes(df, state_col):
    vals = normalize_series(df[state_col]).tolist()
    n = len(vals)
    if n == 0:
        return []
    eps = []
    start = 0
    cur = vals[0]
    for i in range(1, n):
        if vals[i] != cur:
            eps.append((cur, start, i - 1))
            cur = vals[i]
            start = i
    eps.append((cur, start, n - 1))
    return eps


def build_positive_episodes(df, state_col, pos="LOITERING"):
    vals = normalize_series(df[state_col]).tolist()
    eps = []
    start = None
    for i, v in enumerate(vals):
        if v == pos and start is None:
            start = i
        elif v != pos and start is not None:
            eps.append((start, i - 1))
            start = None
    if start is not None:
        eps.append((start, len(vals) - 1))
    return eps


def interval_iou(a, b):
    a0, a1 = a
    b0, b1 = b
    inter = max(0, min(a1, b1) - max(a0, b0) + 1)
    union = (a1 - a0 + 1) + (b1 - b0 + 1) - inter
    return inter / union if union > 0 else 0.0


def interval_overlap_ratio(a, b):
    a0, a1 = a
    b0, b1 = b
    inter = max(0, min(a1, b1) - max(a0, b0) + 1)
    gt_len = (a1 - a0 + 1)
    return inter / gt_len if gt_len > 0 else 0.0


def match_events(gt_eps, pr_eps, min_iou=0.0):
    matches = []
    used_pred = set()
    for gi, g in enumerate(gt_eps):
        best_j = None
        best_iou = -1.0
        for pj, p in enumerate(pr_eps):
            if pj in used_pred:
                continue
            iou = interval_iou(g, p)
            if iou > best_iou:
                best_iou = iou
                best_j = pj
        if best_j is not None and best_iou > min_iou:
            used_pred.add(best_j)
            matches.append((gi, best_j, best_iou))
    unmatched_gt = [i for i in range(len(gt_eps)) if i not in [m[0] for m in matches]]
    unmatched_pr = [j for j in range(len(pr_eps)) if j not in [m[1] for m in matches]]
    return matches, unmatched_gt, unmatched_pr


def calc_frame_binary_metrics(y_true, y_pred):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else np.nan
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else np.nan
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1, "accuracy": acc
    }


def save_block_csv(path, blocks):
    rows_all = []
    for name, df in blocks:
        if df is None:
            continue
        if isinstance(df, pd.DataFrame) and len(df) == 0:
            continue
        tmp = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
        tmp.insert(0, "block", name)
        rows_all.append(tmp)
    out = pd.concat(rows_all, ignore_index=True, sort=False) if rows_all else pd.DataFrame()
    out.to_csv(path, index=False, encoding="utf-8-sig")


# =========================
# 1) gate_state 6x3
# =========================
fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 2.6), sharey=True)
axes = axes.ravel()
for i, seg in enumerate(segs):
    ax = axes[i]
    g = m[m["segment"] == seg].copy()
    if "timestamp" in g.columns and g["timestamp"].notna().any():
        g = g.sort_values("timestamp")
        x = g["timestamp"]
    else:
        g = g.sort_values("frame_id")
        x = np.arange(len(g))

    off_gt = 0.06
    off_pr = -0.06

    if gate_gt and gate_gt in g.columns:
        ax.step(x, gate_int(g[gate_gt]) + off_gt, where="post", label="GT",
                color=PAPER["blue"], linewidth=2.2, alpha=0.95, zorder=3)
    if gate_pr and gate_pr in g.columns:
        ax.step(x, gate_int(g[gate_pr]) + off_pr, where="post", label="Pred",
                color=PAPER["teal"], linewidth=1.8, alpha=0.9, linestyle="--", zorder=2)

    ax.set_title(f"Seg {seg}")
    if cats:
        ax.set_ylim(-0.3, len(cats) - 1 + 0.3)
    ax.grid(True, alpha=0.25, linestyle="--")
    if i == 0:
        ax.legend(fontsize=8, loc="upper right")
    for tick in ax.get_xticklabels():
        tick.set_rotation(20)
for j in range(len(segs), rows * cols):
    axes[j].axis("off")
if cats:
    for ax in axes[:len(segs)]:
        ax.set_yticks(list(range(len(cats))))
        ax.set_yticklabels(cats)
fig.suptitle("Gate State: GT vs Pred (6x3 segments)", y=0.995, fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.99])
fig.savefig(os.path.join(out_dir, "gate_state_6x3.png"), dpi=180, bbox_inches="tight")
plt.close(fig)


# =========================
# 2) loitering status 6x3
# =========================
fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 2.6), sharey=True)
axes = axes.ravel()
for i, seg in enumerate(segs):
    ax = axes[i]
    g = m[m["segment"] == seg].copy()
    if "timestamp" in g.columns and g["timestamp"].notna().any():
        g = g.sort_values("timestamp")
        x = g["timestamp"]
    else:
        g = g.sort_values("frame_id")
        x = np.arange(len(g))

    if status_gt and status_gt in g.columns:
        ax.step(x, bin_loiter(g[status_gt]) + 0.00, where="post", label="GT",
                color=PAPER["blue"], linewidth=2.2, alpha=0.95, zorder=3)
    if status_kf and status_kf in g.columns:
        ax.step(x, bin_loiter(g[status_kf]) + 0.03, where="post", label="status_kf",
                color=PAPER["teal"], linewidth=1.8, alpha=0.9, linestyle="--", zorder=2)
    if status_pre and status_pre in g.columns:
        ax.step(x, bin_loiter(g[status_pre]) - 0.03, where="post", label="status_pre",
                color=PAPER["gold"], linewidth=1.8, alpha=0.9, linestyle=":", zorder=1)

    ax.set_title(f"Seg {seg}")
    ax.set_ylim(-0.12, 1.12)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["non-loiter", "LOITERING"])
    ax.grid(True, alpha=0.25, linestyle="--")
    if i == 0:
        ax.legend(fontsize=8, loc="upper right")
    for tick in ax.get_xticklabels():
        tick.set_rotation(20)
for j in range(len(segs), rows * cols):
    axes[j].axis("off")
fig.suptitle("Loitering Status: GT vs KF vs Pre (6x3 segments)", y=0.995, fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.99])
fig.savefig(os.path.join(out_dir, "status_compare_6x3.png"), dpi=180, bbox_inches="tight")
plt.close(fig)


# =========================
# 3) gate evaluation
# =========================
def draw_confusion(ax, cm, states):
    im = ax.imshow(cm.values, aspect="auto", cmap=CONF_CMAP)
    ax.set_xticks(np.arange(len(states)))
    ax.set_yticks(np.arange(len(states)))
    ax.set_xticklabels(states, rotation=20)
    ax.set_yticklabels(states)
    ax.set_xlabel("Pred")
    ax.set_ylabel("GT")
    ax.set_title("(a) Confusion matrix")
    for r in range(cm.shape[0]):
        row_sum = cm.values[r].sum()
        for c in range(cm.shape[1]):
            v = int(cm.values[r, c])
            pct = 100.0 * v / row_sum if row_sum > 0 else 0.0
            txt_color = "white" if v > cm.values.max() * 0.55 else PAPER["dark"]
            ax.text(c, r, f"{v}\n{pct:.1f}%", ha="center", va="center", fontsize=9, color=txt_color)
    return im


def draw_overall_frames(ax, total_correct, total_errors, total_frames):
    acc_pct = 100.0 * total_correct / total_frames if total_frames else np.nan
    err_pct = 100.0 * total_errors / total_frames if total_frames else np.nan
    sizes = [total_correct, total_errors]
    colors = [PAPER["blue"], PAPER["coral"]]
    explode = [0.0, 0.08 if total_errors > 0 else 0.0]
    wedges, _ = ax.pie(
        sizes,
        startangle=90,
        counterclock=False,
        colors=colors,
        explode=explode,
        wedgeprops=dict(width=0.34, edgecolor="white", linewidth=1.2),
    )
    ax.text(0, 0.10, f"{acc_pct:.3f}%", ha="center", va="center", fontsize=18, fontweight="bold", color=PAPER["dark"])
    ax.text(0, -0.12, "Accuracy", ha="center", va="center", fontsize=10, color=PAPER["gray"])
    ax.text(-1.44, 0.38, f"Correct\n{total_correct:,} frames\n({acc_pct:.3f}%)", ha="left", va="center", fontsize=9.5, color=PAPER["dark"])
    ax.text(1.05, -0.46, f"Error\n{total_errors:,} frames\n({err_pct:.3f}%)", ha="left", va="center", fontsize=9.5, color=PAPER["dark"])
    if len(wedges) >= 2 and total_errors > 0:
        w = wedges[1]
        theta = np.deg2rad((w.theta1 + w.theta2) / 2.0)
        x = np.cos(theta) * 0.95
        y = np.sin(theta) * 0.95
        ax.annotate("error slice", xy=(x, y), xycoords="data", xytext=(0.82, 0.82), textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="-", lw=1.0, color=PAPER["gray"]),
                    ha="left", va="center", fontsize=8.8, color=PAPER["gray"])
    ax.set_title(f"(b) Overall frame-level evaluation\n(all {total_frames} frames)")
    ax.set_aspect("equal")


def draw_delay_box(ax, delay_df, states_delay):
    data = [delay_df.loc[delay_df["state"] == s, "onset_delay_frames"].dropna().values for s in states_delay]
    box = ax.boxplot(
        data, patch_artist=True, widths=0.55, showfliers=False,
        medianprops=dict(color=PAPER["dark"], linewidth=1.6),
        whiskerprops=dict(color=PAPER["gray"], linewidth=1.2),
        capprops=dict(color=PAPER["gray"], linewidth=1.2),
    )
    delay_colors = [PAPER["blue"], PAPER["teal"], PAPER["gold"], "#8c564b"]
    for patch, color in zip(box["boxes"], delay_colors[:len(data)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.68)
        patch.set_edgecolor(PAPER["dark"])
        patch.set_linewidth(1.0)
    for i, vals in enumerate(data, start=1):
        if len(vals) > 0:
            xs = np.linspace(i - 0.12, i + 0.12, len(vals))
            ax.scatter(xs, vals, s=22, color=PAPER["dark"], alpha=0.65, zorder=3)
            ax.scatter([i], [np.mean(vals)], marker="D", s=42, color=PAPER["coral"], zorder=4)
    ax.axhline(0.0, color=PAPER["gray"], linestyle="--", linewidth=1.1)
    ax.set_xticks(np.arange(1, len(states_delay) + 1))
    ax.set_xticklabels(states_delay)
    ax.set_ylabel("Delay (frames)")
    ax.set_title("(c) Event onset delay by state")
    ax.grid(True, axis="y", alpha=0.25, linestyle="--")


def draw_state_metrics(ax, metrics_df):
    x = np.arange(len(metrics_df))
    width = 0.22
    ax.bar(x - width, metrics_df["precision"].fillna(0) * 100.0, width, label="Precision", color=PAPER["blue"], alpha=0.80)
    ax.bar(x, metrics_df["recall"].fillna(0) * 100.0, width, label="Recall", color=PAPER["teal"], alpha=0.80)
    ax.bar(x + width, metrics_df["f1"].fillna(0) * 100.0, width, label="F1", color=PAPER["gold"], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df["state"].tolist())
    ax.set_ylim(0, 105)
    ax.set_ylabel("Score (%)")
    ax.set_title("(d) Per-state frame metrics")
    ax.grid(True, axis="y", alpha=0.25, linestyle="--")
    ax.legend(loc="upper right", frameon=True, fancybox=False, edgecolor="#B8B8B8", framealpha=0.95)


def summarize_segment_for_gate_plot(df, gate_gt_col, gate_pr_col):
    rows = []
    for seg in segs:
        g = df[df["segment"] == seg].copy()
        if "timestamp" in g.columns and g["timestamp"].notna().any():
            g = g.sort_values("timestamp").reset_index(drop=True)
        else:
            g = g.sort_values("frame_id").reset_index(drop=True)
        gt_vals = normalize_series(g[gate_gt_col])
        pr_vals = normalize_series(g[gate_pr_col])
        gt_eps = build_episodes(g, gate_gt_col)
        pr_eps = build_episodes(g, gate_pr_col)
        row = {
            "segment": seg,
            "n_frames": len(g),
            "correct_frames": int((gt_vals == pr_vals).sum()),
            "error_frames": int((gt_vals != pr_vals).sum()),
            "acc_pct": float((gt_vals == pr_vals).mean() * 100.0) if len(g) else np.nan,
            "gt_episode_count": len(gt_eps),
            "pred_episode_count": len(pr_eps),
            "gt_transition_count": max(len(gt_eps) - 1, 0),
            "pred_transition_count": max(len(pr_eps) - 1, 0),
            "gt_start_state": gt_vals.iloc[0] if len(g) else "",
            "gt_end_state": gt_vals.iloc[-1] if len(g) else "",
            "pred_start_state": pr_vals.iloc[0] if len(g) else "",
            "pred_end_state": pr_vals.iloc[-1] if len(g) else "",
        }
        for s in cats:
            row[f"gt_{s}_frames"] = int((gt_vals == s).sum())
            row[f"pred_{s}_frames"] = int((pr_vals == s).sum())
        rows.append(row)
    return pd.DataFrame(rows)


if gate_gt and gate_pr and gate_gt in m.columns and gate_pr in m.columns:
    mg = m.copy()
    if "timestamp" in mg.columns and mg["timestamp"].notna().any():
        mg = mg.sort_values(["timestamp", "segment", "frame_id"]).reset_index(drop=True)
    else:
        mg = mg.sort_values(["segment", "frame_id"]).reset_index(drop=True)

    mg["gate_gt_norm"] = normalize_series(mg[gate_gt])
    mg["gate_pr_norm"] = normalize_series(mg[gate_pr])
    mg["is_correct"] = (mg["gate_gt_norm"] == mg["gate_pr_norm"]).astype(int)
    mg["is_error"] = 1 - mg["is_correct"]

    total_frames = len(mg)
    total_correct = int(mg["is_correct"].sum())
    total_errors = int(mg["is_error"].sum())
    overall_acc = total_correct / total_frames if total_frames else np.nan
    overall_err = total_errors / total_frames if total_frames else np.nan

    states = [c for c in ["CLOSED", "CLOSING", "OPENING", "OPEN"]
              if c in set(mg["gate_gt_norm"]).union(set(mg["gate_pr_norm"]))]
    if not states:
        states = sorted([c for c in set(mg["gate_gt_norm"]).union(set(mg["gate_pr_norm"])) if c != ""])

    metrics_rows = []
    for s in states:
        tp = int(((mg["gate_gt_norm"] == s) & (mg["gate_pr_norm"] == s)).sum())
        fp = int(((mg["gate_gt_norm"] != s) & (mg["gate_pr_norm"] == s)).sum())
        fn = int(((mg["gate_gt_norm"] == s) & (mg["gate_pr_norm"] != s)).sum())
        support = int((mg["gate_gt_norm"] == s).sum())
        pred_n = int((mg["gate_pr_norm"] == s).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else np.nan
        metrics_rows.append({
            "state": s, "support_gt": support, "support_pred": pred_n,
            "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1
        })
    metrics_df = pd.DataFrame(metrics_rows)

    cm = pd.crosstab(mg["gate_gt_norm"], mg["gate_pr_norm"], dropna=False)
    cm = cm.reindex(index=states, columns=states, fill_value=0)

    delay_rows, dur_rows = [], []
    for seg, g in m.groupby("segment"):
        if "timestamp" in g.columns and g["timestamp"].notna().any():
            g = g.sort_values("timestamp").reset_index(drop=True)
        else:
            g = g.sort_values("frame_id").reset_index(drop=True)
        gt_eps = build_episodes(g, gate_gt)
        pr_eps = build_episodes(g, gate_pr)
        i = j = 0
        while i < len(gt_eps) and j < len(pr_eps):
            gt_state, gt_s, gt_e = gt_eps[i]
            pr_state, pr_s, pr_e = pr_eps[j]
            if gt_state == pr_state:
                delay_rows.append({
                    "segment": seg,
                    "state": gt_state,
                    "gt_start_idx": gt_s,
                    "pred_start_idx": pr_s,
                    "onset_delay_frames": int(pr_s - gt_s)
                })
                dur_rows.append({
                    "segment": seg,
                    "state": gt_state,
                    "gt_duration_frames": int(gt_e - gt_s + 1),
                    "pred_duration_frames": int(pr_e - pr_s + 1),
                    "duration_error_frames": int((pr_e - pr_s) - (gt_e - gt_s))
                })
                i += 1
                j += 1
            else:
                if gt_s <= pr_s:
                    i += 1
                else:
                    j += 1

    delay_df = pd.DataFrame(delay_rows)
    dur_df = pd.DataFrame(dur_rows)
    seg_eval = mg.groupby("segment")["is_correct"].agg(["mean", "size"]).reset_index()
    seg_eval["acc_pct"] = seg_eval["mean"] * 100.0
    seg_eval["err_pct"] = (1.0 - seg_eval["mean"]) * 100.0
    seg_eval.rename(columns={"size": "n_frames"}, inplace=True)

    fig, ax = plt.subplots(figsize=(7.4, 6.0))
    im = draw_confusion(ax, cm, states)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "gate_confusion_matrix.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    draw_overall_frames(ax, total_correct, total_errors, total_frames)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "gate_eval_overall.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)

    states_delay = []
    if not delay_df.empty:
        states_delay = [s for s in ["CLOSED", "CLOSING", "OPENING", "OPEN"] if s in delay_df["state"].unique().tolist()]
        fig, ax = plt.subplots(figsize=(7.4, 5.6))
        draw_delay_box(ax, delay_df, states_delay)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "gate_event_delay.png"), dpi=180, bbox_inches="tight")
        plt.close(fig)

    if not metrics_df.empty:
        fig, ax = plt.subplots(figsize=(7.4, 5.6))
        draw_state_metrics(ax, metrics_df)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "gate_state_metrics.png"), dpi=180, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 11.0))
    im = draw_confusion(axes[0, 0], cm, states)
    fig.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.04)
    draw_overall_frames(axes[0, 1], total_correct, total_errors, total_frames)
    if not delay_df.empty and len(states_delay) > 0:
        draw_delay_box(axes[1, 0], delay_df, states_delay)
    else:
        axes[1, 0].axis("off")
    if not metrics_df.empty:
        draw_state_metrics(axes[1, 1], metrics_df)
    else:
        axes[1, 1].axis("off")
    fig.suptitle(f"Gate-state evaluation dashboard | all frames={total_frames}, accuracy={overall_acc*100:.3f}%, error={overall_err*100:.3f}%", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(os.path.join(out_dir, "gate_eval_2x2.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    seg_6x3_df = summarize_segment_for_gate_plot(m, gate_gt, gate_pr)
    overview_df = pd.DataFrame([{
        "total_frames": total_frames,
        "correct_frames": total_correct,
        "error_frames": total_errors,
        "overall_accuracy": overall_acc,
        "overall_error_rate": overall_err,
        "segments_in_gate_state_6x3": len(seg_6x3_df),
        "segment_accuracy_mean_pct": seg_eval["acc_pct"].mean() if not seg_eval.empty else np.nan,
        "segment_accuracy_min_pct": seg_eval["acc_pct"].min() if not seg_eval.empty else np.nan,
        "segment_accuracy_max_pct": seg_eval["acc_pct"].max() if not seg_eval.empty else np.nan,
    }])
    delay_summary = (
        delay_df.groupby("state")["onset_delay_frames"].agg(["count", "mean", "median", "min", "max"]).reindex(states_delay).reset_index()
        if not delay_df.empty else pd.DataFrame(columns=["state", "count", "mean", "median", "min", "max"])
    )
    duration_summary = (
        dur_df.groupby("state")["duration_error_frames"].agg(["count", "mean", "median", "min", "max"]).reset_index()
        if not dur_df.empty else pd.DataFrame(columns=["state", "count", "mean", "median", "min", "max"])
    )
    figure_guide_df = pd.DataFrame([
        {"figure": "gate_eval_2x2.png", "panel": "(a)", "meaning": "All-frame confusion matrix"},
        {"figure": "gate_eval_2x2.png", "panel": "(b)", "meaning": "Overall correct/error frame counts and rates"},
        {"figure": "gate_eval_2x2.png", "panel": "(c)", "meaning": "Per-state onset delay distribution in frames"},
        {"figure": "gate_eval_2x2.png", "panel": "(d)", "meaning": "Per-state precision/recall/F1"},
        {"figure": "gate_state_6x3.png", "panel": "all", "meaning": "18 segments timeline summary for GT vs Pred"},
    ])
    save_block_csv(os.path.join(out_dir, "gate_eval_summary.csv"), [
        ("figure_guide", figure_guide_df),
        ("overall_summary", overview_df),
        ("per_state_metrics", metrics_df),
        ("confusion_matrix", cm.reset_index().rename(columns={cm.index.name or "index": "gt_state"})),
        ("delay_summary", delay_summary),
        ("delay_events", delay_df),
        ("duration_summary", duration_summary),
        ("duration_events", dur_df),
        ("segment_accuracy_all", seg_eval),
        ("segment_timeline_6x3", seg_6x3_df),
    ])


# =========================
# 4) loiter evaluation
# =========================
def collect_loiter_eval(df, gt_col, pred_col, pred_name):
    rows_frame_seg = []
    rows_event_match = []
    rows_event_seg = []
    all_y_true = []
    all_y_pred = []

    for seg, g in df.groupby("segment"):
        if "timestamp" in g.columns and g["timestamp"].notna().any():
            g = g.sort_values("timestamp").reset_index(drop=True)
        else:
            g = g.sort_values("frame_id").reset_index(drop=True)

        y_true = bin_loiter(g[gt_col])
        y_pred = bin_loiter(g[pred_col])
        all_y_true.append(y_true.values)
        all_y_pred.append(y_pred.values)

        fm = calc_frame_binary_metrics(y_true, y_pred)
        fm["segment"] = seg
        fm["method"] = pred_name
        fm["gt_loiter_frames"] = int(y_true.sum())
        fm["pred_loiter_frames"] = int(y_pred.sum())
        rows_frame_seg.append(fm)

        gt_eps = build_positive_episodes(g, gt_col, POS)
        pr_eps = build_positive_episodes(g, pred_col, POS)
        matches, unmatched_gt, unmatched_pr = match_events(gt_eps, pr_eps, min_iou=0.0)

        tp_e = len(matches)
        fp_e = len(unmatched_pr)
        fn_e = len(unmatched_gt)
        prec_e = tp_e / (tp_e + fp_e) if (tp_e + fp_e) > 0 else np.nan
        rec_e = tp_e / (tp_e + fn_e) if (tp_e + fn_e) > 0 else np.nan
        f1_e = 2 * prec_e * rec_e / (prec_e + rec_e) if (prec_e + rec_e) > 0 else np.nan
        rows_event_seg.append({
            "segment": seg,
            "method": pred_name,
            "gt_events": len(gt_eps),
            "pred_events": len(pr_eps),
            "tp_events": tp_e,
            "fp_events": fp_e,
            "fn_events": fn_e,
            "precision_event": prec_e,
            "recall_event": rec_e,
            "f1_event": f1_e,
        })

        for gi, pj, iou in matches:
            g0, g1 = gt_eps[gi]
            p0, p1 = pr_eps[pj]
            inter = max(0, min(g1, p1) - max(g0, p0) + 1)
            union = (g1 - g0 + 1) + (p1 - p0 + 1) - inter
            overlap_gt = interval_overlap_ratio((g0, g1), (p0, p1))
            time_metrics = {}
            if 'timestamp' in g.columns:
                gs,ge = pd.Timestamp(g.loc[g0,'timestamp']),pd.Timestamp(g.loc[g1,'timestamp'])
                ps,pe = pd.Timestamp(g.loc[p0,'timestamp']),pd.Timestamp(g.loc[p1,'timestamp'])
                union_s = (max(ge,pe)-min(gs,ps)).total_seconds()
                intersection_s = max(0.0,(min(ge,pe)-max(gs,ps)).total_seconds())
                time_metrics = dict(onset_delay_s=(ps-gs).total_seconds(),
                    end_delay_s=(pe-ge).total_seconds(), gt_duration_s=(ge-gs).total_seconds(),
                    pred_duration_s=(pe-ps).total_seconds(),
                    duration_error_s=((pe-ps)-(ge-gs)).total_seconds(),
                    temporal_iou=intersection_s/union_s if union_s>0 else np.nan)
            rows_event_match.append({
                **time_metrics,
                "segment": seg,
                "method": pred_name,
                "gt_start": g0,
                "gt_end": g1,
                "pred_start": p0,
                "pred_end": p1,
                "onset_delay_frames": int(p0 - g0),
                "end_delay_frames": int(p1 - g1),
                "gt_duration_frames": int(g1 - g0 + 1),
                "pred_duration_frames": int(p1 - p0 + 1),
                "duration_error_frames": int((p1 - p0 + 1) - (g1 - g0 + 1)),
                "overlap_frames": int(inter),
                "union_frames": int(union),
                "sync_iou": float(iou),
                "sync_ratio_gt": float(overlap_gt),
            })

    all_y_true = np.concatenate(all_y_true) if len(all_y_true) else np.array([])
    all_y_pred = np.concatenate(all_y_pred) if len(all_y_pred) else np.array([])
    overall_frame = calc_frame_binary_metrics(all_y_true, all_y_pred)
    overall_frame["method"] = pred_name
    overall_frame["total_frames"] = len(all_y_true)
    overall_frame["gt_loiter_frames"] = int(all_y_true.sum())
    overall_frame["pred_loiter_frames"] = int(all_y_pred.sum())

    frame_seg_df = pd.DataFrame(rows_frame_seg)
    event_match_df = pd.DataFrame(rows_event_match)
    event_seg_df = pd.DataFrame(rows_event_seg)

    if not event_seg_df.empty:
        tp_e = int(event_seg_df["tp_events"].sum())
        fp_e = int(event_seg_df["fp_events"].sum())
        fn_e = int(event_seg_df["fn_events"].sum())
        prec_e = tp_e / (tp_e + fp_e) if (tp_e + fp_e) > 0 else np.nan
        rec_e = tp_e / (tp_e + fn_e) if (tp_e + fn_e) > 0 else np.nan
        f1_e = 2 * prec_e * rec_e / (prec_e + rec_e) if (prec_e + rec_e) > 0 else np.nan
    else:
        tp_e = fp_e = fn_e = 0
        prec_e = rec_e = f1_e = np.nan

    overall_event = {
        "method": pred_name,
        "tp_events": tp_e,
        "fp_events": fp_e,
        "fn_events": fn_e,
        "precision_event": prec_e,
        "recall_event": rec_e,
        "f1_event": f1_e,
        "gt_events": int(event_seg_df["gt_events"].sum()) if not event_seg_df.empty else 0,
        "pred_events": int(event_seg_df["pred_events"].sum()) if not event_seg_df.empty else 0,
        "mean_onset_delay": event_match_df["onset_delay_frames"].mean() if not event_match_df.empty else np.nan,
        "median_onset_delay": event_match_df["onset_delay_frames"].median() if not event_match_df.empty else np.nan,
        "mean_sync_iou": event_match_df["sync_iou"].mean() if not event_match_df.empty else np.nan,
        "mean_sync_ratio_gt": event_match_df["sync_ratio_gt"].mean() if not event_match_df.empty else np.nan,
        "mean_duration_error": event_match_df["duration_error_frames"].mean() if not event_match_df.empty else np.nan,
        # Keep legacy inclusive-frame values above; physical time metrics are
        # explicitly separate and never change event matching decisions.
        "mean_onset_delay_s": event_match_df['onset_delay_s'].mean() if 'onset_delay_s' in event_match_df else np.nan,
        "mean_temporal_iou": event_match_df['temporal_iou'].mean() if 'temporal_iou' in event_match_df else np.nan,
        "mean_duration_error_s": event_match_df['duration_error_s'].mean() if 'duration_error_s' in event_match_df else np.nan,
    }

    return pd.DataFrame([overall_frame]), pd.DataFrame([overall_event]), frame_seg_df, event_seg_df, event_match_df


def bootstrap_event_intervals(event_segments, replicates=50000, seed=20261012):
    """Paired whole-segment percentile bootstrap; never resample individual frames.

    Intervals describe the observed clips, not independent-scene generalization.
    Zero observed FN makes bootstrap recall degenerate. Wilson recall bounds
    are also provided as a descriptive independent-event binomial sensitivity.
    """
    if replicates <= 0:
        raise ValueError('replicates must be positive')
    segments = sorted(event_segments['segment'].unique())
    rng = np.random.default_rng(seed)
    draws = rng.integers(0,len(segments),size=(replicates,len(segments)))
    result = []
    for method,group in event_segments.groupby('method',sort=False):
        group = group.set_index('segment').reindex(segments)
        if group[['tp_events','fp_events','fn_events']].isna().any().any():
            raise ValueError('Methods must use the same segment set')
        counts = group[['tp_events','fp_events','fn_events']].to_numpy(dtype=int)
        sampled = counts[draws].sum(axis=1)
        tp,fp,fn = sampled.T
        original_tp,original_fp,original_fn = counts.sum(axis=0)
        row = dict(method=method,gt_events=int(original_tp+original_fn),
            pred_events=int(original_tp+original_fp),tp_events=int(original_tp),
            fp_events=int(original_fp),fn_events=int(original_fn),
            bootstrap_replicates=replicates,bootstrap_seed=seed,n_segments=len(segments))
        for name,numerator,denominator,point_n,point_d in [
            ('precision',tp,tp+fp,original_tp,original_tp+original_fp),
            ('recall',tp,tp+fn,original_tp,original_tp+original_fn),
            ('f1',2*tp,2*tp+fp+fn,2*original_tp,2*original_tp+original_fp+original_fn)]:
            values = np.full(replicates,np.nan)
            np.divide(numerator,denominator,out=values,where=denominator>0)
            valid = values[np.isfinite(values)]
            low,high = np.quantile(valid,[.025,.975]) if len(valid) else (np.nan,np.nan)
            row.update({name:float(point_n/point_d) if point_d else np.nan,
                name+'_ci_lower':float(low),name+'_ci_upper':float(high),name+'_valid_replicates':len(valid)})
        n = original_tp+original_fn
        if n:
            z=1.959963984540054; p=original_tp/n; scale=1+z*z/n
            centre=(p+z*z/(2*n))/scale
            radius=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/scale
            row.update(recall_wilson_lower=float(max(0,centre-radius)),
                       recall_wilson_upper=float(min(1,centre+radius)))
        row['ci_scope'] = 'Empirical segment-cluster bootstrap; Wilson assumes independent events'
        result.append(row)
    return pd.DataFrame(result)


def draw_loiter_frame_counts(ax, frame_overall_df):
    plot_df = frame_overall_df.copy()
    if plot_df.empty:
        ax.axis("off")
        return

    methods = plot_df["method"].tolist()

    
    ax.axis("off")
    n = len(methods)

    if n == 1:
        sub_axes = [ax.inset_axes([0.12, 0.10, 0.76, 0.80])]
    elif n == 2:
        sub_axes = [
            ax.inset_axes([0.02, 0.08, 0.46, 0.84]),
            ax.inset_axes([0.52, 0.08, 0.46, 0.84]),
        ]
    else:
        
        sub_axes = []
        w = 0.96 / n
        for i in range(n):
            sub_axes.append(ax.inset_axes([0.02 + i * w, 0.08, w - 0.02, 0.84]))

    for subax, (_, row) in zip(sub_axes, plot_df.iterrows()):
        total_frames = int(row["tp"] + row["fp"] + row["fn"] + row["tn"])
        total_correct = int(row["tp"] + row["tn"])
        total_errors = int(row["fp"] + row["fn"])

        acc_pct = 100.0 * total_correct / total_frames if total_frames else np.nan
        err_pct = 100.0 * total_errors / total_frames if total_frames else np.nan

        sizes = [total_correct, total_errors]
        colors = [PAPER["blue"], PAPER["coral"]]
        explode = [0.0, 0.08 if total_errors > 0 else 0.0]

        wedges, _ = subax.pie(
            sizes,
            startangle=90,
            counterclock=False,
            colors=colors,
            explode=explode,
            wedgeprops=dict(width=0.34, edgecolor="white", linewidth=1.2),
        )

        subax.text(
            0, 0.10, f"{acc_pct:.2f}%",
            ha="center", va="center",
            fontsize=14, fontweight="bold", color=PAPER["dark"]
        )
        subax.text(
            0, -0.12, "Accuracy",
            ha="center", va="center",
            fontsize=9, color=PAPER["gray"]
        )

        subax.text(
            -1.45, 0.40,
            f"Correct\n{total_correct:,}\n({acc_pct:.2f}%)",
            ha="left", va="center",
            fontsize=8.5, color=PAPER["dark"]
        )
        subax.text(
            1.02, -0.42,
            f"Error\n{total_errors:,}\n({err_pct:.2f}%)",
            ha="left", va="center",
            fontsize=8.5, color=PAPER["dark"]
        )

        method_name = str(row["method"]).replace("status_", "").upper()
        subax.set_title(method_name, fontsize=10)
        subax.set_aspect("equal")

        if len(wedges) >= 2 and total_errors > 0:
            w = wedges[1]
            theta = np.deg2rad((w.theta1 + w.theta2) / 2.0)
            x = np.cos(theta) * 0.95
            y = np.sin(theta) * 0.95
            subax.annotate(
                "error",
                xy=(x, y), xycoords="data",
                xytext=(0.82, 0.82), textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-", lw=1.0, color=PAPER["gray"]),
                ha="left", va="center",
                fontsize=8.0, color=PAPER["gray"]
            )

    ax.set_title("(a) Frame-level overall evaluation")


def draw_loiter_event_metrics(ax, event_overall_df):
    plot_df = event_overall_df.copy()
    if plot_df.empty:
        ax.axis("off")
        return

    
    method_labels = (
        plot_df["method"]
        .astype(str)
        .str.replace("status_", "", regex=False)
        .str.upper()
        .tolist()
    )

    x = np.arange(len(plot_df))
    width = 0.22

    precision_vals = plot_df["precision_event"].fillna(0).values * 100.0
    recall_vals    = plot_df["recall_event"].fillna(0).values * 100.0
    f1_vals        = plot_df["f1_event"].fillna(0).values * 100.0

    bars1 = ax.bar(
        x - width, precision_vals, width,
        label="Precision", color=PAPER["blue"], alpha=0.82
    )
    bars2 = ax.bar(
        x, recall_vals, width,
        label="Recall", color=PAPER["teal"], alpha=0.82
    )
    bars3 = ax.bar(
        x + width, f1_vals, width,
        label="F1", color=PAPER["gold"], alpha=0.86
    )

    
    for bars in [bars1, bars2, bars3]:
        for rect in bars:
            h = rect.get_height()
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                h + 1.0,
                f"{h:.2f}",
                ha="center", va="bottom",
                fontsize=8.5, color=PAPER["dark"]
            )

    ax.set_xticks(x)
    ax.set_xticklabels(method_labels)
    ax.set_ylim(0, 112)   
    ax.set_ylabel("Score (%)")
    ax.set_title("(b) Event-level metrics")
    ax.grid(True, axis="y", alpha=0.25, linestyle="--")

    
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
        frameon=True,
        fancybox=False,
        edgecolor="#B8B8B8",
        framealpha=0.95
    )

def _format_box_stats_text(vals, name="KF"):
    vals = np.asarray(vals, dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return f"{name}\nN=0"

    q1 = np.percentile(vals, 25)
    med = np.percentile(vals, 50)
    q3 = np.percentile(vals, 75)
    mean = np.mean(vals)
    vmin = np.min(vals)
    vmax = np.max(vals)

    txt = (
        f"{name}\n"
        f"N={len(vals)}\n"
        f"Mean={mean:.2f}\n"
        f"Median={med:.2f}\n"
        f"Q1={q1:.2f}\n"
        f"Q3={q3:.2f}\n"
        f"Min={vmin:.2f}\n"
        f"Max={vmax:.2f}"
    )
    return txt

def _box_with_points(ax, data, labels, colors, ylabel, title,
                     ylim=None, hline0=False, stats_text_list=None):
    if len(data) == 0:
        ax.axis("off")
        return

    box = ax.boxplot(
        data, patch_artist=True, widths=0.42, showfliers=False,
        medianprops=dict(color=PAPER["dark"], linewidth=1.8),
        whiskerprops=dict(color=PAPER["gray"], linewidth=1.2),
        capprops=dict(color=PAPER["gray"], linewidth=1.2),
        boxprops=dict(linewidth=1.0, edgecolor=PAPER["dark"]),
    )

    for patch, color in zip(box["boxes"], colors[:len(data)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
        patch.set_edgecolor(PAPER["dark"])

    for i, vals in enumerate(data, start=1):
        if len(vals) > 0:
            xs = np.linspace(i - 0.10, i + 0.10, len(vals))
            ax.scatter(xs, vals, s=20, color=PAPER["dark"], alpha=0.45, zorder=3)
            ax.scatter([i], [np.mean(vals)], marker="D", s=42, color=PAPER["coral"], zorder=4)

    if hline0:
        ax.axhline(0.0, color=PAPER["gray"], linestyle="--", linewidth=1.1)

    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25, linestyle="--")

   
    if stats_text_list is not None and len(stats_text_list) > 0:
        txt = "\n\n".join(stats_text_list)
        ax.text(
            0.03, 0.97, txt,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.30",
                      facecolor="white", edgecolor="#B8B8B8", alpha=0.92)
        )


def draw_loiter_delay_box(ax, event_match_all_df):
    vals = event_match_all_df.loc[
        event_match_all_df["method"] == "status_kf",
        "onset_delay_frames"
    ].dropna().values

    stats_text = [_format_box_stats_text(vals, name="KF onset delay stats")]

    _box_with_points(
        ax,
        data=[vals],
        labels=["KF"],
        colors=[PAPER["teal"]],
        ylabel="Delay (frames)",
        title="(c) Event onset delay (KF)",
        hline0=True,
        stats_text_list=stats_text
    )

def draw_loiter_sync_box(ax, event_match_all_df):
    vals = event_match_all_df.loc[
        event_match_all_df["method"] == "status_kf",
        "sync_iou"
    ].dropna().values

    stats_text = [_format_box_stats_text(vals, name="KF temporal IoU stats")]

    _box_with_points(
        ax,
        data=[vals],
        labels=["KF"],
        colors=[PAPER["teal"]],
        ylabel="Temporal IoU",
        title="(d) Event synchronization ratio (KF)",
        ylim=(0, 1.05),
        stats_text_list=stats_text
    )


def draw_loiter_segment_f1(ax, frame_seg_all_df):
    vals = (
        frame_seg_all_df.loc[
            frame_seg_all_df["method"] == "status_kf",
            "f1"
        ].dropna().values * 100.0
    )

    stats_text = [_format_box_stats_text(vals, name="KF segment F1 stats")]

    _box_with_points(
        ax,
        data=[vals],
        labels=["KF"],
        colors=[PAPER["teal"]],
        ylabel="Frame-level F1 (%)",
        title="(e) Segment-wise F1 distribution (KF)",
        ylim=(0, 105),
        stats_text_list=stats_text
    )


def draw_loiter_duration_error(ax, event_match_all_df):
    vals = event_match_all_df.loc[
        event_match_all_df["method"] == "status_kf",
        "duration_error_frames"
    ].dropna().values

    stats_text = [_format_box_stats_text(vals, name="KF duration error stats")]

    _box_with_points(
        ax,
        data=[vals],
        labels=["KF"],
        colors=[PAPER["teal"]],
        ylabel="Duration error (frames)",
        title="(f) Event duration error (KF)",
        hline0=True,
        stats_text_list=stats_text
    )


loiter_frame_overall_df = pd.DataFrame()
loiter_event_overall_df = pd.DataFrame()
loiter_frame_seg_df = pd.DataFrame()
loiter_event_seg_df = pd.DataFrame()
loiter_event_match_df = pd.DataFrame()

if status_gt and status_kf and status_pre and status_gt in m.columns and status_kf in m.columns and status_pre in m.columns:
    frame_overall_list, event_overall_list = [], []
    frame_seg_list, event_seg_list, event_match_list = [], [], []
    for pred_col in [status_pre, status_kf]:
        fo, eo, fs, es, em = collect_loiter_eval(m, status_gt, pred_col, pred_col)
        frame_overall_list.append(fo)
        event_overall_list.append(eo)
        frame_seg_list.append(fs)
        event_seg_list.append(es)
        event_match_list.append(em)
    loiter_frame_overall_df = pd.concat(frame_overall_list, ignore_index=True)
    loiter_event_overall_df = pd.concat(event_overall_list, ignore_index=True)
    loiter_frame_seg_df = pd.concat(frame_seg_list, ignore_index=True)
    loiter_event_seg_df = pd.concat(event_seg_list, ignore_index=True)
    loiter_event_match_df = pd.concat(event_match_list, ignore_index=True)
    if args.bootstrap_replicates > 0:
        bootstrap_event_intervals(loiter_event_seg_df,args.bootstrap_replicates,args.bootstrap_seed).to_csv(
            os.path.join(out_dir,'event_confidence_intervals.csv'),index=False)

    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    draw_loiter_frame_counts(ax, loiter_frame_overall_df)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "loiter_frame_counts.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    draw_loiter_event_metrics(ax, loiter_event_overall_df)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "loiter_event_metrics.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)

    if not loiter_event_match_df.empty:
        fig, ax = plt.subplots(figsize=(7.4, 5.6))
        draw_loiter_delay_box(ax, loiter_event_match_df)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "loiter_event_delay.png"), dpi=180, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.4, 5.6))
        draw_loiter_sync_box(ax, loiter_event_match_df)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "loiter_event_sync.png"), dpi=180, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.4, 5.6))
        draw_loiter_duration_error(ax, loiter_event_match_df)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "loiter_event_duration_error.png"), dpi=180, bbox_inches="tight")
        plt.close(fig)

    if not loiter_frame_seg_df.empty:
        fig, ax = plt.subplots(figsize=(7.4, 5.6))
        draw_loiter_segment_f1(ax, loiter_frame_seg_df)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "loiter_segment_f1_box.png"), dpi=180, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(18.0, 10.5))
    draw_loiter_frame_counts(axes[0, 0], loiter_frame_overall_df)
    draw_loiter_event_metrics(axes[0, 1], loiter_event_overall_df)
    if not loiter_event_match_df.empty:
        draw_loiter_delay_box(axes[0, 2], loiter_event_match_df)
        draw_loiter_sync_box(axes[1, 0], loiter_event_match_df)
        draw_loiter_duration_error(axes[1, 2], loiter_event_match_df)
    else:
        axes[0, 2].axis("off")
        axes[1, 0].axis("off")
        axes[1, 2].axis("off")
    if not loiter_frame_seg_df.empty:
        draw_loiter_segment_f1(axes[1, 1], loiter_frame_seg_df)
    else:
        axes[1, 1].axis("off")
    fig.suptitle("Loitering evaluation dashboard", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(os.path.join(out_dir, "loiter_eval_2x3.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    figure_guide_loiter_df = pd.DataFrame([
        {"figure": "loiter_frame_counts.png", "meaning": "TP/FP/FN/TN frame counts for status_pre and status_kf"},
        {"figure": "loiter_event_metrics.png", "meaning": "Event-level precision/recall/F1 comparison"},
        {"figure": "loiter_event_delay.png", "meaning": "Matched loitering event onset delay distribution"},
        {"figure": "loiter_event_sync.png", "meaning": "Matched loitering event temporal synchronization IoU"},
        {"figure": "loiter_event_duration_error.png", "meaning": "Matched loitering event duration error"},
        {"figure": "loiter_segment_f1_box.png", "meaning": "Segment-wise frame-level F1 distribution"},
        {"figure": "loiter_eval_2x3.png", "meaning": "Combined loitering evaluation dashboard"},
    ])
    save_block_csv(os.path.join(out_dir, "loiter_eval_summary.csv"), [
        ("figure_guide", figure_guide_loiter_df),
        ("frame_overall", loiter_frame_overall_df),
        ("event_overall", loiter_event_overall_df),
        ("frame_segment", loiter_frame_seg_df),
        ("event_segment", loiter_event_seg_df),
        ("event_matches", loiter_event_match_df),
    ])


# =========================
# 5) KF relative improvement over Pre
# =========================

def draw_loiter_kf_vs_pre_big_table(ax,
                                    frame_overall_df,
                                    event_overall_df,
                                    frame_seg_all_df,
                                    event_match_all_df):
    pre_frame = frame_overall_df.loc[frame_overall_df["method"] == "status_pre"]
    kf_frame  = frame_overall_df.loc[frame_overall_df["method"] == "status_kf"]
    pre_event = event_overall_df.loc[event_overall_df["method"] == "status_pre"]
    kf_event  = event_overall_df.loc[event_overall_df["method"] == "status_kf"]

    if pre_frame.empty or kf_frame.empty or pre_event.empty or kf_event.empty:
        ax.axis("off")
        return None

    pre_frame = pre_frame.iloc[0]
    kf_frame  = kf_frame.iloc[0]
    pre_event = pre_event.iloc[0]
    kf_event  = kf_event.iloc[0]

    rows = []

    def add_row(group, metric, pre, kf, improve):
        rows.append({
            "Group": group,
            "Metric": metric,
            "Pre": pre,
            "KF": kf,
            "Improvement": improve
        })

    # -------------------------
    # 1) Frame-level overall
    # -------------------------
    add_row("Frame", "Precision (%)",
            f"{pre_frame['precision'] * 100.0:.2f}",
            f"{kf_frame['precision'] * 100.0:.2f}",
            f"{(kf_frame['precision'] - pre_frame['precision']) * 100.0:+.2f}")

    add_row("Frame", "Recall (%)",
            f"{pre_frame['recall'] * 100.0:.2f}",
            f"{kf_frame['recall'] * 100.0:.2f}",
            f"{(kf_frame['recall'] - pre_frame['recall']) * 100.0:+.2f}")

    add_row("Frame", "F1 (%)",
            f"{pre_frame['f1'] * 100.0:.2f}",
            f"{kf_frame['f1'] * 100.0:.2f}",
            f"{(kf_frame['f1'] - pre_frame['f1']) * 100.0:+.2f}")

    add_row("Frame", "TP (frames)",
            f"{int(pre_frame['tp'])}",
            f"{int(kf_frame['tp'])}",
            f"{int(kf_frame['tp'] - pre_frame['tp']):+d}")

    add_row("Frame", "FP reduction (frames)",
            f"{int(pre_frame['fp'])}",
            f"{int(kf_frame['fp'])}",
            f"{int(pre_frame['fp'] - kf_frame['fp']):+d}")

    add_row("Frame", "FN reduction (frames)",
            f"{int(pre_frame['fn'])}",
            f"{int(kf_frame['fn'])}",
            f"{int(pre_frame['fn'] - kf_frame['fn']):+d}")

    add_row("Frame", "TN (frames)",
            f"{int(pre_frame['tn'])}",
            f"{int(kf_frame['tn'])}",
            f"{int(kf_frame['tn'] - pre_frame['tn']):+d}")

    # -------------------------
    # 2) Event-level overall
    # -------------------------
    add_row("Event", "Precision (%)",
            f"{pre_event['precision_event'] * 100.0:.2f}",
            f"{kf_event['precision_event'] * 100.0:.2f}",
            f"{(kf_event['precision_event'] - pre_event['precision_event']) * 100.0:+.2f}")

    add_row("Event", "Recall (%)",
            f"{pre_event['recall_event'] * 100.0:.2f}",
            f"{kf_event['recall_event'] * 100.0:.2f}",
            f"{(kf_event['recall_event'] - pre_event['recall_event']) * 100.0:+.2f}")

    add_row("Event", "F1 (%)",
            f"{pre_event['f1_event'] * 100.0:.2f}",
            f"{kf_event['f1_event'] * 100.0:.2f}",
            f"{(kf_event['f1_event'] - pre_event['f1_event']) * 100.0:+.2f}")

    add_row("Event", "Mean onset delay (frames)",
            f"{pre_event['mean_onset_delay']:.2f}",
            f"{kf_event['mean_onset_delay']:.2f}",
            f"{(pre_event['mean_onset_delay'] - kf_event['mean_onset_delay']):+.2f}")

    add_row("Event", "Mean sync IoU (%)",
            f"{pre_event['mean_sync_iou'] * 100.0:.2f}",
            f"{kf_event['mean_sync_iou'] * 100.0:.2f}",
            f"{(kf_event['mean_sync_iou'] - pre_event['mean_sync_iou']) * 100.0:+.2f}")

    # -------------------------
    # 3) Segment-level summary
    # -------------------------
    if not frame_seg_all_df.empty:
        pre_seg = frame_seg_all_df.loc[frame_seg_all_df["method"] == "status_pre"].copy()
        kf_seg  = frame_seg_all_df.loc[frame_seg_all_df["method"] == "status_kf"].copy()

        if not pre_seg.empty and not kf_seg.empty:
            seg = pd.merge(
                pre_seg[["segment", "precision", "recall", "f1", "fp", "fn"]],
                kf_seg[["segment", "precision", "recall", "f1", "fp", "fn"]],
                on="segment", suffixes=("_pre", "_kf")
            )
            seg["delta_f1"] = (seg["f1_kf"] - seg["f1_pre"]) * 100.0

            add_row("Segment", "Mean segment F1 (%)",
                    f"{seg['f1_pre'].mean() * 100.0:.2f}",
                    f"{seg['f1_kf'].mean() * 100.0:.2f}",
                    f"{seg['delta_f1'].mean():+.2f}")

            add_row("Segment", "Max segment F1 gain (%)",
                    "-",
                    "-",
                    f"{seg['delta_f1'].max():+.2f}")

            add_row("Segment", "Min segment F1 gain (%)",
                    "-",
                    "-",
                    f"{seg['delta_f1'].min():+.2f}")

            add_row("Segment", "Improved segments (count)",
                    "-",
                    "-",
                    f"{int((seg['delta_f1'] > 0).sum())}")

            add_row("Segment", "Unchanged segments (count)",
                    "-",
                    "-",
                    f"{int((np.isclose(seg['delta_f1'], 0)).sum())}")

    # -------------------------
    # 4) Delay-sync tradeoff summary
    # -------------------------
    if not event_match_all_df.empty:
        pre_df = event_match_all_df.loc[event_match_all_df["method"] == "status_pre"].copy()
        kf_df  = event_match_all_df.loc[event_match_all_df["method"] == "status_kf"].copy()

        if not pre_df.empty and not kf_df.empty:
            pre_ts = pre_df.groupby("segment").agg(
                mean_delay=("onset_delay_frames", "mean"),
                mean_sync=("sync_iou", "mean")
            ).reset_index()

            kf_ts = kf_df.groupby("segment").agg(
                mean_delay=("onset_delay_frames", "mean"),
                mean_sync=("sync_iou", "mean")
            ).reset_index()

            mg = pd.merge(pre_ts, kf_ts, on="segment", suffixes=("_pre", "_kf"))
            mg["delay_improve"] = mg["mean_delay_pre"] - mg["mean_delay_kf"]
            mg["sync_gain"] = (mg["mean_sync_kf"] - mg["mean_sync_pre"]) * 100.0

            add_row("Trade-off", "Mean delay improvement (frames)",
                    f"{mg['mean_delay_pre'].mean():.2f}",
                    f"{mg['mean_delay_kf'].mean():.2f}",
                    f"{mg['delay_improve'].mean():+.2f}")

            add_row("Trade-off", "Mean sync IoU gain (%)",
                    f"{mg['mean_sync_pre'].mean() * 100.0:.2f}",
                    f"{mg['mean_sync_kf'].mean() * 100.0:.2f}",
                    f"{mg['sync_gain'].mean():+.2f}")

            add_row("Trade-off", "Both improved segments (count)",
                    "-",
                    "-",
                    f"{int(((mg['delay_improve'] > 0) & (mg['sync_gain'] > 0)).sum())}")

    table_df = pd.DataFrame(rows)

    ax.axis("off")
    tbl = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc="center",
        loc="center"
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.08, 1.35)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("gray")
        cell.set_linewidth(0.65)

        if r == 0:
            cell.set_facecolor("#DCEAF7")
            cell.set_text_props(weight="bold")
        elif c == 0 and r > 0:
            group_name = cell.get_text().get_text()
            if group_name == "Frame":
                cell.set_facecolor("#EEF5FC")
            elif group_name == "Event":
                cell.set_facecolor("#FDF3E8")
            elif group_name == "Segment":
                cell.set_facecolor("#EEF8EE")
            elif group_name == "Trade-off":
                cell.set_facecolor("#F4F1FB")

        elif c == 4 and r > 0:
            txt = cell.get_text().get_text().strip()
            if txt.startswith("+"):
                cell.set_facecolor("#EAF4EA")
            elif txt.startswith("-"):
                cell.set_facecolor("#FCEAEA")

    ax.set_title("KF vs Pre overall summary table", fontsize=14, pad=12)
    return table_df


improvement_overall_df = pd.DataFrame()
improvement_seg_df = pd.DataFrame()
improvement_tradeoff_df = pd.DataFrame()
improvement_table_df = pd.DataFrame()

if not loiter_frame_overall_df.empty and not loiter_event_overall_df.empty:
    fig, ax = plt.subplots(figsize=(16.0, 10.5))
    improvement_big_table_df = draw_loiter_kf_vs_pre_big_table(
        ax,
        loiter_frame_overall_df,
        loiter_event_overall_df,
        loiter_frame_seg_df,
        loiter_event_match_df
    )
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "loiter_kf_vs_pre_big_table.png"),
                dpi=220, bbox_inches="tight")
    plt.close(fig)

    figure_guide_improve_df = pd.DataFrame([
        {
            "figure": "loiter_kf_vs_pre_big_table.png",
            "meaning": "Single large summary table for KF vs Pre comparison"
        }
    ])

    save_block_csv(os.path.join(out_dir, "loiter_kf_vs_pre_summary.csv"), [
        ("figure_guide", figure_guide_improve_df),
        ("big_table", improvement_big_table_df),
    ])


print("Saved to:", out_dir)
