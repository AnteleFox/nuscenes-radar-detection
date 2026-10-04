"""D2: multi-frame accumulation -> CFAR on accumulated RD map -> DBSCAN -> GT eval.
All matching done in the nearest LIDAR_TOP sample's ego frame with exact rigid transforms
(sweep poses and sample poses both applied); fixes D1's translation-minus-rotation offset
approximation. Reuses cfar2d from cfar_baseline.py. Writes out_d2/ (per-window CSVs + summary).
Run: python multiframe.py [--windows 0,0.25,0.5,1.0] [--pfa 1e-3] [--eps 1.5] [--min-samples 3]
"""
import os
import csv
import glob
import argparse
import numpy as np
from sklearn.cluster import DBSCAN
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from cfar_baseline import cfar2d, RANGE_STEP, DOP_STEP

DATAROOT = os.environ.get('NUSCENES_ROOT', os.path.expanduser('~/fyc~1/nuscenes'))
D1, OUT = 'out_d1', 'out_d2'
os.makedirs(OUT, exist_ok=True)
VIS_CLASSES = {'car', 'truck', 'bus', 'trailer', 'construction_vehicle',
               'pedestrian', 'motorcycle', 'bicycle'}
MATCH_R, MATCH_AZ = 2.0, np.deg2rad(5.0)
RANGE_BINS = np.arange(0, 100.5, RANGE_STEP)
DOP_BINS = np.arange(-20, 20.5, DOP_STEP)

ap = argparse.ArgumentParser()
ap.add_argument('--windows', default='0,0.25,0.5,1.0',
                help='accumulation windows in seconds before current sweep (comma sep)')
ap.add_argument('--pfa', type=float, default=1e-3)
ap.add_argument('--eps', type=float, default=1.5)
ap.add_argument('--min-samples', type=int, default=3)
ap.add_argument('--notch', type=float, default=0.0,
                help='if >0, drop hit points with |vcomp| < notch (static-clutter notch, m/s)')
ap.add_argument('--save-det', action='store_true',
                help='save per-sweep det2_*.npz (centers + accumulated points, sweep-i ego frame)')
args = ap.parse_args()

nusc = NuScenes(version='v1.0-mini', dataroot=DATAROOT, verbose=False)

# sweeps sorted by timestamp; x/y/rcs/vcomp are per-sweep ego-frame values from D1
tokens, T, X, Y, RCS, VC, SP, SQ = [], [], [], [], [], [], [], []
for p in sorted(glob.glob(f'{D1}/pts_*.npz')):
    d = np.load(p)
    tk = os.path.basename(p)[4:-4]
    sd = nusc.get('sample_data', tk)
    ep = nusc.get('ego_pose', sd['ego_pose_token'])
    tokens.append(tk)
    T.append(int(d['ts']))
    X.append(d['x'].astype(np.float64))
    Y.append(d['y'].astype(np.float64))
    RCS.append(d['rcs'].astype(np.float64))
    VC.append(d['vcomp'].astype(np.float64))
    SP.append(np.array(ep['translation']))
    SQ.append(Quaternion(ep['rotation']))
order = np.argsort(T)
tokens = [tokens[i] for i in order]
X, Y, RCS, VC = [X[i] for i in order], [Y[i] for i in order], [RCS[i] for i in order], [VC[i] for i in order]
SP, SQ = [SP[i] for i in order], [SQ[i] for i in order]
T = np.array([T[i] for i in order])
N = len(tokens)
print(f'loaded {N} sweeps, span {(T[-1] - T[0]) / 1e6:.1f} s')

# LIDAR_TOP samples (GT reference frames), same nearest-sample approximation as D1
samples = []
for s in nusc.sample:
    sd = nusc.get('sample_data', s['data']['LIDAR_TOP'])
    ep = nusc.get('ego_pose', sd['ego_pose_token'])
    samples.append((sd['timestamp'], np.array(ep['translation']), Quaternion(ep['rotation']), s['anns']))
samples.sort(key=lambda s: s[0])
ST = np.array([s[0] for s in samples])


def accum_points(i, window_s):
    """points from sweeps in [T[i]-window, T[i]] transformed into ego frame of sweep i.
    Full rigid transform via global frame; motion within window compensated exactly."""
    lo = int(np.searchsorted(T, T[i] - window_s * 1e6, side='left'))
    xs, ys = [X[i]], [Y[i]]
    Ri_inv = SQ[i].inverse.rotation_matrix
    for j in range(lo, i):
        pts = np.stack([X[j], Y[j], np.zeros_like(X[j])], axis=1)
        loc = (pts @ SQ[j].rotation_matrix.T + SP[j] - SP[i]) @ Ri_inv.T
        xs.append(loc[:, 0])
        ys.append(loc[:, 1])
    return (np.concatenate(xs), np.concatenate(ys),
            np.concatenate([RCS[j] for j in range(lo, i + 1)]),
            np.concatenate([VC[j] for j in range(lo, i + 1)]))


def detect(x, y, rcs, vcomp):
    rd, _, _ = np.histogram2d(np.hypot(x, y), vcomp, bins=[RANGE_BINS, DOP_BINS],
                              weights=10 ** (rcs / 10.0))
    mask = cfar2d(rd.astype(np.float64), pfa=args.pfa)
    rng = np.hypot(x, y)
    r_idx = np.clip((rng / RANGE_STEP).astype(int), 0, rd.shape[0] - 1)
    v_idx = np.clip(((vcomp + 20) / DOP_STEP).astype(int), 0, rd.shape[1] - 1)
    hit = mask[r_idx, v_idx]
    if args.notch > 0:
        hit &= np.abs(vcomp) >= args.notch
    centers = np.empty((0, 2))
    xy = np.stack([x[hit], y[hit]], axis=1)
    if len(xy) >= args.min_samples:
        labels = DBSCAN(eps=args.eps, min_samples=args.min_samples).fit(xy).labels_
        keep = [l for l in set(labels) if l != -1]
        if keep:
            centers = np.stack([xy[labels == l].mean(axis=0) for l in keep])
    return centers, int(hit.sum())


def eval_window(window_s, save_det=False):
    rows, tot_gt = [], 0
    tot_match = tot_fa = tot_det = tot_hit = 0
    for i in range(N):
        x, y, rcs, vc = accum_points(i, window_s)
        centers, n_hit = detect(x, y, rcs, vc)
        if save_det:
            np.savez_compressed(f'{OUT}/det2_{tokens[i]}.npz', centers=centers,
                                pts_x=x, pts_y=y, ts=T[i])
        si = int(np.argmin(np.abs(ST - T[i])))
        ts, trans, quat, anns = samples[si]
        # detections: sweep-i ego frame -> sample ego frame
        if len(centers):
            c3 = np.stack([centers[:, 0], centers[:, 1], np.zeros(len(centers))], axis=1)
            g = c3 @ SQ[i].rotation_matrix.T + SP[i]
            loc = (g - trans) @ quat.inverse.rotation_matrix.T
            det_polar = [(np.hypot(u, v), np.arctan2(v, u)) for u, v in loc[:, :2]]
        else:
            det_polar = []
        gt = []
        for a in anns:
            ann = nusc.get('sample_annotation', a)
            cls = ann['category_name'].split('.')[-1]
            if cls not in VIS_CLASSES:
                continue
            c = quat.inverse.rotate(np.array(ann['translation']) - trans)
            r, az = np.hypot(c[0], c[1]), np.arctan2(c[1], c[0])
            if r < 100 and abs(az) < np.deg2rad(60):
                gt.append((r, az))
        matched, used = 0, set()
        for r, az in det_polar:
            for gi, (gr, gaz) in enumerate(gt):
                if gi in used:
                    continue
                if abs(r - gr) < MATCH_R and abs(((az - gaz + np.pi) % (2 * np.pi)) - np.pi) < MATCH_AZ:
                    matched += 1
                    used.add(gi)
                    break
        fa = len(det_polar) - matched
        tot_gt += len(gt); tot_match += matched; tot_fa += fa
        tot_det += len(det_polar); tot_hit += n_hit
        rows.append([tokens[i], len(gt), len(det_polar), matched, fa, n_hit])
    tag = f'w{window_s:g}'.replace('.', 'p')
    with open(f'{OUT}/metrics_{tag}.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['sweep_token', 'n_gt', 'n_det', 'matched', 'false_alarms', 'n_hit_pts'])
        w.writerows(rows)
    pd = tot_match / max(tot_gt, 1)
    return dict(window_s=window_s, gt=tot_gt, det=tot_det, matched=tot_match,
                pd=pd, fa_per_sweep=tot_fa / N, hit_pts_per_sweep=tot_hit / N)


summary = []
windows = [float(v) for v in args.windows.split(',')]
for k, ws in enumerate(windows):
    s = eval_window(ws, save_det=args.save_det and k == len(windows) - 1)
    summary.append(s)
    print(f"[w={s['window_s']:>4g}s] Pd={s['pd']:.3f} FA/sweep={s['fa_per_sweep']:.2f} "
          f"det/sweep={s['det'] / N:.2f} hit_pts/sweep={s['hit_pts_per_sweep']:.0f}")

with open(f'{OUT}/summary.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['window_s', 'n_gt', 'n_det', 'matched', 'Pd', 'FA_per_sweep', 'hit_pts_per_sweep'])
    for s in summary:
        w.writerow([s['window_s'], s['gt'], s['det'], s['matched'],
                    f"{s['pd']:.4f}", f"{s['fa_per_sweep']:.3f}", f"{s['hit_pts_per_sweep']:.1f}"])
print(f"summary -> {OUT}/summary.csv")
