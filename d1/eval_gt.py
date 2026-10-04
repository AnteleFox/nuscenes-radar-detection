"""D1: match CFAR+DBSCAN objects against nuScenes GT -> Pd / false alarms per sweep.
Approximation (note in README): GT visibility filtered geometrically
(range<100m, |az|<60deg, vehicle/pedestrian classes); ego pose taken at nearest
LIDAR_TOP sample (<~40ms from sweep).
Run after cfar_baseline.py: python eval_gt.py
"""
import glob
import os
import csv
import numpy as np
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes

DATAROOT = os.environ.get('NUSCENES_ROOT', os.path.expanduser('~/fyc~1/nuscenes'))
OUT = 'out_d1'
VIS_CLASSES = {'car', 'truck', 'bus', 'trailer', 'construction_vehicle',
               'pedestrian', 'motorcycle', 'bicycle'}
MATCH_R, MATCH_AZ = 2.0, np.deg2rad(5.0)

nusc = NuScenes(version='v1.0-mini', dataroot=DATAROOT, verbose=False)

# sweep token -> (timestamp, nearest sample ego pose)
samples = []
for s in nusc.sample:
    sd = nusc.get('sample_data', s['data']['LIDAR_TOP'])
    ep = nusc.get('ego_pose', sd['ego_pose_token'])
    samples.append((sd['timestamp'], ep['translation'], ep['rotation'], s['anns']))

rows = []
tot_gt = tot_match = tot_fa = 0
for det_path in sorted(glob.glob(f'{OUT}/det_*.npz')):
    token = os.path.basename(det_path)[4:-4]
    pts_meta = np.load(f'{OUT}/pts_{token}.npz')
    ts = int(pts_meta['ts'])
    t_s, trans, rot, anns = min(samples, key=lambda s: abs(s[0] - ts))
    q = Quaternion(rot).inverse
    centers = np.load(det_path)['centers']

    gt = []
    for a in anns:
        ann = nusc.get('sample_annotation', a)
        cls = ann['category_name'].split('.')[-1]
        if cls not in VIS_CLASSES:
            continue
        c = np.array(ann['translation']) - np.array(trans)
        c = q.rotate(c)
        r = np.hypot(c[0], c[1])
        az = np.arctan2(c[1], c[0])
        if r < 100 and abs(az) < np.deg2rad(60):
            gt.append((r, az))

    matched = 0
    used = set()
    for cx, cy in centers:
        r = np.hypot(cx, cy)
        az = np.arctan2(cy, cx)
        for i, (gr, gaz) in enumerate(gt):
            if i in used:
                continue
            if abs(r - gr) < MATCH_R and abs(((az - gaz + np.pi) % (2 * np.pi)) - np.pi) < MATCH_AZ:
                matched += 1
                used.add(i)
                break
    fa = len(centers) - matched
    tot_gt += len(gt)
    tot_match += matched
    tot_fa += fa
    rows.append([token, len(gt), len(centers), matched, fa])

with open(f'{OUT}/metrics.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['sweep_token', 'n_gt', 'n_det', 'matched', 'false_alarms'])
    w.writerows(rows)

pd = tot_match / max(tot_gt, 1)
print(f'sweeps={len(rows)} gt={tot_gt} matched={tot_match} Pd={pd:.3f} '
      f'FA_per_sweep={tot_fa / max(len(rows), 1):.2f}')
