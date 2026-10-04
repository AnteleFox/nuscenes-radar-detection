"""D2: project multi-frame detections onto CAM_FRONT for 3 informative sweeps.
Green boxes = matched GT, red circles = DBSCAN cluster centers, gray dots = accumulated
CFAR hit points (1s window). Run after: python multiframe.py --windows 1.0 --min-samples 2 --save-det
Then: python vis_d2.py
"""
import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes

DATAROOT = os.environ.get('NUSCENES_ROOT', os.path.expanduser('~/fyc~1/nuscenes'))
OUT = 'out_d2'
VIS_CLASSES = {'car', 'truck', 'bus', 'trailer', 'construction_vehicle',
               'pedestrian', 'motorcycle', 'bicycle'}
MATCH_R, MATCH_AZ = 2.0, np.deg2rad(5.0)

nusc = NuScenes(version='v1.0-mini', dataroot=DATAROOT, verbose=False)

cands = []
for p in sorted(glob.glob(f'{OUT}/det2_*.npz')):
    d = np.load(p)
    if len(d['centers']) == 0:
        continue
    tk = os.path.basename(p)[5:-4]
    sd = nusc.get('sample_data', tk)
    ep = nusc.get('ego_pose', sd['ego_pose_token'])
    cands.append((len(d['centers']), tk, sd, ep))
cands.sort(key=lambda c: -c[0])
picks = [cands[0]]
for c in cands[1:]:
    if all(abs(c[2]['timestamp'] - p[2]['timestamp']) > 3e6 for p in picks):
        picks.append(c)
    if len(picks) == 3:
        break

for k, (_, tk, sd, ep) in enumerate(picks):
    d = np.load(f'{OUT}/det2_{tk}.npz')
    centers, px, py = d['centers'], d['pts_x'], d['pts_y']
    sp, sq = np.array(ep['translation']), Quaternion(ep['rotation'])

    # anchor sample: nearest key frame by CAM_FRONT timestamp (sweeps are not key frames)
    near = min(nusc.sample, key=lambda s: abs(
        nusc.get('sample_data', s['data']['CAM_FRONT'])['timestamp'] - sd['timestamp']))
    cam = nusc.get('sample_data', near['data']['CAM_FRONT'])
    lsd = nusc.get('sample_data', near['data']['LIDAR_TOP'])
    rep = nusc.get('ego_pose', lsd['ego_pose_token'])
    strans, squat = np.array(rep['translation']), Quaternion(rep['rotation'])
    csd = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
    K = np.array(csd['camera_intrinsic'])
    R_ce, t_ce = Quaternion(csd['rotation']).rotation_matrix, np.array(csd['translation'])

    def to_sample(x, y):
        """sweep ego frame -> anchor-sample ego frame (same rigid transform as eval)."""
        p3 = np.stack([x, y, np.zeros_like(x)], axis=1)
        g = p3 @ sq.rotation_matrix.T + sp
        return (g - strans) @ squat.rotation_matrix  # R_s^{-1} = R_s.T

    def proj(p_ego, w, h):
        """sample-ego frame -> CAM_FRONT pixels; keeps z>0.5m and in-image points.
        Row-vector form of p_sensor = R^T (p_ego - t) is (p_ego - t) @ R."""
        p_s = (p_ego - t_ce) @ R_ce
        uv = p_s @ K.T
        ok = p_s[:, 2] > 0.5
        u, v = uv[ok, 0] / uv[ok, 2], uv[ok, 1] / uv[ok, 2]
        keep = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        return u[keep], v[keep]

    # match centers against visible GT (replicates multiframe eval logic)
    gt = []
    for a in near['anns']:
        ann = nusc.get('sample_annotation', a)
        if ann['category_name'].split('.')[-1] not in VIS_CLASSES:
            continue
        c = squat.inverse.rotate(np.array(ann['translation']) - strans)
        r, az = np.hypot(c[0], c[1]), np.arctan2(c[1], c[0])
        if r < 100 and abs(az) < np.deg2rad(60):
            gt.append((r, az, ann))
    matched_anns, used = [], set()
    cs = to_sample(centers[:, 0], centers[:, 1])
    for u, v, _ in cs:
        r, az = np.hypot(u, v), np.arctan2(v, u)
        for gi, (gr, gaz, ann) in enumerate(gt):
            if gi in used:
                continue
            if abs(r - gr) < MATCH_R and abs(((az - gaz + np.pi) % (2 * np.pi)) - np.pi) < MATCH_AZ:
                matched_anns.append(ann)
                used.add(gi)
                break

    img = Image.open(os.path.join(DATAROOT, cam['filename']))
    W, H = img.size
    fig, ax = plt.subplots(figsize=(12, 6.6))
    ax.imshow(img)
    ux, uy = proj(to_sample(px, py), W, H)
    ax.scatter(ux, uy, s=3, c='gray', alpha=0.6, label='accumulated CFAR hit points')
    ucx, ucy = proj(cs, W, H)
    ax.scatter(ucx, ucy, s=90, facecolors='none', edgecolors='red', linewidths=1.6,
               label=f'cluster centers ({len(ucx)})')
    for ann in matched_anns:
        # annotation size = [w, l, h]; box x-axis = length, y-axis = width
        c3, R_b = np.array(ann['translation']), Quaternion(ann['rotation']).rotation_matrix
        w, l, h = ann['size']
        corners = np.stack([c3 + R_b @ np.array([dx * l / 2, dy * w / 2, dz * h / 2])
                            for dx in (-1, 1) for dy in (-1, 1) for dz in (-1, 1)])
        corners = (corners - strans) @ squat.rotation_matrix   # global -> sample ego
        p_s = (corners - t_ce) @ R_ce
        uv = p_s @ K.T
        z = p_s[:, 2] > 0.5
        uu, vv = uv[z, 0] / uv[z, 2], uv[z, 1] / uv[z, 2]
        inb = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
        if inb.sum() >= 4:
            uu, vv = uu[inb], vv[inb]
            ax.add_patch(Rectangle((uu.min(), vv.min()), uu.max() - uu.min(), vv.max() - vv.min(),
                                   fill=False, edgecolor='lime', linewidth=1.4))
    ax.set_title(f'sweep {tk[:8]} | window 1.0s | det={len(cs)} matched={len(matched_anns)}')
    ax.legend(loc='upper right', fontsize=8)
    ax.axis('off')
    fig.savefig(f'{OUT}/vis2_{k}.png', dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f'vis2_{k}.png', tk[:8], 'det', len(cs), 'matched', len(matched_anns))
print('done')
