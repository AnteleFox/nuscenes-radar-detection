"""D1: 2D CA-CFAR on RD maps -> hit cells -> back-map to points -> DBSCAN clusters.
Run after make_rd.py: python cfar_baseline.py
"""
import glob
import os
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.cluster import DBSCAN

OUT = 'out_d1'
PFA = 1e-3
GUARD, TRAIN = 2, 5          # cells
RANGE_STEP, DOP_STEP = 0.5, 0.5


def cfar2d(rd, guard=GUARD, train=TRAIN, pfa=PFA):
    g, t = 2 * guard + 1, 2 * train + 1
    n_train = t * t - g * g
    factor = n_train * (pfa ** (-1.0 / n_train) - 1)
    pad_t = np.pad(rd, train, mode='constant')
    pad_g = np.pad(rd, guard, mode='constant')
    total = sliding_window_view(pad_t, (t, t)).sum(axis=(2, 3))
    guard_sum = sliding_window_view(pad_g, (g, g)).sum(axis=(2, 3))
    noise = (total - guard_sum) / n_train
    return rd > noise * factor


def main():
    for rd_path in sorted(glob.glob(f'{OUT}/rd_*.npy')):
        token = os.path.basename(rd_path)[3:-4]
        rd = np.load(rd_path).astype(np.float64)
        mask = cfar2d(rd)
        ri, di = np.nonzero(mask)
        pts = np.load(f'{OUT}/pts_{token}.npz')
        rng = np.hypot(pts['x'], pts['y'])
        r_idx = np.clip((rng / RANGE_STEP).astype(int), 0, rd.shape[0] - 1)
        v_idx = np.clip(((pts['vcomp'] + 20) / DOP_STEP).astype(int), 0, rd.shape[1] - 1)
        hit = mask[r_idx, v_idx]
        xy = np.stack([pts['x'][hit], pts['y'][hit]], axis=1)
        n_obj = 0
        centers = np.empty((0, 2))
        if len(xy) >= 3:
            labels = DBSCAN(eps=1.5, min_samples=3).fit(xy).labels_
            keep = [l for l in set(labels) if l != -1]
            n_obj = len(keep)
            if keep:
                centers = np.stack([xy[labels == l].mean(axis=0) for l in keep])
        np.savez_compressed(f'{OUT}/det_{token}.npz', centers=centers,
                            n_hit=int(hit.sum()), n_pts=len(xy))
        print(token[:8], 'cfar_cells:', int(mask.sum()), 'hit_pts:', len(xy), 'objects:', n_obj)


if __name__ == '__main__':
    main()
