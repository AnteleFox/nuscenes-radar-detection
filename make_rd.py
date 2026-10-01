"""D1: nuScenes RADAR_FRONT sweeps -> range-Doppler maps + point clouds.
Run: python make_rd.py   (env: NUSCENES_ROOT points at dataroot containing v1.0-mini/)
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud

DATAROOT = os.environ.get('NUSCENES_ROOT', os.path.expanduser('~/fyc~1/nuscenes'))
OUT = 'out_d1'
os.makedirs(OUT, exist_ok=True)

nusc = NuScenes(version='v1.0-mini', dataroot=DATAROOT, verbose=True)
scene = nusc.scene[0]
first_sample = nusc.get('sample', scene['first_sample_token'])
sd_token = first_sample['data']['RADAR_FRONT']

RANGE_BINS = np.arange(0, 100.5, 0.5)   # 0-100 m, 0.5 m
DOP_BINS = np.arange(-20, 20.5, 0.5)    # +/-20 m/s, 0.5 m/s

# radar pcd field order: x,y,z=0,1,2 | dyn_prop=3 | id=4 | rcs=5 |
# vel_comp_x,y=6,7 | vel_comp=8
n_sweeps = 0
while sd_token:
    sd = nusc.get('sample_data', sd_token)
    if sd['channel'] == 'RADAR_FRONT':
        pc = RadarPointCloud.from_file(os.path.join(DATAROOT, sd['filename']))
        p = pc.points
        assert p.shape[0] >= 12, p.shape
        cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
        R = Quaternion(cs['rotation']).rotation_matrix
        xyz_ego = R @ p[:3] + np.array(cs['translation'])[:, None]
        x, y = xyz_ego[0], xyz_ego[1]
        rcs, vcomp = p[5], p[8]
        rng = np.hypot(x, y)
        rd, _, _ = np.histogram2d(rng, vcomp, bins=[RANGE_BINS, DOP_BINS],
                                  weights=10 ** (rcs / 10.0))
        np.save(f'{OUT}/rd_{sd["token"]}.npy', rd.astype(np.float32))
        np.savez_compressed(f'{OUT}/pts_{sd["token"]}.npz',
                            x=x, y=y, rcs=rcs, vcomp=vcomp, ts=sd['timestamp'])
        n_sweeps += 1
        if n_sweeps <= 5:
            plt.figure(figsize=(6, 4))
            plt.imshow(np.log1p(rd.T), origin='lower', aspect='auto', cmap='magma',
                       extent=[RANGE_BINS[0], RANGE_BINS[-1], DOP_BINS[0], DOP_BINS[-1]])
            plt.xlabel('range (m)')
            plt.ylabel('doppler (m/s)')
            plt.title(f'RD map #{n_sweeps}')
            plt.savefig(f'{OUT}/rd_preview_{n_sweeps}.png', dpi=100, bbox_inches='tight')
            plt.close()
    sd_token = sd['next']

print('scene:', scene['name'], '| radar sweeps:', n_sweeps)
