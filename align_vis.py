"""D1: radar-camera alignment check — project RADAR_FRONT points into CAM_FRONT.
Run: python align_vis.py
"""
import os
import matplotlib
matplotlib.use('Agg')
from nuscenes.nuscenes import NuScenes

DATAROOT = os.environ.get('NUSCENES_ROOT', os.path.expanduser('~/fyc~1/nuscenes'))
OUT = 'out_d1'
os.makedirs(OUT, exist_ok=True)

nusc = NuScenes(version='v1.0-mini', dataroot=DATAROOT, verbose=False)
scene = nusc.scene[0]
s_token = scene['first_sample_token']

shots = 0
while s_token and shots < 3:
    s = nusc.get('sample', s_token)
    out = f'{OUT}/align_{shots}.png'
    nusc.render_pointcloud_in_image(s['token'], dot_size=4,
                                    pointsensor_channel='RADAR_FRONT',
                                    camera_channel='CAM_FRONT', out_path=out,
                                    render_intensity=False, verbose=False)
    shots += 1
    s_token = s['next']
print('alignment images:', shots)
