# nuScenes 雷达目标检测：单帧 CFAR 基线 → 多帧累积

面向公开数据集雷达感知项目。在 nuScenes v1.0-mini 上搭一条完整的
毫米波雷达目标检测流水线：原始雷达点云 → RD 图 → CFAR 检测 → 聚类 → 与官方 3D 真值对比出 Pd / 虚警率。

选公开数据集的原因：可写进简历、可开源、面试可细讲，不涉及任何保密数据。

## 环境
```bash
pip install nuscenes-devkit scikit-learn matplotlib pyquaternion
# 数据: v1.0-mini (~4GB)。官方 CloudFront 免登录直链:
#   https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-mini.tgz
# 解压后结构: $NUSCENES_ROOT/{v1.0-mini,samples,sweeps,maps}/
export NUSCENES_ROOT=/path/to/nuscenes   # 不设则默认 ~/fyc~1/nuscenes
```

## 运行顺序
```bash
python make_rd.py        # sweeps -> ego系点云 -> RD图 npy + 点云 npz + 预览 png
python align_vis.py      # 雷达点投影 CAM_FRONT, 3 张对齐 png (标定自检)
python cfar_baseline.py  # 2D CA-CFAR(Pfa=1e-3) -> 点反映射 -> DBSCAN 聚类
python eval_gt.py        # GT 匹配 -> out_d1/metrics.csv + Pd / 每sweep虚警
```

## Day-1 结果（基线，scene-0061，247 sweeps）
| 指标 | 值 |
|---|---|
| Pd（检出率） | 0.020 |
| 虚警 / sweep | 0.45 |
| GT 目标数 | 2125 |
| 匹配成功 | 43 |

基线弱的根因（已定位，非代码 bug）：单帧雷达点过稀疏（每帧 CFAR 命中仅 40–60 点），
DBSCAN 聚不出簇，召回上限 ~7%。雷达-相机对齐图与 RD 图均验证数据链路正确。

### 本日修复的实现 bug
1. `scene` 无 `first_sample_data_token` 字段 → 改由 `first_sample_token`→sample→`data['RADAR_FRONT']` 取起始 sweep。
2. 雷达点云默认在**传感器系**，而 GT 在 **ego 系** → make_rd 中用 calibrated_sensor 外参变换到 ego 系（修复后 matched 5→43）。
3. `det_<token>.npz` 前缀为 4 字符，切 token 误用 `[3:-4]` → `[4:-4]`。
4. DBSCAN 全噪声时 `np.stack([])` 崩溃 → 加非空保护。

## 验收清单
- [x] `out_d1/rd_*.npy` = 247（scene-0061 RADAR_FRONT sweep 数）
- [x] `rd_preview_*.png`：RD 图可见零多普勒杂波脊 + 运动目标回波
- [x] `align_*.png`：雷达点落在前方车体/水马位置
- [x] `metrics.csv` 247 行；Pd / FA 可复现

## 已知近似（面试被问=加分点，主动说）
1. radar pcd 字段索引: x,y,z=0,1,2; rcs=5; vel_comp=8（devkit 版本差异时打印 pc.points.shape 对照）。
2. GT 匹配用最近 LIDAR_TOP sample 的 ego pose 近似 sweep 时刻（<~40ms 自运动误差）。
3. "雷达可见 GT" 无官方定义：几何过滤 range<100m, |az|<60°, 车辆/行人类——业界惯例近似。
4. RD 图由点云 (range, Doppler) 直方生成（nuScenes 不提供原始 ADC 数据，真 RD 谱不可得）。

## 下一步（D2）：多帧累积
单帧稀疏是召回瓶颈。改为滑动窗口累积：取连续 N 帧（如 5），用各帧 ego_pose 补偿自车运动、
统一变换到当前帧 ego 系后叠加再聚类，点密度 ×N，使簇可聚、Pd 上台阶；与单帧基线同表对比。
