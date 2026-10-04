# nuScenes 毫米波雷达目标检测：CFAR 基线 → 多帧累积

公开数据集（nuScenes v1.0-mini）上的雷达感知评测流水线：原始雷达点云 → RD 图 →
2D CA-CFAR → DBSCAN 聚类 → 与官方 3D 真值匹配出 Pd / 虚警率；多帧累积（ego 系精确
刚体补偿）把 Pd 从 0.017 提升到 0.481。

## 目录
- `d1/` 单帧基线：`make_rd.py`（RD 图）、`cfar_baseline.py`（CFAR+DBSCAN）、
  `eval_gt.py`（GT 评测）、`align_vis.py`（雷达-相机对齐自检）、`README.md`
- `d2/` 多帧累积：`multiframe.py`（窗口/门限/聚类/杂波凹口全参数扫描）、
  `vis_d2.py`（检测投影可视化）、实验表与结论见 `d2/README.md`、
  组会汇报稿 `d2/组会一页纸.md`

## 快速开始
```bash
pip install nuscenes-devkit scikit-learn matplotlib pyquaternion pillow
# 下载 v1.0-mini（官方免登录直链，~4GB）解压到 $NUSCENES_ROOT
cd d1 && python make_rd.py && python cfar_baseline.py && python eval_gt.py
cd ../d2 && python multiframe.py --windows 0,0.5,1.0 --min-samples 2
```

## 主结果（scene-0061, 247 sweeps）

| 配置 | Pd | 虚警/sweep |
|---|---|---|
| 单帧 CFAR+DBSCAN | 0.017 | 0.47 |
| 多帧累积 1.0 s + DBSCAN(min=2) | 0.481 | 65.1 |
| 1.0 s + 零多普勒凹口 | 0.072 | 4.2 |

结论与已知近似（GT 可见性过滤、最近 sample 位姿、点云直方 RD 图等）见 `d2/README.md`。
所有数字均为本仓库脚本在 v1.0-mini 上的实测值，可复现。
