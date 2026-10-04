# D2：多帧累积检测 + 完整实验表

D1 单帧 CFAR 基线 Pd=0.017（修正后，见下），瓶颈是单帧点稀疏。D2 把滑动窗口内
连续 sweep 的点用各自 ego_pose 做**精确刚体变换**补偿自车运动、统一到当前帧 ego 系后
叠加，再走 CFAR → DBSCAN → GT 评测，扫窗口/聚类/门限/杂波抑制四个维度出对比表。

数据：nuScenes v1.0-mini scene-0061，247 sweeps（19.1 s）。GT 评测体系与 D1 完全一致
（最近 LIDAR_TOP sample、几何可见性过滤、range±2 m / az±5° 匹配），数字可直接同表对比。

## 运行
```bash
# 服务器 a6000，env fyc_dl，工作目录 ~/fyc~1/nuscenes/d1/
python multiframe.py --windows 0,0.25,0.5,1.0        # 窗口扫描（表A 的 ms=3 列）
python multiframe.py --windows 0.5,1.0 --min-samples 2   # 表A 的 ms=2 列
python multiframe.py --windows 1.0 --min-samples 2 --pfa 1e-4   # 表B：换 pfa 重复
python multiframe.py --windows 1.0 --min-samples 2 --notch 0.5  # 表C：静止杂波凹口
python multiframe.py --windows 1.0 --min-samples 2 --save-det   # 存 det2_*.npz 供可视化
python vis_d2.py     # 最优配置检测投影到 CAM_FRONT，3 张 vis2_*.png
```
产物：`out_d2/metrics_w*.csv`（逐 sweep）、`out_d2/summary.csv`（最后一次运行）、
`out_d2/vis2_*.png`。

## 实验表（2026-10-02，a6000/fyc_dl）

表A 窗口 × 聚类（Pfa=1e-3, eps=1.5）

| 配置 | Pd | FA/sweep | 命中点/sweep |
|---|---|---|---|
| w=0（单帧，=D1 基线）, ms=3 | 0.017 | 0.47 | 52 |
| w=0.5s, ms=3 | 0.358 | 31.2 | 253 |
| w=0.5s, ms=2 | 0.440 | 46.8 | 253 |
| w=1.0s, ms=3 | 0.424 | 47.6 | 479 |
| w=1.0s, ms=2（最优） | **0.481** | 65.1 | 479 |

表B CFAR 门限（w=1.0s, ms=2）

| Pfa | Pd | FA/sweep |
|---|---|---|
| 1e-4 | 0.464 | 59.8 |
| 3e-4 | 0.474 | 62.4 |
| 1e-3 | 0.481 | 65.1 |
| 3e-3 | 0.483 | 67.7 |

表C 静止杂波凹口（w=1.0s, ms=2, Pfa=1e-3）

| notch (m/s) | Pd | FA/sweep |
|---|---|---|
| 0（不滤） | 0.481 | 65.1 |
| 0.5 | 0.072 | 4.3 |
| 1.0 | 0.072 | 4.2 |

## 结论（组会可直接讲）
1. **多帧累积是主杠杆**：Pd 0.017 → 0.481（×28），机制是点密度 ×N 使 DBSCAN 可聚；
   代价是虚警同量级上升（0.47 → 65/sweep），Pd-FA 联合上移，不是白赚。
2. **CFAR 门限不是杠杆**：Pfa 跨两个数量级，Pd 仅 0.464→0.483；虚警源是多径/护栏等
   real returns，调门限压不住。
3. **零多普勒凹口是失败杠杆**：滤 |v|<0.5 m/s 后 FA 崩到 4.3 但 Pd 同步崩到 0.072——
   本场景 GT 车辆多为慢速/静止，补偿后多普勒近零，与静止杂波不可分。
   下一个真杠杆是**簇级特征**（RCS 和、速度方差、簇尺寸）与雷达-视觉融合（D3）。

## 对 D1 数字的修正（诚实记录）
D1 eval 用 `translation - rotation` 近似传感器全局位置，坐标系不自洽；D2 改为在最近
sample ego 系做精确刚体变换后重跑 w=0，**D1 基线修正为 Pd=0.017 / FA 0.47**（原报
0.020 / 0.45，差异全部来自匹配坐标系，检测端未变）。后续一切对比以 0.017 为准。

## 已知近似（面试/组会被问=主动说）
1. 继承 D1：GT 取最近 LIDAR_TOP sample（<~40 ms 自运动差）；"雷达可见 GT" 为几何过滤
   （range<100 m, |az|<60°, 车辆/行人类）；RD 图由点云 (range, Doppler) 直方生成
   （nuScenes 无原始 ADC 数据）。
2. D2 新增：累积窗口内目标自运动未补偿（只补自车），1 s 窗口下慢速目标误差 <~1 m，
   与匹配门限同量级；可视化用最近 key frame 的 CAM_FRONT（sweep 非 key frame）。

## 深度网络一栏的状态
原计划"RD 图检测网络复现"需要可训练的公开数据量；v1.0-mini 单场景 19 s 训出的任何
数字都不可信，**如实标"数据不足，顺延"**，不造假数字。trainval 下载已评估并放弃
（共享服务器磁盘余量不足）。网络复现等后续数据/算力条件具备再做，本表先以经典链路
完整实验表收口。
