# random LCFS + psi 散点图脚本与数据（复现用，2026-09-03 补充）

> 论文图脚本与所需数据。LCFS 构造流程见 progress2.md §52.4；psi 网格点采样散点口径见 progress2.md §13。
> run 名 = 归档统一新名（旧名对照 = ../EXPERIMENTS.md 附录 A）。

## 目录结构

```
plot/
├── scripts/
│   ├── plot_lcfs_random.py    # LCFS 双臂图（random split：上排 scratch、下排 ft w150，shot 17833）
│   ├── plot_lcfs_ip.py        # 单模型 LCFS 版（风格基准，lcfs_extract 同源）
│   ├── plot_psi_grid_points.py        # 【2026-09-03】psi 网格点采样散点（锁定风格，见 §psi 散点）
│   ├── plot_psi_scatter_combined.py   # 【2026-09-03】psi 逐样本均值散点（锁定风格）
│   ├── plot_jtor_ip_combined.py       # 【2026-09-03】j_tor–Ip 画图（自包含，读 data/jtor_ip_pairs.npz）
│   ├── compute_jtor_pairs.py          # 【2026-09-03】j_tor–Ip 数组计算（需全量 shot zarr）
│   └── common.py              # 公共配置（旧 LCFS 脚本用；路径按需调整，见"已知限制"）
├── data/
│   ├── raw/mast/17833.zarr    # LCFS 图 shot（M9 test 炮）完整 zarr + machine/17833/ 元件
│   ├── preds/                 # 【生成产物】4 个 run 的预测 npz（约 2.4G，见 §psi 散点）
│   └── jtor_ip_pairs.npz      # 【生成产物】j_tor 预计算数组（1.4MB，见 §psi 散点）
└── out/                       # 【生成产物】psi / j_tor 散点输出图
```

## 脚本需要的 zarr 数据（键路径，zarr v3，均含于 data/raw/mast/<shot>.zarr）

| 用途 | 键 | 内容 |
|---|---|---|
| EFIT LCFS（蓝实线，ground truth） | `equilibrium/lcfs_r`、`equilibrium/lcfs_z` | 每帧边界点（141/145 点/帧），列 = 平衡帧数 |
| 帧时间 | `equilibrium/time` | 平衡帧时间戳，取与样本 target_time 最近帧 |
| Ip(t)（底排曲线） | `magnetics/ip`、`magnetics/time` | 罗氏线圈总电流（A）|
| 机械元件 | `data/raw/mast/machine/<shot>/MAST_{active_coils,passive_coils,wall,limiter}.pickle` | 绘制元件矩形/线 |

> 测试炮的 EFIT LCFS 有 NaN 数据（M9 炮 97.7% 帧 NaN），脚本自动优先选 LCFS 全有限的帧；
> 帧选择只在有限帧不足 4 时回退均匀采样（见脚本 main 内 `finite` 逻辑）。

## LCFS 提取流程（lcfs_extract，与 plot_lcfs_ip.py 逐字节一致）

1. O 点 = `argmax(psi)`（预测 psi 65×65，raw Wb，[R,Z]，R∈[0.06,1.98]、Z∈[−2,2]）
2. 阈值扫描 frac ∈ [0.05, 0.95)，`lv = psi.max() − frac·(psi.max() − psi.min())`
3. `ndimage.label(psi ≥ lv)` 连通域，取含 O 点域
4. 域触及网格边界 → 泄漏判停（break）；否则记录最大闭合域
5. 无候选返回 None；否则 `skimage.measure.find_contours(best, 0.5)` 最长等值线
6. 像素坐标线性映射回物理坐标（`np.interp`）→ 闭合边界 (r, z)（橙虚线）
7. 蓝色实线 = EFIT zarr LCFS（同帧最近时间）

## psi 散点（锁定风格，规范 = ../PLOT_STYLE.md，2026-09-03）

两版脚本均自包含（读 `data/preds/`、写 `out/`），2×2 布局（行 A/C、列 5%/1%），
轴：x = `EFIT psi / ground truth`、y = `psi pred`：

| 脚本 | 口径 | 输出 | 实测 R（A5/A1/C5/C1）|
|---|---|---|---|
| `plot_psi_grid_points.py` | 逐像素均匀采样 0.2%（rng=2026），hexbin gs=200，~18 万点/面板 | `out/psi_scatter_points_combined_1_5pct.png` | 0.9982 / 0.9967 / 0.9985 / 0.9981 |
| `plot_psi_scatter_combined.py` | 逐样本均值，hexbin gs=80 | `out/psi_scatter_combined_1_5pct.png` | 0.9963 / 0.9943 / 0.9966 / 0.9951 |

- ✅ preds 已于 2026-09-03 用归档 checkpoint 生成并保留（4 npz × ~641MB，A/C × 5%/1% s54）；
  删除后重建：`evaluate_tokamind_testset.py --save-predictions plot/data/preds`（命令见归档顶层 README"复现测试"）。
- **j_tor–Ip**（同锁定风格）：画图自包含 = `scripts/plot_jtor_ip_combined.py`（读预计算
  `data/jtor_ip_pairs.npz`，1.4MB，4 run × (Ip_meas, Ip_pred) × 21,350 slice）；
  npz 重建需全量 shot zarr（`scripts/compute_jtor_pairs.py --zarr-root <fusion-workspace>/data/raw/mast`，
  仅 workspace 可跑）。实测 R = 0.9636 / 0.9360 / 0.9922 / 0.9933（归档 preds 口径；早期
  workspace 旧 preds 版 4 位小数差 ±1e-4，浮点级）。

## 复现步骤（LCFS 图）

1. 预测 psi：用归档 checkpoint + test cache 生成（README 顶层"用归档 checkpoint 复现测试"）
   ```
   cp test/test_real_pca.npz <out>/test_cache_split_test_real.npz   # random 用（21,350 行）
   python mast-bridge/scripts/evaluate_tokamind_testset.py --manifest test/split_test_real.jsonl \
     --run-dir checkpoints/<run> --output-json <out>/tmp.json --save-predictions <preds-dir>
   ```
2. 修改脚本路径常量（复制后需按本机布局调整）：
   - `common.py`：`ROOT` = 项目根（脚本按 `parents[2]` 推导，若移动请改）；预测 npz 需放 `<ROOT>/plots/data/preds/`
   - `plot_lcfs_random.py`：`ZARR_ROOT` = 本目录 `plot/data/raw/mast`；`PLOT_DIR`（common）指向输出目录；
     `ARMS` 中 run 名与预测 npz 前缀对应
3. 运行（需 matplotlib/scipy/skimage/zarr）：
   `python plot_lcfs_random.py --shot 17833 --pct 1|5`
   输出：`lcfs_pred_17833_combined_{pct}pct.png`

## 已知限制

- 旧 LCFS 脚本（plot_lcfs_random.py / plot_lcfs_ip.py）仍为上游版本：`ZARR_ROOT`/run 名/输出路径
  按项目根布局写死（需先按上方"复现步骤（LCFS 图）"调整常量或用归档同名源 zarr）；**psi 散点
  两脚本（plot_psi_grid_points.py / plot_psi_scatter_combined.py）是自包含的**（相对路径解析 data/ 与 out/）
- LCFS 提取的是含磁轴的**最大闭合磁面**（limited 近似）；X 点开放 separatrix 未实现（progress2.md plot.md 坑 7）
- 早期爬升段（等离子体未成型/EFIT NaN）可能提取失败 → 返回 None 不画橙线
