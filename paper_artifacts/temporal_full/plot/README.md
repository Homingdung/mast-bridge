# temporal LCFS 图复现（归档内自包含，2026-09-03 更新）

> 论文 LCFS 对比图（EFIT/ground truth vs predicted LCFS）的绘图脚本与数据。
> 本包为**自包含复现包**：zarr + machine 元件 + 脚本均在本地解析，拿到任意机器都能跑
> （不再指向外部 fusion-workspace 绝对路径）。结构 = `<归档>/temporal_full/plot/`，与 random 归档
> 的 `random/plot/` 同构（scripts/ + data/ + README）。
> LCFS 构造算法完整流程见 progress2.md §52.4；run 名 = 归档统一新名（旧名对照 = ../EXPERIMENTS.md 附录 A）。

## 目录结构

```
plot/                      （= <归档>/temporal_full/plot/）
├── scripts/
│   ├── plot_lcfs_temporal.py   # 双臂对比图（上排 scratch、下排 ft、底排 Ip(t)，--pct 1|5）
│   ├── plot_lcfs_ip.py         # 单模型 4 帧风格图（--shot 28631|29412 --model 5pct-ft|...）
├── data/
│   ├── raw/mast/<shot>.zarr      # 图中 shot（M9 test 炮）完整 zarr
│   ├── raw/mast/machine/<shot>/  # MAST_{active_coils,passive_coils,wall,limiter}.pickle
│   └── preds/                    # 【生成产物】预测 psi npz，见下方步骤 1
└── out/                          # 【生成产物】输出图
```

- zarr 覆盖两炮：**28631**（47M，单模型 ip 图用）、**29412**（143M，双臂 combined 图用）。
- 数据键（zarr v3）：`equilibrium/lcfs_r|z|time`、`magnetics/ip|time`；EFIT LCFS 多数帧为 NaN
  （M9 炮 97.7%），脚本自动优先取 LCFS 全有限的帧（见各脚本 main 内 `finite` 逻辑）。

## 复现步骤

**1. 生成预测 psi（必做一次，输出到 `plot/data/preds/`）**

> ✅ 2026-09-03 复现审计时已生成并保留 4 个 run 的预测 npz（约 2.0G，5%/1% 的 A/C s54 各一），
> 拿到本包可直接跳到步骤 2 画图；如需重生成/清空间，删除后按下述命令重建：

用归档 checkpoint（`checkpoints/`）+ 归档 test 数据评估，预测文件名 = `<run 新名>`：
```
cd <归档>/temporal_full
cp test/temporal_test_real.npz /tmp/out/test_cache_split_test_real.npz   # 17,498 行 temporal 缓存
python <repo>/mast-bridge/scripts/evaluate_tokamind_testset.py \
  --manifest test/split_test_real.jsonl \
  --run-dir checkpoints/temporal-scratch-5pct-s54-lr1e4-ep100 \
  --run-dir checkpoints/temporal-ft-5pct-s54-lr1e4-ep100 \
  --run-dir checkpoints/temporal-scratch-1pct-s54-lr1e4-ep100 \
  --run-dir checkpoints/temporal-ft-1pct-s54-lr1e4-ep100 \
  --output-json /tmp/out/tmp.json --save-predictions plot/data/preds
```
（`--save-predictions` 生成 `test_predictions_<run 名>.npz`，内含 `pred_psi`(N,65,65) 与 `sample_ids`）

**2. 画图（脚本自动从自身位置向上找 `data/raw/mast` 与 `data/preds`）**
```
python plot/scripts/plot_lcfs_temporal.py --shot 29412 --pct 5   # -> plot/out/lcfs_pred_29412_combined_5pct.png
python plot/scripts/plot_lcfs_temporal.py --shot 29412 --pct 1   # -> ..._combined_1pct.png
python plot/scripts/plot_lcfs_ip.py --shot 28631 --model 5pct-ft # -> plot/out/lcfs_pred_28631_5pct-ft.png（可换 29412/5pct-scratch/1pct-ft）
```
依赖：python3 + matplotlib + numpy + scipy + scikit-image + zarr。

## 与上游脚本的差异（2026-09-03 为自包含化修改）

- `ZARR_ROOT`/manifest/preds/输出路径由绝对路径改为脚本自身位置推导（`plot/data/...`、`plot/out/`）
- run 名改为归档统一新名（如 `temporal-ft-5pct-s54-lr1e4-ep100`，对应原 `tokamind-temporal-finetune-clean-5pct-s54-u1e4`）
- 原 `common.py` 不再被引用，已从归档移除；上游原版脚本仍在项目根 `plots/script/`
- 图上内容与视觉规范未改动（等值线 8 级、机械元件画法、EFIT 蓝实线 vs 预测橙虚线、Ip(t) 底排）

## 已知限制

- 提取的是含磁轴的**最大闭合磁面**（limited 近似）；X 点开放 separatrix 未实现（progress2.md plot.md 坑 7）
- ramp-up 早期（等离子体未成型/EFIT NaN）可能提取失败 → 返回 None 不画橙线
