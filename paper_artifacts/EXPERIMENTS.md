# paper_artifacts/EXPERIMENTS.md —— 实验总览（新会话从这里开始读）

> **一句话**：同一任务（69 维磁诊断 → 65×65 psi 场，Transformer d64/2/4/128），检验「合成数据预训练
> + 全量微调 (C) vs real 数据从零训练 (A)」在三种数据划分下的价值：
> **Temporal split（时间外推 M5-M7→M8→M9）= 全档正收益且随稀缺单调**（100%→1%：+11.4→+64.0%，30/30 seed 全正）；
> **Random split（逐炮随机）= 仅稀缺档正收益**（5% +13.7%、1% +36.1%，25-100% 为 −2~−6%）；
> **Shape-OOD（未见 diverted 几何）= 稀缺 Real 数据时收益最大**（Test-OOD-only、NEW 1%：+57.6%；Historical 5%：+30.0%）→ 论文"场景依赖"核心证据。
> 本文件 = 精简速览；数值直接来自归档 eval json。逐 run 历史（best@ep/停@ep/best_val）在各 run
> `manifest_training_summary.json` / `checkpoints/best/meta.json`；原始实验日志 = 项目根 `progress2.md` §45–§55。

## 1. 目录地图（三个归档）

| 路径 | 内容 |
|---|---|
| `temporal_full/` | temporal 全 5 档：33 ckpt + 21 eval json + M9 test（17,498 点/507 炮）|
| `random/` | random 全 5 档 × 3 噪声臂：64 ckpt + 61 eval json + test（21,350 点/730 炮）|
| `shape_ood/` | diverted Shape-OOD：frozen split、Test-OOD-only（19,009 slices）、30 个 A/C eval JSON；大型 checkpoint/cache 不提交 |
| `temporal_full/checkpoints/<run>/` | 每 run：`checkpoints/best/`（backbone/modality_heads/output_adapters/token_encoder .pt + meta.json）+ `manifest_scalers.npz` + `manifest_training_summary.json` |
| `temporal_full/eval/`、`random/eval/` | per-run 评估 json（`<run>.json`）；temporal 另有 3 个聚合文件（5%/1%）|
| 各目录 `dataset_split*.csv|json` | slice/shot 级数据划分清单（合作者用）|
| 各目录 `test/` | 划分的 test manifest + cache（行数见 §4）|
| `temporal_full/plot/`、`random/plot/` | LCFS 论文图脚本 + zarr（两归档同构）。temporal 版已自包含（脚本相对路径解析 data/raw/mast + data/preds + out/，含 28631/29412 两炮 zarr，复现步骤见 `temporal_full/plot/README.md`）；random/plot 同结构（LCFS 脚本仍为上游版路径/旧名，**psi 网格点采样散点 `plot_psi_grid_points.py` 自包含**，preds 2.4G 已生成，见 `random/plot/README.md`）|

## 2. 统一命名规则（读 run 名前先看这节）

```
{temporal|random}-{pretrain|scratch|ft[-clean|noisy05|noisy20]}-[{100|50|25|5|1}pct-]s{seed}-lr1e4-ep{50|100|150|500}[-warm10]
```

| token | 含义 |
|---|---|
| `pretrain` | 合成数据预训练（无档位 token）；`scratch` = A 臂 real 从零；`ft` = C 臂，init = 同 seed 同噪声级 pretrain |
| `clean/noisy05/noisy20` | 预训练+微调所用合成数据噪声级；temporal 仅 clean → 省略；random 三臂并存 → 显式 |
| `lr1e4` | 峰值 lr = 1e-4（全部 run 一致）|
| `epN` | **epoch 上限**（实际全部早停，patience=10 由 val 触发；真实停点见各 run summary）|
| `warm10` | cosine + warmup 10%（仅 random 的 ft）|

**旧名（原 `runs/` 目录，progress2.md / plot 脚本 / `artifacts/tokamind_test_eval/` 口径）→ 新名的映射规则见附录 A。**

## 3. 配置（两实验组全部一致，除 §注明的差异）

模型 d_model=64/n_layers=2/n_heads=4/dim_ff=128/dropout 0.05（363,393 参数，全量微调）；输入 69 维磁诊断 →
65×65 raw psi（Wb，R∈[0.06,1.98]m×Z∈[−2,2]m）；z-score 归一化（scaler 随 run 存）；AdamW，loss=embed MSE，
bs=64，wd=0，lr=1e-4（全部），patience=10；seeds 54/55/56（random A 1% = {s54,s56,s58}，s57 备用，2026-09-03 §57 定稿）。

| 家族（新名模板）| 数据 | ep 上限 | scheduler | warmup | init | 实际停@ep |
|---|---|---|---|---|---|---|
| `temporal-pretrain-s{…}-lr1e4-ep50` | synth clean 150,728 行 | 50 | cosine | 无 | 从零 | 27–38 |
| `temporal-scratch-{5 档}pct-s{…}-lr1e4-ep100` | real 档位 | 100 | cosine | 无 | 从零 | 12–98 |
| `temporal-ft-{5 档}pct-s{…}-lr1e4-ep100` | real 档位 | 100 | cosine | 无 | pretrain 同 seed | 12–37 |
| `random-pretrain-s{…}-lr1e4-ep50` | synth（noisy 版未归档）| 50 | cosine | 无 | 从零 | 27–50 |
| `random-scratch-{5,1}pct-s{…}-lr1e4-ep500` | real 5%/1% | 500 | cosine | 无 | 从零 | 135–312 |
| `random-ft-{3 臂}-{5,1}pct-s{…}-lr1e4-ep150-warm10` | real 5%/1% | 150 | cosine | 0.1 | pretrain 同 seed/噪声级 | 14–110 |

## 4. 数据划分（#shot / #slice；档位 = train 炮数嵌套抽样，val/test 全档共享全量）

| 实验 | train 各档（100→1%）| val | test |
|---|---|---|---|
| temporal | 6,046/187,292 → 60/1,917（5 档）| M8：559 / 6,206 | M9：507 / 17,498 |
| random | 5,828/171,427 → 58/1,924（5 档）| 728 / 21,147 | 730 / 21,350 |

## 5. 结果速查（nrmse_per_sample % = 主指标；3 seed 均值±std，std 为总体 ddof=0）

**Temporal**（增益 = (A−C)/A；逐 seed 数值在 `temporal_full/eval/<run>.json` 与聚合 json）

| 档 | A scratch | C ft | 增益 |
|---|---|---|---|
| 100% | 2.489 ± 0.167 | 2.204 ± 0.129 | **+11.4%** |
| 50% | 2.794 ± 0.173 | 2.378 ± 0.165 | **+14.9%** |
| 25% | 3.237 ± 0.151 | 2.249 ± 0.095 | **+30.5%** |
| 5% | 3.799 ± 0.117 | 2.334 ± 0.111 | **+38.6%** |
| 1% | 6.417 ± 0.030 | 2.310 ± 0.136 | **+64.0%** |

**Random**（增益 = **(A−C)/A**，与 temporal 同基准——real scratch 为基准；2026-09-03 统一，原 (A−C)/C 弃用）

| 档 | A（500ep 充分收敛）| ft-clean | ft-noisy05 | ft-noisy20 |
|---|---|---|---|---|
| 5% | 1.107 ± 0.022（s54-56）| 0.974 ± 0.012，**+12.0%** | 0.981，+11.4% | 0.991，+10.5% |
| 1% | 1.483 ± 0.039（s54/56/58）| 1.076 ± 0.010，**+27.5%** | 1.057，+28.7% | 1.147，+22.6% |

> 2026-09-03 §57 更新：1% A 档 seed 集 = {s54, s56, s58}（s55 确定性坏 basin 0.2316 剔除；s58 resume 好 basin
> 0.0435@247，nrmse 1.5374%，见 progress2.md §57.3）。旧 {s54/56/57} = 1.464±0.012（(A−C)/C +36.1%）已由 1.483±0.039 取代。
> ⚠️ 2026-09-03 起随机表增益统一为 **(A−C)/A**（real scratch 基准，与 temporal 一致）→ 1% +27.5%、5% +12.0%；原 (A−C)/C 数字（+37.8/+40.2/+29.2 等）弃用。
> std 均为总体标准差（ddof=0），与 eval json 一致。

⚠️ 论文 LaTeX/§52.1 中 temporal 1% 行 std（±0.04/±0.17）为样本 std（ddof=1），其余档为总体 std——跨源引用需换算。


**Shape-OOD**（主指标 = Test-OOD-only；增益 = `(A−C)/A`，逐 seed 配对；详见 `shape_ood/README.md`）

| 档 | A scratch | C ft | 配对 nRMSE 增益 |
|---|---|---|---|
| 1% | 5.383 ± 0.019 | 2.285 ± 0.202 | **+57.6%** |
| 5% | 2.447 ± 0.126 | 1.706 ± 0.142 | **+30.0%** |
| 25% | 2.071 ± 0.094 | 1.721 ± 0.107 | **+16.7%** |
| 50% | 2.033 ± 0.098 | 1.674 ± 0.206 | **+17.3%** |
| 100% | 2.007 ± 0.079 | 1.858 ± 0.273 | **+7.7%** |

> Shape-OOD is a frozen diverted geometry benchmark (`delta_asym < -0.214`); historical 5/25/50/100% fractions are not strictly nested, and NEW 1% is an explicitly separate low-data ablation. Do not compare its result as a nested-fraction causal ablation.

## 6. 复现评估（任选归档内 run）

```
cp <归档>/test/<cache>.npz <out>/test_cache_split_test_real.npz   # temporal: temporal_test_real.npz(17,498 行) / random: test_real_pca.npz(21,350 行)
python mast-bridge/scripts/evaluate_tokamind_testset.py --manifest <归档>/test/split_test_real.jsonl \
  --run-dir <归档>/checkpoints/<run 新名> --output-json <out>/result.json
```
✅ 已验证逐位一致样例：`temporal-ft-5pct-s54-lr1e4-ep100`（rmae_global=0.008950）、
`temporal-ft-25pct-s56-lr1e4-ep100`、以及 random `-clean-5pct-s54-`（CPU 与原 GPU 差 <1e-6 相对）。
eval json 常用字段：`nrmse_per_sample`（主，%）、`raw_rmse`（Wb）、`rmae_global`、`checkpoint_best_val/epoch`。

## 7. 已知坑（新会话必读）

1. 两个实验共用缓存文件名 `test_cache_split_test_real.npz` 但行数不同（17,498 vs 21,350）——评估前必须用对应 cache 覆盖，严禁依赖 rebuild。
2. temporal 1% scratch 实际上限 100ep（run summary 实测）；progress §52.1 文案「500ep」为笔误。
3. random A 无 s55（1% 坏 basin）→ 归档 s54/56/57；random noisy 预训练 init 未归档。
4. 归档内 summary 的 `fine_tuning.pretraining_run_dir` 指向原 runs/ 绝对路径（溯源用，可悬空）；init 关系见 §3。
5. 高数据档 random eval json（100/50/25%，负收益证据）与 temporal lr=5e-5 复跑（§55，结论同 1e-4）**未归档**（在原 `artifacts/`、`runs/`）。
6. 图产物与**锁定图风格**（规范 = `paper_artifacts/PLOT_STYLE.md`，2026-09-03 定稿）：
   - psi/j_tor 散点统一 2×2 hexbin 风格（A/C 行 × 5%/1% 列；psi x 轴固定 `EFIT psi / ground truth`）。
   - random 侧三图已按锁定风格再生：网格点采样（`psi_scatter_points_combined_1_5pct`，R=0.9982/0.9967/0.9985/0.9981）、
     逐样本均值（`psi_scatter_combined_1_5pct`，R=0.9963/0.9943/0.9966/0.9951）、
     j_tor–Ip（`jtor_ip_combined_1_5pct`，R=0.9636/0.9359/0.9923/0.9933）；
     归档版脚本在 `random/plot/scripts/`（自包含），workspace 上游在 `plots/script/`。
   - `temporal_full/plot/` LCFS 图自包含可复现（zarr 28631+29412、相对路径、run 名统一、preds 已生成）；
     `random/plot/` LCFS 脚本未改造（旧 run 名/外部路径/`plots/data/preds/` 旧名资产）。
   - temporal 侧 psi 网格散点同型图尚未产出（规范已锁定，需要时套用）。

---

## 附录 A：旧名 → 新名映射规则（2026-09-03 统一命名）

旧名（原 `runs/` 下目录名）→ 归档新名（checkpoint 目录名 = per-run eval 文件名）。规则确定性可逆，无需逐行查表：

1. **前缀**：`tokamind-temporal-*` → `temporal-*`；`tokamind-pca01sigma-*` → `random-*`（pca01sigma 即 random 实验内部 cache 名）。
2. **角色**：`synth-pretrain`→`pretrain`；`scratch`→`scratch`；`finetune-clean`→`ft`（temporal）；
   random：`finetune`→`ft-clean`、`noisy05-finetune`→`ft-noisy05`、`noisy20-finetune`→`ft-noisy20`。
3. **档位/seed** 不变：`{tier}pct-s{seed}`（tier ∈ 100/50/25/5/1）。
4. **配置 token**（删除旧后缀编码，按实义重建）：
   | 旧后缀 | 实义 | 新 token |
   |---|---|---|
   | temporal `-u1e4`（含 1% 档 `-1u`）| lr 1e-4，cap 100 | `-lr1e4-ep100` |
   | temporal pretrain `-u1e4` | lr 1e-4，cap 50 | `-lr1e4-ep50` |
   | random pretrain `-sec50` | cap 50 | `-lr1e4-ep50` |
   | random scratch `-scr500`/`-e500`（同配置）| cap 500 | `-lr1e4-ep500` |
   | random ft `-w150` | cap 150 + warmup 10% | `-lr1e4-ep150-warm10` |
5. **噪声词位置**：放在角色后档位前（见规则 2），temporal 无噪声臂 → 不写。

示例：
`tokamind-temporal-finetune-clean-1pct-s54-1u` → `temporal-ft-1pct-s54-lr1e4-ep100`
`tokamind-temporal-synth-pretrain-s55-u1e4` → `temporal-pretrain-s55-lr1e4-ep50`
`tokamind-pca01sigma-scratch-1pct-s54-scr500` → `random-scratch-1pct-s54-lr1e4-ep500`
`tokamind-pca01sigma-noisy20-finetune-1pct-s56-w150` → `random-ft-noisy20-1pct-s56-lr1e4-ep150-warm10`
