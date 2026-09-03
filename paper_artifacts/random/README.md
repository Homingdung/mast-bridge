# paper_artifacts/random —— Random split 稀缺档论文复现档案（2026-09-03 统一命名）

对应 progress2.md §52.2（实验二）与 §50.11/50.12（w150 最终确认）。2026-09-02 归档；
**2026-09-03 统一命名**（`{random}-{pretrain|scratch|ft[-clean|noisy05|noisy20]}-{tier}pct-s{seed}-
lr1e4-ep{50|500|150}[-warm10]`；旧名 → 新名映射见 `../EXPERIMENTS.md` 附录 A）。

## 内容

```
random/
├── eval/         结果 JSON（61 个 per-run，文件名 = run 新名）
├── checkpoints/  64 个 run 的 best checkpoint + scaler + summary
├── test/         Random split test 数据（21,350 点 / 730 炮）
├── dataset_split.csv / dataset_split_info.json   数据集划分清单（slice 级）
├── dataset_split_shots.csv                         数据集划分清单（shot 级）
└── plot/                              LCFS 图脚本 + zarr（LCFS 构造复现，见 plot/README.md）
```

## 数据集划分清单（dataset_split.csv / dataset_split_info.json）

给持有完整数据集的合作者使用：`dataset_split.csv` 为 slice 级划分清单
（213,924 行 = train 171,427 + val 21,147 + test 21,350），列：
`phase`（train/val/test）、`train_fractions_nested`（train 行的档位隶属，如 "100 50 25 5 1"，
嵌套 1%⊂5%⊂10%⊂25%⊂50%⊂100%，空格分隔由大到小）、`shot_id`、`target_time`、`sample_id`。
shot 级清单 `dataset_split_shots.csv`：每炮一行（shot_id, phase, train_fractions_nested, n_slices），直接对应本 README 各表的 #shot 口径。
按 `phase + train_fractions_nested` 过滤即得任意档位训练集；val/test 全档共享。
规则细节与源文件映射见 `dataset_split_info.json`。

## 关键结果（§52.2，nrmse_per_sample %，3 seed 均值；增益 = (A−C)/C）

| 档 | A（充分收敛）| C clean | 增益 | C noisy05 | C noisy20 |
|---|---|---|---|---|---|
| 5% | 1.107 ± 0.022 | 0.974 ± 0.012 | +12.0% | 0.981 | 0.991 |
| 1% | 1.483 ± 0.039 | 1.076 ± 0.010 | +27.5% | 1.057 | 1.147 |

A = `random-scratch-…-lr1e4-ep500`（500ep 上限 + 早停 patience=10，全部提前早停；5% 停@135–145、1% 停@258–312）；
C = `random-ft-…-lr1e4-ep150-warm10`（ft 150ep + warmup 10%，早停 14–110ep）。配置：bs64、cosine+warmup10%、wd=0、lr 全 1e-4、3 seed（std = 总体 ddof=0，与 eval json 一致）。
（2026-09-03 §57 更新：1% A 档 seed 集 s55（确定性坏 basin，0.2316）→ s58（好 basin，0.0435@247，nrmse 1.5374%）替换，
{54,56,58} 均值 **1.483 ± 0.039**（原 {54,56,57} = 1.464 ± 0.012）。⚠️ 2026-09-03 增益口径统一为
> **(A−C)/A**（real scratch 基准，与 temporal 一致）：1% +27.5%、5% +12.0%（原 (A−C)/C 口径 +37.8%/+13.7% 弃用））

## checkpoints/ 说明

64 个 run（布局与原始训练 run 目录一致，仅含推理所需文件）：
`<run>/checkpoints/best/`（backbone.pt + modality_heads.pt + output_adapters.pt + token_encoder.pt + meta.json）
+ `manifest_scalers.npz`（推理必需）+ `manifest_training_summary.json`

| 臂 | run 命名 |
|---|---|
| C clean（ft w150，全 5 档）| `random-ft-clean-{100,50,25,5,1}pct-s{54,55,56}-lr1e4-ep150-warm10`（15 个）|
| C noisy05 / noisy20（ft w150，全 5 档）| `random-ft-{noisy05,noisy20}-{100,50,25,5,1}pct-s{54,55,56}-lr1e4-ep150-warm10`（各 15 个）|
| A scratch 5/1%（e500/scr500）| `random-scratch-{5,1}pct-s{54,55,56}-lr1e4-ep500`（1% s54 = scr500 同名）|
| A scratch 25/50/100%（e500）| `random-scratch-{25,50,100}pct-s{54,55,56}-lr1e4-ep500` |
| A scratch 1% s58 补位 | `random-scratch-1pct-s58-lr1e4-ep500`（s55 确定性坏 basin 剔除；s57 曾补位备用，2026-09-03 §57 定稿 {s54,s56,s58}，见 ../EXPERIMENTS.md 与 progress2.md §57.3）|
| 预训练 init（C warmstart，仅重训需要）| `random-pretrain-s{54,55,56}-lr1e4-ep50`（3 个；noisy 版 init 未归档，如需重训 noisy 臂请从原 runs/ 取，旧名 `tokamind-pca01sigma-synth-pretrain-noisy*-s*-sec50`）|

> A 臂旧名差异说明（`-scr500` vs `-e500` 同为 500ep 配置，仅批次命名不同）在统一命名中已消除。

## eval/ 说明

- per-run = `eval/<run 新名>.json` × 61（C 45 = 3 噪声臂 × 5 档 × 3 seed；A 16 = scratch 5 档 × 3 seed + 1% s58 补位），3 seed 均值需自行聚合
- **2026-09-03 补齐 25/50/100% 档**（A e500 9 run + C w150 27 run 的 ckpt/eval 全部入档）——论文 tab:random_table 五档 × 3 臂可完整复现；数值与 progress2.md §57.4 及 `artifacts/tokamind_test_eval/` 逐位一致

## test/ 说明

- `split_test_real.jsonl`：random split test manifest（21,350 行 / 730 炮）
- `test_real_pca.npz`：test cache（features 69 维 + psi 65×65，21,350 行）

## 用归档 checkpoint 复现测试

1. 把 cache 复制到评估输出目录并命名为 `test_cache_split_test_real.npz`（§50.7 坑：评估前校验行数 == 21,350，严禁依赖 rebuild）：
   `cp test/test_real_pca.npz <out-dir>/test_cache_split_test_real.npz`
2. 运行评估（`--run-dir` 直接指向 checkpoints/<run>，脚本自动加载 best + scalers）：
   `python mast-bridge/scripts/evaluate_tokamind_testset.py --manifest test/split_test_real.jsonl \
    --run-dir checkpoints/random-ft-clean-5pct-s54-lr1e4-ep150-warm10 \
    --output-json <out-dir>/result.json`
3. 核对：nrmse_per_sample 应与 eval/random-ft-clean-5pct-s54-lr1e4-ep150-warm10.json 一致
   （✅ 2026-09-02 端到端验证：CPU 推理与原 GPU 评估数值差 <1e-6 相对（浮点核差异），
   展示到 4 位小数完全相同；checkpoint 文件与 runs/ 逐字节一致）
