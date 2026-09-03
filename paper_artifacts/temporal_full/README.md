# temporal 全 5 档实验归档清单（§54.6 / §48.7 / §48.8 / §55.7，2026-09-03 建档并统一命名）

> 论文写作/复现用归档（拷贝）：结果 JSON + best checkpoint + test 数据集。
> 覆盖 temporal split 完整证据链：**100/50/25/5/1% ×（A real scratch + C clean 预训练+微调）× 3 seed，
> 全部 C 对 A 正增益（30/30，min +6.9%，§55.7）**。
> 配置口径 = lr 1e-4（与 §52 登记及 §54.6 五档曲线一致；§55 的 lr=5e-5 复跑为震荡机制验证，
> 结论与本文档一致且不改变任何论文数字，未归档）。
> **命名规范**（2026-09-03）：`{temporal}-{pretrain|scratch|ft}-{tier}pct-s{seed}-lr1e4-ep{50|100}`；
> 旧名（原 `runs/` 口径，如 `tokamind-temporal-finetune-clean-5pct-s54-u1e4`、1% 的 `-1u`）映射见
> `../EXPERIMENTS.md` 附录 A。run 名指代已统一，**下面所有表均为新名**。
> **代码（mast-bridge 仓库：scripts/、src/、configs/）待全部实验跑完后统一备份，不在此归档**；
> 画图脚本暂未归档（需要时再从 `plots/script/` 拷贝）。

## 0. 目录结构

```
temporal_full/                 （与 random/ 归档同构）
├── eval/           结果 JSON（21 个：18 per-run + 3 聚合）
├── checkpoints/    33 个 run 的 best checkpoint + scaler + summary（3 预训练 + 15 scratch + 15 ft）
├── test/           M9 test 数据（split_test_real.jsonl + temporal_test_real.npz，17,498 点 / 507 炮）
├── dataset_split.csv / dataset_split_info.json / dataset_split_shots.csv   数据集划分清单（见 §0.1）
└── plot/           LCFS 论文图画图脚本+数据（plot/scripts/ + plot/data/；README 见 plot/README.md）
```

## 0.1 数据集划分清单（dataset_split.csv / dataset_split_info.json）

给持有完整数据集的合作者使用：`dataset_split.csv` 为 slice 级划分清单
（210,996 行 = train 187,292 + val 6,206 + test 17,498），列：
`phase`（train/val/test）、`train_fractions_nested`（train 行的档位隶属，如 "100 50 25 5 1"，
嵌套 1%⊂5%⊂10%⊂25%⊂50%⊂100%，空格分隔由大到小）、`shot_id`、`target_time`、`sample_id`。
shot 级清单 `dataset_split_shots.csv`：每炮一行（shot_id, phase, train_fractions_nested, n_slices），直接对应本 README 各表的 #shot 口径。
按 `phase + train_fractions_nested` 过滤即得任意档位训练集；val/test 全档共享。
划分规则（时间序 M5-M7/M8/M9）与源文件映射见 `dataset_split_info.json`。

> **代码依赖**（外部，待统一备份的 mast-bridge 仓库 + vendored mmt）：
> 推理 = `mast-bridge/scripts/evaluate_tokamind_testset.py` + `mast_bridge` 包 +
> `scripts/train_tokamind_manifest.py`（`_build_signal_specs` 依赖）+ `external/tokamind/src/mmt/`。
> 运行环境：python3 + torch + mast_bridge 包。
> ✅ 归档 checkpoint 端到端验证：5pct-ft-s54 与 25pct-ft-s56 用归档内 checkpoint 评估，
> 全部指标与 eval/*.json 逐位一致（含 checkpoint_epoch/best_val）。（2026-09-03 改名仅动目录名与
> json 的 run/run_dir 字段，数值未触碰。）

## 1. 实验结果（eval JSON，M9 test 17,498 点 / 507 炮）

| 文件 | 内容 | 章节 |
|---|---|---|
| `eval/<run>.json` × 18 | 100/50/25% 的 A/C 单 run（run 名 = §3/§4 表）| §54.6 |
| `eval/temporal-5pct-A-scratch-3seed.json` | 5% scratch（s54/55/56，3 run 数组）| §48.7 |
| `eval/temporal-5pct-C-ft-3seed.json` | 5% 预训练+微调（s54/55/56）| §48.7 |
| `eval/temporal-1pct-AC-3seed.json` | 1% scratch + ft 混合（6 run）| §48.8 |

## 2. 模型 checkpoint（33 个，他人可直接用我训练的模型做测试）

`checkpoints/<run 名>/`，每个 run 目录与**原始训练 run 目录布局完全一致**：

```
checkpoints/<run 名>/
├── checkpoints/best/     best 权重（backbone.pt、modality_heads.pt、output_adapters.pt、
│                         token_encoder.pt、meta.json）
├── manifest_scalers.npz  推理必需：输入/输出 z-score 参数（目标为 raw psi）
└── manifest_training_summary.json  训练 history（逐 epoch loss、早停、init 信息）
```

| 臂 | 100/50/25/5%（新名）| 1%（新名）|
|---|---|---|
| 预训练（ep50）| `temporal-pretrain-s{54,55,56}-lr1e4-ep50` | 复用（同 3 个 run）|
| scratch（ep100）| `temporal-scratch-{100,50,25,5}pct-s{54,55,56}-lr1e4-ep100` | `temporal-scratch-1pct-s{54,55,56}-lr1e4-ep100` |
| ft（ep100）| `temporal-ft-{100,50,25,5}pct-s{54,55,56}-lr1e4-ep100` | `temporal-ft-1pct-s{54,55,56}-lr1e4-ep100` |

**用 checkpoint 复现测试**（✅ 已验证：评估结果与 eval/*.json 逐位一致）：
1. 环境：`mast-bridge/.tokamind-train-env/bin/python`（或任意 python3 + torch + mast_bridge 包）
2. 把 `test/temporal_test_real.npz` 复制到评估输出目录命名为 `test_cache_split_test_real.npz`（§00.5 坑 3）
3. 运行（`--run-dir` 指向 `checkpoints/<run 名>/`，脚本自动加载 `checkpoints/best/` 与 `manifest_scalers.npz`）：
   `python mast-bridge/scripts/evaluate_tokamind_testset.py \
      --manifest test/split_test_real.jsonl \
      --run-dir checkpoints/temporal-ft-5pct-s54-lr1e4-ep100 ... \
      --output-json <out>/result.json`
   - 验证样例：`temporal-ft-5pct-s54-lr1e4-ep100` 复现 rmae_global=0.008950、
     nrmse_per_sample=0.022804、raw_rmse=0.004693，与 `eval/temporal-5pct-C-ft-3seed.json` 逐位一致
   - 如需预测输出：加 `--save-predictions <dir>` 生成 `test_predictions_<run>.npz`

## 3. 训练数据资产（保留原位，不复制；test 已归档到 `test/`）

| 用途 | manifest | cache（data/processed/training_cache_pca_01sigma/）|
|---|---|---|
| 预训练 | `data/manifests/training_temporal/run_synth_pretrain.jsonl`（150,728 行）| `temporal_synth_clean.npz` |
| 100% train+val | `data/manifests/training_temporal/run_train_with_val.jsonl`（187,292+6,206）| `temporal_train_with_val.npz` |
| 50% train+val | `data/manifests/training_temporal/subset_run_real_50pct.jsonl`（93,722+6,206）| `temporal_subset_run_real_50pct.npz` |
| 25% train+val | `data/manifests/training_temporal/subset_run_real_25pct.jsonl`（46,840+6,206）| `temporal_subset_run_real_25pct.npz` |
| 5% train+val | `data/manifests/training_temporal/subset_run_real_5pct.jsonl`（8,683+6,206）| `temporal_subset_run_real_5pct.npz` |
| 1% train+val | `data/manifests/training_temporal/subset_run_real_1pct.jsonl`（1,917+6,206）| `temporal_subset_run_real_1pct.npz` |
| val（M8，559 炮）| `data/manifests/training_temporal/split_val_shots.jsonl` | —（`--val-shot` 逐炮传入）|
| test（M9 全量）| `data/manifests/training_temporal/split_test_real.jsonl` | `temporal_test_real.npz` |

- 嵌套关系：1% ⊂ 5% ⊂ 10% ⊂ 25% ⊂ 50%（shot 级嵌套，seed 20260825 shuffle 前缀）
- 5%/1% 构建方式：低档内再 shuffle（seed 20260825/20260827）取子集；
  cache = `mast-bridge/scripts/slice_cache.py --src temporal_train_with_val.npz --manifest subset_run_real_{f}pct.jsonl`

## 4. 原训练 run 目录（旧名 = `runs/` 下目录；仅重跑需要，评估/推理用 §2 的 checkpoints/ 即可）

| 臂 | 100/50/25/5%（旧名 `-u1e4`）| 1%（旧名 `-1u`）|
|---|---|---|
| 预训练（50ep）| `runs/tokamind-temporal-synth-pretrain-s{54,55,56}-u1e4` | 复用 5% 档 |
| scratch | `runs/tokamind-temporal-scratch-{100,50,25,5}pct-s{54,55,56}-u1e4` | `runs/tokamind-temporal-scratch-1pct-s{54,55,56}-1u` |
| finetune | `runs/tokamind-temporal-finetune-clean-{100,50,25,5}pct-s{54,55,56}-u1e4` | `runs/tokamind-temporal-finetune-clean-1pct-s{54,55,56}-1u` |

- 每个 run 目录内：`checkpoints/{best,latest}/`、`manifest_training_summary.json`、`manifest_scalers.npz`
- 说明：§2 `checkpoints/` 即这些 run 的 `checkpoints/best/` + scaler + summary 的拷贝
  （布局与原始 run 目录一致，可直接当 run 目录用）；新旧名一一对应见 `../EXPERIMENTS.md` 附录 A。

## 5. 训练配置（§48.1/48.3/54.1，全部档位完全一致）

- bs64、cosine 无 warmup、wd=0、lr 全 1e-4、epochs 上限 = 100、早停 patience=10、3 seed s54/55/56
  （1% scratch 上限同为 100——progress §52.1 文案「500ep」为笔误，实测 summary epochs=100）
- 模型：d64/2/4/128、dropout 0.05、69 输入特征 → 65×65 psi
- 输入 `--input-mode magnetic-diagnostics`、目标 `--target-mode raw-psi`（raw psi 单位 Wb；
  训练时 z-score，评估反标准化还原，见 `evaluate_tokamind_testset.py:92-93`）
- 原划分：train=M5-M7、val=M8、test=M9 全量

## 6. 关键结果速查（nrmse_per_sample %，3 seed 均值±std；增益 = (A−C)/A）

| 档位 | A real scratch | C pretrain+ft | 增益（NRMSE）| 章节 |
|---|---|---|---|---|
| 100% | 2.489 ± 0.167 | 2.204 ± 0.129 | **+11.4%** | §54.6 |
| 50% | 2.794 ± 0.173 | 2.378 ± 0.165 | **+14.9%** | §54.6 |
| 25% | 3.237 ± 0.151 | 2.249 ± 0.095 | **+30.5%** | §54.6 |
| 5% | 3.80 ± 0.12 | 2.33 ± 0.11 | **+38.6%** | §48.7 |
| 1% | 6.42 ± 0.04 | 2.31 ± 0.17 | **+64.0%** | §48.8 |

- 逐 seed 全正：30/30（单 seed 增益区间 100% +10.2~+12.4 → 1% +61.8~+67.0，无跨档重叠），见 progress2.md §55.7
- C 臂五档平台（2.20-2.38）几乎水平 → 预训练后误差对数据量不敏感；A 臂随稀缺单调上升（2.49 → 6.42）
- 对照 random split（`../random/README.md`）：高数据档 25-100% 预训练为负收益（−2~−6%）→ 场景依赖讨论点
- ⚠️ 5%/1% 行的 std 为样本 std（ddof=1），其余档为总体 std——横向引用见 `../EXPERIMENTS.md` §6 注
