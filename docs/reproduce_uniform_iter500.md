# 复现 uniform_iter500 数据

这是当前 `uniform_random + max_iterations=500` 仿真数据实验的最短复现流程。
该流程只覆盖数据生成、严格过滤和三组对比实验 manifest 构建，不包含模型训练。

所有命令从项目目录运行：

```bash
cd /Users/mingdonghe/pj/fusion-workspace/mast-bridge
source .freegsnke-solve-env/bin/activate
```

## 1. 输入

预期 workspace 中已经存在：

```text
../data/raw/mast/<shot>.zarr
../data/raw/mast/machine/<shot>/
../data/processed/real/lao_parameter_ensemble/all_zarr_lao_parameter_fits.npz
```

machine pickle 标准文件名应包含：

```text
MAST_active_coils.pickle
MAST_limiter.pickle
MAST_magentic_probes.pickle
MAST_passive_coils.pickle
MAST_wall.pickle
```

`MAST_passive_coilds.pickle` 是历史 typo，只作为兼容读取 fallback；新生成的数据
应使用 `MAST_passive_coils.pickle`。

## 2. 生成 Uniform 扰动表

```bash
FIT_PATH=../data/processed/real/lao_parameter_ensemble/all_zarr_lao_parameter_fits.npz
VARIANT_CSV=../data/manifests/lao85_uniform_variants.csv

python scripts/build_lao85_variant_rows.py \
  --fit-path "$FIT_PATH" \
  --variants-per-point 5 \
  --seed 20260729 \
  --min-time 0.10 \
  --max-time 0.30 \
  --output "$VARIANT_CSV"
```

## 3. 运行 FreeGSNKE 求解

使用 `max_iterations=500`。之前设置为 `100` 时，有些本来可能继续收敛的样本
会提前停止，并在后续 `1e-8` strict filter 中被拒绝。

```bash
SYNTH_DIR=../data/processed/synthetic_lao85_uniform_iter500
MANIFEST_DIR=../data/manifests

python scripts/run_lao85_variant_solve_batch.py \
  --variant-csv "$VARIANT_CSV" \
  --data-dir ../data/raw/mast \
  --fit-path "$FIT_PATH" \
  --synthetic-root "$SYNTH_DIR" \
  --manifest-dir "$MANIFEST_DIR" \
  --prefix tokamark_lao85_uniform_iter500 \
  --task task_1-3 \
  --nx 65 \
  --ny 65 \
  --tolerance 1e-8 \
  --max-solver-tolerance 1e-8 \
  --max-iterations 500
```

脚本会跳过已经同时包含 `equilibrium.npz` 和 `metadata.json` 的样本，因此中断后
可以直接重跑。如果只想先试一小段，可以额外加：

```bash
--start-index 870 --limit 100
```

查看进度：

```bash
tail -f ../data/manifests/tokamark_lao85_uniform_iter500_batch_report.jsonl
```

## 4. 重新生成严格过滤 Manifest

这一步不会重新运行 FreeGSNKE，只会扫描已有输出，并按 `1e-8` 收敛准则重新生成
accepted/rejected manifest。

```bash
python scripts/build_synthetic_manifest.py \
  --synthetic-root ../data/processed/synthetic_lao85_uniform_iter500 \
  --output ../data/manifests/tokamark_lao85_uniform_iter500_synthetic_accepted.jsonl \
  --rejected-output ../data/manifests/tokamark_lao85_uniform_iter500_synthetic_rejected.jsonl \
  --task task_1-3 \
  --max-solver-tolerance 1e-8
```

只有 `*_synthetic_accepted.jsonl` 进入后续数据集构建。筛选条件是：

```text
solver_converged == true
solver_final_tolerance <= 1e-8
```

## 5. 构建三组对比实验 Manifest

```bash
python scripts/build_experiment_manifests.py \
  --accepted-synthetic ../data/manifests/tokamark_lao85_uniform_iter500_synthetic_accepted.jsonl \
  --raw-data-dir ../data/raw/mast \
  --fit-path "$FIT_PATH" \
  --output-dir ../data/manifests \
  --prefix tokamark_lao85_uniform_iter500 \
  --task task_1-3
```

输出：

```text
../data/manifests/tokamark_lao85_uniform_iter500_real_only.jsonl
../data/manifests/tokamark_lao85_uniform_iter500_synthetic_only.jsonl
../data/manifests/tokamark_lao85_uniform_iter500_real_plus_synthetic.jsonl
```

当前本地过滤后的计数示例：

```text
synthetic_accepted:     120
synthetic_rejected:      95
real_only:               40
synthetic_only:         120
real_plus_synthetic:    160
```
