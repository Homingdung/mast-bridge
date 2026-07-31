# mast-bridge

`mast-bridge` 把 MAST Level 2 Zarr 数据转换成 FreeGSNKE 可用输入，并以真实 shot 为父样本批量生成 synthetic equilibrium 数据，用于后续 TokaMind 对比实验。

主流程：

```text
download MAST Level 2
  -> filter successful shots
  -> build machine pickles
  -> build Lao/EFIT NPZ
  -> run FreeGSNKE forward solves
  -> synthesize magnetic diagnostics for accepted equilibria
  -> build real/synthetic/mixed manifests
  -> train diagnostics-to-psi TokaMind models
```

数据输入和产出统一放在 workspace 的 `data/` 目录下。`data_analysis_report/` 只用于图片和分析报告，不作为当前数据流水线的输入目录。

## 目录

- [1. 工作区](#1-工作区)
- [2. Python 环境](#2-python-环境)
- [3. 配置运行规模](#3-配置运行规模)
- [4. 批处理流程](#4-批处理流程)
- [5. 当前 uniform_iter500 复现流程](#5-当前-uniform_iter500-复现流程)
- [6. 单个 Shot 命令](#6-单个-shot-命令)
- [7. 数据约定](#7-数据约定)
- [8. TokaMind 小规模训练](#8-tokamind-小规模训练)
- [9. 验证](#9-验证)
- [10. 常见问题](#10-常见问题)

## 1. 工作区

从一个空目录开始：

```bash
mkdir fusion-workspace
cd fusion-workspace

git clone https://github.com/Homingdung/mast-bridge.git mast-bridge

mkdir external
git clone https://github.com/FusionComputingLab/freegsnke.git external/freegsnke
git clone https://github.com/UKAEA-IBM-STFC-Fusion-FMs/tokamind.git external/tokamind
```

当前流水线不依赖 `LARGE_MODEL_FUSION`。`scripts/download_mast_shots.py` 直接通过
`s5cmd` 从 STFC Echo 下载指定 MAST Level 2 shot。`tokamark` 只作为方法参考，
运行本 README 的数据和训练流程不需要安装。

切换到本实验验证过的外部 revision：

```bash
cd mast-bridge
source configs/reproduction/mast_small_13.env
git -C ../external/freegsnke checkout "$FREEGSNKE_REV"
git -C ../external/tokamind checkout "$TOKAMIND_REV"
```

推荐目录：

```text
fusion-workspace/
├── mast-bridge/
│   └── configs/
│       ├── shot_lists/
│       └── time_grids/
├── external/
│   ├── freegsnke/
│   └── tokamind/
├── data/
│   ├── raw/mast/
│   ├── processed/real/
│   ├── processed/synthetic/
│   └── manifests/
├── runs/
└── artifacts/
```

初始化路径配置：

```bash
python3 scripts/bootstrap_workspace.py --write-config
```

这会生成 `configs/paths.local.yaml`。该文件只记录本机路径，不应提交。

## 2. Python 环境

四个阶段使用四个独立环境，避免 FreeGSNKE 依赖污染下载、数据处理和训练阶段。本文
结果使用 Python 3.12.13。
以下命令均从 `fusion-workspace/mast-bridge`（仓库根目录）运行。

### 2.1 mast-download 下载环境

用于调用 `s5cmd` 下载 MAST Level 2 Zarr。`s5cmd` 是独立可执行文件，不是 Python
包。macOS Homebrew 安装：

```bash
brew install peak/tap/s5cmd
```

Linux 可从 `https://github.com/peak/s5cmd/releases` 安装官方预编译文件。当前测试
版本为 `s5cmd v2.3.0`。

```bash
python3.12 -m venv .mast-download-env
source .mast-download-env/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e . --no-deps
```

确认：

```bash
python scripts/doctor.py --skip-imports
s5cmd version
python scripts/download_mast_shots.py --shot 11766 --dry-run
```

### 2.2 mast-process 数据处理环境

用于读取 Zarr、生成 machine pickles、生成 Lao/EFIT NPZ。不导入 FreeGSNKE：

```bash
python3.12 -m venv .mast-process-env
source .mast-process-env/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e . --no-deps
python -m pip install \
  "numpy==2.5.1" \
  "zarr==3.2.1" \
  "xarray==2026.7.0" \
  "scipy==1.18.0" \
  "matplotlib==3.11.1"
```

确认：

```bash
python -m unittest discover -s tests
python -m py_compile \
  scripts/download_mast_shots.py \
  scripts/build_machine_from_zarr.py \
  scripts/build_lao_fit_npz.py \
  scripts/inspect_shot.py
```

### 2.3 freegsnke-solve 正问题求解环境

用于 FreeGSNKE forward solve：

```bash
python3.12 -m venv .freegsnke-solve-env
source .freegsnke-solve-env/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e . --no-deps
python -m pip install -e "../external/freegsnke[freegs4e]"

# Downloaded MAST stores are Zarr 3. Override FreeGSNKE's older pins after install.
python -m pip install --force-reinstall --no-deps \
  "numpy==2.2.6" \
  "scipy==1.15.3" \
  "zarr==3.2.1"
python -m pip install "donfig==0.8.1.post1" "google-crc32c==1.8.0"
```

确认：

```bash
python - <<'PY'
import numpy
import scipy
import freegs4e
import freegsnke

print("numpy:", numpy.__version__)
print("scipy:", scipy.__version__)
print("freegs4e:", freegs4e.__file__)
print("freegsnke:", freegsnke.__file__)
PY
```

如果之后再次运行 `pip install -e "../external/freegsnke[freegs4e]"`，pip 可能把 NumPy 降回 1.26，需要重新执行上面的 override。

### 2.4 tokamind-train 训练环境

训练阶段使用独立环境，放在 `mast-bridge/` 目录下：

```bash
python3.12 -m venv .tokamind-train-env
source .tokamind-train-env/bin/activate
python -m pip install --upgrade pip setuptools wheel

# 本文结果在 macOS Apple Silicon 上使用该版本；CUDA 主机按 PyTorch 官方说明
# 安装同一 torch 版本对应的 wheel。
python -m pip install "torch==2.13.0"

python -m pip install -e . --no-deps
python -m pip install -e ../external/tokamind --no-deps
python -m pip install \
  "numpy==2.5.1" \
  "zarr==3.2.1" \
  "xarray==2026.7.0" \
  "scipy==1.18.0" \
  "matplotlib==3.11.1" \
  "pyyaml==6.0.3" \
  "tqdm==4.70.0" \
  "psutil==7.2.2"
```

确认：

```bash
python - <<'PY'
import torch
import zarr
import mmt

print("torch:", torch.__version__)
print("mps_available:", torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False)
print("zarr:", zarr.__version__)
print("mmt:", mmt.__file__)
PY
```

`.tokamind-train-env/` 是本地环境目录，不应提交。

## 3. 配置运行规模

复现本文记录的 13 炮实验时，不要自行修改 shot 或求解参数。加载已提交的配置：

```bash
source configs/reproduction/mast_small_13.env
cat "$SHOT_LIST"
```

该配置固定：

```text
13 shots
uniform_random
2 variants per fitted shot/time
seed = 20260729
time window = [0.12, 0.24] s
grid = 65 x 65
solver tolerance = 1e-8
max iterations = 500
```

可以另外创建 2-4 炮的 smoke 配置检查环境，但这种自定义数据集不能复现本文的
`624/219/171/390` 计数，也不能直接使用固定的 validation shots 和评估结果。smoke
运行必须使用独立的 `VARIANT_CSV`、`SYNTH_DIR`、`MANIFEST_DIR/PREFIX` 和 run
directory，不能写入本节正式路径。

## 4. 批处理流程

按本节顺序运行。正式复现要求 13 炮全部下载成功；缺少任一炮时应停止，不要用较小的
`ACTIVE_SHOT_LIST` 继续并声称复现了本文结果。

### 4.1 下载数据

```bash
source .mast-download-env/bin/activate
source configs/reproduction/mast_small_13.env
set -euo pipefail
```

先检查命令：

```bash
while read shot; do
  [ -z "$shot" ] && continue
  python scripts/download_mast_shots.py \
    --data-dir "$DATA_DIR" \
    --shot "$shot" \
    --dry-run
done < "$SHOT_LIST"
```

下载并生成实际可处理列表。下载器会检查 `s5cmd` 返回码，并要求每个 `<shot>.zarr`
至少包含 `equilibrium`、`magnetics`、`pf_active`、`pf_passive` 和 `wall` 的 Zarr
metadata。只有命令成功后才写入同目录的
`.<shot>.zarr.mast-bridge-complete` 标记；中断下载没有标记，重新运行时不会被跳过：

```bash
set -euo pipefail
: > "$ACTIVE_SHOT_LIST"
while read shot; do
  [ -z "$shot" ] && continue
  python scripts/download_mast_shots.py \
    --data-dir "$DATA_DIR" \
    --shot "$shot" >&2
  echo "$shot" >> "$ACTIVE_SHOT_LIST"
done < "$SHOT_LIST"

if ! cmp -s "$SHOT_LIST" "$ACTIVE_SHOT_LIST"; then
  echo "The 13-shot download is incomplete; stop before processing." >&2
  diff -u "$SHOT_LIST" "$ACTIVE_SHOT_LIST" || true
  exit 1
fi
```

从这里开始，后续所有 `while read shot` 都以
`done < "$ACTIVE_SHOT_LIST"` 结束，不要继续使用原始 `$SHOT_LIST`。

### 4.2 生成 Machine Pickle

```bash
source .mast-process-env/bin/activate
source configs/reproduction/mast_small_13.env
```

生成 machine pickles：

```bash
while read shot; do
  [ -z "$shot" ] && continue
  if [ ! -d "$DATA_DIR/${shot}.zarr" ]; then
    echo "Skipping missing Zarr: $shot" >&2
    continue
  fi
  python scripts/build_machine_from_zarr.py \
    --data-dir "$DATA_DIR" \
    --output-dir "$DATA_DIR/machine/$shot" \
    --shot "$shot" \
    --overwrite
done < "$ACTIVE_SHOT_LIST"
```

检查：

```bash
while read shot; do
  [ -z "$shot" ] && continue
  python scripts/inspect_shot.py \
    --data-dir "$DATA_DIR" \
    --machine-dir "$DATA_DIR/machine/$shot" \
    --shot "$shot"
done < "$ACTIVE_SHOT_LIST"
```

每个 shot 应生成：

```text
data/raw/mast/machine/<shot>/
├── MAST_active_coils.pickle
├── MAST_limiter.pickle
├── MAST_magentic_probes.pickle
├── MAST_passive_coils.pickle
└── MAST_wall.pickle
```

`magentic` 是为兼容现有 FreeGSNKE loader 保留的历史拼写。早期版本曾写出
`MAST_passive_coilds.pickle`；当前标准文件名已修正为
`MAST_passive_coils.pickle`，loader 仍兼容读取旧 typo 文件名。

### 4.3 生成 Lao/EFIT NPZ

该步骤只从真实 MAST Level 2 Zarr 中拟合 Lao85 profile 参数，供后续扰动 synthetic
样本使用。真实样本本身 **不需要** 运行 FreeGSNKE 正问题；真实标签直接来自 Zarr
里已有的 EFIT equilibrium，例如 `equilibrium/psi`。

FreeGSNKE forward solve 只用于扰动 Lao85 源项后的 synthetic variants。统一 Lao
fit NPZ 路径：

```text
data/processed/real/lao_parameter_ensemble/all_zarr_lao_parameter_fits.npz
```

生成：

```bash
source .mast-process-env/bin/activate
source configs/reproduction/mast_small_13.env

python scripts/build_lao_fit_npz.py \
  --shot-list "$ACTIVE_SHOT_LIST" \
  --data-dir "$DATA_DIR" \
  --output "$FIT_PATH"
```

该脚本从 Level 2 Zarr 读取：

```text
equilibrium/psi_norm
equilibrium/dpressure_dpsi
equilibrium/f_df_dpsi
equilibrium/bvac_rmag
equilibrium/magnetic_axis_r
magnetics/ip
```

并写出：

```text
shot
time
ip
fvac
freegsnke_alpha
freegsnke_beta
```

含义：

- `shot/time`：真实 EFIT equilibrium 的时间点。
- `ip/fvac/freegsnke_alpha/freegsnke_beta`：该 shot/time 对应的 Lao85 profile 参数。
  其中 `fvac = abs(bvac_rmag * magnetic_axis_r)`，即 FreeGSNKE Lao85 需要的
  vacuum toroidal field radius product `R * B_tor`，单位是 `T*m`。
- 这些参数是后续 synthetic 源项扰动的基准；真实数据训练标签仍然读取
  `data/raw/mast/<shot>.zarr/equilibrium/psi`。

### 4.4 可选的 FreeGSNKE 基准求解

```bash
source .freegsnke-solve-env/bin/activate
source configs/reproduction/mast_small_13.env
TIME_GRID=configs/time_grids/baseline_times.txt
BASELINE_SYNTH_DIR=../data/processed/synthetic_baseline
```

注意：这一节是可选的 baseline solve，用来检查某个真实 `(shot,time)` 在未扰动
Lao85 参数时是否容易收敛。正式的数据增强样本建议走 4.5 的 variant CSV +
`run_lao85_variant_solve_batch.py` 批处理。不要对真实数据运行这里的求解来替代
EFIT 标签；真实数据已经在 Zarr 中包含 `equilibrium/psi`。

先检查拟合文件：

```bash
test -f "$FIT_PATH" || {
  echo "Missing Lao fit NPZ: $FIT_PATH" >&2
  echo "Run scripts/build_lao_fit_npz.py in the mast-process environment first." >&2
  exit 1
}
```

批量求解：

```bash
while read shot; do
  [ -z "$shot" ] && continue
  if [ ! -d "$DATA_DIR/${shot}.zarr" ]; then
    echo "Skipping missing Zarr: $shot" >&2
    continue
  fi
  if [ ! -d "$DATA_DIR/machine/$shot" ]; then
    echo "Skipping missing machine directory: $shot" >&2
    continue
  fi
  while read time; do
    [ -z "$time" ] && continue
    output_dir="$BASELINE_SYNTH_DIR/${shot}_t${time}"
    if [ -f "$output_dir/equilibrium.npz" ] && [ -f "$output_dir/metadata.json" ]; then
      echo "Skipping existing sample: ${shot}_t${time}"
      continue
    fi
    python scripts/run_freegsnke_forward.py \
      --data-dir "$DATA_DIR" \
      --machine-dir "$DATA_DIR/machine/$shot" \
      --fit-path "$FIT_PATH" \
      --shot "$shot" \
      --time "$time" \
      --nx 65 \
      --ny 65 \
      --tolerance 1e-8 \
      --max-iterations 500 \
      --output-dir "$output_dir"
  done < "$TIME_GRID"
done < "$ACTIVE_SHOT_LIST"
```

每个 baseline synthetic 样本输出：

```text
data/processed/synthetic_baseline/<shot>_t<time>/
├── equilibrium.npz
├── metadata.json
└── equilibrium.png
```

如果某个 solve 没达到收敛阈值，`metadata.json` 和终端输出会记录状态。小网格 smoke run 可用于检查链路，不代表正式求解参数。

#### 正问题边界类型和源项

当前 `scripts/run_freegsnke_forward.py` 调用的是 FreeGSNKE 官方
`GSstaticsolver.NKGSsolver.solve(..., constrain=None)`。在 FreeGSNKE 中，
`constrain=None` 会进入 forward mode，也就是固定线圈电流后求解非线性
free-boundary Grad-Shafranov 正问题。

因此这个正问题求解是 **free-boundary**，不是预先给定 LCFS 的 fixed-boundary
求解。LCFS/limiter topology 是求解过程中由总磁通、X-point/O-point 和 limiter
关系决定的；这也解释了为什么有些图能画出清晰 LCFS，有些图没有合格 LCFS。
图片只是 QC 视图，训练准入以 solver metadata 的严格收敛和 finite `psi` 为准。

当前正问题的主要源项是：

- active/passive coil currents：从真实 Zarr 的 `pf_active` 和 `pf_passive` 电流插值得到，写入 tokamak 对象后产生 vacuum/tokamak flux `psi_tokamak`。
- Lao85 plasma profile：从拟合 NPZ 读取 `Ip`、`fvac`、`freegsnke_alpha`、`freegsnke_beta`，实例化 `freegsnke.jtor_update.Lao85`，通过 `Jtor(psi_tokamak + psi_plasma)` 产生等离子体环向电流密度源项。
- machine geometry / limiter：由每个 shot 的 machine pickle 提供线圈、被动结构和 limiter/wall 几何；FreeGSNKE 用 Green's function 计算 plasma current 对边界磁通的贡献。
- solve grid：从真实 Zarr 的 `equilibrium/major_radius` 和 `equilibrium/z` 读取
  `Rmin/Rmax/Zmin/Zmax`，使 synthetic `psi` 网格边界和 MAST EFIT 网格一致。

脚本没有传入 magnetic constraints，也不会优化 coil currents；它只在给定真实
coil currents 和 Lao85 profile 的条件下求解 plasma response。

### 4.5 Lao85 参数扰动规则

基于真实数据生成 synthetic variants 时，只扰动已经拟合出来的 Lao85 profile
参数，不直接扰动原始诊断数据：

```text
Ip'      = Ip_fit * ip_scale
fvac'    = fvac_fit * fvac_scale
alpha_i' = alpha_i_fit * alpha_scale + alpha_offset
beta_i'  = beta_i_fit  * beta_scale  + beta_offset
Icoil'   = Icoil_real  * coil_current_scale
```

`scripts/run_freegsnke_forward.py` 支持的扰动参数：

```bash
--ip-scale
--fvac-scale
--alpha-scale
--beta-scale
--alpha-offset
--beta-offset
--coil-current-scale
```

`src/mast_bridge/simulation/variants.py` 当前恢复为最初的 deterministic uniform
random batch variant rows。做法是：对每个已经在
`all_zarr_lao_parameter_fits.npz` 中拟合成功的 `(shot,time)`，每个 variant 独立从
固定边界内做均匀随机扰动。`seed` 固定后 CSV 可复现。

当前默认采样边界恢复为最初的大范围候选：

```text
ip_scale          in [0.95, 1.05]
fvac_scale        in [0.99, 1.01]
alpha_scale       in [0.98, 1.02]
beta_scale        in [0.98, 1.02]
alpha_offset      in [-0.01, 0.01]
beta_offset       in [-0.01, 0.01]
coil_current_scale in [0.97, 1.03]
```

采样代码里有注释说明未来如何替换采样方式：优先修改
`src/mast_bridge/simulation/variants.py` 中的 `build_variant_rows()`。如果以后要改成
Gaussian、Latin Hypercube 或基于基准收敛性的预筛选采样，只需要保持输出 row 字段
不变，后续 FreeGSNKE 批量求解脚本就不需要改。

从拟合 NPZ 生成采样任务 CSV：

```bash
source .freegsnke-solve-env/bin/activate
source configs/reproduction/mast_small_13.env

python scripts/build_lao85_variant_rows.py \
  --fit-path "$FIT_PATH" \
  --variants-per-point "$VARIANTS_PER_POINT" \
  --seed "$SAMPLING_SEED" \
  --min-time "$MIN_TIME" \
  --max-time "$MAX_TIME" \
  --output "$VARIANT_CSV"
```

建议先使用 `--min-time/--max-time` 避开 shot startup 或 ramp-down 阶段。例如从
CSV 开头直接跑时，`11766, t=0.03s` 附近的样本多为 early limiter plasma，图片会
明显比平顶段或较稳定阶段更奇怪；这不一定是采样错误，但不适合作为
第一批训练样本。若研究目标需要包含启动阶段，可以单独建一个 startup 数据集，不要
和主训练集混在一起。

CSV 每行包含：

```text
shot,target_time,variant_id,sampling_method,
ip_scale,fvac_scale,alpha_scale,beta_scale,alpha_offset,beta_offset,coil_current_scale
```

批量正问题求解并立即 strict filter：

```bash
source .freegsnke-solve-env/bin/activate
source configs/reproduction/mast_small_13.env

python scripts/run_lao85_variant_solve_batch.py \
  --variant-csv "$VARIANT_CSV" \
  --data-dir "$DATA_DIR" \
  --fit-path "$FIT_PATH" \
  --synthetic-root "$SYNTH_DIR" \
  --manifest-dir "$MANIFEST_DIR" \
  --prefix "$PREFIX" \
  --task task_1-3 \
  --nx "$NX" \
  --ny "$NY" \
  --tolerance "$SOLVER_TOLERANCE" \
  --max-solver-tolerance "$SOLVER_TOLERANCE" \
  --max-iterations "$MAX_SOLVER_ITERATIONS"
```

正式复现命令不带 `--limit`，会处理 CSV 全部 624 行。smoke 检查只使用
`--limit 20 --dry-run` 查看命令，不要在正式 `SYNTH_DIR/PREFIX` 中先实际求解 20 行。
脚本仅跳过 metadata 中 variant 参数、网格、容差和最大迭代数都与当前 CSV/命令一致的
已有结果；不一致会直接报错，避免静默混入旧 seed 或旧求解配置。需要重算时显式加
`--rerun-existing`，该模式会先删除该样本已有的 equilibrium、metadata、diagnostics
和 QC 图；即使新求解失败，旧结果也不会重新进入 accepted manifest。

注意：正式筛选建议使用 `--max-iterations 500`，不要继续沿用早期 smoke run 的
`100`。部分 accepted 样本需要超过 100 次迭代才达到 `1e-8`；如果只允许 100 次，
会提前截断这类本来可以收敛的样本。

运行时终端会逐个样本打印进度，例如：

```text
[solve-start] row=0 sample=11766_t0.1_v000 shot=11766 time=0.1 variant=v000
[solve-solved] row=0 sample=11766_t0.1_v000 return_code=0
```

如果长时间只看到 `[solve-start]`，通常表示当前 FreeGSNKE 单个正问题仍在求解。脚本
会在每个样本结束后实时更新 batch report，可以另开一个终端查看：

```bash
tail -f "$MANIFEST_DIR/${PREFIX}_batch_report.jsonl"
```

该脚本会写出：

```text
data/processed/synthetic_lao85_uniform_small_iter500/<shot>_t<time>_<variant_id>/
data/manifests/tokamark_lao85_uniform_small_iter500_batch_report.jsonl
data/manifests/tokamark_lao85_uniform_small_iter500_synthetic_accepted.jsonl
data/manifests/tokamark_lao85_uniform_small_iter500_synthetic_rejected.jsonl
```

`batch_report.jsonl` 记录每个采样 row 的执行状态；accepted/rejected manifest 由
同一个脚本在求解结束后调用 `scripts/build_synthetic_manifest.py` 生成。训练时只使用
`*_synthetic_accepted.jsonl`，不要把 rejected 样本并入训练。
过滤器只处理当前 `VARIANT_CSV` 中的 sample ID，即使 `SYNTH_DIR` 中存在其他旧目录也
不会并入正式 manifest。当前 624 行中有 616 个进程正常产出 equilibrium：
`219 accepted + 397 rejected = 616`；另外 8 个进程失败只记录在 batch report。

注意：`coil_current_scale` 会同步作用到 active 和 passive coil currents，并写入
`metadata.json` 的 `coil_currents`。早期已经生成的 synthetic 数据如果 metadata
中没有 `coil_current_scale` 或仍使用硬编码网格，应视为旧版本样本；需要用当前代码
重跑才能得到单位、线圈扰动和网格边界都一致的新样本。

完整语义是：

```text
real sample:
  input  = real shot/time diagnostics
  label  = real Zarr EFIT equilibrium/psi
  solve  = no FreeGSNKE forward solve

synthetic sample:
  input  = same real shot/time machine + coil currents
  source = Lao85 parameters fitted from the corresponding real shot/time, then perturbed
  label  = FreeGSNKE free-boundary forward solve after perturbation
  keep   = only if solver_converged and solver_final_tolerance <= 1e-8
```

### 4.6 过滤收敛的仿真样本

训练 manifest 不直接扫描目录拼接样本，必须先运行严格过滤脚本。默认规则是：

- `solver_converged == true`
- `solver_final_tolerance <= 1e-8`
- `equilibrium.npz` 中 `psi` 是 finite 的 `65x65` 网格

过滤过程写在 `src/mast_bridge/dataset/synthetic_manifest.py`，核心函数是
`rejection_reason()`、`synthetic_entries()` 和 `rejected_samples()`。执行流程是：

1. 扫描 `--synthetic-root` 下、且列在 `--variant-csv` 中的 synthetic sample 目录。
2. 检查是否同时存在 `metadata.json` 和 `equilibrium.npz`。
3. 读取 `metadata.json` 中 FreeGSNKE 写出的求解状态。
4. 若 `solver_converged` 不是 `true`，样本进入 rejected，原因是
   `solver_not_converged`。
5. 若缺少或无法解析 `solver_final_tolerance`，样本进入 rejected，原因是
   `solver_tolerance_missing`。
6. 若 `solver_final_tolerance` 是 NaN/Inf，样本进入 rejected，原因是
   `solver_tolerance_nonfinite`。
7. 若 `solver_final_tolerance > --max-solver-tolerance`，样本进入 rejected，原因是
   `solver_tolerance_above_threshold`。当前默认阈值是 `1e-8`。
8. 若 `equilibrium.npz` 中 `psi` 不存在、不是 `65x65`、无法读取或包含 non-finite
   值，样本进入 rejected，原因是 `invalid_equilibrium`。
9. 只有所有检查都通过的样本才写入 accepted manifest，作为训练候选样本。

未通过的样本不会被物理删除；它们会进入 rejected report，用于后续分析
shot/time/扰动参数为什么不收敛。也就是说，目录里的仿真结果是原始求解输出，
训练集入口必须以 `*_synthetic_accepted.jsonl` 为准。

如果使用 `scripts/run_lao85_variant_solve_batch.py`，脚本会在批量求解结束后自动
调用同一个过滤逻辑生成 accepted/rejected manifest。若求解被中断，或者你手动改了
`--max-solver-tolerance`，可以单独重跑下面这个过滤命令；它不会重新运行
FreeGSNKE，只会重新扫描已有 `metadata.json` 和 `equilibrium.npz`。

```bash
source .freegsnke-solve-env/bin/activate
source configs/reproduction/mast_small_13.env
mkdir -p "$MANIFEST_DIR"

python scripts/build_synthetic_manifest.py \
  --synthetic-root "$SYNTH_DIR" \
  --variant-csv "$VARIANT_CSV" \
  --output "$MANIFEST_DIR/${PREFIX}_synthetic_accepted.jsonl" \
  --rejected-output "$MANIFEST_DIR/${PREFIX}_synthetic_rejected.jsonl" \
  --task task_1-3 \
  --max-solver-tolerance "$SOLVER_TOLERANCE"
```

输出：

```text
data/manifests/
├── tokamark_lao85_uniform_small_iter500_synthetic_accepted.jsonl
└── tokamark_lao85_uniform_small_iter500_synthetic_rejected.jsonl
```

`accepted` 才能进入后续数据集候选；`rejected` 只作为 QC 报告，不进入对比实验数据集。
当前 rejected report 会记录稳定的 rejection reason，例如
`solver_not_converged`、`solver_tolerance_missing`、
`solver_tolerance_nonfinite`、`solver_tolerance_above_threshold` 或
`invalid_equilibrium`。

### 4.7 生成合成磁诊断并构建三组对比实验 Manifest

TokaMind 的输入应是诊断量而不是 Lao85 参数。严格过滤完成后，先为 accepted
synthetic equilibrium 生成 `<synthetic-sample>/diagnostics.npz`。该过程不会重新
运行 Grad-Shafranov 迭代求解，而是从已保存的总 `psi`、求解时的 coil currents 和
Lao85 参数重建最终 equilibrium 状态，再调用 FreeGSNKE 官方 probe calculator。

```bash
source .freegsnke-solve-env/bin/activate
source configs/reproduction/mast_small_13.env

python scripts/build_synthetic_magnetic_diagnostics.py \
  --accepted-manifest "$MANIFEST_DIR/${PREFIX}_synthetic_accepted.jsonl" \
  --data-dir "$DATA_DIR" \
  --report "$MANIFEST_DIR/${PREFIX}_diagnostics_report.jsonl"
```

脚本默认跳过已经存在且通过校验的 `diagnostics.npz`，可以直接断点续跑。若要覆盖重算，
增加 `--overwrite`。report 中 generated 样本的
`psi_reconstruction_max_abs_error` 应接近机器精度，当前数据为 `0.0`。

`diagnostics.npz` 的固定字段为：

```text
schema_version
target_time
magnetics_ip
flux_loop_names, flux_loop_values
pickup_names, pickup_families, pickup_values
active_coil_names, active_coil_values
flux_loop_scale
```

flux-loop 使用 `2*pi` 从 FreeGSNKE 的 `Wb/(2*pi)` 转为 MAST Level 2 的 `Wb`；
pickup 使用修正后的 CCBV/OBR/OBV 方向。文件校验会拒绝缺字段、数组长度不一致、重复
通道以及 NaN/Inf。

随后用通过 solver 和 diagnostics 两层过滤的 synthetic 样本反推出对应真实
`shot/time`，构建真实、仿真和混合三组 manifest：

```bash
python scripts/build_experiment_manifests.py \
  --accepted-synthetic "$MANIFEST_DIR/${PREFIX}_synthetic_accepted.jsonl" \
  --raw-data-dir "$DATA_DIR" \
  --fit-path "$FIT_PATH" \
  --output-dir "$MANIFEST_DIR" \
  --prefix "${PREFIX}_diagnostics" \
  --task task_1-3 \
  --require-synthetic-diagnostics
```

输出：

```text
data/manifests/
├── tokamark_lao85_uniform_small_iter500_diagnostics_real_only.jsonl
├── tokamark_lao85_uniform_small_iter500_diagnostics_synthetic_only.jsonl
└── tokamark_lao85_uniform_small_iter500_diagnostics_real_plus_synthetic.jsonl
```

当前数据统计：

```text
synthetic accepted with valid diagnostics     219
diagnostics real-only                         171
diagnostics synthetic-only                    219
diagnostics real-plus-synthetic               390
```

`--require-synthetic-diagnostics` 要求 synthetic 样本同时通过 `1e-8` solver filter 和
`diagnostics.npz` 内容校验。`real_only` 的 171 个真实点来自这些 synthetic 样本
反推的唯一 `(parent_shot, target_time)`；`real_plus_synthetic` 是两者并集。

real manifest 的 `label_source` 是 `zarr_equilibrium_psi`，真实标签直接读取 EFIT
`equilibrium/psi`，不运行 FreeGSNKE。synthetic manifest 的标签来自
`equilibrium_path`，输入来自 `diagnostics_path`。

## 5. 当前 uniform_iter500 复现流程

当前正式复现顺序只有一条：

1. 按第 1-2 节创建 workspace、固定外部 revision 并安装四个环境。
2. `source configs/reproduction/mast_small_13.env`。
3. 严格按 4.1-4.3 下载 13 炮、生成 machine 和 Lao fit。
4. 4.4 只用于可选 baseline；正式增强直接运行 4.5。
5. 按 4.6 严格过滤，再按 4.7 生成 diagnostics 和三组 manifest。
6. 按第 8 节 dry-run、训练三组模型并统一评估。

阶段计数应为：

```text
downloaded Zarr shots:                   13
Lao fit rows:                           752
variant rows / batch attempts:          624
FreeGSNKE process solved:               616
FreeGSNKE process failed:                 8
strict synthetic accepted:             219
strict synthetic rejected:             397
accepted with valid diagnostics:       219
diagnostics real-only:                  171
diagnostics synthetic-only:             219
diagnostics real-plus-synthetic:        390
```

检查命令：

```bash
source .mast-process-env/bin/activate
source configs/reproduction/mast_small_13.env

cmp "$SHOT_LIST" "$ACTIVE_SHOT_LIST"
wc -l "$ACTIVE_SHOT_LIST"
python -c 'import numpy as np; print(len(np.load("'"$FIT_PATH"'")["shot"]))'
test "$(($(wc -l < "$VARIANT_CSV") - 1))" -eq 624
python -c 'import collections,json,sys; print(collections.Counter(json.loads(x)["batch_status"] for x in open(sys.argv[1])))' \
  "$MANIFEST_DIR/${PREFIX}_batch_report.jsonl"
wc -l \
  "$MANIFEST_DIR/${PREFIX}_synthetic_accepted.jsonl" \
  "$MANIFEST_DIR/${PREFIX}_synthetic_rejected.jsonl" \
  "$MANIFEST_DIR/${PREFIX}_diagnostics_real_only.jsonl" \
  "$MANIFEST_DIR/${PREFIX}_diagnostics_synthetic_only.jsonl" \
  "$MANIFEST_DIR/${PREFIX}_diagnostics_real_plus_synthetic.jsonl"
```

如果计数不同，不要直接进入训练。先检查下载完整性、外部 revision、variant CSV、
batch report 和 rejected reason。不同硬件上的浮点细节可能影响接近 `1e-8` 边界的
个别 FreeGSNKE 样本，因此应同时记录实际 accepted/rejected 数量。

## 6. 单个 Shot 命令

下载一个 shot：

```bash
source .mast-download-env/bin/activate
python scripts/download_mast_shots.py \
  --data-dir ../data/raw/mast \
  --shot 11771
```

生成 machine：

```bash
source .mast-process-env/bin/activate
python scripts/build_machine_from_zarr.py \
  --data-dir ../data/raw/mast \
  --output-dir ../data/raw/mast/machine/11771 \
  --shot 11771 \
  --overwrite
python scripts/inspect_shot.py \
  --data-dir ../data/raw/mast \
  --machine-dir ../data/raw/mast/machine/11771 \
  --shot 11771
```

生成 Lao NPZ：

```bash
source .mast-process-env/bin/activate
printf "11771\n" > /tmp/one_shot.txt
python scripts/build_lao_fit_npz.py \
  --shot-list /tmp/one_shot.txt \
  --data-dir ../data/raw/mast \
  --output ../data/processed/real/lao_parameter_ensemble/all_zarr_lao_parameter_fits.npz
```

求解一个 shot/time：

```bash
source .freegsnke-solve-env/bin/activate
python scripts/run_freegsnke_forward.py \
  --data-dir ../data/raw/mast \
  --machine-dir ../data/raw/mast/machine/11771 \
  --fit-path ../data/processed/real/lao_parameter_ensemble/all_zarr_lao_parameter_fits.npz \
  --shot 11771 \
  --time 0.16 \
  --nx 65 \
  --ny 65 \
  --tolerance 1e-8 \
  --max-iterations 500 \
  --output-dir ../data/processed/synthetic/11771_t0.16
```

## 7. 数据约定

### 7.1 Raw 和 Real 数据

MAST Level 2 Zarr 是默认输入。它通常包含：

```text
summary
pulse_schedule
pf_active
pf_passive
magnetics
equilibrium
wall
```

本项目需要的真实数据产物：

```text
data/raw/mast/<shot>.zarr/
data/raw/mast/machine/<shot>/*.pickle
data/processed/real/lao_parameter_ensemble/all_zarr_lao_parameter_fits.npz
```

真实标签直接来自 `data/raw/mast/<shot>.zarr/equilibrium/psi`。Lao fit NPZ 是
从真实 EFIT profile 拟合出的 profile 参数表，用来构造 synthetic perturbation；
它不是真实样本的标签文件。

Level 1 保留原始诊断名和原始采集形态，不作为默认训练输入。

### 7.2 仿真数据

`run_freegsnke_forward.py` 会保存：

```text
equilibrium.npz:
  psi
  R
  Z
  psi_axis
  psi_bndry

metadata.json:
  parent_shot
  target_time
  fitted_time
  machine_geometry_source
  fit_path
  coil_currents
  Ip
  fvac
  alpha
  beta
  lao85_perturbation
  coil_current_scale
  grid
  target_relative_tolerance
  solver_status
  solver_converged
  solver_final_tolerance
  solver_requested_tolerance
  solver_iterations
  xpt_count
  opt_count
  flag_limiter
```

固定 geometry 和随时间变化的状态分开记录：

```text
fixed machine geometry
  + active/passive coil currents at t
  + Lao/EFIT profile at nearest fitted time
  = one FreeGSNKE solve input
```

不要为每个时间点重复保存完全相同的 machine geometry payload。

### 7.3 磁诊断合成对比约定

`scripts/compare_freegsnke_magnetic_diagnostics.py` 用 FreeGSNKE 求出的
equilibrium 合成磁诊断，并和 MAST Level 2 的真实诊断比较。第一版只比较：

```text
magnetics/flux_loop_flux
magnetics/b_field_pol_probe_{ccbv,obr,obv}_field
```

这一步有几个容易踩坑的数据约定，必须保留：

1. **Flux loop 通道不能按 index 绑定几何。**
   `magnetics/flux_loop_channel` 只有实测信号通道，例如 `CC03`、`P3U/1`；
   `magnetics/flux_loop_geometry_channel` 是完整几何列表，例如 `FL_CC03`、
   `FL_P3U_1`。两者长度和顺序不一定一一对应。构造 FreeGSNKE probe 时必须按通道名
   映射几何：

   ```text
   CC03  -> FL_CC03
   P3U/1 -> FL_P3U_1
   ```

   早期按 index 配对会把 `CC03` 放到 `FL_P2U_1` 的位置，导致 flux-loop 合成量整体
   偏离。

2. **Flux loop 单位需要乘 `2π`。**
   FreeGSNKE/EFIT 的 poloidal flux 约定是 `Wb/(2π)`；MAST Level 2
   `magnetics/flux_loop_flux` 是 Wb 量级。比较脚本默认对 FreeGSNKE flux-loop
   模型值乘 `2π`，并把实际使用的 `flux_loop_scale` 写入
   `diagnostic_summary.json`。

3. **Pickup probe 方向要匹配 Level 2 field 约定。**
   `CCBV` 使用 `[0, 0, 1]`，`OBR` 使用 `[1, 0, 0]`，`OBV` 使用 `[0, 0, 1]`。
   旧 pickle 曾把 `OBR/OBV` 设为相反方向，会造成整族 pickup 符号反转。

比较脚本不会直接修改 `data/raw/mast/machine/<shot>/` 下的旧 pickle；它会复制一份
临时 machine payload，在临时目录中修正 flux-loop 几何和 pickup 方向后再交给
FreeGSNKE。重新运行 `scripts/build_machine_from_zarr.py --overwrite` 生成的新 pickle
会包含这些修正。

示例：

```bash
source .freegsnke-solve-env/bin/activate

python scripts/compare_freegsnke_magnetic_diagnostics.py \
  --shot 11771 \
  --time 0.18 \
  --nx 65 \
  --ny 65 \
  --tolerance 1e-8 \
  --max-iterations 500
```

输出：

```text
data/processed/diagnostic_comparisons/<shot>_t<time>/
  freegsnke_equilibrium.npz
  diagnostic_comparison.npz
  diagnostic_comparison.csv
  diagnostic_summary.json

artifacts/freegsnke_magnetic_diagnostics/<shot>_t<time>/
  diagnostic_observed_vs_model.png
  current_global_constraint.png
```

`diagnostic_comparison.csv` 里的 `model` 已经应用了上述约定：flux-loop 是 Wb，
pickup 是按 Level 2 方向约定得到的 `B.n`。

`current_global_constraint.png` 单独画总等离子体电流约束：`observed` 来自
`magnetics/ip` 在目标时刻的插值，`model` 是传给 Lao85/FreeGSNKE 的 `Ip`。
同样的数值也会写入 `diagnostic_summary.json` 的 `current_constraint` 字段。

### 7.4 Manifest 清单

训练、验证、测试必须按 shot 划分。同一个 shot 的真实样本和所有 synthetic variants 必须在同一个 split，避免信息泄漏。

推荐 manifest：

```text
data/manifests/
├── tokamark_lao85_uniform_small_iter500_synthetic_accepted.jsonl
├── tokamark_lao85_uniform_small_iter500_synthetic_rejected.jsonl
├── tokamark_lao85_uniform_small_iter500_diagnostics_real_only.jsonl
├── tokamark_lao85_uniform_small_iter500_diagnostics_synthetic_only.jsonl
└── tokamark_lao85_uniform_small_iter500_diagnostics_real_plus_synthetic.jsonl
```

`*_synthetic_accepted.jsonl` 由 `scripts/build_synthetic_manifest.py` 生成；不要手写，
也不要把 rejected 样本并入 train/val/test。`*_real_only.jsonl`、
`*_synthetic_only.jsonl` 和 `*_real_plus_synthetic.jsonl` 由
`scripts/build_experiment_manifests.py --require-synthetic-diagnostics` 生成。

每行至少包含：

```json
{
  "sample_id": "11766_t0.155_v000",
  "source": "synthetic",
  "shot_id": "11766_t0.155_v000",
  "parent_shot": "11766",
  "target_time": 0.155,
  "data_path": "/path/to/synthetic/11766_t0.155_v000",
  "equilibrium_path": "/path/to/synthetic/11766_t0.155_v000/equilibrium.npz",
  "diagnostics_path": "/path/to/synthetic/11766_t0.155_v000/diagnostics.npz",
  "task": "task_1-3"
}
```

## 8. TokaMind 小规模训练

当前主训练入口是 `scripts/train_tokamind_diagnostics.py`：

```text
magnetics Ip + flux loops + poloidal pickup probes + active coil currents
  -> TokaMind MultiModalTransformer
  -> 65 x 65 equilibrium/psi
```

不要直接把 diagnostics manifest 交给底层 `train_tokamind_manifest.py`；
该通用入口现在要求显式指定 manifest、run directory 和 input mode，防止误用旧
Lao 参数任务。

真实输入从 MAST Zarr 的 `magnetics` 和 `pf_active` 读取；仿真输入从严格过滤样本的
`diagnostics.npz` 读取。真实标签使用 EFIT `equilibrium/psi`，不运行 FreeGSNKE；
仿真标签使用 accepted 样本的 FreeGSNKE `equilibrium.npz/psi`。Lao85 参数不再作为
这个训练任务的输入。

训练和验证按 `parent_shot`/`shot_id` 分组，同一炮的真实时间片和 synthetic variants
不会跨 split。`train_tokamind_diagnostics.py` 默认把 11768、11775、11780 固定为
三组实验共同的 validation shots，并把 train/validation 炮号和模型结构写入训练摘要。
三组实验还必须使用相同诊断通道。synthetic diagnostics 有 137 个
完整字段，但真实数据部分通道有缺测；当前 mixed manifest 上的公共 finite 特征为
94 维。固定顺序、数量和 SHA-256 digest 写在版本化文件
`configs/diagnostic_features/mast_level2_common_94.json`，训练时不会静默重新求交集。
若数据变化导致任一字段缺失或 non-finite，训练会直接报错，需要显式审查并更新 schema。

先 dry-run：

```bash
source .tokamind-train-env/bin/activate
source configs/reproduction/mast_small_13.env

COMMON="$MANIFEST_DIR/${PREFIX}_diagnostics_real_plus_synthetic.jsonl"
FEATURE_SCHEMA=configs/diagnostic_features/mast_level2_common_94.json

python scripts/train_tokamind_diagnostics.py \
  --manifest "$COMMON" \
  --feature-schema "$FEATURE_SCHEMA" \
  --run-dir ../runs/tokamind-diagnostics-mixed-dry-run \
  --dry-run
```

当前 mixed dry-run：

```text
rows: 390
sources: {"real": 171, "synthetic": 219}
train_windows: 327
val_windows: 63
feature_dim: 94
input_mode: magnetic-diagnostics
target_mode: raw-psi
```

三组对比训练：

```bash
source configs/reproduction/mast_small_13.env
REAL="$MANIFEST_DIR/${PREFIX}_diagnostics_real_only.jsonl"
SYNTHETIC="$MANIFEST_DIR/${PREFIX}_diagnostics_synthetic_only.jsonl"
COMMON="$MANIFEST_DIR/${PREFIX}_diagnostics_real_plus_synthetic.jsonl"
FEATURE_SCHEMA=configs/diagnostic_features/mast_level2_common_94.json

python scripts/train_tokamind_diagnostics.py \
  --manifest "$REAL" \
  --feature-schema "$FEATURE_SCHEMA" \
  --run-dir ../runs/tokamind-diagnostics-real-only \
  --epochs 50 --batch-size 8 --lr 1e-4

python scripts/train_tokamind_diagnostics.py \
  --manifest "$SYNTHETIC" \
  --feature-schema "$FEATURE_SCHEMA" \
  --run-dir ../runs/tokamind-diagnostics-synthetic-only \
  --epochs 50 --batch-size 8 --lr 1e-4

python scripts/train_tokamind_diagnostics.py \
  --manifest "$COMMON" \
  --feature-schema "$FEATURE_SCHEMA" \
  --run-dir ../runs/tokamind-diagnostics-real-plus-synthetic \
  --epochs 50 --batch-size 8 --lr 1e-4

python scripts/plot_tokamind_diagnostics_losses.py
```

模型调用 `external/tokamind/src/mmt` 的 `MultiModalTransformer` 和
`train_finetune`。目标 `psi` 按训练集逐网格点标准化，loss 是 TokaMind
`embed_mse`，即标准化预测 `psi` 与标准化标签 `psi` 的均方误差。每个 run 保存
`manifest_scalers.npz` 和 `manifest_training_summary.json`，其中记录公共
`feature_names`、input/output mean/std、manifest 和 loss history。
最后一个命令从三组训练摘要读取 loss history，把六条 train/validation 曲线画在同一张
图，并输出：

```text
artifacts/tokamind_loss_curves/tokamind_diagnostics_loss_curves.png
artifacts/tokamind_loss_curves/tokamind_diagnostics_loss_summary.csv
```

### 8.1 统一真实验证集评估

三组训练 loss 使用各自的标准化参数，不能直接横向比较。使用相同的真实 EFIT
validation shots（11768、11775、11780）反标准化后计算 `psi` RMSE/MAE：

```bash
source .tokamind-train-env/bin/activate
source configs/reproduction/mast_small_13.env

python scripts/evaluate_tokamind_diagnostics.py \
  --manifest "$MANIFEST_DIR/${PREFIX}_diagnostics_real_only.jsonl" \
  --run-dir ../runs/tokamind-diagnostics-real-only \
  --run-dir ../runs/tokamind-diagnostics-synthetic-only \
  --run-dir ../runs/tokamind-diagnostics-real-plus-synthetic \
  --val-shot 11768 \
  --val-shot 11775 \
  --val-shot 11780
```

该入口固定使用当前 diagnostics real-only manifest、三组 diagnostics checkpoint 和
以下 28 个真实样本：

| Shot | 样本数 | 时间切片 / s |
|---|---:|---|
| 11768 | 7 | 0.160、0.185、0.190、0.195、0.200、0.205、0.210 |
| 11775 | 10 | 0.120、0.125、0.140、0.155、0.160、0.165、0.170、0.210、0.215、0.225 |
| 11780 | 11 | 0.125、0.140、0.145、0.170、0.195、0.205、0.210、0.215、0.225、0.230、0.235 |

当前 28 个真实 validation 样本上的结果：

| 训练数据 | Raw psi RMSE | Raw psi MAE |
|---|---:|---:|
| 仅真实数据 | 0.001947 | 0.001272 |
| 仅仿真数据 | 0.044866 | 0.035819 |
| 真实 + 仿真 | 0.002213 | 0.001681 |

这些炮在训练时未进入 train split，但参与了最佳 checkpoint 的选择，因此应称为
统一真实 validation set，而不是完全独立的 test set。

评估入口会从训练摘要恢复模型结构，并检查待评估炮没有出现在 train split 中；
训练摘要或 checkpoint 缺失时会直接拒绝评估。

当前输入覆盖 flux loop 与 CCBV/OBR/OBV pickup probes，还没有加入 saddle voltage
和时间序列窗口；因此这是 diagnostics-to-psi 的最小任务，不应描述成完整复现
Tokamark 所有诊断任务。

## 9. 验证

项目测试不下载 MAST 数据，也不运行长时间 FreeGSNKE solve：

```bash
source .mast-process-env/bin/activate
python -m unittest discover -s tests
python -m py_compile \
  scripts/download_mast_shots.py \
  scripts/build_machine_from_zarr.py \
  scripts/build_lao_fit_npz.py \
  scripts/inspect_shot.py \
  scripts/plot_mast_geometry.py \
  scripts/run_freegsnke_forward.py \
  scripts/build_synthetic_magnetic_diagnostics.py \
  scripts/build_experiment_manifests.py \
  scripts/train_tokamind_diagnostics.py \
  scripts/plot_tokamind_diagnostics_losses.py \
  scripts/evaluate_tokamind_diagnostics.py
```

阶段检查：

```bash
source configs/reproduction/mast_small_13.env
cmp "$SHOT_LIST" "$ACTIVE_SHOT_LIST"
test -f "$FIT_PATH"
test -f "$VARIANT_CSV"
test -f "$MANIFEST_DIR/${PREFIX}_batch_report.jsonl"
test -f "$MANIFEST_DIR/${PREFIX}_synthetic_accepted.jsonl"
test -f "$MANIFEST_DIR/${PREFIX}_synthetic_rejected.jsonl"
test -f "$MANIFEST_DIR/${PREFIX}_diagnostics_real_only.jsonl"
test -f "$MANIFEST_DIR/${PREFIX}_diagnostics_synthetic_only.jsonl"
test -f "$MANIFEST_DIR/${PREFIX}_diagnostics_real_plus_synthetic.jsonl"
test -f ../runs/tokamind-diagnostics-real-only/manifest_training_summary.json
test -f ../runs/tokamind-diagnostics-real-only/checkpoints/best/meta.json
test -f ../runs/tokamind-diagnostics-synthetic-only/manifest_training_summary.json
test -f ../runs/tokamind-diagnostics-synthetic-only/checkpoints/best/meta.json
test -f ../runs/tokamind-diagnostics-real-plus-synthetic/manifest_training_summary.json
test -f ../runs/tokamind-diagnostics-real-plus-synthetic/checkpoints/best/meta.json
test -f ../artifacts/tokamind_loss_curves/tokamind_diagnostics_loss_curves.png
test -f ../artifacts/tokamind_eval/tokamind_diagnostics_real_val_metrics.json
```

几何图：

```bash
source .mast-process-env/bin/activate
python scripts/plot_mast_geometry.py --shot 11771
```

输出：

```text
data/processed/geometry/11771.png
```

## 10. 常见问题

### 下载的 shot 缺失

正式复现要求 `configs/shot_lists/mast_small_13.txt` 中的 13 炮全部下载完整。重新运行
4.1 的下载循环；成功下载有完成标记时会跳过，没有完成标记时会重新执行 `s5cmd`，
并拒绝缺少必要 Zarr group metadata 的结果。
不要删除 shot 或缩小 `ACTIVE_SHOT_LIST` 后继续声称复现了正式实验。

### Lao fit NPZ 缺失

先在 `.mast-process-env` 运行：

```bash
source configs/reproduction/mast_small_13.env
python scripts/build_lao_fit_npz.py \
  --shot-list "$ACTIVE_SHOT_LIST" \
  --data-dir "$DATA_DIR" \
  --output "$FIT_PATH"
```

### FreeGSNKE 中 NumPy 导入错误

确认当前 shell 使用 `.freegsnke-solve-env`，并重新执行 FreeGSNKE 环境里的 NumPy/SciPy/Zarr override。不要把其他环境的 site-packages 加入 `sys.path`。

### 数据处理环境误导入 FreeGSNKE

不需要。下载、inspect、machine build 和 Lao NPZ 都在 `.mast-process-env` 完成；FreeGSNKE 只在最后求解阶段导入。

### 路径混淆

从 `mast-bridge` 仓库根目录执行命令。正式路径全部由
`configs/reproduction/mast_small_13.env` 定义；原始和生成数据在
`fusion-workspace/data/`，训练结果在 `fusion-workspace/runs/`，图和统一评估结果在
`fusion-workspace/artifacts/`。`data_analysis_report/` 不参与流水线。
