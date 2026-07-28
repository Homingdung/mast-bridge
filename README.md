# mast-bridge

`mast-bridge` 把 MAST Level 2 Zarr 数据转换成 FreeGSNKE 可用输入，并以真实 shot 为父样本批量生成 synthetic equilibrium 数据，用于 Tokamind/Tokamark 训练。

主流程：

```text
download MAST Level 2
  -> filter successful shots
  -> build machine pickles
  -> build Lao/EFIT NPZ
  -> run FreeGSNKE forward solves
  -> train Tokamind/Tokamark from manifests
```

数据输入和产出统一放在 workspace 的 `data/` 目录下。`data_analysis_report/` 只用于图片和分析报告，不作为训练流水线的数据输入目录。

## 1. Workspace

从一个空目录开始：

```bash
mkdir fusion-workspace
cd fusion-workspace

git clone git@github.com:Homingdung/mast-bridge.git mast-bridge

mkdir external
git clone https://github.com/UKAEA-IBM-STFC-Fusion-FMs/LARGE_MODEL_FUSION.git external/LARGE_MODEL_FUSION
git clone https://github.com/FusionComputingLab/freegsnke.git external/freegsnke
git clone https://github.com/UKAEA-IBM-STFC-Fusion-FMs/tokamark.git external/tokamark
git clone https://github.com/UKAEA-IBM-STFC-Fusion-FMs/tokamind.git external/tokamind
```

如果 `LARGE_MODEL_FUSION` 仓库实际以 `LARGE_MODEL_FUSION-master` 目录存在，当前下载脚本也兼容；推荐新 workspace 使用 `external/LARGE_MODEL_FUSION`。

推荐目录：

```text
fusion-workspace/
├── mast-bridge/
│   └── configs/
│       ├── shot_lists/
│       └── time_grids/
├── external/
│   ├── LARGE_MODEL_FUSION/
│   ├── freegsnke/
│   ├── tokamark/
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
cd fusion-workspace/mast-bridge
python3 scripts/bootstrap_workspace.py --write-config
```

这会生成 `configs/paths.local.yaml`。该文件只记录本机路径，不应提交。

## 2. Python Environments

三个阶段使用三个独立环境，避免 FreeGSNKE 依赖污染下载和数据处理阶段。建议 Python 3.12。

### 2.1 mast-download

用于调用 `LARGE_MODEL_FUSION` 下载脚本：

```bash
cd fusion-workspace/mast-bridge
python3.12 -m venv .mast-download-env
source .mast-download-env/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e . --no-deps
python -m pip install s3fs xarray zarr
```

确认：

```bash
python scripts/doctor.py --skip-imports
s5cmd --version
```

### 2.2 mast-process

用于读取 Zarr、生成 machine pickles、生成 Lao/EFIT NPZ。不导入 FreeGSNKE：

```bash
cd fusion-workspace/mast-bridge
python3.12 -m venv .mast-process-env
source .mast-process-env/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e . --no-deps
python -m pip install numpy "zarr>=3,<4" xarray scipy matplotlib
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

### 2.3 freegsnke-solve

用于 FreeGSNKE forward solve：

```bash
cd fusion-workspace/mast-bridge
python3.12 -m venv .freegsnke-solve-env
source .freegsnke-solve-env/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e . --no-deps
python -m pip install -e "../external/freegsnke[freegs4e]"

# Downloaded MAST stores are Zarr 3. Override FreeGSNKE's older pins after install.
python -m pip install --force-reinstall --no-deps \
  "numpy>=2.0,<2.3" \
  "scipy==1.15.3" \
  "zarr>=3,<4"
python -m pip install "donfig>=0.8" "google-crc32c>=1.5"
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

## 3. Configure Run Size

所有规模都由 shot list、time grid 和路径变量控制。同一套命令可以先本地 smoke test，再放到服务器跑 full run。

推荐三档：

```text
local_smoke   2-4 shots, each 2-3 times
local_dev     10-20 shots, each 5 times
server_full   200-2000 shots, each 5-20 times
```

本地 smoke 配置：

```bash
mkdir -p configs/shot_lists configs/time_grids

cat > configs/shot_lists/local_smoke.txt <<'EOF'
11771
11772
11773
EOF

cat > configs/time_grids/smoke_times.txt <<'EOF'
0.16
0.20
EOF
```

本地变量：

```bash
SHOT_LIST=configs/shot_lists/local_smoke.txt
ACTIVE_SHOT_LIST=configs/shot_lists/downloaded_success.txt
TIME_GRID=configs/time_grids/smoke_times.txt
DATA_DIR=../data/raw/mast
FIT_PATH=../data/processed/real/lao_parameter_ensemble/all_zarr_lao_parameter_fits.npz
SYNTH_DIR=../data/processed/synthetic
```

服务器只替换变量，例如：

```bash
SHOT_LIST=configs/shot_lists/server_full.txt
ACTIVE_SHOT_LIST=configs/shot_lists/downloaded_success.txt
TIME_GRID=configs/time_grids/full_times.txt
DATA_DIR=/data/mast/raw
FIT_PATH=/data/mast/processed/real/lao_parameter_ensemble/all_zarr_lao_parameter_fits.npz
SYNTH_DIR=/data/mast/processed/synthetic
```

## 4. Batch Pipeline

按本节顺序运行。不要跳过 `ACTIVE_SHOT_LIST`，它用于过滤下载失败或远端不存在的 shot。

### 4.1 Download

```bash
source .mast-download-env/bin/activate

SHOT_LIST=configs/shot_lists/local_smoke.txt
ACTIVE_SHOT_LIST=configs/shot_lists/downloaded_success.txt
DATA_DIR=../data/raw/mast
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

下载：

```bash
while read shot; do
  [ -z "$shot" ] && continue
  python scripts/download_mast_shots.py \
    --data-dir "$DATA_DIR" \
    --shot "$shot"
done < "$SHOT_LIST"
```

生成实际可处理列表：

```bash
while read shot; do
  [ -z "$shot" ] && continue
  if [ -d "$DATA_DIR/${shot}.zarr" ]; then
    echo "$shot"
  else
    echo "Missing downloaded shot: $shot" >&2
  fi
done < "$SHOT_LIST" > "$ACTIVE_SHOT_LIST"
```

从这里开始，后续所有 `while read shot` 都使用：

```bash
done < "$ACTIVE_SHOT_LIST"
```

不要继续使用原始 `$SHOT_LIST`。

### 4.2 Build Machine Pickles

```bash
source .mast-process-env/bin/activate

ACTIVE_SHOT_LIST=configs/shot_lists/downloaded_success.txt
DATA_DIR=../data/raw/mast
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
├── MAST_passive_coilds.pickle
└── MAST_wall.pickle
```

`magentic` 和 `coilds` 是为兼容现有 FreeGSNKE loader 保留的历史拼写。

### 4.3 Build Lao/EFIT NPZ

FreeGSNKE forward solve 需要每个 shot/time 对应的 profile 参数。统一 NPZ 路径：

```text
data/processed/real/lao_parameter_ensemble/all_zarr_lao_parameter_fits.npz
```

生成：

```bash
source .mast-process-env/bin/activate

ACTIVE_SHOT_LIST=configs/shot_lists/downloaded_success.txt
DATA_DIR=../data/raw/mast
FIT_PATH=../data/processed/real/lao_parameter_ensemble/all_zarr_lao_parameter_fits.npz

python scripts/build_lao_fit_npz.py \
  --shot-list "$ACTIVE_SHOT_LIST" \
  --data-dir "$DATA_DIR" \
  --output "$FIT_PATH"
```

该脚本从 Level 2 Zarr 读取：

```text
equilibrium/dpressure_dpsi
equilibrium/f_df_dpsi
equilibrium/bvac_rmag
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

### 4.4 Run FreeGSNKE Solves

```bash
source .freegsnke-solve-env/bin/activate

ACTIVE_SHOT_LIST=configs/shot_lists/downloaded_success.txt
TIME_GRID=configs/time_grids/smoke_times.txt
DATA_DIR=../data/raw/mast
FIT_PATH=../data/processed/real/lao_parameter_ensemble/all_zarr_lao_parameter_fits.npz
SYNTH_DIR=../data/processed/synthetic
```

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
    output_dir="$SYNTH_DIR/${shot}_t${time}"
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
      --tolerance 1e-3 \
      --max-iterations 100 \
      --output-dir "$output_dir"
  done < "$TIME_GRID"
done < "$ACTIVE_SHOT_LIST"
```

每个 synthetic 样本输出：

```text
data/processed/synthetic/<shot>_t<time>/
├── equilibrium.npz
├── metadata.json
└── equilibrium.png
```

如果某个 solve 没达到收敛阈值，`metadata.json` 和终端输出会记录状态。小网格 smoke run 可用于检查链路，不代表正式训练参数。

## 5. Single-Shot Commands

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
  --tolerance 1e-3 \
  --max-iterations 100 \
  --output-dir ../data/processed/synthetic/11771_t0.16
```

## 6. Data Contract

### 6.1 Raw and Real Data

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

Level 1 保留原始诊断名和原始采集形态，不作为默认训练输入。

### 6.2 Synthetic Data

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
  grid
  solver_status
```

固定 geometry 和随时间变化的状态分开记录：

```text
fixed machine geometry
  + active/passive coil currents at t
  + Lao/EFIT profile at nearest fitted time
  = one FreeGSNKE solve input
```

不要为每个时间点重复保存完全相同的 machine geometry payload。

### 6.3 Manifest

训练、验证、测试必须按 shot 划分。同一个 shot 的真实样本和所有 synthetic variants 必须在同一个 split，避免信息泄漏。

推荐 manifest：

```text
data/manifests/
├── tokamark_simple_real.jsonl
├── tokamark_simple_synthetic.jsonl
├── tokamark_simple_train.jsonl
├── tokamark_simple_val.jsonl
└── tokamark_simple_test.jsonl
```

每行至少包含：

```json
{
  "sample_id": "11771_t0.16",
  "source": "synthetic",
  "parent_shot": "11771",
  "time": 0.16,
  "data_path": "data/processed/synthetic/11771_t0.16/equilibrium.npz",
  "metadata_path": "data/processed/synthetic/11771_t0.16/metadata.json",
  "split": "train",
  "task": "tokamark_simple"
}
```

## 7. Tokamind/Tokamark Training Input

第一阶段保持任务简单：从 machine + currents + Lao profile 预测 equilibrium flux map。

输入建议：

```text
machine geometry reference
coil currents
Ip / Lao profile parameters
R grid
Z grid
```

目标：

```text
psi
psi_axis
psi_bndry
```

扩展顺序：

```text
2 shots x 2 times       smoke test
10-20 shots x 5 times   local dev
200-2000 shots x 5-20 times server training data
```

跑通后再加入 Lao 参数扰动，为每个 shot/time 生成多个 variant。

## 8. Verification

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
  scripts/run_freegsnke_forward.py
```

阶段检查：

```bash
test -d ../data/raw/mast/11771.zarr
test -d ../data/raw/mast/machine/11771
test -f ../data/processed/real/lao_parameter_ensemble/all_zarr_lao_parameter_fits.npz
test -f ../data/processed/synthetic/11771_t0.16/equilibrium.npz
test -f ../data/processed/synthetic/11771_t0.16/metadata.json
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

## 9. FAQ

### Missing downloaded shot

如果原始 `SHOT_LIST` 里有 `11770`，但本地没有 `11770.zarr`，后续步骤会失败。先生成 `ACTIVE_SHOT_LIST`，后续步骤只读 `ACTIVE_SHOT_LIST`。

### Missing Lao fit NPZ

先在 `.mast-process-env` 运行：

```bash
python scripts/build_lao_fit_npz.py \
  --shot-list configs/shot_lists/downloaded_success.txt \
  --data-dir ../data/raw/mast \
  --output ../data/processed/real/lao_parameter_ensemble/all_zarr_lao_parameter_fits.npz
```

### NumPy import error in FreeGSNKE

确认当前 shell 使用 `.freegsnke-solve-env`，并重新执行 FreeGSNKE 环境里的 NumPy/SciPy/Zarr override。不要把其他环境的 site-packages 加入 `sys.path`。

### Processing environment imports FreeGSNKE

不需要。下载、inspect、machine build 和 Lao NPZ 都在 `.mast-process-env` 完成；FreeGSNKE 只在最后求解阶段导入。

### Path confusion

从 `mast-bridge` 目录执行命令。默认数据目录是 `fusion-workspace/data/...`，不是 `external/LARGE_MODEL_FUSION-master/mast_data`，也不是 `data_analysis_report/`。
