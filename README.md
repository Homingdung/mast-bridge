# mast-bridge

`mast-bridge` 用于把 MAST 实验数据转换成 FreeGSNKE 可以使用的输入，并以真实 MAST shot 为父样本生成仿真数据，服务于数据增强。

本项目的工作流拆成三个阶段：

1. 从 `LARGE_MODEL_FUSION-master` 下载 MAST 原始 Zarr 数据；
2. 在不安装 FreeGSNKE 的环境中读取 Zarr，提取几何信息并生成 machine configuration；
3. 在独立的 FreeGSNKE 环境中，根据某个 shot 和某个时间点的电流状态进行平衡求解。

三个阶段通过 workspace 中的文件传递数据，不要求使用同一个 Python 环境。

## 1. Workspace 结构

建议从一个空目录开始：

```text
fusion-workspace/
├── mast-bridge/
├── external/
│   ├── LARGE_MODEL_FUSION-master/
│   └── freegsnke/
├── data/
│   ├── raw/
│   │   └── mast/
│   │       ├── <shot>.zarr/
│   │       └── machine/<shot>/
│   ├── processed/
│   │   ├── real/<shot>/
│   │   └── synthetic/<shot>_variant_<id>/
│   └── manifests/
├── runs/
└── artifacts/
```

克隆本仓库和 FreeGSNKE：

```bash
mkdir fusion-workspace
cd fusion-workspace
git clone <mast-bridge-repo-url> mast-bridge
mkdir -p external
git clone https://github.com/FusionComputingLab/freegsnke.git external/freegsnke
```

将 `LARGE_MODEL_FUSION-master` 放到：

```text
fusion-workspace/external/LARGE_MODEL_FUSION-master/
```

初始化本地路径配置：

```bash
cd mast-bridge
python3 scripts/bootstrap_workspace.py --write-config
```

这会生成 `configs/paths.local.yaml`。该文件只记录本机路径，不应提交到版本库。

## 2. 三个独立的 Python 环境

所有环境建议使用 Python 3.12。FreeGSNKE 当前要求 Python `>=3.10,<3.14`。

### 2.1 下载环境：`mast-download`

这个环境只负责调用 `LARGE_MODEL_FUSION-master` 的下载脚本，不安装 FreeGSNKE：

```bash
cd fusion-workspace/mast-bridge
python3.12 -m venv .mast-download-env
source .mast-download-env/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e . --no-deps
python -m pip install s3fs xarray zarr
```

`LARGE_MODEL_FUSION-master` 可能还有自己的运行时依赖，请按照该仓库的 README 安装，并只安装到这个环境。

确认工具：

```bash
python scripts/doctor.py --skip-imports
s5cmd --version
```

先使用 `--dry-run` 检查命令：

```bash
python scripts/download_mast_shots.py \
  --shot 11766 \
  --shot 11767 \
  --dry-run
```

确认后下载：

```bash
python scripts/download_mast_shots.py \
  --shot 11766 \
  --shot 11767
```

默认输出：

```text
fusion-workspace/data/raw/mast/11766.zarr/
fusion-workspace/data/raw/mast/11767.zarr/
```

也可以指定输出目录：

```bash
python scripts/download_mast_shots.py \
  --data-dir ../data/raw/mast \
  --shot 11766
```

完成后：

```bash
deactivate
```

### 2.2 数据处理环境：`mast-process`

这个环境负责读取 Zarr、检查 shot，以及从 shot 自身的几何字段生成 FreeGSNKE-compatible machine pickles。此阶段不导入 `freegsnke`：

```bash
cd fusion-workspace/mast-bridge
python3.12 -m venv .mast-process-env
source .mast-process-env/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e . --no-deps
python -m pip install numpy "zarr>=3,<4" xarray scipy matplotlib
```

运行项目测试：

```bash
python -m unittest discover -s tests
python -m py_compile \
  scripts/build_machine_from_zarr.py \
  scripts/inspect_shot.py
```

从 shot `11766` 生成 machine configuration：

```bash
python scripts/build_machine_from_zarr.py --shot 11766
```

输出目录：

```text
fusion-workspace/data/raw/mast/machine/11766/
```

其中包含：

```text
MAST_active_coils.pickle
MAST_limiter.pickle
MAST_magentic_probes.pickle
MAST_passive_coilds.pickle
MAST_wall.pickle
```

`magentic` 和 `coilds` 是为兼容现有 FreeGSNKE machine loader 保留的历史拼写。

覆盖已有文件时必须显式指定：

```bash
python scripts/build_machine_from_zarr.py \
  --shot 11766 \
  --overwrite
```

检查 shot：

```bash
python scripts/inspect_shot.py --shot 11766
```

生成不包含磁通面的原始装置几何检查图：

```bash
source .mast-process-env/bin/activate
python scripts/plot_mast_geometry.py --shot 11766
```

输出为 `data/processed/geometry/11766.png`。这张图直接使用 shot Zarr 中的 wall、active/passive structures 和 magnetic probes，用于检查 machine geometry；它不运行 FreeGSNKE，也不绘制求解结果。

在 Python 中读取：

```python
from mast_bridge.mast.reader import ShotReader

record = ShotReader("../data/raw/mast").read("11766")
print(record.shot_id)
print(record.zarr_path)
print(record.signals.keys())
print(record.equilibrium.keys())
print(record.machine.files)
```

`record.signals` 和 `record.equilibrium` 是惰性打开的 Zarr group；读取 shot 不会启动 FreeGSNKE 求解。

### 2.3 FreeGSNKE 求解环境：`freegsnke-solve`

这个环境只负责求解：

```bash
cd fusion-workspace/mast-bridge
python3.12 -m venv .freegsnke-solve-env
source .freegsnke-solve-env/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e . --no-deps
python -m pip install -e "../external/freegsnke[freegs4e]"

# The downloaded MAST stores are Zarr 3. FreeGSNKE 3.0.1 currently
# declares an older NumPy/SciPy combination, so override those pins after
# installing its dependency set.
python -m pip install --force-reinstall --no-deps \
  "numpy>=2.0,<2.3" \
  "scipy==1.15.3" \
  "zarr>=3,<4"
python -m pip install "donfig>=0.8" "google-crc32c>=1.5"
```

安装完成后，在全新的 Python 进程中验证：

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

这里的 NumPy 2/SciPy 1.15/Zarr 3 组合是为了匹配当前下载数据和 macOS ARM 二进制 wheel；FreeGSNKE 的声明依赖仍可能让 `pip check` 报版本冲突，但实际导入和命令行求解需要这个组合。不要在同一环境中再次运行 `pip install -e "../external/freegsnke[freegs4e]"`，否则 pip 会把 NumPy 降回 1.26。

求解环境验证通过后，直接运行命令行脚本，不需要 Jupyter：

```bash
python scripts/run_freegsnke_forward.py \
  --shot 11766 \
  --time 0.18
```

默认输出：

```text
data/processed/synthetic/11766_t0.18/equilibrium.npz
data/processed/synthetic/11766_t0.18/metadata.json
data/processed/synthetic/11766_t0.18/equilibrium.png
```

脚本会从该 shot 的 Zarr 读取目标时间的 active/passive coil currents，从 Lao 拟合 NPZ 中选择最近时间的参数，并运行一个 `65 x 65` 的 FreeGSNKE 静态 forward solve。求解完成后使用 FreeGS4E 官方的 `plotEquilibrium` 绘制磁通面、分离面、磁轴、X 点、wall 和 limiter，再叠加 active coils 与 passive structures，保存为 `equilibrium.png`。

常用参数：

```bash
python scripts/run_freegsnke_forward.py \
  --shot 11766 \
  --time 0.18 \
  --nx 65 \
  --ny 65 \
  --tolerance 1e-3 \
  --max-iterations 100 \
  --output-dir data/processed/synthetic/11766_variant_0001
```

如果数据或拟合结果不在默认位置，可以显式指定：

```bash
python scripts/run_freegsnke_forward.py \
  --shot 11766 \
  --time 0.18 \
  --data-dir ../data/raw/mast \
  --machine-dir ../data/raw/mast/machine/11766 \
  --fit-path ../data_analysis_report/efit_lao_freegsnke_forward/lao_parameter_ensemble/all_zarr_lao_parameter_fits.npz
```

## 3. 从真实 shot 到仿真增强数据

### 3.1 数据和几何的边界

一个 shot 的 Zarr 通常包含：

- 装置几何：PF active coils、PF passive structures、wall/limiter、magnetic probes；
- 随时间变化的信号：active-coil current、passive current、磁测量等；
- equilibrium 或 EFIT 相关数据。

`build_machine_from_zarr.py` 只把装置几何转换为五个 machine pickle。它不会为每个时间点生成不同的几何文件，因为同一 shot 的机械元件几何通常不随时间改变。

所谓“不同时间的 machine 状态”应表示为：

```text
固定 machine geometry
  + t 时刻的 active-coil currents
  + t 时刻的 passive-structure currents
  + t 时刻的 plasma / Lao 参数
  = t 时刻的 FreeGSNKE solve input
```

因此，求解脚本应保存 `shot_id`、`time`、电流、Lao 参数和 solver 状态，而不是重复保存完全相同的几何 payload。

### 3.2 推荐的数据目录

真实 shot：

```text
data/processed/real/11766/
├── shot_record.json
├── machine/
│   ├── MAST_active_coils.pickle
│   ├── MAST_limiter.pickle
│   ├── MAST_magentic_probes.pickle
│   ├── MAST_passive_coilds.pickle
│   └── MAST_wall.pickle
└── equilibrium/
    ├── efit.npz
    └── lao_fit.json
```

仿真 variant：

```text
data/processed/synthetic/11766_variant_0001/
├── equilibrium.npz
├── metadata.json
└── machine_link.json
```

`metadata.json` 至少记录：

```text
parent_shot
target_time
machine_geometry_source
coil_currents
passive_currents
lao_parameters
random_seed
solver_status
solver_iterations
```

用户自定义 Lao 参数放在独立配置中：

```bash
cp configs/simulation/lao_custom.example.yaml \
  configs/simulation/lao_custom.yaml
```

真实 shot 拟合得到的参数和增强时采样的参数必须分开保存。

### 3.3 当前 FreeGSNKE 适配接口

当前仓库已经提供 machine geometry 到 FreeGSNKE machine 的适配层：

```python
from mast_bridge.mast.reader import ShotReader
from mast_bridge.simulation.freegsnke_runner import build_machine

record = ShotReader("../data/raw/mast").read("11766")
tokamak = build_machine(record.machine)
```

如果需要开发新的求解策略，可以复用以下底层接口；正常读者只需要运行上面的命令行脚本。

实际求解接口示例：

```python
from freegsnke import equilibrium_update, GSstaticsolver
from freegsnke.jtor_update import Lao85

eq = equilibrium_update.Equilibrium(
    tokamak=tokamak,
    Rmin=0.1,
    Rmax=2.0,
    Zmin=-2.0,
    Zmax=2.0,
    nx=65,
    ny=65,
)

profiles = Lao85(
    eq=eq,
    Ip=Ip,
    fvac=fvac,
    alpha=alpha,
    beta=beta,
)

solver = GSstaticsolver.NKGSsolver(eq)
solver.solve(
    eq=eq,
    profiles=profiles,
    constrain=None,
    target_relative_tolerance=1e-3,
    max_solving_iterations=100,
)
```

这里的 `Ip`、`fvac`、`alpha` 和 `beta` 应来自真实 shot 的 EFIT/Lao 处理结果，或来自明确记录的增强参数采样过程。

## 4. 阶段之间的验收标准

下载阶段：

```bash
test -d ../data/raw/mast/11766.zarr
```

并确认 Zarr 至少包含 `pf_active`、`pf_passive`、`magnetics` 和 `wall` 等所需 group。

处理阶段：

```bash
python scripts/inspect_shot.py --shot 11766
```

五个 machine 文件都必须存在并可读取。处理环境不需要能够执行 `import freegsnke`。

求解阶段：

```bash
python -c "import numpy, scipy, freegs4e, freegsnke; print('FreeGSNKE import OK')"
```

然后再运行求解脚本。求解输出必须同时保存数值结果和 metadata，不能只依赖终端屏幕输出。

## 5. 数据划分和可复现性

真实数据和仿真 variant 使用 JSONL manifest 统一描述。训练集、验证集和测试集必须按 shot 划分；同一 shot 产生的所有 variant 必须和原始真实 shot 位于同一个 split，避免同一 shot 的信息泄漏到不同数据集。

每个仿真样本必须记录：

- 父 shot 和目标时间；
- machine geometry 的来源目录；
- active/passive coil currents；
- Lao 参数；
- 随机种子；
- FreeGSNKE、NumPy、SciPy 版本；
- solver 是否收敛、迭代次数和容差。

## 6. 项目测试

项目测试不下载 MAST 数据，也不启动 FreeGSNKE 长时间求解：

```bash
source .mast-process-env/bin/activate
python -m unittest discover -s tests
python -m py_compile \
  scripts/download_mast_shots.py \
  scripts/build_machine_from_zarr.py \
  scripts/inspect_shot.py \
  scripts/plot_mast_geometry.py \
  scripts/run_freegsnke_forward.py
```

### 6.1 测试原始几何图

在数据处理环境中运行：

```bash
source .mast-process-env/bin/activate
python scripts/plot_mast_geometry.py --shot 11766
```

生成的图片位于：

```text
data/processed/geometry/11766.png
```

这张图只检查 Zarr 中的装置几何，不需要 FreeGSNKE 环境，也不会执行平衡求解。

### 6.2 测试 FreeGSNKE 求解图

在 FreeGSNKE 求解环境中运行：

```bash
source .freegsnke-solve-env/bin/activate
python scripts/run_freegsnke_forward.py \
  --shot 11766 \
  --time 0.18
```

生成的求解结果和图片位于：

```text
data/processed/synthetic/11766_t0.18/equilibrium.npz
data/processed/synthetic/11766_t0.18/metadata.json
data/processed/synthetic/11766_t0.18/equilibrium.png
```

`equilibrium.png` 使用 FreeGS4E 官方 `plotEquilibrium` 绘制磁通面、分离面、磁轴、X 点、wall 和 limiter；legend 位于图像右侧，不遮挡装置或磁通面。

真实数据下载和 FreeGSNKE 求解都属于读者主动执行的阶段，不会在安装项目时自动发生。

## 7. 常见问题

### `ImportError: numpy._core.multiarray failed to import`

这是 FreeGSNKE/FreeGS4E 环境中的 NumPy/SciPy 二进制兼容问题。确认当前 shell 使用 `.freegsnke-solve-env`，并在全新的 Python 进程中测试导入。不要把其他环境的 site-packages 加入 `sys.path`。

### 处理环境导入了 FreeGSNKE

这是不必要的。生成 machine pickles 的代码只需要 NumPy、Zarr 和标准库；FreeGSNKE 只在最后的求解阶段导入。

### machine 文件或原始数据找不到

从 `mast-bridge` 目录执行脚本，或显式传入 `--data-dir`、`--output-dir` 和 `--machine-dir`。脚本默认使用 `fusion-workspace/data/raw/mast`，不是 `external/LARGE_MODEL_FUSION-master/mast_data`。
