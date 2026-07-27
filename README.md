# mast-bridge

`mast-bridge` 是 MAST 数据与 FreeGSNKE 仿真之间的桥接仓库，负责 MAST 数据读取、machine geometry 标准化、仿真数据准备，以及后续 TokaMark/TokaMind 训练数据组织。

`external/` 下的仓库保持为外部依赖，不把它们的代码复制到 `mast-bridge`。`Lao85.ipynb` 仍然只是探索草稿，正式接口位于 `src/mast_bridge/`。

## 1. 创建 workspace

读者从一个空目录开始：

```bash
mkdir fusion-workspace
cd fusion-workspace
git clone <mast-bridge-repo-url> mast-bridge
mkdir external
git clone https://github.com/UKAEA-IBM-STFC-Fusion-FMs/tokamark.git external/tokamark
git clone https://github.com/UKAEA-IBM-STFC-Fusion-FMs/tokamind.git external/tokamind
git clone https://github.com/FusionComputingLab/freegsnke.git external/freegsnke
# 将 LARGE_MODEL_FUSION 放到 external/LARGE_MODEL_FUSION/
```

推荐目录：

```text
fusion-workspace/
├── mast-bridge/
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
cd mast-bridge
python scripts/bootstrap_workspace.py --write-config
python scripts/doctor.py --skip-imports
```

## 2. 配置 Python 环境

`mast-bridge` 的完整 FreeGSNKE 仿真环境要求 Python `>=3.10,<3.14`。推荐 Python 3.12 或 3.13；请只为 `mast-bridge` 项目创建独立虚拟环境，并在同一个环境中安装 `mast-bridge`、FreeGSNKE、TokaMark 和 TokaMind。Python 3.14 会被 bootstrap 拒绝，因为 FreeGSNKE 3.0.1 的 NumPy/SciPy 依赖尚未覆盖该版本。

```bash
cd mast-bridge
python3 --version
python3.12 -m venv .mast-bridge-env
source .mast-bridge-env/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
```

Windows PowerShell 激活方式：

```powershell
.mast-bridge-env\Scripts\Activate.ps1
```

初始化 workspace 路径并检查外部仓库：

```bash
python scripts/bootstrap_workspace.py --write-config
python scripts/doctor.py --skip-imports
```

### 2.1 只下载和读取 MAST 数据

这条路径不需要安装 FreeGSNKE、TokaMark 或 TokaMind，只安装 MAST 下载和 Zarr 读取依赖：

```bash
brew install s5cmd
python -m pip install s3fs xarray zarr
```

如果不使用 macOS Homebrew，请安装 `s5cmd` 的其他平台版本，并确保下面命令可以运行：

```bash
s5cmd --version
```

### 2.2 完整仿真和训练环境

如果还要调用 FreeGSNKE，以及后续使用 TokaMark/TokaMind 训练，在同一个已激活的 `.mast-bridge-env` 中执行：

```bash
python scripts/bootstrap_workspace.py \
  --install-editable \
  --with-deps
python scripts/doctor.py

# 验证当前激活的环境确实能导入 FreeGSNKE
python -c "import freegsnke; print(freegsnke.__file__)"
```

`--with-deps` 会从 `external/freegsnke` 以 editable 方式安装 FreeGSNKE，并允许 pip 安装外部仓库声明的运行时依赖，可能需要较长时间和较大的磁盘空间。只做 MAST 下载时不要执行这一步。`doctor.py` 若报告数据目录缺失，不影响上面的 Python 导入检查；准备完整 workspace 后再按提示补齐路径即可。

## 3. 下载指定 shot

下载脚本由 `LARGE_MODEL_FUSION` 提供，`mast-bridge` 只提供选 shot 的薄封装。

先查看命令，不产生网络请求：

```bash
python scripts/download_mast_shots.py \
  --shot 11766 \
  --shot 11767 \
  --dry-run
```

确认后执行下载：

```bash
python scripts/download_mast_shots.py \
  --shot 11766 \
  --shot 11767
```

默认输出：

```text
../data/raw/mast/11766.zarr/
../data/raw/mast/11767.zarr/
```

也可以显式指定输出目录：

```bash
python scripts/download_mast_shots.py \
  --data-dir ../data/raw/mast \
  --shot 11766
```

建议重复使用 `--shot` 指定确切 shot。外部脚本的 `--limit` 只限制 S3 返回列表的数量，不保证得到指定的 shot。

脚本会逐炮调用：

```text
external/LARGE_MODEL_FUSION*/scripts/download/download_data_v2.py
```

不会修改外部仓库源码。

## 4. 检查 shot 和 machine geometry

下载后，先从该 shot 自身的 Zarr 几何字段生成 machine configuration：

```bash
python scripts/build_machine_from_zarr.py --shot 11766
```

输出目录为：

```text
../data/raw/mast/machine/11766/
```

这个步骤不读取 `external/freegsnke/machine_configs/` 中的通用或预生成配置；五个 pickle 全部由当前 shot 的 Zarr 生成。

如果需要重新生成已有配置，显式允许覆盖：

```bash
python scripts/build_machine_from_zarr.py \
  --shot 11766 \
  --overwrite
```

生成后检查一炮：

```bash
python scripts/inspect_shot.py --shot 11766
```

每炮的 machine configuration 必须包含：

```text
MAST_active_coils.pickle
MAST_limiter.pickle
MAST_magentic_probes.pickle
MAST_passive_coilds.pickle
MAST_wall.pickle
```

当前实现会在 `<shot>.zarr/`、Zarr 子目录、`data/raw/mast/machine/<shot>/` 中查找这些文件，也可以显式指定：

```bash
python scripts/inspect_shot.py \
  --shot 11766 \
  --data-dir ../data/raw/mast \
  --machine-dir ../data/raw/mast/machine/11766
```

`magentic` 和 `coilds` 是当前读取器兼容的既有拼写；源目录中的标准文件名分别为 `MAST_magnetic_probes.pickle` 和 `MAST_passive_coils.pickle`，复制时需按上面的目标文件名保存。`MAST_wall.picklet` 应为 `MAST_wall.pickle`。

代码中统一通过 `MachineGeometry` 表示：

```python
from mast_bridge.mast.reader import ShotReader

record = ShotReader("../data/raw/mast").read("11766")
machine = record.machine
machine_payloads = machine.load_pickles()
```

`record.machine` 可同时用于未来的装置总览图、EFIT 图、磁探针位置图和 FreeGSNKE 求解。真实 Zarr 的 signals 和 equilibrium group 会以惰性 Zarr group 形式挂在 `record.signals` 与 `record.equilibrium`。

## 5. 数据处理布局

真实 shot 的处理结果建议保存为：

```text
data/processed/real/11766/
├── machine/
├── equilibrium/
│   ├── efit.npz
│   └── lao_fit.json
└── shot_record.json
```

`lao_fit.json` 保存该炮 EFIT 拟合得到的 `P'`、`FF'` Lao 参数和拟合质量。它是真实数据基准，不被用户自定义参数覆盖。

用户自己的 Lao 参数放在独立配置中：

```bash
cp configs/simulation/lao_custom.example.yaml configs/simulation/lao_custom.yaml
```

配置可以使用固定系数，也可以使用 `{min, max}` 范围。示例文件中的范围用于未来生成多个 FreeGSNKE variant；每个 variant 都应把实际使用的系数、父 shot、随机种子和 solver 状态写入 metadata。

## 6. FreeGSNKE 和训练边界

未来的求解流程为：

```text
ShotReader
  -> MachineGeometry
  -> EFIT / lao_fit.json 或 lao_custom.yaml
  -> FreeGSNKE build_machine.tokamak(...)
  -> Equilibrium + NKGSsolver.solve(...)
  -> data/processed/synthetic/<shot>_variant_<id>/
```

当前已经提供 machine 到 FreeGSNKE 的路径适配层，但不会自动求解：

```python
from mast_bridge.mast.reader import ShotReader
from mast_bridge.simulation.freegsnke_runner import build_machine

record = ShotReader("../data/raw/mast").read("11766")
tokamak = build_machine(record.machine)
```

真正的 `Equilibrium`、Lao profile 注入和 `NKGSsolver.solve(...)` 应由后续仿真脚本显式控制，避免读取数据时产生长时间计算。

真实和仿真样本通过 JSONL manifest 统一描述：

```python
from pathlib import Path
from mast_bridge.dataset.manifest import ManifestEntry, write_manifest

write_manifest(
    [
        ManifestEntry(
            sample_id="11766",
            source="real",
            shot_id="11766",
            data_path=Path("../data/raw/mast/11766.zarr"),
        )
    ],
    "../data/manifests/all.jsonl",
)
```

仿真样本需要额外记录：

```text
parent_shot
lao parameters
coil currents
random seed
solver_status
```

训练集、验证集和测试集必须按 shot 划分。同一炮产生的所有仿真 variant 必须和原始真实 shot 位于同一个 split，避免数据泄漏。

## 7. 当前验证

代码测试不下载 MAST 数据，也不启动 FreeGSNKE 求解：

```bash
python3 -m unittest discover -s tests
python3 -m py_compile scripts/download_mast_shots.py scripts/inspect_shot.py
```

真实下载和求解属于用户主动执行的后续步骤。
