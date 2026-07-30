# 本地 FreeGSNKE 数据增强生产实施计划

> **给 agentic workers：** 必须使用 `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans` 按任务逐项执行本计划。步骤使用 checkbox（`- [ ]`）语法跟踪进度。

**目标：** 建立一个先数据后训练的本地 pipeline：批量下载 MAST 数据，批量处理成 FreeGSNKE 输入，批量求解生成 synthetic equilibrium 数据，并在样本数量和质量达标后再单独设计训练方案。

**架构：** `mast-bridge` 负责数据工厂：下载 MAST Level 2 Zarr，构建 machine pickles，构建 Lao/EFIT NPZ，生成 FreeGSNKE variant plan，批量运行 FreeGSNKE forward solves，校验 synthetic equilibrium，输出 manifest 和质量报告。本计划不启动 TokaMind 训练；训练阶段必须等数据量、成功率、物理质量和 TokaMark-compatible 数据组织方式验收通过后，再参考 `external/tokamark` 官方 pipeline 单独制定。

**技术栈：** Python 3.12 虚拟环境、MAST Level 2 Zarr、NumPy、Zarr、FreeGSNKE/FreeGS4E、JSONL manifests、`unittest`；训练设计阶段再引入 TokaMark official `tokamark.data` / `tokamark.tasks` / `tokamark.evaluator` 和 TokaMind model/training loop。

## 全局约束

- 除非步骤明确说明切换目录，所有 `mast-bridge` 命令都从 `/Users/mingdonghe/pj/fusion-workspace/mast-bridge` 执行。
- 数据输入和产出统一放在 `/Users/mingdonghe/pj/fusion-workspace/data/`，不要放进 `data_analysis_report/`。
- train/val/test 必须按 `parent_shot` 划分；synthetic 样本必须和它的真实父 shot 在同一个 split。
- 第一目标任务是 `task_1-3`，因为 FreeGSNKE 直接产出 `equilibrium-psi`；第二目标任务是 `task_2-3`。
- 不修改 TokaMark 官方 benchmark task YAML；所有训练样本必须通过 TokaMark 官方 task config、window segmenter、transform map、split 和 evaluator 对齐。
- 本计划结束前不启动任何 TokaMind 训练，也不设计自定义训练参数。
- 数据量未达标前，不讨论 mixed training、warmstart、loss、batch size 或 evaluator glue。
- FreeGSNKE 第一轮网格固定为 `nx=65`、`ny=65`，匹配 TokaMark `equilibrium-psi`。
- 网络下载、HuggingFace 下载、长时间 FreeGSNKE 批量运行都作为单独需要确认的操作。

---

## 文件结构

- 创建 `configs/shot_lists/local_tiny.txt`：本地 smoke 的 3 个 shot。
- 创建 `configs/time_grids/local_tiny_times.txt`：每个 shot 的 2 个时间点。
- 创建 `src/mast_bridge/dataset/splits.py`：按 `parent_shot` 做 split 分配。
- 创建 `src/mast_bridge/dataset/synthetic_manifest.py`：扫描 synthetic 样本目录，校验 `equilibrium.npz` 和 `metadata.json`，输出 manifest entries。
- 创建 `scripts/build_synthetic_manifest.py`：synthetic manifest CLI。
- 创建 `src/mast_bridge/simulation/variants.py`：确定性生成 `Ip`、`fvac`、Lao `alpha/beta`、coil current 的扰动计划。
- 创建 `scripts/build_freegsnke_variant_plan.py`：生成 FreeGSNKE variant job JSONL。
- 修改 `scripts/run_freegsnke_forward.py`：支持扰动参数，并把扰动记录到 metadata。
- 创建 `scripts/run_freegsnke_variant_batch.py`：按 JSONL plan 幂等执行 FreeGSNKE variant jobs。
- 创建 `tests/test_splits.py`：验证 synthetic 样本和 parent shot 同 split。
- 创建 `tests/test_synthetic_manifest.py`：验证 manifest 扫描和无效 equilibrium 拒收。
- 创建 `tests/test_variants.py`：验证 variant 生成确定性和扰动边界。
- 创建 `scripts/write_augmentation_report.py`：统计下载、处理、求解、校验结果，输出数据生产报告。
- 创建 `docs/local-augmentation-results.md`：记录数据规模、成功率、失败原因和是否进入训练设计阶段。
- 修改 `README.md`：记录本地数据增强 smoke 和 batch workflow 的完整命令。

### 任务 1：创建本地 tiny 数据输入

**文件：**
- 创建：`configs/shot_lists/local_tiny.txt`
- 创建：`configs/time_grids/local_tiny_times.txt`
- 修改：`README.md`

**接口：**
- 使用已有脚本：`scripts/download_mast_shots.py`、`scripts/build_machine_from_zarr.py`、`scripts/build_lao_fit_npz.py`、`scripts/inspect_shot.py`
- 产出：后续 FreeGSNKE 和 manifest 任务使用的小规模 shot/time 配置

- [ ] **步骤 1：创建 tiny shot list**

创建 `configs/shot_lists/local_tiny.txt`：

```text
11771
11772
11773
```

- [ ] **步骤 2：创建 tiny time grid**

创建 `configs/time_grids/local_tiny_times.txt`：

```text
0.16
0.20
```

- [ ] **步骤 3：把本地 smoke 下载命令写入 README**

在 `README.md` 增加 `Local augmentation smoke run` 小节，包含：

```bash
cd /Users/mingdonghe/pj/fusion-workspace/mast-bridge
source .mast-download-env/bin/activate
SHOT_LIST=configs/shot_lists/local_tiny.txt
ACTIVE_SHOT_LIST=configs/shot_lists/local_tiny_downloaded.txt
DATA_DIR=../data/raw/mast

while read shot; do
  [ -z "$shot" ] && continue
  python scripts/download_mast_shots.py --data-dir "$DATA_DIR" --shot "$shot"
done < "$SHOT_LIST"

while read shot; do
  [ -z "$shot" ] && continue
  if [ -d "$DATA_DIR/${shot}.zarr" ]; then
    echo "$shot"
  fi
done < "$SHOT_LIST" > "$ACTIVE_SHOT_LIST"
```

- [ ] **步骤 4：验证文件存在**

运行：

```bash
test -f configs/shot_lists/local_tiny.txt
test -f configs/time_grids/local_tiny_times.txt
rg -n "Local augmentation smoke run" README.md
```

期望：所有命令退出码为 `0`。

- [ ] **步骤 5：提交**

```bash
git add README.md configs/shot_lists/local_tiny.txt configs/time_grids/local_tiny_times.txt
git commit -m "docs: add local augmentation smoke inputs"
```

### 任务 2：实现 split-safe manifest helper

**文件：**
- 创建：`src/mast_bridge/dataset/splits.py`
- 创建：`tests/test_splits.py`

**接口：**
- 输入：带有 `shot_id`、可选 `parent_shot`、`source` 的 JSONL row dict
- 输出：`assign_parent_shot_splits(rows: list[dict], train_fraction: float, val_fraction: float, seed: int) -> dict[str, str]`
- 输出：`split_for_row(row: dict, assignments: dict[str, str]) -> str`

- [ ] **步骤 1：先写失败测试**

创建 `tests/test_splits.py`：

```python
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from mast_bridge.dataset.splits import assign_parent_shot_splits, split_for_row


class SplitTests(unittest.TestCase):
    def test_synthetic_uses_parent_shot_split(self):
        rows = [
            {"sample_id": "11771", "source": "real", "shot_id": "11771"},
            {"sample_id": "11771_t0.16_v000", "source": "synthetic", "shot_id": "11771_t0.16_v000", "parent_shot": "11771"},
            {"sample_id": "11772", "source": "real", "shot_id": "11772"},
            {"sample_id": "11773", "source": "real", "shot_id": "11773"},
        ]
        assignments = assign_parent_shot_splits(rows, train_fraction=0.67, val_fraction=0.0, seed=7)

        self.assertEqual(
            split_for_row(rows[1], assignments),
            split_for_row(rows[0], assignments),
        )

    def test_requires_parent_for_synthetic(self):
        rows = [{"sample_id": "bad", "source": "synthetic", "shot_id": "bad"}]

        with self.assertRaisesRegex(ValueError, "parent_shot"):
            assign_parent_shot_splits(rows, train_fraction=0.8, val_fraction=0.1, seed=1)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **步骤 2：运行测试确认失败**

运行：

```bash
python -m unittest tests/test_splits.py
```

期望：失败，错误包含 `ModuleNotFoundError: No module named 'mast_bridge.dataset.splits'`。

- [ ] **步骤 3：实现 helper**

创建 `src/mast_bridge/dataset/splits.py`：

```python
from __future__ import annotations

import random
from typing import Any


def parent_shot_for_row(row: dict[str, Any]) -> str:
    source = row.get("source")
    if source == "synthetic":
        parent = row.get("parent_shot")
        if not isinstance(parent, str) or not parent.strip():
            raise ValueError("synthetic manifest rows must include parent_shot")
        return parent
    shot = row.get("shot_id")
    if not isinstance(shot, str) or not shot.strip():
        raise ValueError("manifest rows must include shot_id")
    return shot


def assign_parent_shot_splits(
    rows: list[dict[str, Any]],
    train_fraction: float,
    val_fraction: float,
    seed: int,
) -> dict[str, str]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError("train_fraction + val_fraction must be less than 1")

    parents = sorted({parent_shot_for_row(row) for row in rows})
    rng = random.Random(seed)
    rng.shuffle(parents)

    n_total = len(parents)
    n_train = max(1, int(round(n_total * train_fraction))) if n_total else 0
    n_val = int(round(n_total * val_fraction))
    if n_train + n_val >= n_total and n_total >= 2:
        n_train = n_total - 1
        n_val = 0

    assignments: dict[str, str] = {}
    for index, parent in enumerate(parents):
        if index < n_train:
            assignments[parent] = "train"
        elif index < n_train + n_val:
            assignments[parent] = "val"
        else:
            assignments[parent] = "test"
    return assignments


def split_for_row(row: dict[str, Any], assignments: dict[str, str]) -> str:
    parent = parent_shot_for_row(row)
    try:
        return assignments[parent]
    except KeyError as exc:
        raise KeyError(f"parent shot {parent!r} has no split assignment") from exc
```

- [ ] **步骤 4：运行测试确认通过**

运行：

```bash
python -m unittest tests/test_splits.py
```

期望：通过。

- [ ] **步骤 5：提交**

```bash
git add src/mast_bridge/dataset/splits.py tests/test_splits.py
git commit -m "feat: add parent-shot split helpers"
```

### 任务 3：实现 synthetic manifest builder

**文件：**
- 创建：`src/mast_bridge/dataset/synthetic_manifest.py`
- 创建：`scripts/build_synthetic_manifest.py`
- 创建：`tests/test_synthetic_manifest.py`

**接口：**
- 输入：包含 `equilibrium.npz` 和 `metadata.json` 的 synthetic sample 目录
- 输出：`synthetic_entries(synthetic_root: Path, task: str) -> list[ManifestEntry]`
- CLI：`python scripts/build_synthetic_manifest.py --synthetic-root ../data/processed/synthetic --output ../data/manifests/tokamark_synthetic.jsonl --task task_1-3`

- [ ] **步骤 1：先写失败测试**

创建 `tests/test_synthetic_manifest.py`：

```python
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from mast_bridge.dataset.synthetic_manifest import synthetic_entries


class SyntheticManifestTests(unittest.TestCase):
    def test_scans_completed_synthetic_sample(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sample = root / "11771_t0.16_v000"
            sample.mkdir()
            np.savez_compressed(sample / "equilibrium.npz", psi=np.zeros((65, 65)), R=np.zeros((65, 65)), Z=np.zeros((65, 65)), psi_axis=0.0, psi_bndry=1.0)
            (sample / "metadata.json").write_text(json.dumps({"source": "synthetic", "parent_shot": "11771", "target_time": 0.16, "solver_status": "completed"}))

            rows = synthetic_entries(root, task="task_1-3")

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].sample_id, "11771_t0.16_v000")
        self.assertEqual(rows[0].parent_shot, "11771")
        self.assertEqual(rows[0].metadata["task"], "task_1-3")

    def test_rejects_nonfinite_psi(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sample = root / "11771_t0.16_v000"
            sample.mkdir()
            np.savez_compressed(sample / "equilibrium.npz", psi=np.array([[float("nan")]]), R=np.zeros((1, 1)), Z=np.zeros((1, 1)), psi_axis=0.0, psi_bndry=1.0)
            (sample / "metadata.json").write_text(json.dumps({"source": "synthetic", "parent_shot": "11771", "target_time": 0.16, "solver_status": "completed"}))

            rows = synthetic_entries(root, task="task_1-3")

        self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **步骤 2：运行测试确认失败**

运行：

```bash
python -m unittest tests/test_synthetic_manifest.py
```

期望：失败，错误包含 `ModuleNotFoundError: No module named 'mast_bridge.dataset.synthetic_manifest'`。

- [ ] **步骤 3：实现 manifest scanner**

创建 `src/mast_bridge/dataset/synthetic_manifest.py`：

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from mast_bridge.dataset.manifest import ManifestEntry


def _load_metadata(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _valid_equilibrium(path: Path) -> bool:
    try:
        payload = np.load(path)
        psi = np.asarray(payload["psi"])
        psi_axis = float(payload["psi_axis"])
        psi_bndry = float(payload["psi_bndry"])
    except (KeyError, OSError, ValueError):
        return False
    return bool(np.isfinite(psi).all() and np.isfinite(psi_axis) and np.isfinite(psi_bndry) and psi_axis < psi_bndry)


def synthetic_entries(synthetic_root: Path, task: str) -> list[ManifestEntry]:
    root = synthetic_root.expanduser().resolve()
    entries: list[ManifestEntry] = []
    for sample_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        eq_path = sample_dir / "equilibrium.npz"
        metadata_path = sample_dir / "metadata.json"
        if not eq_path.is_file() or not metadata_path.is_file():
            continue
        metadata = _load_metadata(metadata_path)
        if metadata.get("solver_status") != "completed":
            continue
        parent_shot = metadata.get("parent_shot")
        if not isinstance(parent_shot, str) or not parent_shot:
            continue
        if not _valid_equilibrium(eq_path):
            continue
        entries.append(
            ManifestEntry(
                sample_id=sample_dir.name,
                source="synthetic",
                shot_id=sample_dir.name,
                data_path=eq_path,
                equilibrium_path=eq_path,
                label_path=metadata_path,
                parent_shot=parent_shot,
                solver_status="completed",
                metadata={
                    "task": task,
                    "time": metadata.get("target_time"),
                    "metadata_path": str(metadata_path),
                },
            )
        )
    return entries
```

- [ ] **步骤 4：实现 CLI**

创建 `scripts/build_synthetic_manifest.py`：

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from mast_bridge.dataset.manifest import write_manifest
from mast_bridge.dataset.synthetic_manifest import synthetic_entries


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a JSONL manifest for completed FreeGSNKE synthetic samples.")
    parser.add_argument("--synthetic-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    entries = synthetic_entries(args.synthetic_root, task=args.task)
    write_manifest(entries, args.output)
    print(f"wrote {len(entries)} synthetic entries to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **步骤 5：运行测试和编译检查**

运行：

```bash
python -m unittest tests/test_synthetic_manifest.py
python -m py_compile scripts/build_synthetic_manifest.py
```

期望：测试通过，编译退出码为 `0`。

- [ ] **步骤 6：提交**

```bash
git add src/mast_bridge/dataset/synthetic_manifest.py scripts/build_synthetic_manifest.py tests/test_synthetic_manifest.py
git commit -m "feat: build synthetic sample manifests"
```

### 任务 4：生成确定性的 FreeGSNKE variant plan

**文件：**
- 创建：`src/mast_bridge/simulation/variants.py`
- 创建：`scripts/build_freegsnke_variant_plan.py`
- 创建：`tests/test_variants.py`

**接口：**
- 输入：active shot list 文本文件、time grid 文本文件
- 输出：JSONL rows，字段包括 `shot`、`time`、`variant_id`、`ip_scale`、`fvac_scale`、`alpha_scale`、`beta_scale`、`coil_current_scale`
- 输出函数：`build_variant_rows(shots: list[str], times: list[float], variants_per_point: int, seed: int) -> list[dict]`

- [ ] **步骤 1：先写失败测试**

创建 `tests/test_variants.py`：

```python
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from mast_bridge.simulation.variants import build_variant_rows


class VariantTests(unittest.TestCase):
    def test_builds_deterministic_rows(self):
        first = build_variant_rows(["11771"], [0.16], variants_per_point=2, seed=123)
        second = build_variant_rows(["11771"], [0.16], variants_per_point=2, seed=123)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)
        self.assertEqual(first[0]["variant_id"], "v000")

    def test_scales_are_bounded(self):
        rows = build_variant_rows(["11771"], [0.16], variants_per_point=20, seed=123)

        for row in rows:
            self.assertGreaterEqual(row["ip_scale"], 0.95)
            self.assertLessEqual(row["ip_scale"], 1.05)
            self.assertGreaterEqual(row["fvac_scale"], 0.99)
            self.assertLessEqual(row["fvac_scale"], 1.01)
            self.assertGreaterEqual(row["coil_current_scale"], 0.97)
            self.assertLessEqual(row["coil_current_scale"], 1.03)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **步骤 2：运行测试确认失败**

运行：

```bash
python -m unittest tests/test_variants.py
```

期望：失败，错误包含 `ModuleNotFoundError: No module named 'mast_bridge.simulation.variants'`。

- [ ] **步骤 3：实现 variant 生成**

创建 `src/mast_bridge/simulation/variants.py`：

```python
from __future__ import annotations

import random


def _uniform_scale(rng: random.Random, width: float) -> float:
    return 1.0 + rng.uniform(-width, width)


def build_variant_rows(
    shots: list[str],
    times: list[float],
    variants_per_point: int,
    seed: int,
) -> list[dict[str, float | str]]:
    if variants_per_point < 1:
        raise ValueError("variants_per_point must be at least 1")
    rows: list[dict[str, float | str]] = []
    for shot in sorted(str(value) for value in shots):
        for time_value in sorted(float(value) for value in times):
            for variant_index in range(variants_per_point):
                rng = random.Random(f"{seed}:{shot}:{time_value:g}:{variant_index}")
                rows.append(
                    {
                        "shot": shot,
                        "time": time_value,
                        "variant_id": f"v{variant_index:03d}",
                        "ip_scale": _uniform_scale(rng, 0.05),
                        "fvac_scale": _uniform_scale(rng, 0.01),
                        "alpha_scale": _uniform_scale(rng, 0.03),
                        "beta_scale": _uniform_scale(rng, 0.03),
                        "coil_current_scale": _uniform_scale(rng, 0.03),
                    }
                )
    return rows
```

- [ ] **步骤 4：实现 variant-plan CLI**

创建 `scripts/build_freegsnke_variant_plan.py`：

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mast_bridge.simulation.variants import build_variant_rows


def _read_shots(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_times(path: Path) -> list[float]:
    return [float(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a deterministic FreeGSNKE augmentation job plan.")
    parser.add_argument("--shot-list", type=Path, required=True)
    parser.add_argument("--time-grid", type=Path, required=True)
    parser.add_argument("--variants-per-point", type=int, default=2)
    parser.add_argument("--seed", type=int, default=54)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = build_variant_rows(
        _read_shots(args.shot_list),
        _read_times(args.time_grid),
        variants_per_point=args.variants_per_point,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"wrote {len(rows)} variant jobs to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **步骤 5：运行测试和编译检查**

运行：

```bash
python -m unittest tests/test_variants.py
python -m py_compile scripts/build_freegsnke_variant_plan.py
```

期望：测试通过，编译退出码为 `0`。

- [ ] **步骤 6：提交**

```bash
git add src/mast_bridge/simulation/variants.py scripts/build_freegsnke_variant_plan.py tests/test_variants.py
git commit -m "feat: add freegsnke variant planning"
```

### 任务 5：在 FreeGSNKE solve 中应用扰动

**文件：**
- 修改：`scripts/run_freegsnke_forward.py`
- 修改：`tests/test_forward_script.py`

**接口：**
- 输入 variant plan 字段：`ip_scale`、`fvac_scale`、`alpha_scale`、`beta_scale`、`coil_current_scale`、`variant_id`
- 输出 metadata 字段：`variant_id`、`perturbation`

- [ ] **步骤 1：先写失败测试**

向 `tests/test_forward_script.py` 追加：

```python
    def test_scale_vector_multiplies_values(self):
        self.assertEqual(MODULE.scale_vector([1.0, -2.0], 1.5), [1.5, -3.0])

    def test_variant_output_dir_includes_variant_id(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = MODULE.variant_output_dir(Path(temp_dir), "11766", 0.18, "v002")

        self.assertEqual(
            output,
            Path(temp_dir) / "data" / "processed" / "synthetic" / "11766_t0.18_v002",
        )
```

- [ ] **步骤 2：运行测试确认失败**

运行：

```bash
python -m unittest tests/test_forward_script.py
```

期望：失败，错误包含 `AttributeError`，缺少 `scale_vector` 和 `variant_output_dir`。

- [ ] **步骤 3：加入纯函数 helper**

在 `scripts/run_freegsnke_forward.py` 的 `default_output_dir` 附近加入：

```python
def variant_output_dir(workspace_root: Path, shot: str, target_time: float, variant_id: str) -> Path:
    time_label = f"{target_time:g}"
    return workspace_root / "data" / "processed" / "synthetic" / f"{shot}_t{time_label}_{variant_id}"


def scale_vector(values: list[float], scale: float) -> list[float]:
    return [float(value) * float(scale) for value in values]
```

- [ ] **步骤 4：加入 CLI 参数**

在 parser 中加入：

```python
    parser.add_argument("--variant-id", default=None)
    parser.add_argument("--ip-scale", type=float, default=1.0)
    parser.add_argument("--fvac-scale", type=float, default=1.0)
    parser.add_argument("--alpha-scale", type=float, default=1.0)
    parser.add_argument("--beta-scale", type=float, default=1.0)
    parser.add_argument("--coil-current-scale", type=float, default=1.0)
```

调整默认 output dir：当传入 `--variant-id v000` 且没有显式 `--output-dir` 时，输出到 `11771_t0.16_v000`。

- [ ] **步骤 5：应用扰动**

读取 `Ip`、`fvac`、`alpha`、`beta` 后改为：

```python
    Ip = Ip * args.ip_scale
    fvac = fvac * args.fvac_scale
    alpha = scale_vector(alpha, args.alpha_scale)
    beta = scale_vector(beta, args.beta_scale)
```

`_apply_currents(...)` 之后，当 `args.coil_current_scale != 1.0` 时，把 tokamak 中的 active/passive coil current 和返回的 metadata current 同步乘以该 scale。

- [ ] **步骤 6：记录扰动 metadata**

在 `metadata` 中加入：

```python
        "variant_id": args.variant_id,
        "perturbation": {
            "ip_scale": args.ip_scale,
            "fvac_scale": args.fvac_scale,
            "alpha_scale": args.alpha_scale,
            "beta_scale": args.beta_scale,
            "coil_current_scale": args.coil_current_scale,
        },
```

- [ ] **步骤 7：运行测试**

运行：

```bash
python -m unittest tests/test_forward_script.py
```

期望：通过。

- [ ] **步骤 8：提交**

```bash
git add scripts/run_freegsnke_forward.py tests/test_forward_script.py
git commit -m "feat: perturb freegsnke forward solves"
```

### 任务 6：实现幂等 variant batch runner

**文件：**
- 创建：`scripts/run_freegsnke_variant_batch.py`
- 修改：`README.md`

**接口：**
- 输入：`scripts/build_freegsnke_variant_plan.py` 生成的 JSONL
- 输出：`../data/processed/synthetic/<shot>_t<time>_<variant_id>/` 下的 synthetic sample 目录

- [ ] **步骤 1：创建 batch runner**

创建 `scripts/run_freegsnke_variant_batch.py`：

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run FreeGSNKE variant jobs from a JSONL plan.")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("../data/raw/mast"))
    parser.add_argument("--fit-path", type=Path, default=Path("../data/processed/real/lao_parameter_ensemble/all_zarr_lao_parameter_fits.npz"))
    parser.add_argument("--synthetic-root", type=Path, default=Path("../data/processed/synthetic"))
    parser.add_argument("--nx", type=int, default=65)
    parser.add_argument("--ny", type=int, default=65)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--max-iterations", type=int, default=100)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    count = 0
    for line in args.plan.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        time_label = f"{float(row['time']):g}"
        output_dir = args.synthetic_root / f"{row['shot']}_t{time_label}_{row['variant_id']}"
        if (output_dir / "equilibrium.npz").is_file() and (output_dir / "metadata.json").is_file():
            print(f"skip existing {output_dir}")
            continue
        command = [
            sys.executable,
            str(SCRIPT_DIR / "run_freegsnke_forward.py"),
            "--data-dir", str(args.data_dir),
            "--machine-dir", str(args.data_dir / "machine" / str(row["shot"])),
            "--fit-path", str(args.fit_path),
            "--shot", str(row["shot"]),
            "--time", str(row["time"]),
            "--variant-id", str(row["variant_id"]),
            "--ip-scale", str(row["ip_scale"]),
            "--fvac-scale", str(row["fvac_scale"]),
            "--alpha-scale", str(row["alpha_scale"]),
            "--beta-scale", str(row["beta_scale"]),
            "--coil-current-scale", str(row["coil_current_scale"]),
            "--nx", str(args.nx),
            "--ny", str(args.ny),
            "--tolerance", str(args.tolerance),
            "--max-iterations", str(args.max_iterations),
            "--output-dir", str(output_dir),
        ]
        subprocess.run(command, check=True)
        count += 1
    print(f"ran {count} FreeGSNKE variant jobs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **步骤 2：把 variant 命令写入 README**

加入：

```bash
source .freegsnke-solve-env/bin/activate
python scripts/build_freegsnke_variant_plan.py \
  --shot-list configs/shot_lists/local_tiny_downloaded.txt \
  --time-grid configs/time_grids/local_tiny_times.txt \
  --variants-per-point 2 \
  --seed 54 \
  --output ../data/manifests/freegsnke_local_tiny_plan.jsonl

python scripts/run_freegsnke_variant_batch.py \
  --plan ../data/manifests/freegsnke_local_tiny_plan.jsonl \
  --data-dir ../data/raw/mast \
  --fit-path ../data/processed/real/lao_parameter_ensemble/all_zarr_lao_parameter_fits.npz \
  --synthetic-root ../data/processed/synthetic \
  --nx 65 \
  --ny 65
```

- [ ] **步骤 3：编译检查**

运行：

```bash
python -m py_compile scripts/run_freegsnke_variant_batch.py
```

期望：退出码为 `0`。

- [ ] **步骤 4：提交**

```bash
git add README.md scripts/run_freegsnke_variant_batch.py
git commit -m "feat: run freegsnke variants in batch"
```

### 任务 7：本地获取数据并跑 FreeGSNKE smoke

**文件：**
- 使用：`configs/shot_lists/local_tiny.txt`
- 使用：`configs/time_grids/local_tiny_times.txt`
- 运行时产出：`../data/raw/mast`、`../data/processed/real`、`../data/processed/synthetic`、`../data/manifests`

**接口：**
- 输入：可用的 `.mast-download-env`、`.mast-process-env`、`.freegsnke-solve-env`
- 输出：至少一个 completed synthetic sample 和一个 synthetic manifest

- [ ] **步骤 1：下载 tiny local shots**

运行：

```bash
cd /Users/mingdonghe/pj/fusion-workspace/mast-bridge
source .mast-download-env/bin/activate
SHOT_LIST=configs/shot_lists/local_tiny.txt
ACTIVE_SHOT_LIST=configs/shot_lists/local_tiny_downloaded.txt
DATA_DIR=../data/raw/mast

while read shot; do
  [ -z "$shot" ] && continue
  python scripts/download_mast_shots.py --data-dir "$DATA_DIR" --shot "$shot"
done < "$SHOT_LIST"

while read shot; do
  [ -z "$shot" ] && continue
  if [ -d "$DATA_DIR/${shot}.zarr" ]; then
    echo "$shot"
  fi
done < "$SHOT_LIST" > "$ACTIVE_SHOT_LIST"
```

期望：`configs/shot_lists/local_tiny_downloaded.txt` 至少包含一个 shot。

- [ ] **步骤 2：生成 machine pickles**

运行：

```bash
source .mast-process-env/bin/activate
ACTIVE_SHOT_LIST=configs/shot_lists/local_tiny_downloaded.txt
DATA_DIR=../data/raw/mast

while read shot; do
  [ -z "$shot" ] && continue
  python scripts/build_machine_from_zarr.py \
    --data-dir "$DATA_DIR" \
    --output-dir "$DATA_DIR/machine/$shot" \
    --shot "$shot" \
    --overwrite
  python scripts/inspect_shot.py \
    --data-dir "$DATA_DIR" \
    --machine-dir "$DATA_DIR/machine/$shot" \
    --shot "$shot"
done < "$ACTIVE_SHOT_LIST"
```

期望：每个 active shot 都有 `../data/raw/mast/machine/<shot>/MAST_active_coils.pickle`。

- [ ] **步骤 3：生成 Lao/EFIT NPZ**

运行：

```bash
FIT_PATH=../data/processed/real/lao_parameter_ensemble/all_zarr_lao_parameter_fits.npz
python scripts/build_lao_fit_npz.py \
  --shot-list configs/shot_lists/local_tiny_downloaded.txt \
  --data-dir ../data/raw/mast \
  --output "$FIT_PATH"
```

期望：`$FIT_PATH` 存在。

- [ ] **步骤 4：生成 variant plan**

运行：

```bash
python scripts/build_freegsnke_variant_plan.py \
  --shot-list configs/shot_lists/local_tiny_downloaded.txt \
  --time-grid configs/time_grids/local_tiny_times.txt \
  --variants-per-point 2 \
  --seed 54 \
  --output ../data/manifests/freegsnke_local_tiny_plan.jsonl
```

期望：JSONL 行数等于 downloaded shot 数量乘以 `2` 个 time 点再乘以 `2` 个 variant。

- [ ] **步骤 5：运行 FreeGSNKE variants**

运行：

```bash
source .freegsnke-solve-env/bin/activate
python scripts/run_freegsnke_variant_batch.py \
  --plan ../data/manifests/freegsnke_local_tiny_plan.jsonl \
  --data-dir ../data/raw/mast \
  --fit-path ../data/processed/real/lao_parameter_ensemble/all_zarr_lao_parameter_fits.npz \
  --synthetic-root ../data/processed/synthetic \
  --nx 65 \
  --ny 65 \
  --tolerance 1e-8 \
  --max-iterations 100
```

期望：`../data/processed/synthetic` 下至少有一个目录包含 `equilibrium.npz` 和 `metadata.json`。

- [ ] **步骤 6：生成 synthetic manifest**

运行：

```bash
source .mast-process-env/bin/activate
python scripts/build_synthetic_manifest.py \
  --synthetic-root ../data/processed/synthetic \
  --output ../data/manifests/tokamark_task_1-3_synthetic_local_tiny.jsonl \
  --task task_1-3
```

期望：输出 JSONL 至少包含一行。

- [ ] **步骤 7：只提交代码状态**

不要提交下载的数据。运行：

```bash
git status --short
```

期望：`../data` 下的数据不在 `mast-bridge` 仓库里，或已经被 ignore。

### 任务 8：扩大到批量下载、批量处理和批量求解

**文件：**
- 创建：`configs/shot_lists/local_batch.txt`
- 创建：`configs/time_grids/local_batch_times.txt`
- 修改：`README.md`
- 运行时产出：`../data/raw/mast`、`../data/processed/real`、`../data/processed/synthetic`、`../data/manifests`

**接口：**
- 输入：已经通过任务 7 验证的 tiny pipeline
- 输出：一批可用于训练设计评估的 real shot 数据和 synthetic equilibrium 样本

- [ ] **步骤 1：创建 batch shot list**

先用本地已有且处理链路已验证过的 shot 建立第一版 batch list。创建 `configs/shot_lists/local_batch.txt`：

```text
11766
11767
11768
11769
11771
11772
11773
11774
11775
11776
16431
21719
```

如果后续要扩大到 50、100、500 个 shot，只追加到这个文件，不改脚本逻辑。

- [ ] **步骤 2：创建 batch time grid**

创建 `configs/time_grids/local_batch_times.txt`：

```text
0.12
0.14
0.16
0.18
0.20
0.22
0.24
0.26
0.28
0.30
```

这会给每个 shot 10 个目标时间点。第一批 12 个 shot、每点 4 个 variant 时，目标是 `12 * 10 * 4 = 480` 个 synthetic solve jobs。

- [ ] **步骤 3：批量下载**

运行：

```bash
cd /Users/mingdonghe/pj/fusion-workspace/mast-bridge
source .mast-download-env/bin/activate
SHOT_LIST=configs/shot_lists/local_batch.txt
ACTIVE_SHOT_LIST=configs/shot_lists/local_batch_downloaded.txt
DATA_DIR=../data/raw/mast

while read shot; do
  [ -z "$shot" ] && continue
  python scripts/download_mast_shots.py --data-dir "$DATA_DIR" --shot "$shot"
done < "$SHOT_LIST"

while read shot; do
  [ -z "$shot" ] && continue
  if [ -d "$DATA_DIR/${shot}.zarr" ]; then
    echo "$shot"
  else
    echo "Missing downloaded shot: $shot" >&2
  fi
done < "$SHOT_LIST" > "$ACTIVE_SHOT_LIST"
```

期望：`configs/shot_lists/local_batch_downloaded.txt` 至少包含 10 个 shot。低于 10 个时不要进入批量求解，先补 shot。

- [ ] **步骤 4：批量构建 machine pickles**

运行：

```bash
source .mast-process-env/bin/activate
ACTIVE_SHOT_LIST=configs/shot_lists/local_batch_downloaded.txt
DATA_DIR=../data/raw/mast

while read shot; do
  [ -z "$shot" ] && continue
  python scripts/build_machine_from_zarr.py \
    --data-dir "$DATA_DIR" \
    --output-dir "$DATA_DIR/machine/$shot" \
    --shot "$shot" \
    --overwrite
  python scripts/inspect_shot.py \
    --data-dir "$DATA_DIR" \
    --machine-dir "$DATA_DIR/machine/$shot" \
    --shot "$shot"
done < "$ACTIVE_SHOT_LIST"
```

期望：每个 active shot 都有 machine pickle 目录；失败的 shot 从 active list 中移除并记录原因。

- [ ] **步骤 5：批量构建 Lao/EFIT NPZ**

运行：

```bash
FIT_PATH=../data/processed/real/lao_parameter_ensemble/all_zarr_lao_parameter_fits.npz
python scripts/build_lao_fit_npz.py \
  --shot-list configs/shot_lists/local_batch_downloaded.txt \
  --data-dir ../data/raw/mast \
  --output "$FIT_PATH"
```

期望：`$FIT_PATH` 存在，且包含每个 active shot 的多个 fitted time slices。

- [ ] **步骤 6：生成 batch variant plan**

运行：

```bash
python scripts/build_freegsnke_variant_plan.py \
  --shot-list configs/shot_lists/local_batch_downloaded.txt \
  --time-grid configs/time_grids/local_batch_times.txt \
  --variants-per-point 4 \
  --seed 54 \
  --output ../data/manifests/freegsnke_local_batch_plan.jsonl
```

期望：plan 行数等于 active shot 数量乘以 10 个 time 点再乘以 4 个 variant。

- [ ] **步骤 7：批量 FreeGSNKE 求解**

运行：

```bash
source .freegsnke-solve-env/bin/activate
python scripts/run_freegsnke_variant_batch.py \
  --plan ../data/manifests/freegsnke_local_batch_plan.jsonl \
  --data-dir ../data/raw/mast \
  --fit-path ../data/processed/real/lao_parameter_ensemble/all_zarr_lao_parameter_fits.npz \
  --synthetic-root ../data/processed/synthetic \
  --nx 65 \
  --ny 65 \
  --tolerance 1e-8 \
  --max-iterations 100
```

期望：已有样本被跳过；新样本写入 `../data/processed/synthetic/<shot>_t<time>_<variant_id>/`。

- [ ] **步骤 8：生成 batch synthetic manifest**

运行：

```bash
source .mast-process-env/bin/activate
python scripts/build_synthetic_manifest.py \
  --synthetic-root ../data/processed/synthetic \
  --output ../data/manifests/tokamark_task_1-3_synthetic_local_batch.jsonl \
  --task task_1-3
```

期望：manifest 行数大于等于 300。低于 300 时，先解决 solve 失败或扩大 shot/time/variant，不进入训练设计。

- [ ] **步骤 9：提交代码和配置，不提交数据**

```bash
git status --short
git add README.md configs/shot_lists/local_batch.txt configs/time_grids/local_batch_times.txt
git commit -m "config: add batch augmentation inputs"
```

不要提交 `../data` 下的下载数据、NPZ、PNG 或 JSONL 大文件。

### 任务 9：数据质量报告和训练前门槛

**文件：**
- 修改：`README.md`
- 创建：`scripts/write_augmentation_report.py`
- 创建：`docs/local-augmentation-results.md`

**接口：**
- 输入：batch plan、synthetic manifest、synthetic sample metadata
- 输出：样本数量、solve 成功率、无效样本数量、是否进入训练设计阶段

- [ ] **步骤 1：创建报告脚本**

创建 `scripts/write_augmentation_report.py`：

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _count_jsonl(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _metadata_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(root.glob("*/metadata.json"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write a summary report for local FreeGSNKE augmentation data.")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--synthetic-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-valid-samples", type=int, default=300)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    planned = _count_jsonl(args.plan)
    valid = _count_jsonl(args.manifest)
    metadata_files = _metadata_files(args.synthetic_root)
    completed = 0
    failed = 0
    for path in metadata_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("solver_status") == "completed":
            completed += 1
        else:
            failed += 1
    success_rate = completed / planned if planned else 0.0
    ready = valid >= args.min_valid_samples and success_rate >= 0.70
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(
            [
                "# Local Augmentation Data Report",
                "",
                "Date: 2026-07-28",
                "",
                "## Counts",
                "",
                f"- Planned solve jobs: {planned}",
                f"- Metadata files: {len(metadata_files)}",
                f"- Completed solves: {completed}",
                f"- Failed or non-completed solves: {failed}",
                f"- Valid manifest samples: {valid}",
                f"- Solve success rate: {success_rate:.3f}",
                "",
                "## Decision",
                "",
                f"- Ready for training design: {'yes' if ready else 'no'}",
                "- Gate: valid samples >= min-valid-samples and solve success rate >= 0.70",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}")
    return 0 if ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **步骤 2：运行单元测试和编译检查**

从 `mast-bridge` 运行：

```bash
cd /Users/mingdonghe/pj/fusion-workspace/mast-bridge
python -m unittest discover -s tests
python -m py_compile \
  scripts/download_mast_shots.py \
  scripts/build_machine_from_zarr.py \
  scripts/build_lao_fit_npz.py \
  scripts/inspect_shot.py \
  scripts/run_freegsnke_forward.py \
  scripts/build_freegsnke_variant_plan.py \
  scripts/run_freegsnke_variant_batch.py \
  scripts/build_synthetic_manifest.py \
  scripts/write_augmentation_report.py
```

期望：测试通过，所有脚本编译退出码为 `0`。

- [ ] **步骤 3：生成数据质量报告**

运行：

```bash
python scripts/write_augmentation_report.py \
  --plan ../data/manifests/freegsnke_local_batch_plan.jsonl \
  --manifest ../data/manifests/tokamark_task_1-3_synthetic_local_batch.jsonl \
  --synthetic-root ../data/processed/synthetic \
  --output docs/local-augmentation-results.md \
  --min-valid-samples 300
```

期望：如果退出码为 `0`，说明数据量和 solve 成功率达到训练设计门槛；如果退出码为 `2`，继续扩大数据或修复 solve 失败原因。

- [ ] **步骤 4：人工检查报告**

打开 `docs/local-augmentation-results.md`，检查：

- `Valid manifest samples >= 300`
- `Solve success rate >= 0.70`
- 失败样本没有集中在某个 shot 或某个 time grid
- synthetic 样本目录命名都能追溯到 `parent_shot`、`target_time`、`variant_id`

- [ ] **步骤 5：训练设计阶段的进入条件**

只有满足下面条件，才新建一份训练计划：

```text
1. 至少 300 个 valid synthetic samples
2. 至少 10 个 parent shots
3. solve success rate >= 70%
4. 所有 synthetic samples 都有 parent_shot 和 target_time
5. 没有把 synthetic samples 放入 validation/test 的计划
6. 明确下一阶段训练必须参考 external/tokamark 官方 dataset/split/evaluator 方法
```

- [ ] **步骤 6：提交报告脚本和报告**

```bash
git add README.md scripts/write_augmentation_report.py docs/local-augmentation-results.md
git commit -m "docs: record augmentation data readiness"
```

## 自检

- 需求覆盖：计划覆盖批量下载、批量处理、批量 FreeGSNKE 求解、manifest 生成、数据质量报告和训练前门槛。
- 占位检查：没有未解决的占位标记。
- 类型一致性：`build_variant_rows`、`synthetic_entries`、`assign_parent_shot_splits`、`split_for_row` 的签名在后续任务使用前已经定义。
- 范围说明：本计划不启动训练。训练计划必须在 `docs/local-augmentation-results.md` 显示 ready 后另起，并参考 `external/tokamark` 官方 dataset、split、windowing 和 evaluator。
