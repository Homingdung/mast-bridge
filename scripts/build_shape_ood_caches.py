#!/usr/bin/env python3
"""Shape-OOD 步骤 [6] 准备：嵌套子集 manifest + cache 切片 + 合成预训练 cache。

- subsets: 从 split_train_real.jsonl（diverted train 176,319 行）按炮分层嵌套抽 50/25/5%，
  拼接 shape_ood val（2,835 行）→ subset_run_real_{f}pct.jsonl（run 版，与 §19 模式一致）
- caches: 从 real_full.npz 按 sample_id 切片 → subset_run_real_{f}pct.npz
- synth C: split_synth_pretrain.jsonl → synth_ood_pretrain_clean.npz（从 synth_pretrain_clean.npz 切片）
- test caches: split_test_real.jsonl → test_real.npz；split_test_ood_only_real.jsonl → test_ood_only_real.npz
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = SCRIPT_ROOT.parent
sys.path.insert(0, str(SCRIPT_ROOT))

from scripts.slice_cache import main as slice_cache_main  # noqa: E402

SEED = 20260828
M = WORKSPACE_ROOT / "data/manifests/shape_ood"
C = WORKSPACE_ROOT / "data/processed/training_cache_pca_01sigma"
OUT = C


def run_slice(src: str, manifest: str, out: str) -> None:
    import subprocess
    import sys as _sys

    cmd = [_sys.executable, str(SCRIPT_ROOT / "scripts/slice_cache.py"),
           "--src", str(C / src), "--manifest", str(M / manifest), "--out", str(OUT / out)]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> int:
    train_rows = [json.loads(l) for l in (M / "split_train_real.jsonl").open()]
    val_rows = [json.loads(l) for l in (M / "split_val_real.jsonl").open()]
    rng = random.Random(SEED)
    shots = sorted({r["shot_id"] for r in train_rows})
    by_shot = {}
    for r in train_rows:
        by_shot.setdefault(r["shot_id"], []).append(r)

    for frac in (100, 50, 25, 5):
        if frac == 100:
            sel_shots = set(shots)
        else:
            n = max(1, round(len(shots) * frac / 100))
            sel_shots = set(rng.sample(shots, n))
        sel = [r for s in sel_shots for r in by_shot[s]]
        run_rows = sel + val_rows
        (M / f"subset_run_real_{frac}pct.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in run_rows), encoding="utf-8")
        print(f"subset {frac}%: train_rows={len(sel)} + val={len(val_rows)} = {len(run_rows)}", flush=True)
        run_slice("real_full.npz", f"subset_run_real_{frac}pct.jsonl", f"subset_run_real_{frac}pct.npz")

    # 合成预训练 cache
    run_slice("synth_pretrain_clean.npz", "split_synth_pretrain.jsonl", "synth_ood_pretrain_clean.npz")
    # test caches
    run_slice("real_full.npz", "split_test_real.jsonl", "test_real.npz")
    run_slice("real_full.npz", "split_test_ood_only_real.jsonl", "test_ood_only_real.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
