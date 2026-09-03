#!/usr/bin/env python3
"""limiter 版 Shape-OOD split：topology=limited，OOD = delta_asym > P90。
输出 shape_ood_limiter/ 目录：split train/val/test/test_ood_only + subsets 100/50/25 + synth C。
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
from mast_bridge.training.tokamind_manifest import load_manifest_rows  # noqa: E402

OOD_THRESHOLD = 0.010  # delta_asym > P90 (limited)
R_TEST, R_VAL_LO = 0.5, 0.1
MAX_VAL_SHOTS = 300
SEED = 20260828
META = WORKSPACE_ROOT / "data/processed/shape_ood"
MANIFESTS = WORKSPACE_ROOT / "data/manifests"
OUT = MANIFESTS / "shape_ood_limiter"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> int:
    real_rows = [r for r in load_manifest_rows(MANIFESTS / "training_pca_01sigma/real.jsonl") if r.get("source") == "real"]
    shape = {json.loads(l)["sample_id"]: json.loads(l) for l in (META / "real_shape_metadata.jsonl").open()}
    synth_rows = [r for r in load_manifest_rows(MANIFESTS / "fullsolve_pca_01sigma_accepted.jsonl") if r.get("source") == "synthetic"]
    synth_shape = {json.loads(l)["sample_id"]: json.loads(l) for l in (META / "synthetic_shape_metadata.jsonl").open()}

    occ = {}
    for r in real_rows:
        d = shape.get(r["sample_id"])
        if d is None or not d["lcfs_valid"] or d.get("topology") != "limited":
            continue
        s = r["shot_id"]
        occ.setdefault(s, {"n": 0, "ood": 0})
        occ[s]["n"] += 1
        occ[s]["ood"] += int(d["delta_asym"] > OOD_THRESHOLD)
    for s in occ:
        occ[s]["r"] = occ[s]["ood"] / occ[s]["n"]
    shots_all = sorted(occ, key=lambda s: -occ[s]["r"])

    test_shots = {s for s in shots_all if occ[s]["r"] > R_TEST}
    rest = [s for s in shots_all if occ[s]["r"] <= R_TEST and occ[s]["r"] > R_VAL_LO]
    val_shots = set(rest[:MAX_VAL_SHOTS])
    train_shots = {s for s in shots_all if s not in test_shots and s not in val_shots}
    assert not (train_shots & val_shots) and not (train_shots & test_shots) and not (val_shots & test_shots)

    def dump(shots, path, ood_only=False):
        kept = 0
        with (OUT / path).open("w", encoding="utf-8") as f:
            for r in real_rows:
                s = r["shot_id"]
                d = shape.get(r["sample_id"])
                if s not in shots or d is None or not d["lcfs_valid"] or d.get("topology") != "limited":
                    continue
                if ood_only and not (d["delta_asym"] > OOD_THRESHOLD):
                    continue
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                kept += 1
        return kept

    n_tr = dump(train_shots, "split_train_real.jsonl")
    n_va = dump(val_shots, "split_val_real.jsonl")
    n_te = dump(test_shots, "split_test_real.jsonl")
    n_ood = dump(test_shots, "split_test_ood_only_real.jsonl", ood_only=True)

    # subsets 100/50/25（嵌套按炮 + val）
    val_rows = [json.loads(l) for l in (OUT / "split_val_real.jsonl").open()]
    train_rows = [json.loads(l) for l in (OUT / "split_train_real.jsonl").open()]
    by_shot = {}
    for r in train_rows:
        by_shot.setdefault(r["shot_id"], []).append(r)
    all_shots = sorted(by_shot)
    rng = random.Random(SEED)
    for frac in (100, 50, 25):
        sel = all_shots if frac == 100 else rng.sample(all_shots, max(1, round(len(all_shots) * frac / 100)))
        rows = [r for s in sel for r in by_shot[s]] + val_rows
        (OUT / f"subset_run_real_{frac}pct.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        print(f"subset {frac}%: {len(rows)} rows", flush=True)

    # Synthetic C：limited + train-parent
    train_keys = set()
    for r in real_rows:
        d = shape.get(r["sample_id"])
        if r["shot_id"] in train_shots and d is not None and d["lcfs_valid"] and d.get("topology") == "limited":
            train_keys.add((r["shot_id"], round(float(r["target_time"]), 9)))
    n_c = 0
    with (OUT / "split_synth_pretrain.jsonl").open("w", encoding="utf-8") as f:
        for r in synth_rows:
            d = synth_shape.get(r["sample_id"])
            if d is None or not d["lcfs_valid"] or d.get("topology") != "limited":
                continue
            if (r["parent_shot"], round(float(r["target_time"]), 9)) not in train_keys:
                continue
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n_c += 1

    print(f"shots: train={len(train_shots)} val={len(val_shots)} test={len(test_shots)}")
    print(f"slices: train={n_tr} val={n_va} test={n_te} test_ood_only={n_ood}")
    print(f"synthetic C: {n_c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
