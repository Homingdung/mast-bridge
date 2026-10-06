#!/usr/bin/env python3
"""Build the frozen diverted-only, shot-level Shape-OOD split.

An OOD slice has delta_asym < -0.214 (Real diverted P10).
For each shot, r_OOD is the fraction of OOD slices: Test > 0.5;
Validation (0.1, 0.5], ranked from high to low for approximately 500 shots;
Train <= 0.1. Write Real train/validation/test and OOD-only manifests.
Synthetic pretraining uses topology-aligned, exact Train-parent matches only."""
from __future__ import annotations

import os

import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(os.environ.get("MAST_WORKSPACE_ROOT", SCRIPT_ROOT.parent)).expanduser().resolve()
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from mast_bridge.training.tokamind_manifest import load_manifest_rows  # noqa: E402

OOD_THRESHOLD = -0.214
R_TEST, R_VAL_LO = 0.5, 0.1
MAX_VAL_SHOTS = 500
SEED = 20260828
META = WORKSPACE_ROOT / "data/processed/shape_ood"
MANIFESTS = WORKSPACE_ROOT / "data/manifests"
OUT = MANIFESTS / "shape_ood"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> int:
    real_rows = [r for r in load_manifest_rows(MANIFESTS / "training_pca_01sigma/real.jsonl") if r.get("source") == "real"]
    shape = {}
    for l in (META / "real_shape_metadata.jsonl").open():
        d = json.loads(l)
        shape[d["sample_id"]] = d
    synth_rows = [r for r in load_manifest_rows(MANIFESTS / "fullsolve_pca_01sigma_accepted.jsonl") if r.get("source") == "synthetic"]
    synth_shape = {}
    for l in (META / "synthetic_shape_metadata.jsonl").open():
        d = json.loads(l)
        synth_shape[d["sample_id"]] = d

    # --- Per-shot OOD occupancy: diverted Real slices with valid LCFS only ---
    occ = {}
    for r in real_rows:
        sid = r["sample_id"]
        d = shape.get(sid)
        if d is None or not d["lcfs_valid"] or d.get("topology") != "diverted":
            continue
        s = r["shot_id"]
        occ.setdefault(s, {"n": 0, "ood": 0})
        occ[s]["n"] += 1
        occ[s]["ood"] += int(d["delta_asym"] < OOD_THRESHOLD)
    for s in occ:
        occ[s]["r"] = occ[s]["ood"] / occ[s]["n"]
    shots_all = sorted(occ, key=lambda s: -occ[s]["r"])

    test_shots = {s for s in shots_all if occ[s]["r"] > R_TEST}
    rest = [s for s in shots_all if occ[s]["r"] <= R_TEST and occ[s]["r"] > R_VAL_LO]
    val_shots = set(rest[:MAX_VAL_SHOTS])
    train_shots = {s for s in shots_all if s not in test_shots and s not in val_shots}
    assert not (train_shots & val_shots) and not (train_shots & test_shots) and not (val_shots & test_shots)

    # --- Write Real split manifests: diverted topology and valid LCFS only ---
    def dump(shots, path, ood_only=False):
        kept = 0
        with (OUT / path).open("w", encoding="utf-8") as f:
            for r in real_rows:
                s = r["shot_id"]
                d = shape.get(r["sample_id"])
                if s not in shots or d is None or not d["lcfs_valid"] or d.get("topology") != "diverted":
                    continue
                if ood_only and not (d["delta_asym"] < OOD_THRESHOLD):
                    continue
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                kept += 1
        return kept

    n_tr = dump(train_shots, "split_train_real.jsonl")
    n_va = dump(val_shots, "split_val_real.jsonl")
    n_te = dump(test_shots, "split_test_real.jsonl")
    n_ood = dump(test_shots, "split_test_ood_only_real.jsonl", ood_only=True)

    # --- Synthetic C: diverted topology and exact Train-parent matching ---
    train_keys = set()
    for r in real_rows:
        s = r["shot_id"]
        d = shape.get(r["sample_id"])
        if s in train_shots and d is not None and d["lcfs_valid"] and d.get("topology") == "diverted":
            train_keys.add((s, round(float(r["target_time"]), 9)))
    n_c = 0
    with (OUT / "split_synth_pretrain.jsonl").open("w", encoding="utf-8") as f:
        for r in synth_rows:
            d = synth_shape.get(r["sample_id"])
            if d is None or not d["lcfs_valid"] or d.get("topology") != "diverted":
                continue
            if (r["parent_shot"], round(float(r["target_time"]), 9)) not in train_keys:
                continue
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n_c += 1

    print(f"shots: train={len(train_shots)} val={len(val_shots)} test={len(test_shots)}")
    print(f"real slices: train={n_tr} val={n_va} test={n_te} test_ood_only={n_ood}")
    print(f"synthetic C (diverted, train-parent): {n_c}")
    with (OUT / "split_summary.txt").open("w") as f:
        f.write(f"OOD_THRESHOLD delta_asym < {OOD_THRESHOLD}\n")
        f.write(f"shots: train={len(train_shots)} val={len(val_shots)} test={len(test_shots)}\n")
        f.write(f"real slices: train={n_tr} val={n_va} test={n_te} test_ood_only={n_ood}\n")
        f.write(f"synthetic C: {n_c}\n")
        f.write(f"test r_OOD range: {min(occ[s]['r'] for s in test_shots):.2f}-{max(occ[s]['r'] for s in test_shots):.2f}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
