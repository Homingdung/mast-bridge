#!/usr/bin/env python3
"""Random shot-level split for the PCA 01σ dataset (no campaign stratification).

Semantics:
  - Split at the SHOT level (no leakage): all time slices of a shot go to one
    of train/val/test.  Real and synthetic samples share the same shot lists.
  - Pure random shuffle of the full shot pool, 80/10/10, seed 20260825.
  - Real rows come from real.jsonl; synthetic rows from the accepted manifest
    (filtered to the same shot/time pairs).  Only train shots are used for
    training; val/test are real-only evaluation sets.

Outputs (data/manifests/training_pca_01sigma/):
  split_train_shots.jsonl / split_val_shots.jsonl / split_test_shots.jsonl
  split_train_real.jsonl / split_val_real.jsonl / split_test_real.jsonl
  split_train_synth.jsonl
  run_real.jsonl                     (train_real + val_real)
  run_synth_pretrain.jsonl           (train-synth only)
  run_mixed_100pct.jsonl             (train_real + train_synth + val_real)
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = SCRIPT_ROOT.parent
DEFAULT_OUT = WORKSPACE_ROOT / "data" / "manifests" / "training_pca_01sigma"


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_shots(path: Path, shots: list[str]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for shot in shots:
            handle.write(json.dumps({"shot_id": shot}) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--real-manifest", type=Path, default=None)
    parser.add_argument("--synth-manifest", type=Path, default=None)
    args = parser.parse_args(argv)

    out_dir = (args.out_dir or DEFAULT_OUT).expanduser().resolve()
    real_manifest = (args.real_manifest or out_dir / "real.jsonl").resolve()
    synth_manifest = (
        args.synth_manifest or WORKSPACE_ROOT / "data" / "manifests" / "fullsolve_pca_01sigma_accepted.jsonl"
    ).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    real_rows = load_rows(real_manifest)
    synth_rows = load_rows(synth_manifest)

    # synthetic rows: keep only the accepted samples whose (shot, time) is in real
    # (they are 1:1 paired by construction)
    real_by_pair = {(str(r["shot_id"]), round(float(r["target_time"]), 6)) for r in real_rows}
    synth_paired = [
        r for r in synth_rows
        if (str(r["parent_shot"]), round(float(r["target_time"]), 6)) in real_by_pair
    ]
    print(f"real rows: {len(real_rows)}")
    print(f"synth rows (accepted, paired): {len(synth_paired)}")

    shots = sorted({str(r["shot_id"]) for r in real_rows})
    print(f"shot pool: {len(shots)}")

    rng = random.Random(args.seed)
    pool = list(shots)
    rng.shuffle(pool)
    n_train = int(len(pool) * 0.8)
    n_val = int(len(pool) * 0.1)
    train_shots = pool[:n_train]
    val_shots = pool[n_train:n_train + n_val]
    test_shots = pool[n_train + n_val:]
    assert not (set(train_shots) & set(val_shots))
    assert not (set(train_shots) & set(test_shots))
    assert not (set(val_shots) & set(test_shots))
    print(f"shots: train={len(train_shots)} val={len(val_shots)} test={len(test_shots)}")

    train_set, val_set, test_set = set(train_shots), set(val_shots), set(test_shots)

    def by_shot(rows: list[dict], key: str) -> dict[str, list[dict]]:
        out: dict[str, list[dict]] = {}
        for r in rows:
            out.setdefault(str(r[key]), []).append(r)
        return out

    real_by_shot = by_shot(real_rows, "shot_id")
    train_real = [r for s in train_shots for r in real_by_shot[s]]
    val_real = [r for s in val_shots for r in real_by_shot[s]]
    test_real = [r for s in test_shots for r in real_by_shot[s]]

    synth_by_shot = by_shot(synth_paired, "parent_shot")
    train_synth = [r for s in train_shots for r in synth_by_shot[s]]
    print(f"rows: train_real={len(train_real)} val_real={len(val_real)} test_real={len(test_real)}")
    print(f"      train_synth={len(train_synth)}")

    # sanity: synthetic train rows have parent shots in train only
    assert all(str(r["parent_shot"]) in train_set for r in train_synth)

    write_shots(out_dir / "split_train_shots.jsonl", train_shots)
    write_shots(out_dir / "split_val_shots.jsonl", val_shots)
    write_shots(out_dir / "split_test_shots.jsonl", test_shots)
    write_rows(out_dir / "split_train_real.jsonl", train_real)
    write_rows(out_dir / "split_val_real.jsonl", val_real)
    write_rows(out_dir / "split_test_real.jsonl", test_real)
    write_rows(out_dir / "split_train_synth.jsonl", train_synth)
    write_rows(out_dir / "run_real.jsonl", train_real + val_real)
    write_rows(out_dir / "run_synth_pretrain.jsonl", train_synth)
    write_rows(out_dir / "run_mixed_100pct.jsonl", train_real + train_synth + val_real)

    for name in (
        "split_train_shots", "split_val_shots", "split_test_shots",
        "split_train_real", "split_val_real", "split_test_real",
        "split_train_synth", "run_real", "run_synth_pretrain", "run_mixed_100pct",
    ):
        p = out_dir / f"{name}.jsonl"
        print(f"  {p.name}: {sum(1 for _ in p.open())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
