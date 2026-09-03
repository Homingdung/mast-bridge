#!/usr/bin/env python3
"""Split v2 real/synthetic data into campaign-stratified train/val/test sets.

Split policy (progress.md 16.1):
  - Shot-level split (no leakage), campaign-stratified 80/10/10, seed 20260812
  - Pool = 9,995 shots covered by the real manifest
  - Synthetic training rows: parent_shot in train shots only
  - val/test contain real rows only (no synthetic)

Outputs (data/manifests/training_v2_statwidth/):
  split_{train,val,test}_shots.jsonl   -- shot lists
  split_{train_real,val_real,test_real}.jsonl
  split_train_synth_{clean,noisy}.jsonl
  run_real.jsonl                       -- train_real + val_real (for training runs)
  run_synth_{clean,noisy}.jsonl        -- train_synth_* + val_real
  run_mixed_{clean,noisy}.jsonl        -- train_real + train_synth_* + val_real

Caches (data/processed/training_cache_v2_statwidth/):
  {train_real,val_real,test_real}.npz
  train_synth_{clean,noisy}.npz
  mixed_{clean,noisy}.npz              -- train_real + train_synth_* (56,495 rows)
  run_*.npz                            -- matching run_*.jsonl row order
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = SCRIPT_ROOT.parent


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def write_shots(path: Path, shots: list[str]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for shot in shots:
            handle.write(json.dumps({"shot_id": shot}) + "\n")


def slice_cache(src_npz: Path, rows: list[dict], dst_npz: Path) -> None:
    with np.load(src_npz, allow_pickle=False) as data:
        cached_ids = [str(value) for value in data["sample_ids"].tolist()]
        index = {sample_id: i for i, sample_id in enumerate(cached_ids)}
        order = [index[str(row["sample_id"])] for row in rows]
        features = np.asarray(data["features"], dtype=np.float32)[order]
        psi = np.asarray(data["psi"], dtype=np.float32)[order]
        sample_ids = np.asarray([str(row["sample_id"]) for row in rows])
    np.savez_compressed(dst_npz, features=features, psi=psi, sample_ids=sample_ids)
    print(f"  cache: {dst_npz.name} ({len(rows)} rows)")


def concat_caches(parts: list[tuple[Path, list[dict]]], dst_npz: Path) -> None:
    features: list[np.ndarray] = []
    psi: list[np.ndarray] = []
    sample_ids: list[str] = []
    for cache_path, rows in parts:
        with np.load(cache_path, allow_pickle=False) as data:
            cached_ids = [str(value) for value in data["sample_ids"].tolist()]
            index = {sample_id: i for i, sample_id in enumerate(cached_ids)}
            order = [index[str(row["sample_id"])] for row in rows]
            features.append(np.asarray(data["features"], dtype=np.float32)[order])
            psi.append(np.asarray(data["psi"], dtype=np.float32)[order])
        sample_ids.extend(str(row["sample_id"]) for row in rows)
    np.savez_compressed(
        dst_npz,
        features=np.concatenate(features, axis=0),
        psi=np.concatenate(psi, axis=0),
        sample_ids=np.asarray(sample_ids),
    )
    print(f"  cache: {dst_npz.name} ({len(sample_ids)} rows)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--real-manifest", type=Path, default=None)
    parser.add_argument("--synth-manifest", type=Path, default=None)
    parser.add_argument("--clean-pool", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    out_dir = (args.out_dir or WORKSPACE_ROOT / "data" / "manifests" / "training_v2_statwidth").resolve()
    cache_dir = (args.cache_dir or WORKSPACE_ROOT / "data" / "processed" / "training_cache_v2_statwidth").resolve()
    real_manifest = args.real_manifest or out_dir / "tokamark_statwidth_k4_real.jsonl"
    synth_manifest = args.synth_manifest or out_dir / "tokamark_statwidth_k4_synthetic_accepted.jsonl"
    clean_pool = args.clean_pool or WORKSPACE_ROOT / "data" / "processed" / "real" / "clean_pool" / "clean_pool.csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    real_rows = load_rows(real_manifest)
    synth_rows = load_rows(synth_manifest)

    shots = list({str(row["shot_id"]) for row in real_rows})
    if len(shots) != 9995:
        print(f"WARNING: real manifest covers {len(shots)} shots (expected 9995)", flush=True)

    campaign: dict[str, str] = {}
    with clean_pool.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            campaign.setdefault(str(row["shot_id"]), row["campaign"])
    missing = [s for s in shots if s not in campaign]
    if missing:
        print(f"WARNING: {len(missing)} shots without campaign (default Unknown)", flush=True)
    for shot in missing:
        campaign[shot] = "Unknown"

    per_campaign: dict[str, list[str]] = {}
    for shot in shots:
        per_campaign.setdefault(campaign[shot], []).append(shot)

    rng = random.Random(args.seed)
    train_shots: list[str] = []
    val_shots: list[str] = []
    test_shots: list[str] = []
    summary: dict[str, dict[str, int]] = {}
    for camp in sorted(per_campaign):
        bucket = list(per_campaign[camp])
        rng.shuffle(bucket)
        n_train = int(len(bucket) * 0.8)
        n_val = round(len(bucket) * 0.1)
        train_shots.extend(bucket[:n_train])
        val_shots.extend(bucket[n_train:n_train + n_val])
        test_shots.extend(bucket[n_train + n_val:])
        summary[camp] = {"pool": len(bucket), "train": n_train, "val": n_val, "test": len(bucket) - n_train - n_val}

    train_set = set(train_shots)
    val_set = set(val_shots)
    test_set = set(test_shots)
    assert len(train_set) == len(train_shots) and len(val_set) == len(val_shots) and len(test_set) == len(test_shots)
    assert not (train_set & val_set) and not (train_set & test_set) and not (val_set & test_set)

    print("split summary (campaign stratified):")
    for camp, counts in summary.items():
        print(f"  {camp:8s} {counts}")
    print(f"  total: {len(shots)} shots -> train {len(train_shots)} / val {len(val_shots)} / test {len(test_shots)}")

    write_shots(out_dir / "split_train_shots.jsonl", train_shots)
    write_shots(out_dir / "split_val_shots.jsonl", val_shots)
    write_shots(out_dir / "split_test_shots.jsonl", test_shots)

    def by_shot(rows: list[dict], shot_key: str, allowed: set[str]) -> list[dict]:
        return [row for row in rows if str(row.get(shot_key)) in allowed]

    train_real = by_shot(real_rows, "shot_id", train_set)
    val_real = by_shot(real_rows, "shot_id", val_set)
    test_real = by_shot(real_rows, "shot_id", test_set)
    write_rows(out_dir / "split_train_real.jsonl", train_real)
    write_rows(out_dir / "split_val_real.jsonl", val_real)
    write_rows(out_dir / "split_test_real.jsonl", test_real)
    print(f"real rows: train {len(train_real)} / val {len(val_real)} / test {len(test_real)}")

    synth_train = by_shot(synth_rows, "parent_shot", train_set)
    leftover = [row for row in synth_rows if str(row.get("parent_shot")) not in train_set]
    print(f"synth rows: train {len(synth_train)} / excluded (val+test shots) {len(leftover)}")

    def inject_diagnostics(rows: list[dict], name: str) -> list[dict]:
        out: list[dict] = []
        for row in rows:
            row = dict(row)
            row["diagnostics_path"] = str(Path(row["data_path"]).expanduser().resolve() / name)
            out.append(row)
        return out

    synth_train_clean = inject_diagnostics(synth_train, "diagnostics.npz")
    synth_train_noisy = inject_diagnostics(synth_train, "diagnostics_noisy.npz")
    write_rows(out_dir / "split_train_synth_clean.jsonl", synth_train_clean)
    write_rows(out_dir / "split_train_synth_noisy.jsonl", synth_train_noisy)

    assert {str(r["sample_id"]) for r in train_real} & {str(r["sample_id"]) for r in val_real} == set()
    assert {str(r["sample_id"]) for r in train_real} & {str(r["sample_id"]) for r in test_real} == set()
    synth_sample_ids = {str(r["sample_id"]) for r in synth_train_clean}
    assert len(synth_sample_ids) == len(synth_train_clean)

    real_npz = cache_dir / "real.npz"
    synth_clean_npz = cache_dir / "synthetic_clean.npz"
    synth_noisy_npz = cache_dir / "synthetic_noisy.npz"
    for name, rows, src in [
        ("train_real.npz", train_real, real_npz),
        ("val_real.npz", val_real, real_npz),
        ("test_real.npz", test_real, real_npz),
        ("train_synth_clean.npz", synth_train_clean, synth_clean_npz),
        ("train_synth_noisy.npz", synth_train_noisy, synth_noisy_npz),
    ]:
        slice_cache(src, rows, cache_dir / name)

    print("concat run caches:")
    concat_caches([(real_npz, train_real)], cache_dir / "run_real.npz")
    concat_caches([(synth_clean_npz, synth_train_clean)], cache_dir / "run_synth_clean.npz")
    concat_caches([(synth_noisy_npz, synth_train_noisy)], cache_dir / "run_synth_noisy.npz")
    concat_caches([(real_npz, train_real), (synth_clean_npz, synth_train_clean)], cache_dir / "mixed_clean.npz")
    concat_caches([(real_npz, train_real), (synth_noisy_npz, synth_train_noisy)], cache_dir / "mixed_noisy.npz")
    concat_caches([(real_npz, train_real), (real_npz, val_real)], cache_dir / "run_real_with_val.npz")
    concat_caches([(synth_clean_npz, synth_train_clean), (real_npz, val_real)], cache_dir / "run_synth_clean_with_val.npz")
    concat_caches([(synth_noisy_npz, synth_train_noisy), (real_npz, val_real)], cache_dir / "run_synth_noisy_with_val.npz")
    concat_caches(
        [(real_npz, train_real), (synth_clean_npz, synth_train_clean), (real_npz, val_real)],
        cache_dir / "run_mixed_clean_with_val.npz",
    )
    concat_caches(
        [(real_npz, train_real), (synth_noisy_npz, synth_train_noisy), (real_npz, val_real)],
        cache_dir / "run_mixed_noisy_with_val.npz",
    )

    print("run manifests (train+val for training script):")
    write_rows(out_dir / "run_real.jsonl", train_real + val_real)
    write_rows(out_dir / "run_synth_clean.jsonl", synth_train_clean + val_real)
    write_rows(out_dir / "run_synth_noisy.jsonl", synth_train_noisy + val_real)
    write_rows(out_dir / "run_mixed_clean.jsonl", train_real + synth_train_clean + val_real)
    write_rows(out_dir / "run_mixed_noisy.jsonl", train_real + synth_train_noisy + val_real)
    for name in ["run_real", "run_synth_clean", "run_synth_noisy", "run_mixed_clean", "run_mixed_noisy"]:
        rows = load_rows(out_dir / f"{name}.jsonl")
        print(f"  {name}.jsonl: {len(rows)} rows")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
