#!/usr/bin/env python3
"""Build data-scarcity experiment manifests and dataset caches.

Stage 1:
  - synthetic nested subsets M in {357, 715, 1430, 2860, 5720, 7157}
    from train_synthetic_noisy.jsonl (fixed-seed shuffle, prefix subsets)
  - real n=700 subset from train_real.jsonl (stratified by shot-id prefix)
Each manifest gets a sliced dataset cache derived from the full caches.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20260806)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    manifest_dir = root / "data" / "manifests" / "training"
    cache_dir = root / "data" / "processed" / "training_cache"
    out = (args.out_dir or (root / "data" / "manifests" / "scarcity")).resolve()
    cache_out = (root / "data" / "processed" / "scarcity_cache").resolve()
    out.mkdir(parents=True, exist_ok=True)
    cache_out.mkdir(parents=True, exist_ok=True)

    synthetic_sizes = [357, 715, 1430, 2860, 5720, 7157]

    # --- synthetic nested subsets (noisy, main arm) ---
    synth_rows = load_rows(manifest_dir / "train_synthetic_noisy.jsonl")
    if len(synth_rows) != max(synthetic_sizes):
        raise ValueError(f"synthetic pool size mismatch: {len(synth_rows)}")
    rng = random.Random(args.seed)
    rng.shuffle(synth_rows)
    print(f"synthetic pool: {len(synth_rows)} rows, nested subsets {synthetic_sizes}")
    for size in synthetic_sizes:
        subset = synth_rows[:size]
        base = out / f"synth_noisy_M{size}.jsonl"
        write_rows(base, subset)
        slice_cache(cache_dir / "synthetic_noisy.npz", subset, cache_out / f"synth_noisy_M{size}.npz")

    # --- real n=700 subset (stratified by shot-id prefix) ---
    real_rows = load_rows(manifest_dir / "train_real.jsonl")
    strata = Counter(row["shot_id"][:3] for row in real_rows)
    buckets: dict[str, list[dict]] = {}
    for row in real_rows:
        buckets.setdefault(row["shot_id"][:3], []).append(row)
    n_real = 700
    chosen: list[dict] = []
    for prefix, bucket in sorted(buckets.items()):
        share = max(0, round(n_real * len(bucket) / len(real_rows)))
        bucket_sorted = list(bucket)
        rng.shuffle(bucket_sorted)
        chosen.extend(bucket_sorted[:share])
    deficit = n_real - len(chosen)
    if deficit > 0:
        remaining = [row for row in real_rows if row not in chosen]
        rng.shuffle(remaining)
        chosen.extend(remaining[:deficit])
    if len(chosen) != n_real or len({row["shot_id"] for row in chosen}) != n_real:
        raise ValueError("real subset sampling failed")
    real_subset = out / "real_n700.jsonl"
    write_rows(real_subset, chosen)
    print(f"real subset: {len(chosen)} rows (stratified by shot prefix)")
    slice_cache(cache_dir / "real.npz", chosen, cache_out / "real_n700.npz")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
