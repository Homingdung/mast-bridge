#!/usr/bin/env python3
"""Slice a large cache npz by a run manifest (keeps manifest row order).

Usage:
  slice_cache.py --src real_full.npz --manifest subset_run_real_10pct.jsonl \
                 --out subset_run_real_10pct.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    rows = [json.loads(line) for line in args.manifest.open(encoding="utf-8") if line.strip()]
    with np.load(args.src, allow_pickle=False) as data:
        cached_ids = [str(v) for v in data["sample_ids"].tolist()]
        index = {sid: i for i, sid in enumerate(cached_ids)}
        missing = [str(r.get("sample_id")) for r in rows if str(r.get("sample_id")) not in index]
        if missing:
            print(f"WARNING: {len(missing)} rows missing from src cache, first: {missing[:5]}")
        order = [index[str(r.get("sample_id"))] for r in rows if str(r.get("sample_id")) in index]
        features = np.asarray(data["features"], dtype=np.float32)[order]
        psi = np.asarray(data["psi"], dtype=np.float32)[order]
        sample_ids = np.asarray([str(r.get("sample_id")) for r in rows if str(r.get("sample_id")) in index])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, features=features, psi=psi, sample_ids=sample_ids)
    print(f"slice: {args.out} ({len(order)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
