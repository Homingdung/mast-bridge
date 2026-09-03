#!/usr/bin/env python
"""Parallel passive-current precheck for variant rows.

Groups rows by shot so each shot zarr is opened once, then checks every
target time of the shot against the pf_passive current arrays. Drops rows
whose passive currents are not finite at their target time.

Usage:
  precheck_variant_rows.py --variant-csv in.csv --output out.csv [--workers 32]
"""

from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import zarr

SCRIPT_ROOT = Path(__file__).resolve().parents[0]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from build_production_variants import passive_current_is_finite  # noqa: E402


def check_shot(args: tuple[Path, str, list[float]]) -> tuple[str, list[float]]:
    data_dir, shot, times = args
    finite = [t for t in times if passive_current_is_finite(data_dir / f"{shot}.zarr", t)]
    return shot, finite


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Parallel passive-current precheck for variant rows.")
    parser.add_argument("--variant-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=SCRIPT_ROOT.parents[1] / "data" / "raw" / "mast")
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args(argv)

    data_dir = args.data_dir.expanduser().resolve()
    with args.variant_csv.expanduser().resolve().open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)

    by_shot: dict[str, set[float]] = {}
    for row in rows:
        by_shot.setdefault(row["shot"], set()).add(float(row["target_time"]))

    tasks = [(data_dir, shot, sorted(times)) for shot, times in by_shot.items()]
    kept: dict[str, set[float]] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, (shot, finite) in enumerate(pool.map(check_shot, tasks, chunksize=8)):
            kept[shot] = set(finite)
            if (i + 1) % 250 == 0 or i + 1 == len(tasks):
                print(f"[precheck] {i + 1}/{len(tasks)} shots checked", flush=True)

    kept_rows = [r for r in rows if float(r["target_time"]) in kept.get(r["shot"], set())]
    dropped = len(rows) - len(kept_rows)
    args.output.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    with args.output.expanduser().resolve().open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept_rows)
    print(f"[precheck] rows {len(rows)} -> {len(kept_rows)} (dropped {dropped}, {100 * dropped / len(rows):.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
