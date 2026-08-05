#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_ROOT / "src"))

import numpy as np  # noqa: E402
import zarr  # noqa: E402

from mast_bridge.simulation.variants import build_variant_rows  # noqa: E402
from mast_bridge.workspace import discover_workspace  # noqa: E402


def passive_current_is_finite(shot_zarr: Path, target_time: float) -> bool:
    try:
        pg = zarr.open_group(str(shot_zarr), mode="r")["pf_passive"]
        times = np.asarray(pg["time"][:], dtype=float)
        for key in sorted(pg.array_keys()):
            if not key.endswith("_current"):
                continue
            values = np.asarray(pg[key][:], dtype=float)
            if values.ndim == 1:
                rows = values[None, :]
            elif values.ndim == 2 and values.shape[1] == times.size:
                rows = values
            else:
                continue
            for row in rows:
                if not np.isfinite(float(np.interp(target_time, times, row))):
                    return False
        return True
    except Exception:
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build production variant CSV from the clean flat-top pool with precheck."
    )
    parser.add_argument("--clean-pool", type=Path, required=True, help="clean_pool.csv from build_clean_pool_analysis.py.")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--points-per-shot", type=int, default=1)
    parser.add_argument("--variants-per-point", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument(
        "--criterion",
        choices=["ip", "full"],
        default="ip",
        help="'ip': in_window & ip_ok (Ip-based flat top); 'full': also require metadata flat-top window.",
    )
    parser.add_argument("--skip-precheck", action="store_true")
    args = parser.parse_args(argv)

    layout = discover_workspace(SCRIPT_ROOT)
    data_dir = (args.data_dir or layout.data_root / "raw" / "mast").expanduser().resolve()
    df = __import__("pandas").read_csv(args.clean_pool, dtype={"shot": str})
    if args.criterion == "ip":
        clean = df[df["in_window"] & df["ip_ok"]]
    else:
        clean = df[df["clean"]]

    points: list[tuple[str, float]] = []
    for shot, grp in clean.groupby("shot"):
        times = sorted(grp["time"].tolist())
        k = min(args.points_per_shot, len(times))
        if k == 1:
            selected = [times[len(times) // 2]]
        else:
            step = (len(times) - 1) / (k - 1)
            selected = [times[round(i * step)] for i in range(k)]
        points.extend((shot, float(t)) for t in selected)
    print(f"[variants] clean pool shots={clean['shot'].nunique()} points={len(points)}", flush=True)

    if not args.skip_precheck:
        bad = []
        good = []
        for shot, t in points:
            if passive_current_is_finite(data_dir / f"{shot}.zarr", t):
                good.append((shot, t))
            else:
                bad.append((shot, t))
        print(f"[variants] precheck dropped {len(bad)} points (non-finite passive current)", flush=True)
        points = good

    rows: list[dict] = []
    for index, (shot, t) in enumerate(sorted(points)):
        rows.extend(
            build_variant_rows(
                [shot], [t], args.variants_per_point, seed=args.seed + index
            )
        )
    columns = [
        "shot", "target_time", "variant_id", "sampling_method",
        "ip_scale", "fvac_scale", "alpha_scale", "beta_scale",
        "alpha_offset", "beta_offset", "coil_current_scale",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"[variants] wrote {len(rows)} variant rows -> {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
