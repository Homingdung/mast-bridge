#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = SCRIPT_ROOT.parent
sys.path.insert(0, str(SCRIPT_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mast_bridge.workspace import discover_workspace  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Regenerate clean flat-top pool and per-shot statistics for paper figures."
    )
    parser.add_argument("--fit-path", type=Path, default=None)
    parser.add_argument("--metadata-parquet", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--t-window", type=float, nargs=2, default=[0.12, 0.24])
    parser.add_argument("--ip-fraction", type=float, default=0.85)
    parser.add_argument("--plateau-quantile", type=float, default=0.95)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    layout = discover_workspace(SCRIPT_ROOT)
    data_root = layout.data_root / "raw" / "mast"
    fit_path = args.fit_path or (
        layout.data_root / "processed" / "real" / "lao_parameter_ensemble"
        / "all_zarr_lao_parameter_fits.npz"
    )
    metadata_path = args.metadata_parquet or (data_root / ".shots_metadata.parquet")
    output_dir = (
        args.output_dir
        or layout.data_root / "processed" / "real" / "clean_pool"
    ).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    t_lo, t_hi = args.t_window

    fit = np.load(fit_path)
    df = pd.DataFrame({"shot": fit["shot"], "time": fit["time"], "ip": fit["ip"]})
    meta = pd.read_parquet(metadata_path)[
        ["shot_id", "campaign", "plasma_flat_top_start_time", "plasma_flat_top_end_time"]
    ].astype({"shot_id": str})
    meta = meta.dropna(subset=["plasma_flat_top_start_time", "plasma_flat_top_end_time"])
    df = df.merge(meta, left_on="shot", right_on="shot_id", how="inner")

    pos = df[df["ip"] > 0].groupby("shot")["ip"]
    plateau = pos.quantile(args.plateau_quantile).rename("ip_plateau")
    df = df.merge(plateau, on="shot", how="left")
    df["ip_fraction"] = df["ip"] / df["ip_plateau"]

    df["in_window"] = (df["time"] >= t_lo) & (df["time"] <= t_hi)
    df["in_flat_top"] = (df["time"] >= df["plasma_flat_top_start_time"]) & (
        df["time"] <= df["plasma_flat_top_end_time"]
    )
    df["ip_ok"] = df["ip_fraction"] >= args.ip_fraction
    df["clean"] = df["in_window"] & df["in_flat_top"] & df["ip_ok"]

    per_shot = (
        df[df["ip_ok"]]
        .groupby("shot")
        .agg(onset=("time", "min"), end=("time", "max"), n_high_ip=("time", "size"))
    )
    clean_counts = df[df["clean"]].groupby("shot").size().rename("n_clean")
    per_shot = per_shot.merge(clean_counts, left_index=True, right_index=True, how="left")
    per_shot["n_clean"] = per_shot["n_clean"].fillna(0).astype(int)
    per_shot = per_shot.reset_index().merge(
        meta[["shot_id", "campaign"]].astype({"shot_id": str}), left_on="shot", right_on="shot_id"
    )

    n_total = len(df)
    win = df[df["in_window"]]
    funnel = {
        "fit_rows_total": int(n_total),
        "fit_shots": int(df["shot"].nunique()),
        "in_window_rows": int(df["in_window"].sum()),
        "in_flat_top_rows": int(df["in_flat_top"].sum()),
        "ip_ok_rows": int(df["ip_ok"].sum()),
        "clean_rows": int(df["clean"].sum()),
        "clean_shots": int(clean_counts.notna().sum()),
        "window_conditioned": {
            "in_window_rows": int(len(win)),
            "plus_in_flat_top": int(win["in_flat_top"].sum()),
            "plus_ip_ok": int(win["ip_ok"].sum()),
            "clean": int(win["clean"].sum()),
        },
        "t_window": [t_lo, t_hi],
        "ip_fraction": args.ip_fraction,
        "plateau_quantile": args.plateau_quantile,
        "window_onset_end_percentiles": {
            "onset": [float(x) for x in np.percentile(per_shot["onset"], [10, 25, 50, 75, 90])],
            "end": [float(x) for x in np.percentile(per_shot["end"], [10, 25, 50, 75, 90])],
        },
        "pct_onset_le_t_lo": float((per_shot["onset"] <= t_lo).mean()),
        "pct_end_ge_t_hi": float((per_shot["end"] >= t_hi).mean()),
        "pct_full_window": float(((per_shot["onset"] <= t_lo) & (per_shot["end"] >= t_hi)).mean()),
    }

    df.to_csv(output_dir / "clean_pool.csv", index=False)
    per_shot.to_csv(output_dir / "per_shot_flat_top_stats.csv", index=False)
    with (output_dir / "funnel_counts.json").open("w", encoding="utf-8") as f:
        json.dump(funnel, f, indent=2)

    print(f"output dir: {output_dir}")
    print("funnel:", json.dumps(funnel, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
