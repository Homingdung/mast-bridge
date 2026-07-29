#!/usr/bin/env python
"""Write original uniform-random Lao85 perturbation rows from a fitted Lao NPZ."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mast_bridge.simulation.variants import rows_from_lao_fit_npz  # noqa: E402


FIELDNAMES = [
    "shot",
    "target_time",
    "variant_id",
    "sampling_method",
    "ip_scale",
    "fvac_scale",
    "alpha_scale",
    "beta_scale",
    "alpha_offset",
    "beta_offset",
    "coil_current_scale",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create original uniform-random Lao85 perturbation CSV rows."
    )
    parser.add_argument("--fit-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variants-per-point", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument(
        "--min-time",
        type=float,
        default=None,
        help="Keep only fitted rows with time >= this value.",
    )
    parser.add_argument(
        "--max-time",
        type=float,
        default=None,
        help="Keep only fitted rows with time <= this value.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = rows_from_lao_fit_npz(
        args.fit_path,
        variants_per_point=args.variants_per_point,
        seed=args.seed,
        min_time=args.min_time,
        max_time=args.max_time,
    )

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"variant_rows: {len(rows)} -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
