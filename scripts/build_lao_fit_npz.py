#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = SCRIPT_ROOT.parent
sys.path.insert(0, str(SCRIPT_ROOT / "src"))

from mast_bridge.equilibrium.lao_from_zarr import (  # noqa: E402
    default_lao_fit_path,
    write_lao_fit_npz,
)
from mast_bridge.workspace import discover_workspace  # noqa: E402


def _read_shot_list(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a FreeGSNKE Lao85 fit NPZ from downloaded MAST Level 2 Zarr shots."
    )
    parser.add_argument(
        "--shot-list",
        type=Path,
        required=True,
        help="Text file containing one downloaded shot ID per line.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing <shot>.zarr; default is workspace/data/raw/mast.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output NPZ; default is data/processed/real/lao_parameter_ensemble/all_zarr_lao_parameter_fits.npz.",
    )
    parser.add_argument("--n-alpha", type=int, default=3)
    parser.add_argument("--n-beta", type=int, default=3)
    parser.add_argument("--min-finite-points", type=int, default=8)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    layout = discover_workspace(SCRIPT_ROOT)
    data_dir = (args.data_dir or layout.data_root / "raw" / "mast").expanduser().resolve()
    output = (args.output or default_lao_fit_path(WORKSPACE_ROOT)).expanduser().resolve()
    shots = _read_shot_list(args.shot_list.expanduser().resolve())
    zarr_paths = [data_dir / f"{shot}.zarr" for shot in shots]
    table = write_lao_fit_npz(
        zarr_paths,
        output,
        n_alpha=args.n_alpha,
        n_beta=args.n_beta,
        min_finite_points=args.min_finite_points,
    )
    print(f"output: {output}")
    print(f"rows: {len(table['shot'])}")
    print(f"shots: {len(set(table['shot'].tolist()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
