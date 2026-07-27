#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_ROOT / "src"))

from mast_bridge.mast.machine_from_zarr import write_machine_pickles
from mast_bridge.workspace import discover_workspace


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build FreeGSNKE-compatible machine pickles from one shot Zarr."
    )
    parser.add_argument("--shot", required=True, help="Shot ID, for example 11766.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Directory containing <shot>.zarr.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for machine pickles; default is workspace/data/raw/mast/machine.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing machine pickles.",
    )
    args = parser.parse_args(argv)

    layout = discover_workspace(SCRIPT_ROOT)
    data_dir = (args.data_dir or layout.data_root / "raw" / "mast").expanduser().resolve()
    output_dir = (
        args.output_dir
        or layout.data_root / "raw" / "mast" / "machine" / str(args.shot)
    ).expanduser().resolve()
    paths = write_machine_pickles(
        data_dir / f"{args.shot}.zarr",
        output_dir,
        overwrite=args.overwrite,
    )
    print(f"shot_zarr: {data_dir / f'{args.shot}.zarr'}")
    print(f"machine_dir: {output_dir}")
    for key, path in paths.items():
        print(f"{key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
