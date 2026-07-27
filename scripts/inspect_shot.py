#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_ROOT / "src"))

from mast_bridge.mast.reader import ShotReader
from mast_bridge.workspace import discover_workspace


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect one local MAST shot and its machine geometry.")
    parser.add_argument("--shot", required=True, help="Shot ID, for example 11766.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Directory containing <shot>.zarr.")
    parser.add_argument("--machine-dir", type=Path, default=None, help="Optional directory containing machine pickles.")
    args = parser.parse_args(argv)

    layout = discover_workspace(SCRIPT_ROOT)
    data_dir = args.data_dir or layout.data_root / "raw" / "mast"
    record = ShotReader(data_dir, machine_dir=args.machine_dir).read(args.shot)
    print(f"shot_id: {record.shot_id}")
    print(f"zarr_path: {record.zarr_path}")
    print("machine_files:")
    for key, path in record.machine.files.items():
        print(f"  {key}: {path}")
    print(f"signal_groups: {', '.join(sorted(record.signals)) or '(not opened)'}")
    print(f"equilibrium_groups: {', '.join(sorted(record.equilibrium)) or '(not found)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
