#!/usr/bin/env python
"""Build a real-data manifest for the (shot, time) points of a variant CSV.

Each row references the downloaded MAST zarr as data path and the real
EFIT equilibrium psi (label_source=zarr_equilibrium_psi) as label, so the
real manifest pairs 1:1 with the synthetic samples of the same CSV.

Usage:
  build_real_manifest.py --variant-csv FILE --output FILE
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parents[0]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

sys.path.insert(0, str(SCRIPT_ROOT.parents[0] / "src"))

from mast_bridge.workspace import discover_workspace  # noqa: E402

FIT_PATH = (
    Path("/inspire/qb-ilm/project/ai-for-fusion/public/fusion-workspace")
    / "data" / "processed" / "real" / "lao_parameter_ensemble" / "all_zarr_lao_parameter_fits.npz"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build real manifest from a variant CSV.")
    parser.add_argument("--variant-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", default="task_1-3")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    workspace_root = SCRIPT_ROOT.parents[1]
    data_root = workspace_root / "data"

    with args.variant_csv.expanduser().resolve().open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    manifest: list[dict] = []
    for row in rows:
        shot = row["shot"]
        target_time = float(row["target_time"])
        manifest.append(
            {
                "comparison_group": "real_only",
                "data_path": str(data_root / "raw" / "mast" / f"{shot}.zarr"),
                "equilibrium_path": None,
                "fit_path": str(FIT_PATH),
                "label_path": None,
                "label_source": "zarr_equilibrium_psi",
                "machine_config_path": str(data_root / "raw" / "mast" / "machine" / shot),
                "parent_shot": None,
                "profile_parameter_source": "lao_fit_npz",
                "sample_id": f"{shot}_t{target_time:g}_real",
                "shot_id": shot,
                "solver_status": None,
                "source": "real",
                "target_time": target_time,
                "task": args.task,
            }
        )

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for item in manifest:
            handle.write(json.dumps(item, sort_keys=True) + "\n")
    print(f"real manifest: {len(manifest)} rows -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
