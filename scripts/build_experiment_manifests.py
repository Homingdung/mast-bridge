#!/usr/bin/env python3
"""Build small comparison manifests for real, synthetic, and mixed experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_ROOT / "src"))

from mast_bridge.dataset.manifest import ManifestEntry, write_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create real-only, synthetic-only, and mixed experiment manifests."
    )
    parser.add_argument("--accepted-synthetic", type=Path, required=True)
    parser.add_argument("--raw-data-dir", type=Path, required=True)
    parser.add_argument("--fit-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="small")
    parser.add_argument("--task", default="task_1-3")
    return parser


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.expanduser().resolve().open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _time_label(value: Any) -> str:
    return f"{float(value):g}"


def build_real_entries(
    synthetic_rows: list[dict[str, Any]],
    raw_data_dir: Path,
    fit_path: Path,
    task: str,
    comparison_group: str,
) -> list[ManifestEntry]:
    roots = raw_data_dir.expanduser().resolve()
    fits = fit_path.expanduser().resolve()
    unique_points = sorted(
        {
            (str(row["parent_shot"]), float(row["target_time"]))
            for row in synthetic_rows
            if row.get("parent_shot") and row.get("target_time") is not None
        }
    )

    entries: list[ManifestEntry] = []
    for shot, target_time in unique_points:
        zarr_path = roots / f"{shot}.zarr"
        machine_path = roots / "machine" / shot
        if not zarr_path.is_dir() or not machine_path.is_dir():
            continue
        time_label = _time_label(target_time)
        entries.append(
            ManifestEntry(
                sample_id=f"{shot}_t{time_label}_real",
                source="real",
                shot_id=shot,
                data_path=zarr_path,
                machine_config_path=machine_path,
                metadata={
                    "target_time": target_time,
                    "task": task,
                    "comparison_group": comparison_group,
                    "label_source": "zarr_equilibrium_psi",
                    "fit_path": str(fits),
                    "profile_parameter_source": "lao_fit_npz",
                },
            )
        )
    return entries


def build_synthetic_entries(
    synthetic_rows: list[dict[str, Any]],
    comparison_group: str,
) -> list[ManifestEntry]:
    entries: list[ManifestEntry] = []
    for row in synthetic_rows:
        metadata = dict(row)
        metadata["comparison_group"] = comparison_group
        entries.append(
            ManifestEntry(
                sample_id=str(row["sample_id"]),
                source="synthetic",
                shot_id=str(row.get("shot_id", row["sample_id"])),
                data_path=Path(str(row["data_path"])),
                equilibrium_path=Path(str(row["equilibrium_path"])),
                parent_shot=str(row["parent_shot"]),
                solver_status=row.get("solver_status"),
                metadata=metadata,
            )
        )
    return entries


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    synthetic_rows = _read_jsonl(args.accepted_synthetic)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    real_only = build_real_entries(
        synthetic_rows,
        args.raw_data_dir,
        args.fit_path,
        args.task,
        "real_only",
    )
    synthetic_only = build_synthetic_entries(synthetic_rows, "synthetic_only")
    mixed = build_real_entries(
        synthetic_rows,
        args.raw_data_dir,
        args.fit_path,
        args.task,
        "real_plus_synthetic",
    ) + build_synthetic_entries(synthetic_rows, "real_plus_synthetic")

    real_path = output_dir / f"{args.prefix}_real_only.jsonl"
    synthetic_path = output_dir / f"{args.prefix}_synthetic_only.jsonl"
    mixed_path = output_dir / f"{args.prefix}_real_plus_synthetic.jsonl"

    write_manifest(real_only, real_path)
    write_manifest(synthetic_only, synthetic_path)
    write_manifest(mixed, mixed_path)

    print(f"real_only: {len(real_only)} -> {real_path}")
    print(f"synthetic_only: {len(synthetic_only)} -> {synthetic_path}")
    print(f"real_plus_synthetic: {len(mixed)} -> {mixed_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
