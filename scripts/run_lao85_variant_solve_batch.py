#!/usr/bin/env python3
"""Run FreeGSNKE forward solves for sampled Lao85 variants, then strict-filter them."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import build_synthetic_manifest  # noqa: E402


VARIANT_FIELDNAMES = [
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


def sample_name(row: dict[str, str]) -> str:
    time_label = f"{float(row['target_time']):g}"
    return f"{row['shot']}_t{time_label}_{row['variant_id']}"


def read_variant_rows(path: Path) -> list[dict[str, str]]:
    with path.expanduser().resolve().open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    missing = set(VARIANT_FIELDNAMES) - set(rows[0].keys() if rows else [])
    if missing:
        raise ValueError(f"Variant CSV is missing required columns: {sorted(missing)}")
    return rows


def build_forward_command(
    row: dict[str, str],
    *,
    python_executable: Path,
    data_dir: Path,
    fit_path: Path,
    synthetic_root: Path,
    nx: int,
    ny: int,
    tolerance: float,
    max_iterations: int,
) -> list[str]:
    output_dir = synthetic_root.expanduser().resolve() / sample_name(row)
    machine_dir = data_dir.expanduser().resolve() / "machine" / row["shot"]
    return [
        str(python_executable),
        str(REPO_ROOT / "scripts" / "run_freegsnke_forward.py"),
        "--data-dir",
        str(data_dir.expanduser().resolve()),
        "--machine-dir",
        str(machine_dir),
        "--fit-path",
        str(fit_path.expanduser().resolve()),
        "--shot",
        row["shot"],
        "--time",
        str(float(row["target_time"])),
        "--nx",
        str(nx),
        "--ny",
        str(ny),
        "--tolerance",
        f"{tolerance:.17g}",
        "--max-iterations",
        str(max_iterations),
        "--ip-scale",
        row["ip_scale"],
        "--fvac-scale",
        row["fvac_scale"],
        "--alpha-scale",
        row["alpha_scale"],
        "--beta-scale",
        row["beta_scale"],
        f"--alpha-offset={row['alpha_offset']}",
        f"--beta-offset={row['beta_offset']}",
        "--coil-current-scale",
        row["coil_current_scale"],
        "--output-dir",
        str(output_dir),
    ]


def run_subprocess(command: list[str]) -> int:
    return subprocess.run(command, check=False).returncode


def enrich_metadata(output_dir: Path, row: dict[str, str], command: list[str]) -> None:
    metadata_path = output_dir / "metadata.json"
    if not metadata_path.is_file():
        return
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "variant_id": row["variant_id"],
            "sampling_method": row["sampling_method"],
            "variant_row": {key: row.get(key) for key in VARIANT_FIELDNAMES},
            "batch_forward_command": command,
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def validate_existing_output(
    output_dir: Path,
    row: dict[str, str],
    *,
    nx: int,
    ny: int,
    tolerance: float,
    max_iterations: int,
) -> None:
    metadata_path = output_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_text = {
        "parent_shot": row["shot"],
        "variant_id": row["variant_id"],
        "sampling_method": row["sampling_method"],
    }
    for key, expected in expected_text.items():
        if str(metadata.get(key)) != str(expected):
            raise ValueError(f"existing sample {output_dir.name} has mismatched {key}")

    variant_row = metadata.get("variant_row")
    if not isinstance(variant_row, dict):
        raise ValueError(f"existing sample {output_dir.name} has no variant_row identity")
    numeric_fields = (
        "target_time",
        "ip_scale",
        "fvac_scale",
        "alpha_scale",
        "beta_scale",
        "alpha_offset",
        "beta_offset",
        "coil_current_scale",
    )
    for key in numeric_fields:
        try:
            actual = float(variant_row[key])
            expected = float(row[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"existing sample {output_dir.name} has invalid {key}"
            ) from exc
        if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError(f"existing sample {output_dir.name} has mismatched {key}")

    grid = metadata.get("grid", {})
    checks = {
        "nx": (grid.get("nx"), nx),
        "ny": (grid.get("ny"), ny),
        "solver_requested_tolerance": (
            metadata.get("solver_requested_tolerance"),
            tolerance,
        ),
        "solver_max_iterations": (
            metadata.get("solver_max_iterations"),
            max_iterations,
        ),
    }
    for key, (actual, expected) in checks.items():
        if actual is None or not math.isclose(
            float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-15
        ):
            raise ValueError(f"existing sample {output_dir.name} has mismatched {key}")


def write_batch_report(path: Path, rows: list[dict[str, str]]) -> None:
    output = path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def clear_generated_sample_outputs(output_dir: Path) -> None:
    for filename in (
        "equilibrium.npz",
        "metadata.json",
        "diagnostics.npz",
        "equilibrium.png",
    ):
        (output_dir / filename).unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run sampled Lao85 FreeGSNKE forward solves and strict-filter outputs."
    )
    parser.add_argument("--variant-csv", type=Path, required=True)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=WORKSPACE_ROOT / "data" / "raw" / "mast",
    )
    parser.add_argument(
        "--fit-path",
        type=Path,
        default=WORKSPACE_ROOT
        / "data"
        / "processed"
        / "real"
        / "lao_parameter_ensemble"
        / "all_zarr_lao_parameter_fits.npz",
    )
    parser.add_argument(
        "--synthetic-root",
        type=Path,
        default=WORKSPACE_ROOT / "data" / "processed" / "synthetic_lao85_uniform",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=WORKSPACE_ROOT / "data" / "manifests",
    )
    parser.add_argument("--prefix", default="tokamark_lao85_uniform")
    parser.add_argument("--task", default="task_1-3")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--nx", type=int, default=65)
    parser.add_argument("--ny", type=int, default=65)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--max-solver-tolerance", type=float, default=1e-8)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rerun-existing", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Only run solves; do not build accepted/rejected manifests.",
    )
    parser.add_argument(
        "--batch-report",
        type=Path,
        default=None,
        help="JSONL status report; default is <manifest-dir>/<prefix>_batch_report.jsonl.",
    )
    return parser


def main(
    argv: list[str] | None = None,
    *,
    run_command: Callable[[list[str]], int] = run_subprocess,
) -> int:
    args = build_parser().parse_args(argv)
    rows = read_variant_rows(args.variant_csv)
    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative")
    if args.limit is not None and args.limit < 0:
        raise ValueError("--limit must be non-negative")

    selected = rows[args.start_index :]
    if args.limit is not None:
        selected = selected[: args.limit]

    synthetic_root = args.synthetic_root.expanduser().resolve()
    manifest_dir = args.manifest_dir.expanduser().resolve()
    report_path = args.batch_report or manifest_dir / f"{args.prefix}_batch_report.jsonl"
    report_rows: list[dict[str, str]] = []

    for row_index, row in enumerate(selected, start=args.start_index):
        output_dir = synthetic_root / sample_name(row)
        sample_id = sample_name(row)
        command = build_forward_command(
            row,
            python_executable=args.python,
            data_dir=args.data_dir,
            fit_path=args.fit_path,
            synthetic_root=synthetic_root,
            nx=args.nx,
            ny=args.ny,
            tolerance=args.tolerance,
            max_iterations=args.max_iterations,
        )

        if args.dry_run:
            print(f"[dry-run] row={row_index} sample={sample_id}", flush=True)
            print(" ".join(command), flush=True)
            report_rows.append(
                {**row, "row_index": str(row_index), "sample_id": sample_id, "batch_status": "dry_run"}
            )
            write_batch_report(report_path, report_rows)
            continue

        if (
            not args.rerun_existing
            and (output_dir / "equilibrium.npz").is_file()
            and (output_dir / "metadata.json").is_file()
        ):
            validate_existing_output(
                output_dir,
                row,
                nx=args.nx,
                ny=args.ny,
                tolerance=args.tolerance,
                max_iterations=args.max_iterations,
            )
            print(f"[skip] row={row_index} sample={sample_id} existing output", flush=True)
            report_rows.append(
                {
                    **row,
                    "row_index": str(row_index),
                    "sample_id": sample_id,
                    "output_dir": str(output_dir),
                    "batch_status": "solved",
                    "return_code": "0",
                }
            )
            write_batch_report(report_path, report_rows)
            continue
        if args.rerun_existing:
            clear_generated_sample_outputs(output_dir)

        print(
            f"[solve-start] row={row_index} sample={sample_id} "
            f"shot={row['shot']} time={row['target_time']} variant={row['variant_id']}",
            flush=True,
        )
        return_code = run_command(command)
        if return_code == 0:
            enrich_metadata(output_dir, row, command)
            status = "solved"
        else:
            status = "failed"
        print(
            f"[solve-{status}] row={row_index} sample={sample_id} return_code={return_code}",
            flush=True,
        )
        report_rows.append(
            {
                **row,
                "row_index": str(row_index),
                "sample_id": sample_id,
                "output_dir": str(output_dir),
                "batch_status": status,
                "return_code": str(return_code),
            }
        )
        write_batch_report(report_path, report_rows)
        if return_code != 0 and args.fail_fast:
            return return_code

    write_batch_report(report_path, report_rows)

    if not args.no_filter and not args.dry_run:
        manifest_dir.mkdir(parents=True, exist_ok=True)
        build_synthetic_manifest.main(
            [
                "--synthetic-root",
                str(synthetic_root),
                "--variant-csv",
                str(args.variant_csv.expanduser().resolve()),
                "--output",
                str(manifest_dir / f"{args.prefix}_synthetic_accepted.jsonl"),
                "--rejected-output",
                str(manifest_dir / f"{args.prefix}_synthetic_rejected.jsonl"),
                "--task",
                args.task,
                "--max-solver-tolerance",
                f"{args.max_solver_tolerance:.17g}",
            ]
        )

    print(f"variant_rows_total: {len(rows)}")
    print(f"variant_rows_selected: {len(selected)}")
    print(f"synthetic_root: {synthetic_root}")
    print(f"batch_report: {report_path.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
