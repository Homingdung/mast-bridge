#!/usr/bin/env python3
"""Build strict accepted/rejected manifests for synthetic FreeGSNKE samples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_ROOT / "src"))

from mast_bridge.dataset.manifest import write_manifest
from mast_bridge.dataset.synthetic_manifest import (
    STRICT_SOLVER_TOLERANCE,
    rejected_samples,
    synthetic_entries,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Filter synthetic FreeGSNKE samples into accepted and rejected manifests."
    )
    parser.add_argument(
        "--synthetic-root",
        type=Path,
        required=True,
        help="Directory containing one subdirectory per synthetic sample.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Accepted manifest JSONL output path.",
    )
    parser.add_argument(
        "--rejected-output",
        type=Path,
        required=True,
        help="Rejected sample report JSONL output path.",
    )
    parser.add_argument("--task", default=None, help="Optional task label to include.")
    parser.add_argument(
        "--max-solver-tolerance",
        type=float,
        default=STRICT_SOLVER_TOLERANCE,
        help="Maximum accepted FreeGSNKE final tolerance.",
    )
    return parser


def _write_jsonl(rows: list[dict], output: Path) -> None:
    path = output.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    accepted = synthetic_entries(
        args.synthetic_root,
        task=args.task,
        max_solver_tolerance=args.max_solver_tolerance,
    )
    rejected = rejected_samples(
        args.synthetic_root,
        max_solver_tolerance=args.max_solver_tolerance,
    )

    write_manifest(accepted, args.output)
    _write_jsonl(rejected, args.rejected_output)

    print(f"synthetic_root: {args.synthetic_root.expanduser().resolve()}")
    print(f"accepted: {len(accepted)}")
    print(f"rejected: {len(rejected)}")
    print(f"max_solver_tolerance: {args.max_solver_tolerance:.2e}")
    print(f"manifest: {args.output.expanduser().resolve()}")
    print(f"rejected_report: {args.rejected_output.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
