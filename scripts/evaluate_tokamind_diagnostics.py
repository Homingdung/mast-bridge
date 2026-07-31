#!/usr/bin/env python3
"""Evaluate diagnostics-to-psi runs on one fixed real EFIT validation set."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = SCRIPT_ROOT.parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from scripts import evaluate_tokamind_manifest  # noqa: E402


DEFAULT_MANIFEST = (
    WORKSPACE_ROOT
    / "data"
    / "manifests"
    / "tokamark_lao85_uniform_small_iter500_diagnostics_real_only.jsonl"
)
DEFAULT_RUN_DIRS = [
    WORKSPACE_ROOT / "runs" / "tokamind-diagnostics-real-only",
    WORKSPACE_ROOT / "runs" / "tokamind-diagnostics-synthetic-only",
    WORKSPACE_ROOT / "runs" / "tokamind-diagnostics-real-plus-synthetic",
]
DEFAULT_VAL_SHOTS = ["11768", "11775", "11780"]
DEFAULT_OUTPUT_JSON = (
    WORKSPACE_ROOT
    / "artifacts"
    / "tokamind_eval"
    / "tokamind_diagnostics_real_val_metrics.json"
)
DEFAULT_OUTPUT_CSV = (
    WORKSPACE_ROOT
    / "artifacts"
    / "tokamind_eval"
    / "tokamind_diagnostics_real_val_metrics.csv"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate diagnostics-to-psi models on a fixed real EFIT validation set."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--run-dir", type=Path, action="append", default=None)
    parser.add_argument("--val-shot", action="append", default=None)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser


def _manifest_args(args: argparse.Namespace) -> list[str]:
    forwarded = ["--manifest", str(args.manifest)]
    for run_dir in args.run_dir or DEFAULT_RUN_DIRS:
        forwarded.extend(["--run-dir", str(run_dir)])
    for shot in args.val_shot or DEFAULT_VAL_SHOTS:
        forwarded.extend(["--val-shot", str(shot)])
    forwarded.extend(
        [
            "--output-json",
            str(args.output_json),
            "--output-csv",
            str(args.output_csv),
            "--batch-size",
            str(args.batch_size),
        ]
    )
    return forwarded


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return evaluate_tokamind_manifest.main(_manifest_args(args))


if __name__ == "__main__":
    raise SystemExit(main())
