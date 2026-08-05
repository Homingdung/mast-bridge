#!/usr/bin/env python3
"""Train TokaMind on magnetic diagnostics as input and equilibrium psi as output."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = SCRIPT_ROOT.parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))
if str(SCRIPT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT / "src"))

from mast_bridge.training.tokamind_manifest import (  # noqa: E402
    INPUT_MAGNETIC_DIAGNOSTICS,
    TARGET_MODES,
    TARGET_RAW_PSI,
)
from scripts import train_tokamind_manifest  # noqa: E402


DEFAULT_MANIFEST = (
    WORKSPACE_ROOT
    / "data"
    / "manifests"
    / "tokamark_lao85_uniform_small_iter500_diagnostics_real_only.jsonl"
)
DEFAULT_FEATURE_REFERENCE_MANIFEST = (
    WORKSPACE_ROOT
    / "data"
    / "manifests"
    / "tokamark_lao85_uniform_small_iter500_diagnostics_real_plus_synthetic.jsonl"
)
DEFAULT_FEATURE_SCHEMA = (
    SCRIPT_ROOT
    / "configs"
    / "diagnostic_features"
    / "mast_level2_common_94.json"
)
DEFAULT_RUN_DIR = WORKSPACE_ROOT / "runs" / "tokamind-diagnostics-real-only"
DEFAULT_VAL_SHOTS = ["11768", "11775", "11780"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train TokaMind with magnetic diagnostics input and equilibrium psi output."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--feature-schema",
        type=Path,
        default=DEFAULT_FEATURE_SCHEMA,
        help="Versioned common diagnostic feature schema.",
    )
    parser.add_argument(
        "--feature-reference-manifest",
        type=Path,
        default=None,
        help="Optional manifest checked against the versioned feature schema.",
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument(
        "--val-fraction-only",
        action="store_true",
        help="Split train/val by seeded shot-based val_fraction instead of fixed val shots.",
    )
    parser.add_argument("--val-shot", action="append", default=None)
    parser.add_argument("--seed", type=int, default=54)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--dim-ff", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument(
        "--init-run-dir",
        type=Path,
        default=None,
        help="Synthetic pretraining run used to initialize fine-tuning.",
    )
    parser.add_argument(
        "--finetune-method",
        choices=("lora", "full"),
        default="lora",
    )
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--target-mode", choices=sorted(TARGET_MODES), default=TARGET_RAW_PSI)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _manifest_args(args: argparse.Namespace) -> list[str]:
    forwarded = [
        "--manifest",
        str(args.manifest),
        "--run-dir",
        str(args.run_dir),
        "--input-mode",
        INPUT_MAGNETIC_DIAGNOSTICS,
        "--target-mode",
        str(args.target_mode),
        "--val-fraction",
        str(args.val_fraction),
        "--seed",
        str(args.seed),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--d-model",
        str(args.d_model),
        "--n-layers",
        str(args.n_layers),
        "--n-heads",
        str(args.n_heads),
        "--dim-ff",
        str(args.dim_ff),
        "--dropout",
        str(args.dropout),
    ]
    if args.feature_reference_manifest is not None:
        forwarded.extend(
            [
                "--feature-reference-manifest",
                str(args.feature_reference_manifest),
            ]
        )
    if args.feature_schema is not None:
        forwarded.extend(["--feature-schema", str(args.feature_schema)])
    if args.init_run_dir is not None:
        forwarded.extend(
            [
                "--init-run-dir",
                str(args.init_run_dir),
                "--finetune-method",
                str(args.finetune_method),
            ]
        )
        if args.finetune_method == "lora":
            forwarded.extend(
                [
                    "--lora-rank",
                    str(args.lora_rank),
                    "--lora-alpha",
                    str(args.lora_alpha),
                ]
            )
    if args.val_shot:
        for shot in args.val_shot:
            forwarded.extend(["--val-shot", str(shot)])
    elif args.val_fraction_only:
        forwarded.extend(["--val-fraction", str(args.val_fraction)])
    else:
        for shot in DEFAULT_VAL_SHOTS:
            forwarded.extend(["--val-shot", str(shot)])
    if args.dry_run:
        forwarded.append("--dry-run")
    return forwarded


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return train_tokamind_manifest.main(_manifest_args(args))


if __name__ == "__main__":
    raise SystemExit(main())
