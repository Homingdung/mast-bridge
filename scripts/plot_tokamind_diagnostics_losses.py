#!/usr/bin/env python3
"""Plot the three diagnostics-to-psi training histories in one figure."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = SCRIPT_ROOT.parent
DEFAULT_RUNS = (
    (
        "Real only",
        WORKSPACE_ROOT / "runs" / "tokamind-diagnostics-real-only",
        "tokamark_lao85_uniform_small_iter500_diagnostics_real_only.jsonl",
    ),
    (
        "Synthetic only",
        WORKSPACE_ROOT / "runs" / "tokamind-diagnostics-synthetic-only",
        "tokamark_lao85_uniform_small_iter500_diagnostics_synthetic_only.jsonl",
    ),
    (
        "Real + synthetic",
        WORKSPACE_ROOT / "runs" / "tokamind-diagnostics-real-plus-synthetic",
        "tokamark_lao85_uniform_small_iter500_diagnostics_real_plus_synthetic.jsonl",
    ),
)
DEFAULT_OUTPUT = (
    WORKSPACE_ROOT
    / "artifacts"
    / "tokamind_loss_curves"
    / "tokamind_diagnostics_loss_curves.png"
)
DEFAULT_CSV = (
    WORKSPACE_ROOT
    / "artifacts"
    / "tokamind_loss_curves"
    / "tokamind_diagnostics_loss_summary.csv"
)


def load_loss_rows(
    run_dir: Path,
    experiment: str,
    expected_manifest: str | None = None,
) -> list[dict[str, Any]]:
    summary_path = run_dir.expanduser().resolve() / "manifest_training_summary.json"
    with summary_path.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    if expected_manifest is not None:
        actual_manifest = Path(str(summary.get("manifest", ""))).name
        if actual_manifest != expected_manifest:
            raise ValueError(
                f"{summary_path} uses {actual_manifest!r}, expected {expected_manifest!r}"
            )
    history = summary.get("history", {}).get("stages", {}).get("manifest_scratch")
    if not isinstance(history, list) or not history:
        raise ValueError(f"missing manifest_scratch history: {summary_path}")

    rows: list[dict[str, Any]] = []
    for item in history:
        train_loss = float(item["train_loss"])
        val_loss = float(item["val_loss"])
        if not math.isfinite(train_loss) or not math.isfinite(val_loss):
            raise ValueError(f"non-finite loss in {summary_path}")
        rows.append(
            {
                "experiment": experiment,
                "epoch": int(item["epoch_global"]),
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )
    return rows


def write_loss_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("experiment", "epoch", "train_loss", "val_loss"),
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_losses(path: Path, grouped_rows: list[list[dict[str, Any]]]) -> None:
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "mast-bridge-matplotlib"),
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ("#1677b3", "#cc5a00", "#00966f")
    figure, axis = plt.subplots(figsize=(12, 7))
    for rows, color in zip(grouped_rows, colors, strict=True):
        label = str(rows[0]["experiment"])
        epochs = [row["epoch"] for row in rows]
        axis.plot(
            epochs,
            [row["train_loss"] for row in rows],
            color=color,
            linewidth=2,
            label=f"{label} - train",
        )
        axis.plot(
            epochs,
            [row["val_loss"] for row in rows],
            color=color,
            linewidth=2,
            linestyle="--",
            label=f"{label} - val",
        )

    axis.set_title("TokaMind Diagnostics-to-Psi Training Curves")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Normalized Embed MSE")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plot real, synthetic, and mixed TokaMind loss curves together."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        default=None,
        help="Run directory in real/synthetic/mixed order; repeat exactly three times.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args(argv)

    if args.run_dir is not None and len(args.run_dir) != len(DEFAULT_RUNS):
        parser.error("--run-dir must be repeated exactly three times")
    run_dirs = args.run_dir or [run_dir for _, run_dir, _ in DEFAULT_RUNS]
    grouped_rows = [
        load_loss_rows(run_dir, experiment, expected_manifest)
        for run_dir, (experiment, _, expected_manifest) in zip(
            run_dirs, DEFAULT_RUNS, strict=True
        )
    ]
    rows = [row for group in grouped_rows for row in group]
    write_loss_csv(args.output_csv.expanduser().resolve(), rows)
    plot_losses(args.output.expanduser().resolve(), grouped_rows)
    print(f"loss_plot: {args.output.expanduser().resolve()}")
    print(f"loss_csv: {args.output_csv.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
