#!/usr/bin/env python3
"""Evaluate trained TokaMind manifest runs on one fixed real-validation set."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = SCRIPT_ROOT.parent
TOKAMIND_SRC = WORKSPACE_ROOT / "external" / "tokamind" / "src"
if TOKAMIND_SRC.is_dir() and str(TOKAMIND_SRC) not in sys.path:
    sys.path.insert(0, str(TOKAMIND_SRC))
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))
if str(SCRIPT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT / "src"))

from mast_bridge.training.tokamind_manifest import (  # noqa: E402
    OUTPUT_SIGNAL_ID,
    TARGET_MODES,
    TARGET_RAW_PSI,
    ManifestWindowDataset,
    load_manifest_rows,
)
from scripts.train_tokamind_manifest import _build_signal_specs  # noqa: E402


DEFAULT_MANIFEST = (
    WORKSPACE_ROOT
    / "data"
    / "manifests"
    / "tokamark_lao85_uniform_small_iter500_real_only.jsonl"
)
DEFAULT_RUNS = [
    WORKSPACE_ROOT / "runs" / "tokamark_lao85_uniform_small_iter500-real-only",
    WORKSPACE_ROOT / "runs" / "tokamark_lao85_uniform_small_iter500-synthetic-only",
    WORKSPACE_ROOT / "runs" / "tokamark_lao85_uniform_small_iter500-real-plus-synthetic",
]
DEFAULT_VAL_SHOTS = ["11768", "11775", "11780"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate one or more TokaMind manifest runs on fixed real EFIT validation shots."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--run-dir", type=Path, action="append", default=None)
    parser.add_argument("--val-shot", action="append", default=None)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=WORKSPACE_ROOT / "artifacts" / "tokamind_eval" / "real_val_metrics.json",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=WORKSPACE_ROOT / "artifacts" / "tokamind_eval" / "real_val_metrics.csv",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--dim-ff", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-mode",
        choices=sorted(TARGET_MODES),
        default=None,
        help="Evaluation target mode; default reads manifest_scalers.npz, falling back to raw-psi.",
    )
    return parser


def select_real_validation_rows(rows: list[dict[str, Any]], val_shots: list[str]) -> list[dict[str, Any]]:
    """Return real rows whose shot_id is in the fixed validation shot set."""
    allowed = {str(shot) for shot in val_shots}
    return [
        row
        for row in rows
        if row.get("source") == "real" and str(row.get("shot_id")) in allowed
    ]


def compute_metrics(
    *,
    prediction_std: np.ndarray,
    target_std: np.ndarray,
    output_mean: np.ndarray,
    output_std: np.ndarray,
) -> dict[str, float | int]:
    """Compute standardized and de-standardized psi error metrics."""
    pred_std = np.asarray(prediction_std, dtype=np.float64)
    true_std = np.asarray(target_std, dtype=np.float64)
    mean = np.asarray(output_mean, dtype=np.float64).reshape(1, -1)
    std = np.maximum(np.asarray(output_std, dtype=np.float64).reshape(1, -1), 1e-6)

    diff_std = pred_std - true_std
    pred_raw = pred_std * std + mean
    true_raw = true_std * std + mean
    diff_raw = pred_raw - true_raw

    raw_mse = float(np.mean(diff_raw**2))
    return {
        "samples": int(pred_std.shape[0]),
        "standardized_mse": float(np.mean(diff_std**2)),
        "raw_mse": raw_mse,
        "raw_rmse": float(np.sqrt(raw_mse)),
        "raw_mae": float(np.mean(np.abs(diff_raw))),
    }


def _load_scalers(run_dir: Path) -> dict[str, np.ndarray | list[str]]:
    scaler_path = run_dir / "manifest_scalers.npz"
    if not scaler_path.is_file():
        raise FileNotFoundError(f"Missing scaler file: {scaler_path}")
    with np.load(scaler_path, allow_pickle=True) as data:
        scalers: dict[str, Any] = {
            "feature_names": [str(value) for value in data["feature_names"].tolist()],
            "input_mean": np.asarray(data["input_mean"], dtype=np.float32),
            "input_std": np.asarray(data["input_std"], dtype=np.float32),
            "output_mean": np.asarray(data["output_mean"], dtype=np.float32),
            "output_std": np.asarray(data["output_std"], dtype=np.float32),
        }
        if "target_mode" in data:
            scalers["target_mode"] = str(np.asarray(data["target_mode"]).item())
        return scalers


def resolve_target_mode(cli_target_mode: str | None, scalers: dict[str, Any]) -> str:
    """Choose target mode from CLI, scaler metadata, or the legacy raw default."""
    target_mode = cli_target_mode or str(scalers.get("target_mode") or TARGET_RAW_PSI)
    if target_mode not in TARGET_MODES:
        raise ValueError(f"Unknown target_mode {target_mode!r}; expected one of {sorted(TARGET_MODES)}")
    return target_mode


def _build_dataset(rows: list[dict[str, Any]], scalers: dict[str, Any], target_mode: str) -> ManifestWindowDataset:
    return ManifestWindowDataset.from_rows(
        rows,
        feature_names=scalers["feature_names"],
        input_mean=scalers["input_mean"],
        input_std=scalers["input_std"],
        output_mean=scalers["output_mean"],
        output_std=scalers["output_std"],
        target_mode=target_mode,
    )


def evaluate_run(
    *,
    run_dir: Path,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Load one trained run and evaluate it on fixed real validation rows."""
    import torch
    from torch.utils.data import DataLoader

    from mmt.checkpoints import load_best_weights
    from mmt.data import MMTCollate
    from mmt.models import MultiModalTransformer
    from mmt.train.loop_utils import move_batch_to_device

    resolved_run = run_dir.expanduser().resolve()
    scalers = _load_scalers(resolved_run)
    target_mode = resolve_target_mode(args.target_mode, scalers)
    dataset = _build_dataset(rows, scalers, target_mode)
    signal_specs = _build_signal_specs(
        feature_dim=len(dataset.feature_names),
        output_dim=65 * 65,
    )
    model = MultiModalTransformer(
        signal_specs=signal_specs,
        d_model=int(args.d_model),
        n_layers=int(args.n_layers),
        n_heads=int(args.n_heads),
        dim_ff=int(args.dim_ff),
        dropout=float(args.dropout),
        max_positions=1,
        modality_heads_cfg={
            "timeseries": {"hidden": int(args.d_model), "out_dim": int(args.d_model)},
            "video": {"hidden": int(args.d_model), "out_dim": int(args.d_model)},
        },
        output_adapters_cfg={
            "hidden_dim": {"default": int(args.d_model), "bucketed": {"enable": False}, "manual": {}}
        },
        backbone_activation="gelu",
        debug_tokens=False,
    )
    epoch_best, best_val, checkpoint_meta = load_best_weights(str(resolved_run), model, map_location="cpu")

    collate = MMTCollate(
        {
            "p_drop_inputs": 0.0,
            "p_drop_outputs": 0.0,
            "p_drop_actuators": 0.0,
            "p_drop_inputs_chunks": 0.0,
            "p_drop_actuators_chunks": 0.0,
        }
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        drop_last=False,
        num_workers=0,
        collate_fn=collate,
    )

    device = torch.device("cpu")
    model.to(device)
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch=batch, device=device)
            output = model(batch)
            predictions.append(output["pred"][OUTPUT_SIGNAL_ID].detach().cpu().numpy())
            targets.append(batch["output_emb"][OUTPUT_SIGNAL_ID].detach().cpu().numpy())

    metrics = compute_metrics(
        prediction_std=np.concatenate(predictions, axis=0),
        target_std=np.concatenate(targets, axis=0),
        output_mean=np.asarray(scalers["output_mean"], dtype=np.float32),
        output_std=np.asarray(scalers["output_std"], dtype=np.float32),
    )
    return {
        "run": resolved_run.name,
        "run_dir": str(resolved_run),
        "checkpoint_epoch": int(epoch_best),
        "checkpoint_best_val": float(best_val),
        "checkpoint_meta": checkpoint_meta,
        "target_mode": target_mode,
        "validation_shots": sorted({str(row["shot_id"]) for row in rows}),
        **metrics,
    }


def _write_outputs(results: list[dict[str, Any]], output_json: Path, output_csv: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    fieldnames = [
        "run",
        "samples",
        "validation_shots",
        "checkpoint_epoch",
        "checkpoint_best_val",
        "target_mode",
        "standardized_mse",
        "raw_mse",
        "raw_rmse",
        "raw_mae",
        "run_dir",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = {key: result.get(key) for key in fieldnames}
            row["validation_shots"] = " ".join(result["validation_shots"])
            writer.writerow(row)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = args.manifest.expanduser().resolve()
    val_shots = args.val_shot or DEFAULT_VAL_SHOTS
    rows = select_real_validation_rows(load_manifest_rows(manifest), val_shots)
    if not rows:
        raise ValueError(f"No real validation rows found in {manifest} for shots {val_shots}")

    run_dirs = args.run_dir or DEFAULT_RUNS
    results = [evaluate_run(run_dir=run_dir, rows=rows, args=args) for run_dir in run_dirs]
    _write_outputs(
        results,
        args.output_json.expanduser().resolve(),
        args.output_csv.expanduser().resolve(),
    )

    for result in results:
        print(
            f"{result['run']}: samples={result['samples']} "
            f"std_mse={result['standardized_mse']:.6g} "
            f"raw_rmse={result['raw_rmse']:.6g} raw_mae={result['raw_mae']:.6g}"
        )
    print(f"output_json: {args.output_json.expanduser().resolve()}")
    print(f"output_csv: {args.output_csv.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
