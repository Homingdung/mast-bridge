#!/usr/bin/env python3
"""Evaluate the 9 large TokaMind runs on the held-out 300-shot test manifest (test_real.jsonl).

Reports raw RMSE, raw MAE, and RMAE (relative MAE) for the 65x65 equilibrium psi field.
"""

from __future__ import annotations

import argparse
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
    INPUT_MODES,
    OUTPUT_SIGNAL_ID,
    TARGET_MODES,
    ManifestWindowDataset,
    load_manifest_rows,
)
from scripts.train_tokamind_manifest import _build_signal_specs  # noqa: E402


def load_run_summary(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "manifest_training_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing training summary: {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def resolve_model_config(summary: dict[str, Any]) -> dict[str, float | int]:
    config = summary.get("model_config")
    required = {"d_model", "n_layers", "n_heads", "dim_ff", "dropout"}
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("Training summary is missing the exact model_config")
    return {
        "d_model": int(config["d_model"]),
        "n_layers": int(config["n_layers"]),
        "n_heads": int(config["n_heads"]),
        "dim_ff": int(config["dim_ff"]),
        "dropout": float(config["dropout"]),
    }


def load_scalers(run_dir: Path) -> dict[str, np.ndarray | list[str]]:
    with np.load(run_dir / "manifest_scalers.npz", allow_pickle=True) as data:
        return {
            "feature_names": [str(v) for v in data["feature_names"].tolist()],
            "input_mean": np.asarray(data["input_mean"], dtype=np.float32),
            "input_std": np.asarray(data["input_std"], dtype=np.float32),
            "output_mean": np.asarray(data["output_mean"], dtype=np.float32),
            "output_std": np.asarray(data["output_std"], dtype=np.float32),
            "target_mode": str(np.asarray(data["target_mode"]).item()),
            "input_mode": str(np.asarray(data["input_mode"]).item()),
        }


def resolve_lora_config(summary: dict[str, Any]):
    peft = summary.get("peft")
    if peft is None or peft.get("method") != "lora":
        return None
    from mast_bridge.training.tokamind_lora import LoRAConfig

    return LoRAConfig(rank=int(peft["rank"]), alpha=float(peft["alpha"]), targets=str(peft["targets"]))


def metrics_from_predictions(
    pred_std: np.ndarray,
    true_std: np.ndarray,
    output_mean: np.ndarray,
    output_std: np.ndarray,
) -> dict[str, float]:
    pred_std = np.asarray(pred_std, dtype=np.float64)
    true_std = np.asarray(true_std, dtype=np.float64)
    mean = np.asarray(output_mean, dtype=np.float64).reshape(1, -1)
    std = np.maximum(np.asarray(output_std, dtype=np.float64).reshape(1, -1), 1e-6)

    pred = pred_std * std + mean
    true = true_std * std + mean
    diff = pred - true

    per_sample_rmse = np.sqrt(np.mean(diff**2, axis=1))
    per_sample_mae = np.mean(np.abs(diff), axis=1)
    per_sample_range = np.max(true, axis=1) - np.min(true, axis=1)
    per_sample_sigma = np.std(true, axis=1)

    global_range = float(np.max(true) - np.min(true))
    global_sigma = float(np.std(true))
    return {
        "samples": int(pred.shape[0]),
        "raw_rmse": float(np.sqrt(np.mean(diff**2))),
        "raw_mae": float(np.mean(np.abs(diff))),
        "rmae_per_sample": float(np.mean(per_sample_mae / np.maximum(per_sample_range, 1e-12))),
        "rmae_global": float(np.mean(np.abs(diff)) / global_range) if global_range > 0 else float("nan"),
        "nrmse_per_sample": float(np.mean(per_sample_rmse / np.maximum(per_sample_range, 1e-12))),
        "nrmse_global": float(np.sqrt(np.mean(diff**2)) / global_range) if global_range > 0 else float("nan"),
        "nrmse_sigma_per_sample": float(np.mean(per_sample_rmse / np.maximum(per_sample_sigma, 1e-12))),
        "nrmse_sigma_global": float(np.sqrt(np.mean(diff**2)) / global_sigma) if global_sigma > 0 else float("nan"),
        "std_rmse_per_sample_mean": float(np.mean(np.sqrt(np.mean((pred_std - true_std) ** 2, axis=1)))),
    }


def evaluate_run(
    run_dir: Path,
    rows: list[dict[str, Any]],
    batch_size: int,
    cached_features: np.ndarray | None,
    cached_psi: np.ndarray | None,
    save_predictions: Path | None = None,
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    from mmt.checkpoints import load_best_weights
    from mmt.data import MMTCollate
    from mmt.models import MultiModalTransformer
    from mmt.train.loop_utils import move_batch_to_device

    resolved = run_dir.expanduser().resolve()
    summary = load_run_summary(resolved)
    model_config = resolve_model_config(summary)
    scalers = load_scalers(resolved)
    input_mode = str(scalers["input_mode"])
    target_mode = str(scalers["target_mode"])
    if input_mode not in INPUT_MODES or target_mode not in TARGET_MODES:
        raise ValueError(f"Bad mode in {resolved.name}: {input_mode}/{target_mode}")

    train_shots = {str(s) for s in summary.get("train_shots", [])}
    eval_shots = {str(r["shot_id"]) for r in rows}
    overlap = eval_shots & train_shots
    if overlap:
        raise ValueError(f"{resolved.name}: test shots overlap training split: {sorted(overlap)}")

    dataset = ManifestWindowDataset.from_rows(
        rows,
        feature_names=scalers["feature_names"],
        input_mean=scalers["input_mean"],
        input_std=scalers["input_std"],
        output_mean=scalers["output_mean"],
        output_std=scalers["output_std"],
        target_mode=target_mode,
        input_mode=input_mode,
        cached_features=cached_features,
        cached_targets=cached_psi,
    )
    specs = _build_signal_specs(feature_dim=len(dataset.feature_names), output_dim=65 * 65)
    model = MultiModalTransformer(
        signal_specs=specs,
        d_model=int(model_config["d_model"]),
        n_layers=int(model_config["n_layers"]),
        n_heads=int(model_config["n_heads"]),
        dim_ff=int(model_config["dim_ff"]),
        dropout=float(model_config["dropout"]),
        max_positions=1,
        modality_heads_cfg={
            "timeseries": {"hidden": int(model_config["d_model"]), "out_dim": int(model_config["d_model"])},
            "video": {"hidden": int(model_config["d_model"]), "out_dim": int(model_config["d_model"])},
        },
        output_adapters_cfg={"hidden_dim": {"default": int(model_config["d_model"]), "bucketed": {"enable": False}, "manual": {}}},
        backbone_activation="gelu",
        debug_tokens=False,
    )
    lora_config = resolve_lora_config(summary)
    if lora_config is not None:
        from mast_bridge.training.tokamind_lora import inject_lora_backbone

        inject_lora_backbone(model, lora_config)
    epoch_best, best_val, checkpoint_meta = load_best_weights(str(resolved), model, map_location="cpu")
    if int(epoch_best) < 0:
        raise FileNotFoundError(f"No usable checkpoint in {resolved}")

    collate = MMTCollate(
        {
            "p_drop_inputs": 0.0,
            "p_drop_outputs": 0.0,
            "p_drop_actuators": 0.0,
            "p_drop_inputs_chunks": 0.0,
            "p_drop_actuators_chunks": 0.0,
        }
    )
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False, drop_last=False, num_workers=0, collate_fn=collate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch=batch, device=device)
            output = model(batch)
            predictions.append(output["pred"][OUTPUT_SIGNAL_ID].detach().cpu().numpy())
            targets.append(batch["output_emb"][OUTPUT_SIGNAL_ID].detach().cpu().numpy())

    metrics = metrics_from_predictions(
        pred_std=np.concatenate(predictions, axis=0),
        true_std=np.concatenate(targets, axis=0),
        output_mean=np.asarray(scalers["output_mean"], dtype=np.float32),
        output_std=np.asarray(scalers["output_std"], dtype=np.float32),
    )
    result = {
        "run": resolved.name,
        "run_dir": str(resolved),
        "checkpoint_epoch": int(epoch_best),
        "checkpoint_best_val": float(best_val),
        "test_shots": len(eval_shots),
        **metrics,
    }
    if save_predictions is not None:
        # Denormalize predictions back to raw psi units so downstream plotting
        # can load them without re-running inference or re-applying scalers.
        mean = np.asarray(scalers["output_mean"], dtype=np.float64).reshape(1, -1)
        std = np.maximum(np.asarray(scalers["output_std"], dtype=np.float64).reshape(1, -1), 1e-6)
        pred_std = np.asarray(np.concatenate(predictions, axis=0), dtype=np.float64)
        true_std = np.asarray(np.concatenate(targets, axis=0), dtype=np.float64)
        pred_raw = (pred_std * std + mean).astype(np.float32).reshape(len(rows), 65, 65)
        true_raw = (true_std * std + mean).astype(np.float32).reshape(len(rows), 65, 65)
        sample_ids = np.asarray([str(r.get("sample_id")) for r in rows], dtype=str)
        out_path = save_predictions.expanduser().resolve() / f"test_predictions_{resolved.name}.npz"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path,
            sample_ids=sample_ids,
            pred_psi=pred_raw,
            true_psi=true_raw,
        )
        result["predictions_path"] = str(out_path)
    return result


def _build_test_cache(rows: list[dict[str, Any]], cache_dir: Path, cache_key: str, run_dirs: list[Path] | None = None) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    """Extract features/psi once for the whole test set and reuse it across all runs.

    Returns the (possibly reduced) row list and cached matrices; rows with non-finite
    features or psi are dropped so every evaluated row is well-defined.
    """
    import time

    from mast_bridge.training.tokamind_manifest import (
        _feature_vector,
        _psi_for_row,
        TARGET_RAW_PSI,
        INPUT_MAGNETIC_DIAGNOSTICS,
    )

    cache_dir = cache_dir.expanduser().resolve()
    cache_path = cache_dir / f"test_cache_{cache_key}.npz"
    ids = [str(r.get("sample_id")) for r in rows]
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as data:
            cached_ids = [str(v) for v in data["sample_ids"].tolist()]
            if cached_ids == ids:
                print(f"cache hit: {cache_path}", flush=True)
                return rows, np.asarray(data["features"], dtype=np.float32), np.asarray(data["psi"], dtype=np.float32)
            index = {sid: i for i, sid in enumerate(ids)}
            try:
                positions = [index[sid] for sid in cached_ids]
            except KeyError:
                positions = None
            if positions is not None and positions == sorted(positions):
                by_id = {r.get("sample_id"): r for r in rows}
                cached_rows = [by_id[sid] for sid in cached_ids]
                print(f"cache hit (subset {len(cached_ids)}/{len(ids)} rows): {cache_path}", flush=True)
                return cached_rows, np.asarray(data["features"], dtype=np.float32), np.asarray(data["psi"], dtype=np.float32)
        print(f"cache mismatch, rebuilding: {cache_path}", flush=True)

    import concurrent.futures

    feature_names = list(rows[0].get("_feature_names", []))
    if not feature_names:
        for cand in run_dirs or []:
            scalers_path = cand / "manifest_scalers.npz"
            if scalers_path.is_file():
                with np.load(scalers_path, allow_pickle=True) as data:
                    feature_names = [str(v) for v in data["feature_names"].tolist()]
                break
    if not feature_names:
        raise FileNotFoundError("feature_names not found (no run-dir scalers available)")

    t0 = time.time()
    good_rows: list[dict[str, Any]] = []
    feature_list: list[np.ndarray] = []
    psi_list: list[np.ndarray] = []
    bad: list[str] = []

    def extract(row: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray, np.ndarray] | None:
        try:
            fv = _feature_vector(row, feature_names, INPUT_MAGNETIC_DIAGNOSTICS)
            pv = _psi_for_row(row, TARGET_RAW_PSI).reshape(-1)
            if not (np.isfinite(fv).all() and np.isfinite(pv).all()):
                return None
            return row, fv, pv
        except Exception as exc:  # noqa: BLE001 - per-row tolerance for held-out data
            return exc

    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
        for result in pool.map(extract, rows):
            if result is None:
                continue
            if isinstance(result, Exception):
                bad.append(str(result))
                continue
            good_rows.append(result[0])
            feature_list.append(result[1])
            psi_list.append(result[2])
    if bad:
        print(f"WARNING: dropped {len(bad)} rows: {bad[0]}", flush=True)
    if not feature_list:
        raise ValueError("No test rows survived feature extraction")
    features = np.stack(feature_list, axis=0)
    psi = np.stack(psi_list, axis=0)
    print(f"test cache extracted: {time.time() - t0:.1f}s ({features.shape[0]} x {features.shape[1]})", flush=True)
    ids = [str(r.get("sample_id")) for r in good_rows]
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, sample_ids=np.asarray(ids, dtype=str), features=features, psi=psi)
    return good_rows, features.astype(np.float32), psi.astype(np.float32)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate 9 large runs on the 300-shot real test set.")
    parser.add_argument("--manifest", type=Path, default=WORKSPACE_ROOT / "data/manifests/training/test_real.jsonl")
    parser.add_argument("--run-dir", type=Path, action="append")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-json", type=Path, default=WORKSPACE_ROOT / "artifacts" / "tokamind_test_eval" / "test_metrics.json")
    parser.add_argument(
        "--save-predictions",
        type=Path,
        default=None,
        help="Directory for denormalized prediction npz per run (test_predictions_<run>.npz).",
    )
    args = parser.parse_args(argv)

    rows = [r for r in load_manifest_rows(args.manifest.expanduser().resolve()) if r.get("source") == "real"]
    if not rows:
        raise ValueError("No real rows in test manifest")

    default_runs = [
        "tokamind-large-real-scratch-100e",
        "tokamind-large-synthetic-clean-scratch-100e",
        "tokamind-large-mixed-clean-scratch-50e",
        "tokamind-large-clean-A-finetune-real-100e",
        "tokamind-large-clean-B-finetune-real-100e",
        "tokamind-large-synthetic-noisy-scratch-100e",
        "tokamind-large-mixed-noisy-scratch-50e",
        "tokamind-large-noisy-A-finetune-real-100e",
        "tokamind-large-noisy-B-finetune-real-100e",
    ]
    run_dirs = [d.expanduser().resolve() for d in (args.run_dir or [])]
    if not run_dirs:
        run_dirs = [WORKSPACE_ROOT / "runs" / name for name in default_runs]
    missing = [str(d) for d in run_dirs if not (d / "manifest_training_summary.json").is_file()]
    if missing:
        raise FileNotFoundError(f"Runs without training summary: {missing}")

    rows, cached_features, cached_psi = _build_test_cache(rows, args.output_json.parent, args.manifest.expanduser().resolve().stem, run_dirs)
    print(f"evaluating on {len(rows)} test rows ({len({r['shot_id'] for r in rows})} shots)", flush=True)

    results = []
    for i, d in enumerate(run_dirs, start=1):
        print(f"[{i}/{len(run_dirs)}] evaluating {d.name} ...", flush=True)
        results.append(evaluate_run(run_dir=d, rows=rows, batch_size=args.batch_size,
                                    cached_features=cached_features, cached_psi=cached_psi,
                                    save_predictions=args.save_predictions))
    out = args.output_json.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"test rows: {len(rows)} | shots: {len({r['shot_id'] for r in rows})}")
    print(f"{'run':<45} {'RMSE':>10} {'MAE':>10} {'RMAE(sample)':>13} {'RMAE(global)':>13}")
    for r in results:
        print(
            f"{r['run']:<45} {r['raw_rmse']:>10.6g} {r['raw_mae']:>10.6g} "
            f"{r['rmae_per_sample']:>13.6g} {r['rmae_global']:>13.6g}"
        )
    print(f"output_json: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
