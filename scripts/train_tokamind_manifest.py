#!/usr/bin/env python3
"""Train a small TokaMind MMT model from a mast-bridge comparison manifest."""

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

from mast_bridge.training.tokamind_manifest import (  # noqa: E402
    INPUT_SIGNAL_ID,
    OUTPUT_SIGNAL_ID,
    TARGET_MODES,
    TARGET_RAW_PSI,
    ManifestWindowDataset,
    build_manifest_datasets,
    load_manifest_rows,
)


DEFAULT_MANIFEST = (
    WORKSPACE_ROOT
    / "data"
    / "manifests"
    / "tokamark_lao85_uniform_iter500_real_plus_synthetic.jsonl"
)
DEFAULT_RUN_DIR = WORKSPACE_ROOT / "runs" / "tokamind-lao85-uniform-iter500-real-plus-synthetic"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a small TokaMind MMT model on a mast-bridge JSONL manifest."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=54)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--dim-ff", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--target-mode", choices=sorted(TARGET_MODES), default=TARGET_RAW_PSI)
    parser.add_argument("--dry-run", action="store_true", help="Validate manifest loading without importing torch.")
    return parser


def _write_scalers(run_dir: Path, train_dataset: ManifestWindowDataset, target_mode: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        run_dir / "manifest_scalers.npz",
        feature_names=np.asarray(train_dataset.feature_names, dtype=object),
        input_mean=train_dataset.input_mean,
        input_std=train_dataset.input_std,
        output_mean=train_dataset.output_mean,
        output_std=train_dataset.output_std,
        target_mode=np.asarray(target_mode),
    )


def _summary(
    rows: list[dict[str, Any]],
    train_dataset: ManifestWindowDataset,
    val_dataset: ManifestWindowDataset,
    target_mode: str,
) -> str:
    sources: dict[str, int] = {}
    for row in rows:
        source = str(row.get("source", "unknown"))
        sources[source] = sources.get(source, 0) + 1
    return "\n".join(
        [
            f"rows: {len(rows)}",
            f"sources: {json.dumps(sources, sort_keys=True)}",
            f"train_windows: {len(train_dataset)}",
            f"val_windows: {len(val_dataset)}",
            f"feature_dim: {len(train_dataset.feature_names)}",
            f"target_mode: {target_mode}",
            "input_signal: fusion-state",
            "output_signal: equilibrium-psi",
        ]
    )


def _build_signal_specs(feature_dim: int, output_dim: int):
    from mmt.data import SignalSpec
    from mmt.data.signal_spec import SignalSpecRegistry

    specs = [
        SignalSpec(
            name="fusion-state",
            role="input",
            modality="timeseries",
            encoder_name="identity",
            encoder_kwargs={},
            signal_id=INPUT_SIGNAL_ID,
            embedding_dim=feature_dim,
            values_shape=(),
            native_shape=(1, 1, feature_dim),
        ),
        SignalSpec(
            name="equilibrium-psi",
            role="output",
            modality="video",
            encoder_name="identity",
            encoder_kwargs={},
            signal_id=OUTPUT_SIGNAL_ID,
            embedding_dim=output_dim,
            values_shape=(65, 65),
            native_shape=(65, 65, 1),
        ),
    ]
    return SignalSpecRegistry(specs=specs)


def _train(args: argparse.Namespace, train_dataset: ManifestWindowDataset, val_dataset: ManifestWindowDataset) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    from mmt.data import MMTCollate
    from mmt.models import MultiModalTransformer
    from mmt.train import train_finetune
    from mmt.utils import set_seed

    set_seed(seed=int(args.seed), deterministic=True, warn_only=True)

    signal_specs = _build_signal_specs(
        feature_dim=len(train_dataset.feature_names),
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
        output_adapters_cfg={"hidden_dim": {"default": int(args.d_model), "bucketed": {"enable": False}, "manual": {}}},
        backbone_activation="gelu",
        debug_tokens=False,
    )

    collate = MMTCollate(
        {
            "p_drop_inputs": 0.0,
            "p_drop_outputs": 0.0,
            "p_drop_actuators": 0.0,
            "p_drop_inputs_chunks": 0.0,
            "p_drop_actuators_chunks": 0.0,
        }
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        drop_last=False,
        num_workers=0,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        drop_last=False,
        num_workers=0,
        collate_fn=collate,
    )

    train_cfg = {
        "resume": False,
        "early_stop": {"patience": max(2, int(args.epochs)), "delta": 0.0},
        "amp": {"enable": torch.cuda.is_available()},
        "loss": {"terms": [{"type": "embed_mse", "weight": 1.0}], "output_weights": {}},
        "optimizer": {"use_adamw": True},
        "stages": [
            {
                "name": "manifest_scratch",
                "epochs": int(args.epochs),
                "scheduler": {"grad_accum_steps": 1, "warmup_steps_fraction": 0.0},
                "optimizer": {
                    "lr": {
                        "token_encoder": float(args.lr),
                        "backbone": float(args.lr),
                        "modality_heads": float(args.lr),
                        "output_adapters": float(args.lr),
                    },
                    "wd": {
                        "token_encoder": 0.0,
                        "backbone": 0.0,
                        "modality_heads": 0.0,
                        "output_adapters": 0.0,
                    },
                },
                "freeze": {
                    "token_encoder": False,
                    "backbone": False,
                    "modality_heads": False,
                    "output_adapters": False,
                },
            }
        ],
    }
    loader_cfg = {"batch_size": int(args.batch_size), "batches_per_epoch": None}
    return train_finetune(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        run_dir=str(args.run_dir.expanduser().resolve()),
        train_cfg=train_cfg,
        loader_cfg=loader_cfg,
        output_decoders=None,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = args.manifest.expanduser().resolve()
    rows = load_manifest_rows(manifest)
    train_dataset, val_dataset = build_manifest_datasets(
        rows,
        val_fraction=float(args.val_fraction),
        seed=int(args.seed),
        target_mode=str(args.target_mode),
    )

    print(_summary(rows, train_dataset, val_dataset, str(args.target_mode)))

    if args.dry_run:
        return 0

    run_dir = args.run_dir.expanduser().resolve()
    _write_scalers(run_dir, train_dataset, str(args.target_mode))
    history = _train(args, train_dataset, val_dataset)
    (run_dir / "manifest_training_summary.json").write_text(
        json.dumps(
            {
                "manifest": str(manifest),
                "rows": len(rows),
                "train_windows": len(train_dataset),
                "val_windows": len(val_dataset),
                "target_mode": str(args.target_mode),
                "feature_names": train_dataset.feature_names,
                "history": history,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"run_dir: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
