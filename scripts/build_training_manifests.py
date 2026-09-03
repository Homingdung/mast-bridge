#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_ROOT / "src"))

import pandas as pd  # noqa: E402

from mast_bridge.workspace import discover_workspace  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build training/test manifests for the large-scale TokaMind experiments."
    )
    parser.add_argument("--real-manifest", type=Path, required=True)
    parser.add_argument("--synthetic-manifest", type=Path, required=True)
    parser.add_argument("--clean-pool", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--test-shots", type=int, default=300)
    parser.add_argument("--fixed-val-shots", nargs="*", default=["11768", "11775", "11780"])
    parser.add_argument("--seed", type=int, default=20260804)
    args = parser.parse_args(argv)

    real_rows = load_jsonl(args.real_manifest)
    synth_rows = load_jsonl(args.synthetic_manifest)
    pool = pd.read_csv(args.clean_pool, dtype={"shot": str})
    criterion = pool["in_window"] & pool["ip_ok"]

    available = sorted({str(r["shot_id"]) for r in real_rows})
    rng = random.Random(args.seed)
    fixed = [str(s) for s in args.fixed_val_shots]
    chosen = fixed + sorted(rng.sample([s for s in available if s not in fixed], args.test_shots - len(fixed)))
    chosen_set = set(chosen)
    print(f"test shots: {len(chosen_set)} (fixed {len(fixed)} + random {len(chosen_set) - len(fixed)})")

    test_rows = []
    for shot in chosen:
        sub = pool[(pool["shot"] == shot) & criterion]
        if sub.empty:
            print(f"  warning: {shot} has no clean slice, skipping")
            continue
        time = float(sub["time"].sort_values().iloc[len(sub) // 2])
        test_rows.append(
            {
                "comparison_group": "test",
                "data_path": str(discover_workspace(SCRIPT_ROOT).data_root / "raw" / "mast" / f"{shot}.zarr"),
                "equilibrium_path": None,
                "fit_path": "/inspire/qb-ilm/project/ai-for-fusion/public/fusion-workspace/data/processed/real/lao_parameter_ensemble/all_zarr_lao_parameter_fits.npz",
                "label_path": None,
                "label_source": "zarr_equilibrium_psi",
                "machine_config_path": str(discover_workspace(SCRIPT_ROOT).data_root / "raw" / "mast" / "machine" / shot),
                "parent_shot": None,
                "profile_parameter_source": "lao_fit_npz",
                "sample_id": f"{shot}_t{time:g}_test",
                "shot_id": shot,
                "solver_status": None,
                "source": "real",
                "target_time": time,
                "task": "task_1-3",
            }
        )
    print(f"test rows: {len(test_rows)}")

    real_train = [r for r in real_rows if str(r["shot_id"]) not in chosen_set]
    synth_train = [r for r in synth_rows if str(r["parent_shot"]) not in chosen_set]
    synth_noisy = [
        {**r, "diagnostics_path": str(Path(r["diagnostics_path"]).with_name("diagnostics_noisy.npz"))}
        for r in synth_train
    ]
    print(f"real_train: {len(real_train)} | synth_clean_train: {len(synth_train)} | synth_noisy_train: {len(synth_noisy)}")

    base = args.output_dir.expanduser().resolve()
    write_jsonl(base / "test_real.jsonl", test_rows)
    write_jsonl(base / "train_real.jsonl", real_train)
    write_jsonl(base / "train_synthetic_clean.jsonl", synth_train)
    write_jsonl(base / "train_synthetic_noisy.jsonl", synth_noisy)
    write_jsonl(base / "train_mixed_clean.jsonl", real_train + synth_train)
    write_jsonl(base / "train_mixed_noisy.jsonl", real_train + synth_noisy)
    print(f"wrote manifests to {base}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
