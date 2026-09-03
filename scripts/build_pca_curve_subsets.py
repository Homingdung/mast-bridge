#!/usr/bin/env python3
"""Build nested random shot subsets of PCA 01σ train real data.

Semantics: fraction = fraction of the FULL train shot pool, sampled as a PREFIX
of a single random shuffle (seed 20260825, same as split). Guarantees nesting:
  5% ⊂ 10% ⊂ 25% ⊂ 50% ⊂ 100% (train)
No campaign stratification (temporal-split experiments come separately).

Outputs (data/manifests/training_pca_01sigma/):
  subset_real_{f}pct.jsonl      subset rows only
  subset_run_real_{f}pct.jsonl  subset + val rows (run-file pattern)
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = SCRIPT_ROOT.parent
DEFAULT_OUT = WORKSPACE_ROOT / "data" / "manifests" / "training_pca_01sigma"


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--fractions", default="5,10,25,50")
    args = parser.parse_args(argv)

    out_dir = (args.out_dir or DEFAULT_OUT).expanduser().resolve()
    fractions = [float(x) / 100 for x in args.fractions.split(",")]

    train_rows = load_rows(out_dir / "split_train_real.jsonl")
    val_rows = load_rows(out_dir / "split_val_real.jsonl")
    by_shot: dict[str, list[dict]] = {}
    for row in train_rows:
        by_shot.setdefault(str(row["shot_id"]), []).append(row)
    shots = sorted(by_shot)

    rng = random.Random(args.seed)
    pool = list(shots)
    rng.shuffle(pool)
    print(f"train shots: {len(shots)} (master shuffle seed={args.seed})")

    for frac in sorted(fractions):
        n = int(round(len(pool) * frac))
        chosen = pool[:n]
        chosen_set = set(chosen)
        assert len(chosen_set) == len(chosen)
        subset_rows = [row for shot in chosen for row in by_shot[shot]]
        label = f"{frac*100:g}pct"
        subset_path = out_dir / f"subset_real_{label}.jsonl"
        run_path = out_dir / f"subset_run_real_{label}.jsonl"
        write_rows(subset_path, subset_rows)
        write_rows(run_path, subset_rows + val_rows)
        print(f"  {label:5s} shots={len(chosen):5d} rows={len(subset_rows):7d} "
              f"({len(subset_rows)/len(train_rows):.1%} of train) run_rows={len(subset_rows)+len(val_rows):7d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
