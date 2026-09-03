#!/usr/bin/env python3
"""Build temporal v2 split: M5-M7 + M9 first 20% (early) -> train, M8 -> val,
M9 last 80% (late) -> test. Simulates rolling forecast (history + recent
data finetune -> predict future).

Outputs (data/manifests/training_temporal_v2/):
  split_train_real.jsonl            train rows only (M5-M7 + M9 early 20%)
  split_val_real.jsonl              val rows (M8)
  split_test_real.jsonl             test rows (M9 late 80%)
  split_train_shots.jsonl / split_val_shots.jsonl / split_test_shots.jsonl
  split_train_with_val.jsonl        train + val (run-file pattern, 100% tier)
  subset_run_real_{50,25,5}pct.jsonl  nested train shot subsets + val (run-file)
"""

from __future__ import annotations

import json
import random
from pathlib import Path

SRC = Path("data/manifests/training_temporal")
OUT = Path("data/manifests/training_temporal_v2")


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_shots(path: Path, shot_ids: list[str]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for sid in shot_ids:
            handle.write(json.dumps({"shot_id": sid}) + "\n")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    train_rows = load_rows(SRC / "split_train_real.jsonl")
    val_rows = load_rows(SRC / "split_val_real.jsonl")
    m9_rows = load_rows(SRC / "split_test_real.jsonl")

    by_shot: dict[str, list[dict]] = {}
    for row in m9_rows:
        by_shot.setdefault(str(row["shot_id"]), []).append(row)
    m9_ids = sorted(by_shot)
    n20 = int(round(len(m9_ids) * 0.2))
    early, late = m9_ids[:n20], m9_ids[n20:]

    train_rows_v2 = train_rows + [r for s in early for r in by_shot[s]]
    test_rows_v2 = [r for s in late for r in by_shot[s]]

    write_rows(OUT / "split_train_real.jsonl", train_rows_v2)
    write_rows(OUT / "split_val_real.jsonl", val_rows)
    write_rows(OUT / "split_test_real.jsonl", test_rows_v2)
    write_shots(OUT / "split_train_shots.jsonl", sorted({str(r["shot_id"]) for r in train_rows_v2}))
    write_shots(OUT / "split_val_shots.jsonl", sorted({str(r["shot_id"]) for r in val_rows}))
    write_shots(OUT / "split_test_shots.jsonl", late)
    write_rows(OUT / "split_train_with_val.jsonl", train_rows_v2 + val_rows)

    print(f"train: {len(train_rows_v2)} rows / {len({str(r['shot_id']) for r in train_rows_v2})} shots")
    print(f"val:   {len(val_rows)} rows")
    print(f"test:  {len(test_rows_v2)} rows / {len(late)} shots (M9 late 80%)")

    rng = random.Random(20260831)
    train_shots = sorted({str(r["shot_id"]) for r in train_rows_v2})
    pool = list(train_shots)
    rng.shuffle(pool)
    by_shot_v2: dict[str, list[dict]] = {}
    for row in train_rows_v2:
        by_shot_v2.setdefault(str(row["shot_id"]), []).append(row)
    for frac in (0.5, 0.25, 0.05):
        n = int(round(len(pool) * frac))
        chosen = pool[:n]
        subset_rows = [row for shot in chosen for row in by_shot_v2[shot]]
        label = f"{frac*100:g}pct"
        write_rows(OUT / f"subset_real_{label}.jsonl", subset_rows)
        write_rows(OUT / f"subset_run_real_{label}.jsonl", subset_rows + val_rows)
        print(f"  {label}: shots={len(chosen):5d} subset_rows={len(subset_rows):7d} "
              f"run_rows={len(subset_rows) + len(val_rows):7d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
