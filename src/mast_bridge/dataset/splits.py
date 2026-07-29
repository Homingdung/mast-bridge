from __future__ import annotations

import random
from typing import Any, Iterable


def _parent_key(row: dict[str, Any]) -> str:
    if row.get("source") == "synthetic":
        parent = row.get("parent_shot")
        if parent is None:
            raise ValueError("synthetic rows require parent_shot")
        return str(parent)
    shot = row.get("shot_id", row.get("sample_id"))
    if shot is None:
        raise ValueError("rows require shot_id or sample_id")
    return str(shot)


def assign_parent_shot_splits(
    rows: Iterable[dict[str, Any]],
    train_fraction: float,
    val_fraction: float,
    seed: int,
) -> dict[str, str]:
    """Assign splits by parent shot so synthetic children cannot leak."""
    if train_fraction < 0 or val_fraction < 0 or train_fraction + val_fraction > 1:
        raise ValueError("split fractions must be non-negative and sum to <= 1")

    parents = sorted({_parent_key(row) for row in rows})
    rng = random.Random(seed)
    rng.shuffle(parents)

    train_count = int(round(len(parents) * train_fraction))
    val_count = int(round(len(parents) * val_fraction))
    if train_count + val_count > len(parents):
        val_count = len(parents) - train_count

    assignments: dict[str, str] = {}
    for index, parent in enumerate(parents):
        if index < train_count:
            split = "train"
        elif index < train_count + val_count:
            split = "val"
        else:
            split = "test"
        assignments[parent] = split
    return assignments


def split_for_row(row: dict[str, Any], assignments: dict[str, str]) -> str:
    return assignments[_parent_key(row)]
