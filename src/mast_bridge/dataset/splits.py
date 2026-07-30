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

    row_list = list(rows)
    parent_counts: dict[str, int] = {}
    for row in row_list:
        parent = _parent_key(row)
        parent_counts[parent] = parent_counts.get(parent, 0) + 1

    parents = sorted(parent_counts)
    rng = random.Random(seed)
    rng.shuffle(parents)
    shuffle_order = {parent: index for index, parent in enumerate(parents)}

    train_count = int(round(len(parents) * train_fraction))
    val_count = int(round(len(parents) * val_fraction))
    if len(parents) >= 2 and train_fraction > 0 and train_count == 0:
        train_count = 1
    if len(parents) >= 2 and val_fraction > 0 and val_count == 0:
        val_count = 1
    if train_count + val_count > len(parents):
        if val_fraction > 0:
            train_count = max(1, len(parents) - val_count)
        else:
            val_count = len(parents) - train_count
    train_count = min(train_count, len(parents) - val_count)

    val_parents: set[str] = set()
    if val_count:
        val_parents = set(
            sorted(parents, key=lambda parent: (parent_counts[parent], shuffle_order[parent]))[
                :val_count
            ]
        )

    train_candidates = [parent for parent in parents if parent not in val_parents]
    train_parents = set(train_candidates[:train_count])

    assignments: dict[str, str] = {}
    for parent in parents:
        if parent in train_parents:
            split = "train"
        elif parent in val_parents:
            split = "val"
        else:
            split = "test"
        assignments[parent] = split
    return assignments


def split_for_row(row: dict[str, Any], assignments: dict[str, str]) -> str:
    return assignments[_parent_key(row)]
