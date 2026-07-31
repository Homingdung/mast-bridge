from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .manifest import ManifestEntry

STRICT_SOLVER_TOLERANCE = 1e-8


def _is_finite_equilibrium(path: Path) -> bool:
    try:
        with np.load(path) as equilibrium:
            psi = np.asarray(equilibrium["psi"])
            return bool(psi.shape == (65, 65) and np.isfinite(psi).all())
    except (OSError, KeyError, ValueError):
        return False


def rejection_reason(
    metadata: dict[str, Any],
    equilibrium_path: Path,
    max_solver_tolerance: float = STRICT_SOLVER_TOLERANCE,
) -> str | None:
    """Return None for accepted samples, otherwise a stable rejection reason."""
    if metadata.get("solver_converged") is not True:
        return "solver_not_converged"
    try:
        final_tolerance = float(metadata["solver_final_tolerance"])
    except (KeyError, TypeError, ValueError):
        return "solver_tolerance_missing"
    if not np.isfinite(final_tolerance):
        return "solver_tolerance_nonfinite"
    if final_tolerance > max_solver_tolerance:
        return "solver_tolerance_above_threshold"
    if not _is_finite_equilibrium(equilibrium_path):
        return "invalid_equilibrium"
    return None


def _is_converged(
    metadata: dict[str, Any], max_solver_tolerance: float = STRICT_SOLVER_TOLERANCE
) -> bool:
    try:
        final_tolerance = float(metadata["solver_final_tolerance"])
    except (KeyError, TypeError, ValueError):
        return False
    return (
        metadata.get("solver_converged") is True
        and np.isfinite(final_tolerance)
        and final_tolerance <= max_solver_tolerance
    )


def synthetic_entries(
    root: str | Path,
    task: str | None = None,
    max_solver_tolerance: float = STRICT_SOLVER_TOLERANCE,
) -> list[ManifestEntry]:
    """Scan converged synthetic samples into manifest entries."""
    synthetic_root = Path(root).expanduser().resolve()
    rows: list[ManifestEntry] = []
    if not synthetic_root.is_dir():
        return rows

    for sample_dir in sorted(path for path in synthetic_root.iterdir() if path.is_dir()):
        metadata_path = sample_dir / "metadata.json"
        equilibrium_path = sample_dir / "equilibrium.npz"
        if not metadata_path.is_file() or not equilibrium_path.is_file():
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if rejection_reason(metadata, equilibrium_path, max_solver_tolerance) is not None:
            continue

        parent_shot = str(metadata.get("parent_shot", ""))
        if not parent_shot:
            continue
        row_metadata = dict(metadata)
        if task is not None:
            row_metadata["task"] = task

        rows.append(
            ManifestEntry(
                sample_id=sample_dir.name,
                source="synthetic",
                shot_id=sample_dir.name,
                data_path=sample_dir,
                equilibrium_path=equilibrium_path,
                parent_shot=parent_shot,
                solver_status=metadata.get("solver_status"),
                metadata=row_metadata,
            )
        )
    return rows


def rejected_samples(
    root: str | Path,
    max_solver_tolerance: float = STRICT_SOLVER_TOLERANCE,
) -> list[dict[str, Any]]:
    """Scan synthetic samples that are excluded from the strict manifest."""
    synthetic_root = Path(root).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    if not synthetic_root.is_dir():
        return rows

    for sample_dir in sorted(path for path in synthetic_root.iterdir() if path.is_dir()):
        metadata_path = sample_dir / "metadata.json"
        equilibrium_path = sample_dir / "equilibrium.npz"
        if not metadata_path.is_file():
            rows.append({"sample_id": sample_dir.name, "reason": "metadata_missing"})
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            rows.append({"sample_id": sample_dir.name, "reason": "metadata_invalid_json"})
            continue
        reason = rejection_reason(metadata, equilibrium_path, max_solver_tolerance)
        if reason is None:
            continue
        rows.append(
            {
                "sample_id": sample_dir.name,
                "reason": reason,
                "parent_shot": metadata.get("parent_shot"),
                "target_time": metadata.get("target_time"),
                "solver_status": metadata.get("solver_status"),
                "solver_converged": metadata.get("solver_converged"),
                "solver_final_tolerance": metadata.get("solver_final_tolerance"),
                "max_solver_tolerance": max_solver_tolerance,
                "metadata_path": str(metadata_path),
            }
        )
    return rows
