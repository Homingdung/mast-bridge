from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from mast_bridge.dataset.splits import assign_parent_shot_splits, split_for_row


INPUT_SIGNAL_ID = 0
OUTPUT_SIGNAL_ID = 1
TIMESERIES_MODALITY_ID = 0
ROLE_CONTEXT = 0
TARGET_RAW_PSI = "raw-psi"
TARGET_PSI_NORM = "psi-norm"
TARGET_MODES = {TARGET_RAW_PSI, TARGET_PSI_NORM}


def load_manifest_rows(path: str | Path) -> list[dict[str, Any]]:
    """Load JSONL manifest rows."""
    manifest_path = Path(path).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _nearest_index(times: Any, target_time: float) -> int:
    values = np.asarray(times, dtype=float)
    if values.size == 0:
        raise ValueError("Cannot select nearest time from an empty time array")
    return int(np.argmin(np.abs(values - float(target_time))))


def _fit_parameters_for_row(row: dict[str, Any]) -> dict[str, Any]:
    fit_path = row.get("fit_path") or row.get("label_path")
    if not fit_path:
        raise KeyError(f"Real row {row.get('sample_id')!r} is missing fit_path/label_path")

    with np.load(Path(fit_path).expanduser().resolve()) as fit:
        shot_values = np.asarray(fit["shot"]).astype(str)
        shot = str(row["shot_id"])
        candidates = np.flatnonzero(shot_values == shot)
        if candidates.size == 0:
            raise ValueError(f"No Lao fit row found for shot {shot!r}")
        fit_times = np.asarray(fit["time"], dtype=float)
        index = int(candidates[np.argmin(np.abs(fit_times[candidates] - float(row["target_time"])) )])
        return {
            "Ip": float(fit["ip"][index]),
            "fvac": float(fit["fvac"][index]),
            "alpha": np.asarray(fit["freegsnke_alpha"][index], dtype=float).reshape(-1).tolist(),
            "beta": np.asarray(fit["freegsnke_beta"][index], dtype=float).reshape(-1).tolist(),
        }


def _active_currents_from_real_row(row: dict[str, Any]) -> dict[str, float]:
    try:
        import zarr
    except ModuleNotFoundError as exc:
        raise RuntimeError("Reading real manifest rows requires the optional 'zarr' package") from exc

    root = zarr.open_group(str(Path(row["data_path"]).expanduser().resolve()), mode="r")
    active = root["pf_active"]
    index = _nearest_index(active["time"][:], float(row["target_time"]))
    channels = [str(value) for value in active["current_channel"][:]]
    currents = np.asarray(active["coil_current"][:], dtype=float)
    return {channel: float(currents[channel_index, index]) for channel_index, channel in enumerate(channels)}


def _real_psi(row: dict[str, Any]) -> np.ndarray:
    try:
        import zarr
    except ModuleNotFoundError as exc:
        raise RuntimeError("Reading real manifest rows requires the optional 'zarr' package") from exc

    root = zarr.open_group(str(Path(row["data_path"]).expanduser().resolve()), mode="r")
    equilibrium = root["equilibrium"]
    index = _nearest_index(equilibrium["time"][:], float(row["target_time"]))
    psi = np.asarray(equilibrium["psi"][:, :, index], dtype=np.float32)
    return _validate_psi(psi, row)


def _interp_grid_value(r_grid: np.ndarray, z_grid: np.ndarray, values: np.ndarray, r_value: float, z_value: float) -> float:
    """Bilinear interpolation on the rectilinear MAST EFIT grid."""
    r = np.asarray(r_grid, dtype=float)
    z = np.asarray(z_grid, dtype=float)
    field = np.asarray(values, dtype=float)
    if r.ndim != 1 or z.ndim != 1:
        raise ValueError("Expected 1D major_radius and z grids for psi normalization")
    if field.shape != (r.size, z.size):
        raise ValueError(f"Expected psi shape {(r.size, z.size)}, got {field.shape}")
    if not (r[0] <= r_value <= r[-1]) or not (z[0] <= z_value <= z[-1]):
        raise ValueError(f"Interpolation point ({r_value}, {z_value}) is outside the EFIT grid")

    i = int(np.searchsorted(r, r_value, side="right") - 1)
    j = int(np.searchsorted(z, z_value, side="right") - 1)
    i = min(max(i, 0), r.size - 2)
    j = min(max(j, 0), z.size - 2)
    r0, r1 = float(r[i]), float(r[i + 1])
    z0, z1 = float(z[j]), float(z[j + 1])
    tr = 0.0 if r1 == r0 else (float(r_value) - r0) / (r1 - r0)
    tz = 0.0 if z1 == z0 else (float(z_value) - z0) / (z1 - z0)
    return float(
        (1 - tr) * (1 - tz) * field[i, j]
        + tr * (1 - tz) * field[i + 1, j]
        + (1 - tr) * tz * field[i, j + 1]
        + tr * tz * field[i + 1, j + 1]
    )


def normalize_psi(psi: np.ndarray, *, psi_axis: float, psi_bndry: float) -> np.ndarray:
    """Normalize a 2D psi grid using axis and boundary flux values."""
    denominator = float(psi_bndry) - float(psi_axis)
    if not np.isfinite(denominator) or abs(denominator) < 1e-12:
        raise ValueError("Cannot normalize psi with non-finite or degenerate axis/boundary flux")
    return ((np.asarray(psi, dtype=np.float32) - float(psi_axis)) / denominator).astype(np.float32)


def _finite_lcfs_points(r_values: np.ndarray, z_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r = np.asarray(r_values, dtype=float).reshape(-1)
    z = np.asarray(z_values, dtype=float).reshape(-1)
    mask = np.isfinite(r) & np.isfinite(z)
    return r[mask], z[mask]


def _real_psi_norm(row: dict[str, Any]) -> np.ndarray:
    try:
        import zarr
    except ModuleNotFoundError as exc:
        raise RuntimeError("Reading real manifest rows requires the optional 'zarr' package") from exc

    root = zarr.open_group(str(Path(row["data_path"]).expanduser().resolve()), mode="r")
    equilibrium = root["equilibrium"]
    index = _nearest_index(equilibrium["time"][:], float(row["target_time"]))
    psi = _real_psi(row)
    r_grid = np.asarray(equilibrium["major_radius"][:], dtype=float)
    z_grid = np.asarray(equilibrium["z"][:], dtype=float)
    psi_axis = _interp_grid_value(
        r_grid,
        z_grid,
        psi,
        float(equilibrium["magnetic_axis_r"][index]),
        float(equilibrium["magnetic_axis_z"][index]),
    )
    lcfs_r, lcfs_z = _finite_lcfs_points(equilibrium["lcfs_r"][:, index], equilibrium["lcfs_z"][:, index])
    if lcfs_r.size == 0:
        raise ValueError(f"No finite LCFS points for real sample {row.get('sample_id')!r}")
    boundary_values = [
        _interp_grid_value(r_grid, z_grid, psi, float(radius), float(vertical))
        for radius, vertical in zip(lcfs_r, lcfs_z)
        if r_grid[0] <= radius <= r_grid[-1] and z_grid[0] <= vertical <= z_grid[-1]
    ]
    if not boundary_values:
        raise ValueError(f"No LCFS points inside EFIT grid for real sample {row.get('sample_id')!r}")
    return _validate_psi(
        normalize_psi(psi, psi_axis=psi_axis, psi_bndry=float(np.nanmedian(boundary_values))),
        row,
    )


def _synthetic_psi(row: dict[str, Any]) -> np.ndarray:
    equilibrium_path = row.get("equilibrium_path")
    if not equilibrium_path:
        data_path = Path(row["data_path"]).expanduser().resolve()
        equilibrium_path = data_path / "equilibrium.npz"
    with np.load(Path(equilibrium_path).expanduser().resolve()) as equilibrium:
        psi = np.asarray(equilibrium["psi"], dtype=np.float32)
    return _validate_psi(psi, row)


def _synthetic_psi_norm(row: dict[str, Any]) -> np.ndarray:
    equilibrium_path = row.get("equilibrium_path")
    if not equilibrium_path:
        data_path = Path(row["data_path"]).expanduser().resolve()
        equilibrium_path = data_path / "equilibrium.npz"
    with np.load(Path(equilibrium_path).expanduser().resolve()) as equilibrium:
        psi = np.asarray(equilibrium["psi"], dtype=np.float32)
        psi_axis = float(equilibrium["psi_axis"])
        psi_bndry = float(equilibrium["psi_bndry"])
    return _validate_psi(normalize_psi(psi, psi_axis=psi_axis, psi_bndry=psi_bndry), row)


def _validate_psi(psi: np.ndarray, row: dict[str, Any]) -> np.ndarray:
    if psi.shape != (65, 65):
        raise ValueError(f"Expected 65x65 psi for {row.get('sample_id')!r}, got {psi.shape}")
    if not np.isfinite(psi).all():
        raise ValueError(f"Non-finite psi values in {row.get('sample_id')!r}")
    return psi


def _row_parameters(row: dict[str, Any]) -> dict[str, Any]:
    if row.get("source") == "real":
        params = _fit_parameters_for_row(row)
        params["coil_currents"] = {"active": _active_currents_from_real_row(row)}
        return params
    return row


def _coil_names(rows: Iterable[dict[str, Any]]) -> list[str]:
    names: set[str] = set()
    for row in rows:
        params = _row_parameters(row)
        active = ((params.get("coil_currents") or {}).get("active") or {})
        names.update(str(name) for name in active)
    return sorted(names)


def _feature_names(rows: Iterable[dict[str, Any]]) -> list[str]:
    return (
        ["target_time", "Ip", "fvac"]
        + [f"alpha_{index}" for index in range(3)]
        + [f"beta_{index}" for index in range(3)]
        + [f"coil_active_{name}" for name in _coil_names(rows)]
    )


def _feature_vector(row: dict[str, Any], feature_names: list[str]) -> np.ndarray:
    params = _row_parameters(row)
    alpha = list(params.get("alpha") or [])
    beta = list(params.get("beta") or [])
    active = ((params.get("coil_currents") or {}).get("active") or {})

    values: dict[str, float] = {
        "target_time": float(row["target_time"]),
        "Ip": float(params["Ip"]),
        "fvac": float(params["fvac"]),
    }
    for index in range(3):
        values[f"alpha_{index}"] = float(alpha[index]) if index < len(alpha) else 0.0
        values[f"beta_{index}"] = float(beta[index]) if index < len(beta) else 0.0
    for name in active:
        values[f"coil_active_{name}"] = float(active[name])

    return np.asarray([values.get(name, 0.0) for name in feature_names], dtype=np.float32)


def _psi_for_row(row: dict[str, Any], target_mode: str = TARGET_RAW_PSI) -> np.ndarray:
    if target_mode not in TARGET_MODES:
        raise ValueError(f"Unknown target_mode {target_mode!r}; expected one of {sorted(TARGET_MODES)}")
    if row.get("source") == "real":
        return _real_psi_norm(row) if target_mode == TARGET_PSI_NORM else _real_psi(row)
    return _synthetic_psi_norm(row) if target_mode == TARGET_PSI_NORM else _synthetic_psi(row)


def _standardize(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((values - mean) / np.maximum(std, 1e-6)).astype(np.float32)


@dataclass
class ManifestWindowDataset:
    """Small manifest-backed dataset that emits MMT-compatible window dictionaries."""

    rows: list[dict[str, Any]]
    feature_names: list[str]
    input_mean: np.ndarray
    input_std: np.ndarray
    output_mean: np.ndarray
    output_std: np.ndarray
    target_mode: str = TARGET_RAW_PSI

    @classmethod
    def from_rows(
        cls,
        rows: list[dict[str, Any]],
        *,
        feature_names: list[str] | None = None,
        input_mean: np.ndarray | None = None,
        input_std: np.ndarray | None = None,
        output_mean: np.ndarray | None = None,
        output_std: np.ndarray | None = None,
        target_mode: str = TARGET_RAW_PSI,
    ) -> "ManifestWindowDataset":
        if not rows:
            raise ValueError("ManifestWindowDataset requires at least one row")
        if target_mode not in TARGET_MODES:
            raise ValueError(f"Unknown target_mode {target_mode!r}; expected one of {sorted(TARGET_MODES)}")

        features = feature_names or _feature_names(rows)
        input_matrix = np.stack([_feature_vector(row, features) for row in rows], axis=0)
        output_matrix = np.stack([_psi_for_row(row, target_mode).reshape(-1) for row in rows], axis=0)

        return cls(
            rows=list(rows),
            feature_names=list(features),
            input_mean=np.asarray(input_mean if input_mean is not None else input_matrix.mean(axis=0), dtype=np.float32),
            input_std=np.asarray(input_std if input_std is not None else input_matrix.std(axis=0), dtype=np.float32),
            output_mean=np.asarray(
                output_mean if output_mean is not None else output_matrix.mean(axis=0), dtype=np.float32
            ),
            output_std=np.asarray(output_std if output_std is not None else output_matrix.std(axis=0), dtype=np.float32),
            target_mode=target_mode,
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        features = _standardize(_feature_vector(row, self.feature_names), self.input_mean, self.input_std)
        psi = _standardize(_psi_for_row(row, self.target_mode).reshape(-1), self.output_mean, self.output_std)

        return {
            "shot_id": str(row.get("shot_id") or row.get("sample_id") or index),
            "window_index": int(index),
            "t_cut": float(row["target_time"]),
            "emb_chunks": [features],
            "pos": np.asarray([0], dtype=np.int32),
            "id": np.asarray([INPUT_SIGNAL_ID], dtype=np.int32),
            "mod": np.asarray([TIMESERIES_MODALITY_ID], dtype=np.int16),
            "role": np.asarray([ROLE_CONTEXT], dtype=np.int8),
            "signal_name": np.asarray(["fusion-state"], dtype=object),
            "output_emb": {OUTPUT_SIGNAL_ID: psi},
            "output_shapes": {OUTPUT_SIGNAL_ID: tuple(psi.shape)},
            "output_names": {OUTPUT_SIGNAL_ID: "equilibrium-psi"},
        }


def build_manifest_datasets(
    rows: list[dict[str, Any]],
    *,
    val_fraction: float = 0.2,
    seed: int = 54,
    target_mode: str = TARGET_RAW_PSI,
) -> tuple[ManifestWindowDataset, ManifestWindowDataset]:
    """Build train/validation datasets with shared feature and normalization statistics."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    if len(rows) < 2:
        raise ValueError("At least two manifest rows are required for train/val split")

    assignments = assign_parent_shot_splits(
        rows,
        train_fraction=1.0 - val_fraction,
        val_fraction=val_fraction,
        seed=seed,
    )
    train_rows = [row for row in rows if split_for_row(row, assignments) == "train"]
    val_rows = [row for row in rows if split_for_row(row, assignments) == "val"]
    if not train_rows:
        raise ValueError("Train split is empty")
    if not val_rows:
        raise ValueError("Validation split is empty")

    feature_names = _feature_names(rows)
    base = ManifestWindowDataset.from_rows(train_rows, feature_names=feature_names, target_mode=target_mode)
    val = ManifestWindowDataset.from_rows(
        val_rows,
        feature_names=feature_names,
        input_mean=base.input_mean,
        input_std=base.input_std,
        output_mean=base.output_mean,
        output_std=base.output_std,
        target_mode=target_mode,
    )
    return base, val
