from __future__ import annotations

import concurrent.futures
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from mast_bridge.dataset.splits import assign_parent_shot_splits, split_for_row
from mast_bridge.simulation.magnetic_diagnostics import (
    observed_flux_loop_signals,
    observed_pickup_signals,
    observed_plasma_current,
)
from mast_bridge.simulation.synthetic_diagnostics import (
    load_synthetic_diagnostic_values,
)


INPUT_SIGNAL_ID = 0
OUTPUT_SIGNAL_ID = 1
TIMESERIES_MODALITY_ID = 0
ROLE_CONTEXT = 0
TARGET_RAW_PSI = "raw-psi"
TARGET_PSI_NORM = "psi-norm"
TARGET_MODES = {TARGET_RAW_PSI, TARGET_PSI_NORM}
INPUT_LAO_PARAMS = "lao-params"
INPUT_MAGNETIC_DIAGNOSTICS = "magnetic-diagnostics"
INPUT_MODES = {INPUT_LAO_PARAMS, INPUT_MAGNETIC_DIAGNOSTICS}
FEATURE_SCHEMA_VERSION = 1


def load_manifest_rows(path: str | Path) -> list[dict[str, Any]]:
    """Load JSONL manifest rows."""
    manifest_path = Path(path).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def feature_schema_digest(feature_names: Iterable[str]) -> str:
    return hashlib.sha256(
        "\n".join(str(name) for name in feature_names).encode("utf-8")
    ).hexdigest()


def load_feature_schema(path: str | Path) -> list[str]:
    """Load and integrity-check a versioned feature-name schema."""
    schema_path = Path(path).expanduser().resolve()
    payload = json.loads(schema_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported feature schema version in {schema_path}")
    names = payload.get("feature_names")
    if (
        not isinstance(names, list)
        or not names
        or any(not isinstance(name, str) or not name for name in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError(f"Invalid feature names in {schema_path}")
    if payload.get("feature_count") != len(names):
        raise ValueError(f"Feature schema count mismatch in {schema_path}")
    if payload.get("feature_names_sha256") != feature_schema_digest(names):
        raise ValueError(f"Feature schema digest mismatch in {schema_path}")
    return names


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
    times = np.asarray(active["time"][:], dtype=float)
    channels = [str(value) for value in active["current_channel"][:]]
    currents = np.asarray(active["coil_current"][:], dtype=float)
    return {
        channel: float(
            np.interp(float(row["target_time"]), times, currents[channel_index])
        )
        for channel_index, channel in enumerate(channels)
    }


def _real_magnetics_group(row: dict[str, Any]) -> Any:
    try:
        import zarr
    except ModuleNotFoundError as exc:
        raise RuntimeError("Reading real manifest rows requires the optional 'zarr' package") from exc

    root = zarr.open_group(str(Path(row["data_path"]).expanduser().resolve()), mode="r")
    if "magnetics" not in root:
        raise ValueError(f"Real row {row.get('sample_id')!r} is missing magnetics group")
    return root["magnetics"]


def _real_psi(row: dict[str, Any]) -> np.ndarray:
    try:
        import zarr
    except ModuleNotFoundError as exc:
        raise RuntimeError("Reading real manifest rows requires the optional 'zarr' package") from exc

    root = zarr.open_group(str(Path(row["data_path"]).expanduser().resolve()), mode="r")
    equilibrium = root["equilibrium"]
    index = _nearest_index(equilibrium["time"][:], float(row["target_time"]))
    # Downloaded MAST Level-2 psi is stored as [Z, R, time].  The bridge and
    # FreeGSNKE use [R, Z], so convert at the real-data boundary.
    psi_zr = np.asarray(equilibrium["psi"][:, :, index], dtype=np.float32)
    psi = np.ascontiguousarray(psi_zr.T)
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


def _lao_feature_names(rows: Iterable[dict[str, Any]]) -> list[str]:
    return (
        ["target_time", "Ip", "fvac"]
        + [f"alpha_{index}" for index in range(3)]
        + [f"beta_{index}" for index in range(3)]
        + [f"coil_active_{name}" for name in _coil_names(rows)]
    )


def _lao_feature_vector(row: dict[str, Any], feature_names: list[str]) -> np.ndarray:
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


def _real_diagnostic_values(row: dict[str, Any]) -> dict[str, float]:
    target_time = float(row["target_time"])
    magnetics = _real_magnetics_group(row)
    flux = observed_flux_loop_signals(magnetics, target_time)
    pickups = observed_pickup_signals(magnetics, target_time)
    active = _active_currents_from_real_row(row)
    values: dict[str, float] = {
        "target_time": target_time,
        "magnetics_ip": observed_plasma_current(magnetics, target_time),
    }
    values.update({f"flux_loop_{name}": float(value) for name, value in zip(flux.names, flux.values)})
    if pickups.families is None:
        values.update({f"pickup_{name}": float(value) for name, value in zip(pickups.names, pickups.values)})
    else:
        values.update(
            {
                f"pickup_{family}_{name}": float(value)
                for family, name, value in zip(pickups.families, pickups.names, pickups.values)
            }
        )
    values.update({f"coil_active_{name}": float(value) for name, value in active.items()})
    return values


def _synthetic_diagnostic_values(row: dict[str, Any]) -> dict[str, float]:
    diagnostics_path = row.get("diagnostics_path")
    if not diagnostics_path:
        diagnostics_path = (
            Path(row["data_path"]).expanduser().resolve() / "diagnostics.npz"
        )
    return load_synthetic_diagnostic_values(diagnostics_path)


def _diagnostic_values(row: dict[str, Any]) -> dict[str, float]:
    if row.get("source") == "real":
        return _real_diagnostic_values(row)
    if row.get("source") == "synthetic":
        return _synthetic_diagnostic_values(row)
    raise ValueError(
        f"Unknown diagnostic row source {row.get('source')!r} for "
        f"{row.get('sample_id')!r}"
    )


def diagnostic_feature_names(rows: Iterable[dict[str, Any]]) -> list[str]:
    row_values = [_diagnostic_values(row) for row in rows]
    names: set[str] = {"target_time", "magnetics_ip"}
    for values in row_values:
        names.update(values)
    finite_names = {
        name
        for name in names
        if all(np.isfinite(values.get(name, np.nan)) for values in row_values)
    }
    flux_names = sorted(name for name in names if name.startswith("flux_loop_"))
    pickup_names = sorted(name for name in names if name.startswith("pickup_"))
    coil_names = sorted(name for name in names if name.startswith("coil_active_"))
    extra_names = sorted(
        name
        for name in names
        if name not in {"target_time", "magnetics_ip"}
        and not name.startswith(("flux_loop_", "pickup_", "coil_active_"))
    )
    ordered = ["target_time", "magnetics_ip"] + flux_names + pickup_names + coil_names + extra_names
    return [name for name in ordered if name in finite_names]


def diagnostic_feature_vector(row: dict[str, Any], feature_names: list[str]) -> np.ndarray:
    values = _diagnostic_values(row)
    entries = []
    for name in feature_names:
        if name not in values and name.startswith("coil_active_"):
            entries.append(0.0)
        else:
            entries.append(values.get(name, np.nan))
    vector = np.asarray(entries, dtype=np.float32)
    if not np.isfinite(vector).all():
        bad = [feature_names[index] for index in np.flatnonzero(~np.isfinite(vector))]
        raise ValueError(f"Non-finite diagnostic features for {row.get('sample_id')!r}: {bad}")
    return vector


def _feature_names(rows: Iterable[dict[str, Any]], input_mode: str = INPUT_LAO_PARAMS) -> list[str]:
    if input_mode == INPUT_LAO_PARAMS:
        return _lao_feature_names(rows)
    if input_mode == INPUT_MAGNETIC_DIAGNOSTICS:
        return diagnostic_feature_names(rows)
    raise ValueError(f"Unknown input_mode {input_mode!r}; expected one of {sorted(INPUT_MODES)}")


def feature_names_for_rows(
    rows: Iterable[dict[str, Any]],
    input_mode: str = INPUT_LAO_PARAMS,
) -> list[str]:
    """Return the ordered finite feature schema for a reference row set."""
    return _feature_names(rows, input_mode)


def _feature_vector(row: dict[str, Any], feature_names: list[str], input_mode: str = INPUT_LAO_PARAMS) -> np.ndarray:
    if input_mode == INPUT_LAO_PARAMS:
        return _lao_feature_vector(row, feature_names)
    if input_mode == INPUT_MAGNETIC_DIAGNOSTICS:
        return diagnostic_feature_vector(row, feature_names)
    raise ValueError(f"Unknown input_mode {input_mode!r}; expected one of {sorted(INPUT_MODES)}")


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
    features: np.ndarray = None
    targets: np.ndarray = None
    target_mode: str = TARGET_RAW_PSI
    input_mode: str = INPUT_LAO_PARAMS

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
        input_mode: str = INPUT_LAO_PARAMS,
        cached_features: np.ndarray | None = None,
        cached_targets: np.ndarray | None = None,
    ) -> "ManifestWindowDataset":
        if not rows:
            raise ValueError("ManifestWindowDataset requires at least one row")
        if target_mode not in TARGET_MODES:
            raise ValueError(f"Unknown target_mode {target_mode!r}; expected one of {sorted(TARGET_MODES)}")
        if input_mode not in INPUT_MODES:
            raise ValueError(f"Unknown input_mode {input_mode!r}; expected one of {sorted(INPUT_MODES)}")

        features = feature_names or _feature_names(rows, input_mode)
        if cached_features is not None and cached_targets is not None:
            if len(cached_features) != len(rows) or len(cached_targets) != len(rows):
                raise ValueError("Cached dataset size does not match rows")
            input_matrix = cached_features
            output_matrix = cached_targets
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
                feature_vectors = list(
                    pool.map(lambda row: _feature_vector(row, features, input_mode), rows)
                )
                psi_vectors = list(
                    pool.map(lambda row: _psi_for_row(row, target_mode).reshape(-1), rows)
                )
            input_matrix = np.stack(feature_vectors, axis=0)
            output_matrix = np.stack(psi_vectors, axis=0)

        input_mean = np.asarray(input_mean if input_mean is not None else input_matrix.mean(axis=0), dtype=np.float32)
        input_std = np.asarray(input_std if input_std is not None else input_matrix.std(axis=0), dtype=np.float32)
        output_mean = np.asarray(
            output_mean if output_mean is not None else output_matrix.mean(axis=0), dtype=np.float32
        )
        output_std = np.asarray(
            output_std if output_std is not None else output_matrix.std(axis=0), dtype=np.float32
        )
        return cls(
            rows=list(rows),
            feature_names=list(features),
            input_mean=input_mean,
            input_std=input_std,
            output_mean=output_mean,
            output_std=output_std,
            features=_standardize(input_matrix, input_mean, input_std).astype(np.float32),
            targets=_standardize(output_matrix, output_mean, output_std).astype(np.float32),
            target_mode=target_mode,
            input_mode=input_mode,
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        features = self.features[index]
        psi = self.targets[index]
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


def _load_dataset_cache(cache_path: str | Path, rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray] | None:
    """Load pre-extracted (features, psi) matrices when sample order matches."""
    import os

    path = Path(cache_path).expanduser().resolve()
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as data:
        cached_ids = [str(value) for value in data["sample_ids"].tolist()]
        row_ids = [str(row.get("sample_id")) for row in rows]
        if cached_ids != row_ids:
            return None
        features = np.asarray(data["features"], dtype=np.float32)
        psi = np.asarray(data["psi"], dtype=np.float32)
    return features, psi


def build_manifest_datasets(
    rows: list[dict[str, Any]],
    *,
    val_fraction: float = 0.2,
    seed: int = 54,
    val_shots: list[str] | None = None,
    target_mode: str = TARGET_RAW_PSI,
    input_mode: str = INPUT_LAO_PARAMS,
    feature_names: list[str] | None = None,
    cache_path: str | Path | None = None,
) -> tuple[ManifestWindowDataset, ManifestWindowDataset]:
    """Build train/validation datasets with shared feature and normalization statistics."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    if len(rows) < 2:
        raise ValueError("At least two manifest rows are required for train/val split")

    if val_shots is None:
        assignments = assign_parent_shot_splits(
            rows,
            train_fraction=1.0 - val_fraction,
            val_fraction=val_fraction,
            seed=seed,
        )
        train_rows = [row for row in rows if split_for_row(row, assignments) == "train"]
        val_rows = [row for row in rows if split_for_row(row, assignments) == "val"]
    else:
        requested = {str(shot) for shot in val_shots}
        if not requested:
            raise ValueError("val_shots cannot be empty")

        def parent_shot(row: dict[str, Any]) -> str:
            return str(
                row.get("parent_shot")
                if row.get("source") == "synthetic"
                else row.get("shot_id")
            )

        available = {parent_shot(row) for row in rows}
        missing = sorted(requested - available)
        if missing:
            raise ValueError(f"Validation shots are missing from manifest: {missing}")
        train_rows = [row for row in rows if parent_shot(row) not in requested]
        val_rows = [row for row in rows if parent_shot(row) in requested]
    if not train_rows:
        raise ValueError("Train split is empty")
    if not val_rows:
        raise ValueError("Validation split is empty")

    resolved_feature_names = feature_names or _feature_names(rows, input_mode)
    cached = _load_dataset_cache(cache_path, rows) if cache_path else None
    if cached is not None:
        cache_features, cache_psi = cached
        all_rows = rows
        train_index = {id(row): i for i, row in enumerate(all_rows)}
        train_idx = [train_index[id(r)] for r in train_rows]
        val_idx = [train_index[id(r)] for r in val_rows]
        base = ManifestWindowDataset.from_rows(
            train_rows,
            feature_names=resolved_feature_names,
            target_mode=target_mode,
            input_mode=input_mode,
            cached_features=cache_features[train_idx],
            cached_targets=cache_psi[train_idx],
        )
        val = ManifestWindowDataset.from_rows(
            val_rows,
            feature_names=resolved_feature_names,
            input_mean=base.input_mean,
            input_std=base.input_std,
            output_mean=base.output_mean,
            output_std=base.output_std,
            target_mode=target_mode,
            input_mode=input_mode,
            cached_features=cache_features[val_idx],
            cached_targets=cache_psi[val_idx],
        )
        return base, val
    base = ManifestWindowDataset.from_rows(
        train_rows,
        feature_names=resolved_feature_names,
        target_mode=target_mode,
        input_mode=input_mode,
    )
    val = ManifestWindowDataset.from_rows(
        val_rows,
        feature_names=resolved_feature_names,
        input_mean=base.input_mean,
        input_std=base.input_std,
        output_mean=base.output_mean,
        output_std=base.output_std,
        target_mode=target_mode,
        input_mode=input_mode,
    )
    return base, val
