from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from mast_bridge.simulation.magnetic_diagnostics import LEVEL2_FLUX_LOOP_SCALE


SCHEMA_VERSION = 1
REQUIRED_KEYS = {
    "schema_version",
    "target_time",
    "magnetics_ip",
    "flux_loop_names",
    "flux_loop_values",
    "pickup_names",
    "pickup_families",
    "pickup_values",
    "active_coil_names",
    "active_coil_values",
    "flux_loop_scale",
}


def _string_array(values: Iterable[str]) -> np.ndarray:
    items = [str(value) for value in values]
    width = max((len(value) for value in items), default=1)
    return np.asarray(items, dtype=f"U{width}")


def write_synthetic_diagnostics(
    path: str | Path,
    *,
    target_time: float,
    magnetics_ip: float,
    flux_loop_names: Iterable[str],
    flux_loop_values: Iterable[float],
    pickup_names: Iterable[str],
    pickup_families: Iterable[str],
    pickup_values: Iterable[float],
    active_coil_currents: Mapping[str, float],
    flux_loop_scale: float,
) -> Path:
    """Write one synthetic magnetic-diagnostics payload without pickle objects."""
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    active_names = sorted(str(name) for name in active_coil_currents)
    with tempfile.NamedTemporaryFile(
        dir=output.parent,
        prefix=f".{output.stem}-",
        suffix=".npz",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(
            temporary,
            schema_version=np.asarray(SCHEMA_VERSION, dtype=np.int16),
            target_time=np.asarray(float(target_time), dtype=np.float64),
            magnetics_ip=np.asarray(float(magnetics_ip), dtype=np.float64),
            flux_loop_names=_string_array(flux_loop_names),
            flux_loop_values=np.asarray(list(flux_loop_values), dtype=np.float64),
            pickup_names=_string_array(pickup_names),
            pickup_families=_string_array(pickup_families),
            pickup_values=np.asarray(list(pickup_values), dtype=np.float64),
            active_coil_names=_string_array(active_names),
            active_coil_values=np.asarray(
                [float(active_coil_currents[name]) for name in active_names],
                dtype=np.float64,
            ),
            flux_loop_scale=np.asarray(float(flux_loop_scale), dtype=np.float64),
        )
        reason = synthetic_diagnostics_rejection_reason(
            temporary,
            expected_target_time=target_time,
        )
        if reason is not None:
            raise ValueError(f"Invalid synthetic diagnostics: {reason}")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _one_dimensional(payload: Any, key: str) -> np.ndarray:
    values = np.asarray(payload[key])
    if values.ndim != 1:
        raise ValueError("diagnostics_shape_mismatch")
    return values


def _scalar(payload: Any, key: str) -> float:
    value = np.asarray(payload[key])
    if value.shape != ():
        raise ValueError("diagnostics_shape_mismatch")
    return float(value.item())


def _validated_payload(
    path: str | Path,
    *,
    expected_target_time: float | None = None,
) -> dict[str, Any]:
    diagnostics_path = Path(path).expanduser().resolve()
    if not diagnostics_path.is_file():
        raise ValueError("diagnostics_missing")
    try:
        with np.load(diagnostics_path, allow_pickle=False) as payload:
            if not REQUIRED_KEYS.issubset(payload.files):
                raise ValueError("diagnostics_keys_missing")
            if int(np.asarray(payload["schema_version"]).item()) != SCHEMA_VERSION:
                raise ValueError("diagnostics_schema_unsupported")

            raw_flux_names = _one_dimensional(payload, "flux_loop_names")
            raw_pickup_names = _one_dimensional(payload, "pickup_names")
            raw_pickup_families = _one_dimensional(payload, "pickup_families")
            raw_active_names = _one_dimensional(payload, "active_coil_names")
            if any(
                values.dtype.kind != "U"
                for values in (
                    raw_flux_names,
                    raw_pickup_names,
                    raw_pickup_families,
                    raw_active_names,
                )
            ):
                raise ValueError("diagnostics_channel_schema_invalid")

            flux_names = raw_flux_names.astype(str)
            flux_values = _one_dimensional(payload, "flux_loop_values").astype(float)
            pickup_names = raw_pickup_names.astype(str)
            pickup_families = raw_pickup_families.astype(str)
            pickup_values = _one_dimensional(payload, "pickup_values").astype(float)
            active_names = raw_active_names.astype(str)
            active_values = _one_dimensional(payload, "active_coil_values").astype(float)

            if (
                flux_names.size != flux_values.size
                or pickup_names.size != pickup_values.size
                or pickup_families.size != pickup_values.size
                or active_names.size != active_values.size
            ):
                raise ValueError("diagnostics_shape_mismatch")

            if (
                flux_names.size == 0
                or pickup_names.size == 0
                or active_names.size == 0
                or any(not str(name).strip() for name in flux_names)
                or any(not str(name).strip() for name in pickup_names)
                or any(not str(name).strip() for name in active_names)
                or any(
                    str(family) not in {"CCBV", "OBR", "OBV"}
                    for family in pickup_families
                )
            ):
                raise ValueError("diagnostics_channel_schema_invalid")

            pickup_keys = [
                (str(family), str(name))
                for family, name in zip(pickup_families, pickup_names)
            ]
            if (
                len(set(flux_names.tolist())) != flux_names.size
                or len(set(pickup_keys)) != len(pickup_keys)
                or len(set(active_names.tolist())) != active_names.size
            ):
                raise ValueError("diagnostics_duplicate_channels")

            target_time = _scalar(payload, "target_time")
            magnetics_ip = _scalar(payload, "magnetics_ip")
            flux_loop_scale = _scalar(payload, "flux_loop_scale")
            if (
                expected_target_time is not None
                and not np.isclose(
                    target_time,
                    float(expected_target_time),
                    rtol=0.0,
                    atol=1e-12,
                )
            ):
                raise ValueError("diagnostics_target_time_mismatch")
            if not np.isclose(
                flux_loop_scale,
                LEVEL2_FLUX_LOOP_SCALE,
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError("diagnostics_flux_loop_scale_invalid")
            numeric = np.concatenate(
                [
                    np.asarray([target_time, magnetics_ip, flux_loop_scale]),
                    flux_values,
                    pickup_values,
                    active_values,
                ]
            )
            if not np.isfinite(numeric).all():
                raise ValueError("diagnostics_nonfinite")

            return {
                "target_time": target_time,
                "magnetics_ip": magnetics_ip,
                "flux_loop_names": flux_names.tolist(),
                "flux_loop_values": flux_values.tolist(),
                "pickup_names": pickup_names.tolist(),
                "pickup_families": pickup_families.tolist(),
                "pickup_values": pickup_values.tolist(),
                "active_coil_names": active_names.tolist(),
                "active_coil_values": active_values.tolist(),
                "flux_loop_scale": flux_loop_scale,
            }
    except (OSError, KeyError, TypeError) as exc:
        raise ValueError("diagnostics_unreadable") from exc


def synthetic_diagnostics_rejection_reason(
    path: str | Path,
    *,
    expected_target_time: float | None = None,
) -> str | None:
    """Return a stable reason for an invalid diagnostics payload."""
    try:
        _validated_payload(path, expected_target_time=expected_target_time)
    except ValueError as exc:
        return str(exc)
    return None


def load_synthetic_diagnostic_values(path: str | Path) -> dict[str, float]:
    """Load a synthetic payload into the feature names used by TokaMind."""
    payload = _validated_payload(path)
    values: dict[str, float] = {
        "target_time": payload["target_time"],
        "magnetics_ip": payload["magnetics_ip"],
    }
    values.update(
        {
            f"flux_loop_{name}": float(value)
            for name, value in zip(
                payload["flux_loop_names"], payload["flux_loop_values"]
            )
        }
    )
    values.update(
        {
            f"pickup_{family}_{name}": float(value)
            for family, name, value in zip(
                payload["pickup_families"],
                payload["pickup_names"],
                payload["pickup_values"],
            )
        }
    )
    values.update(
        {
            f"coil_active_{name}": float(value)
            for name, value in zip(
                payload["active_coil_names"], payload["active_coil_values"]
            )
        }
    )
    return values
