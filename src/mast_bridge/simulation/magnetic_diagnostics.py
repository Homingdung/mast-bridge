from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


COMPARISON_DTYPE = [
    ("diagnostic_type", "U16"),
    ("channel", "U64"),
    ("model", "f8"),
    ("observed", "f8"),
    ("error", "f8"),
    ("abs_error", "f8"),
]

# FreeGSNKE/EFIT psi is Wb/(2*pi); MAST Level 2 flux_loop_flux is Wb.
LEVEL2_FLUX_LOOP_SCALE = 2.0 * np.pi

PICKUP_FAMILIES = (
    ("CCBV", "b_field_pol_probe_ccbv"),
    ("OBR", "b_field_pol_probe_obr"),
    ("OBV", "b_field_pol_probe_obv"),
)

MAST_LEVEL2_PICKUP_ORIENTATIONS = {
    "CCBV": np.asarray([0.0, 0.0, 1.0]),
    "OBR": np.asarray([1.0, 0.0, 0.0]),
    "OBV": np.asarray([0.0, 0.0, 1.0]),
}


@dataclass(frozen=True)
class NamedSignals:
    names: list[str]
    values: np.ndarray
    families: list[str] | None = None


def correct_mast_level2_pickup_orientations(probe_payload: dict[str, Any]) -> None:
    """Mutate a FreeGSNKE magnetic-probe payload to match MAST Level 2 fields."""
    for pickup in probe_payload.get("pickups", []):
        family = str(pickup.get("family", ""))
        if family in MAST_LEVEL2_PICKUP_ORIENTATIONS:
            pickup["orientation_vector"] = MAST_LEVEL2_PICKUP_ORIENTATIONS[family].copy()


def mast_level2_flux_loop_geometry_name(channel: str) -> str:
    # Level 2 stores measured channels separately from full geometry channels.
    # The reliable join key is the normalized channel name, not array index.
    return "FL_" + str(channel).replace("/", "_")


def correct_mast_level2_flux_loop_positions(
    probe_payload: dict[str, Any], magnetics: Any
) -> None:
    geometry_channels = [str(value) for value in array_values(magnetics, "flux_loop_geometry_channel")]
    radii = array_values(magnetics, "flux_loop_r")
    verticals = array_values(magnetics, "flux_loop_z")
    geometry_by_name = {
        name: np.asarray([radii[index], verticals[index]], dtype=float)
        for index, name in enumerate(geometry_channels)
        if index < radii.size and index < verticals.size
    }
    for flux_loop in probe_payload.get("flux_loops", []):
        source_channel = flux_loop.get("source_signal_channel")
        if source_channel is not None:
            geometry_name = mast_level2_flux_loop_geometry_name(str(source_channel))
        elif flux_loop.get("measurement_status") == "virtual":
            geometry_name = str(flux_loop.get("geometry_name", ""))
        else:
            # Compatibility with older measured-only machine pickles.
            geometry_name = mast_level2_flux_loop_geometry_name(
                str(flux_loop.get("name", ""))
            )
        if geometry_name not in geometry_by_name:
            continue
        flux_loop["geometry_name"] = geometry_name
        flux_loop["position"] = geometry_by_name[geometry_name].copy()


def array_values(group: Any, key: str) -> np.ndarray:
    values = group[key]
    if hasattr(values, "__getitem__") and not isinstance(values, np.ndarray):
        try:
            values = values[:]
        except (TypeError, ValueError, AttributeError):
            pass
    return np.asarray(values)


def interpolate_channel_rows(
    times: Any, values: Any, target_time: float
) -> np.ndarray:
    time_values = np.asarray(times, dtype=float)
    signal_values = np.asarray(values, dtype=float)
    if signal_values.ndim == 1:
        return np.asarray([np.interp(target_time, time_values, signal_values)])
    if signal_values.ndim != 2:
        raise ValueError(f"Expected 1D or 2D diagnostic signal, got {signal_values.ndim}D")
    if signal_values.shape[1] == time_values.size:
        rows = signal_values
    elif signal_values.shape[0] == time_values.size:
        rows = signal_values.T
    else:
        raise ValueError(
            "Diagnostic signal shape does not align with time axis: "
            f"{signal_values.shape} vs {time_values.size}"
        )
    return np.asarray([np.interp(target_time, time_values, row) for row in rows])


def observed_flux_loop_signals(magnetics: Any, target_time: float) -> NamedSignals:
    names = [str(value) for value in array_values(magnetics, "flux_loop_channel")]
    values = interpolate_channel_rows(
        array_values(magnetics, "time"),
        array_values(magnetics, "flux_loop_flux"),
        target_time,
    )
    return NamedSignals(names=names, values=values)


def observed_pickup_signals(magnetics: Any, target_time: float) -> NamedSignals:
    names: list[str] = []
    values: list[float] = []
    families: list[str] = []
    times = array_values(magnetics, "time")
    for family, prefix in PICKUP_FAMILIES:
        channel_key = f"{prefix}_channel"
        field_key = f"{prefix}_field"
        if channel_key not in magnetics or field_key not in magnetics:
            continue
        family_names = [str(value) for value in array_values(magnetics, channel_key)]
        family_values = interpolate_channel_rows(
            times, array_values(magnetics, field_key), target_time
        )
        names.extend(family_names)
        values.extend(float(value) for value in family_values)
        families.extend([family] * len(family_names))
    return NamedSignals(names=names, values=np.asarray(values, dtype=float), families=families)


def observed_plasma_current(magnetics: Any, target_time: float) -> float:
    """Return Level 2 magnetics/ip at the target time."""
    return float(
        np.interp(
            target_time,
            np.asarray(array_values(magnetics, "time"), dtype=float),
            np.asarray(array_values(magnetics, "ip"), dtype=float),
        )
    )


def current_constraint_comparison(
    *, model_ip: float, magnetics: Any, target_time: float
) -> dict[str, float | str]:
    observed = observed_plasma_current(magnetics, target_time)
    model = float(model_ip)
    error = model - observed
    denominator = abs(observed)
    relative_error = error / denominator if denominator > 0 else float("nan")
    return {
        "diagnostic_type": "plasma_current",
        "channel": "ip",
        "model": model,
        "observed": observed,
        "error": error,
        "abs_error": abs(error),
        "relative_error": relative_error,
    }


def probe_order(probes: Any, attr: str, fallback_key: str) -> list[str]:
    if hasattr(probes, attr):
        return [str(value) for value in getattr(probes, attr)]
    return [str(item["name"]) for item in getattr(probes, fallback_key)]


def modeled_flux_loop_signals(
    tokamak: Any, eq: Any, scale: float = LEVEL2_FLUX_LOOP_SCALE
) -> NamedSignals:
    probes = tokamak.probes
    # calculate_fluxloop_value returns psi in FreeGSNKE units; scale before
    # comparing with Level 2 flux_loop_flux.
    values = np.asarray(probes.calculate_fluxloop_value(eq), dtype=float) * float(scale)
    names = probe_order(probes, "floop_order", "floops")
    return NamedSignals(names=names, values=values)


def modeled_pickup_signals(tokamak: Any, eq: Any) -> NamedSignals:
    probes = tokamak.probes
    values = np.asarray(probes.calculate_pickup_value(eq), dtype=float)
    names = probe_order(probes, "pickup_order", "pickups")
    families = [
        str(item.get("family", "")) for item in getattr(probes, "pickups", [])
    ] or None
    return NamedSignals(names=names, values=values, families=families)


def compare_named_signals(
    *,
    diagnostic_type: str,
    model_names: Iterable[str],
    model_values: Any,
    observed_names: Iterable[str],
    observed_values: Any,
) -> np.ndarray:
    observed_by_name = {
        str(name): float(value)
        for name, value in zip(observed_names, np.asarray(observed_values, dtype=float))
        if np.isfinite(value)
    }
    rows: list[tuple[str, str, float, float, float, float]] = []
    for name, model_value in zip(model_names, np.asarray(model_values, dtype=float)):
        channel = str(name)
        model = float(model_value)
        if channel not in observed_by_name or not np.isfinite(model):
            continue
        observed = observed_by_name[channel]
        error = model - observed
        rows.append((diagnostic_type, channel, model, observed, error, abs(error)))
    return np.asarray(rows, dtype=COMPARISON_DTYPE)


def combine_comparisons(*comparisons: np.ndarray) -> np.ndarray:
    non_empty = [rows for rows in comparisons if rows.size]
    if not non_empty:
        return np.asarray([], dtype=COMPARISON_DTYPE)
    return np.concatenate(non_empty).astype(COMPARISON_DTYPE, copy=False)


def _error_metrics(rows: np.ndarray) -> dict[str, float | int | None]:
    count = int(rows.size)
    if count == 0:
        return {
            "count": 0,
            "mean_error": None,
            "mean_abs_error": None,
            "max_abs_error": None,
            "rmse": None,
        }
    return {
        "count": count,
        "mean_error": float(np.mean(rows["error"])),
        "mean_abs_error": float(np.mean(rows["abs_error"])),
        "max_abs_error": float(np.max(rows["abs_error"])),
        "rmse": float(np.sqrt(np.mean(rows["error"] ** 2))),
    }


def summarize_comparisons(rows: np.ndarray) -> dict[str, Any]:
    summary: dict[str, Any] = {"total": _error_metrics(rows), "by_type": {}}
    for diagnostic_type in sorted(set(str(value) for value in rows["diagnostic_type"])):
        mask = rows["diagnostic_type"] == diagnostic_type
        summary["by_type"][diagnostic_type] = _error_metrics(rows[mask])
    return summary


def build_magnetic_diagnostic_comparison(
    tokamak: Any,
    eq: Any,
    magnetics: Any,
    target_time: float,
    *,
    flux_loop_scale: float = LEVEL2_FLUX_LOOP_SCALE,
) -> np.ndarray:
    model_flux = modeled_flux_loop_signals(tokamak, eq, scale=flux_loop_scale)
    observed_flux = observed_flux_loop_signals(magnetics, target_time)
    flux_rows = compare_named_signals(
        diagnostic_type="flux_loop",
        model_names=model_flux.names,
        model_values=model_flux.values,
        observed_names=observed_flux.names,
        observed_values=observed_flux.values,
    )

    model_pickups = modeled_pickup_signals(tokamak, eq)
    observed_pickups = observed_pickup_signals(magnetics, target_time)
    pickup_rows = compare_named_signals(
        diagnostic_type="pickup",
        model_names=model_pickups.names,
        model_values=model_pickups.values,
        observed_names=observed_pickups.names,
        observed_values=observed_pickups.values,
    )

    return combine_comparisons(flux_rows, pickup_rows)
