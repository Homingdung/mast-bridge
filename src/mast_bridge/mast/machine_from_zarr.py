from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np


ACTIVE_GROUPS = {
    "Solenoid": ("SOL", "sol"),
    "P2IL": ("P2IL FEED", "p2_inner_lower"),
    "P2IU": ("P2IU FEED", "p2_inner_upper"),
    "P2OL": ("P2OL FEED", "p2_outer_lower"),
    "P2OU": ("P2OU FEED", "p2_outer_upper"),
    "P3L": ("P3L FEED", "p3_lower"),
    "P3U": ("P3U FEED", "p3_upper"),
    "P4L": ("P4L FEED", "p4_lower"),
    "P4U": ("P4U FEED", "p4_upper"),
    "P5L": ("P5L FEED", "p5_lower"),
    "P5U": ("P5U FEED", "p5_upper"),
    "P6L": ("P6L", "p6_lower"),
    "P6U": ("P6U", "p6_upper"),
}

ACTIVE_RESISTIVITY = 1.55e-8
PASSIVE_RESISTIVITY = 7.1e-7
PASSIVE_SUM_FAMILIES = frozenset({"botcol", "topcol"})

REQUIRED_OUTPUTS = {
    "active_coils": "MAST_active_coils.pickle",
    "limiter": "MAST_limiter.pickle",
    "magnetic_probes": "MAST_magentic_probes.pickle",
    "passive_coils": "MAST_passive_coils.pickle",
    "wall": "MAST_wall.pickle",
}


def _group(root: Any, name: str) -> Any:
    if name not in root:
        raise ValueError(f"Shot Zarr is missing required group: {name}")
    return root[name]


def _values(group: Any, name: str) -> list[Any]:
    if name not in group:
        raise ValueError(f"Shot Zarr group {group.name!r} is missing signal: {name}")
    values = np.asarray(group[name][:])
    return values.tolist() if values.ndim else [values.item()]


def _optional_values(group: Any, name: str, length: int, default: Any) -> list[Any]:
    if name not in group:
        return [default] * length
    values = _values(group, name)
    if len(values) not in (1, length):
        raise ValueError(f"Signal {name} has incompatible geometry length")
    return values


def _scalar(values: list[Any], name: str) -> Any:
    if not values:
        raise ValueError(f"Signal {name} is empty")
    first = values[0]
    if not all(np.isclose(value, first) for value in values):
        raise ValueError(f"Signal {name} must be constant across geometry elements")
    return first


def _active_payload(root: Any) -> dict[str, dict[str, Any]]:
    group = _group(root, "pf_active")
    channels = _values(group, "current_channel")
    by_channel = {str(channel): index for index, channel in enumerate(channels)}
    payload: dict[str, dict[str, Any]] = {}

    for name, (channel, geometry) in ACTIVE_GROUPS.items():
        if channel not in by_channel:
            continue
        r = _values(group, f"{geometry}_r")
        z = _values(group, f"{geometry}_z")
        widths = _values(group, f"{geometry}_width")
        heights = _values(group, f"{geometry}_height")
        if len(widths) != len(r) or len(heights) != len(r):
            raise ValueError(f"{geometry} geometry and size signals have different lengths")
        common = {
            "polarity": 1,
            "resistivity": ACTIVE_RESISTIVITY,
            "multiplier": 1,
            "source_channel": channel,
            "source_geometry": geometry,
        }
        if len(set(widths)) == 1 and len(set(heights)) == 1:
            payload[name] = {
                "R": r,
                "Z": z,
                "dR": widths[0],
                "dZ": heights[0],
                **common,
            }
        else:
            payload[name] = {
                f"{geometry}_{index}": {
                    "R": [r[index]],
                    "Z": [z[index]],
                    "dR": widths[index],
                    "dZ": heights[index],
                    **common,
                }
                for index in range(len(r))
            }
    return payload


def _passive_payload(root: Any) -> list[dict[str, Any]]:
    group = _group(root, "pf_passive")
    geometry_groups = sorted(
        name.removesuffix("_geometry_channel")
        for name in group.array_keys()
        if name.endswith("_geometry_channel")
    )
    payload: list[dict[str, Any]] = []
    for geometry in geometry_groups:
        r = _values(group, f"{geometry}_r")
        z = _values(group, f"{geometry}_z")
        width = _values(group, f"{geometry}_width")
        height = _values(group, f"{geometry}_height")
        angle1 = _optional_values(group, f"{geometry}_shapeAngle1", len(r), 0.0)
        angle2 = _optional_values(group, f"{geometry}_shapeAngle2", len(r), 0.0)
        channels = _values(group, f"{geometry}_geometry_channel")
        current_channels = (
            _values(group, f"{geometry}_current_channel")
            if f"{geometry}_current_channel" in group
            else []
        )
        current_key = f"{geometry}_current"
        if current_channels and len(current_channels) == len(r):
            source_groups = [[str(channel)] for channel in current_channels]
        elif (
            current_channels
            and len(r) == 1
            and geometry in PASSIVE_SUM_FAMILIES
        ):
            # FAIR-MAST represents BOTCOL/TOPCOL as six measured currents
            # against one shared axisymmetric geometry in the shots audited by
            # this project.  Their sum is an explicit MAST-specific equivalent
            # current approximation; do not generalize it to unknown families.
            source_groups = [[str(channel) for channel in current_channels]]
        elif current_channels:
            raise ValueError(
                f"{geometry} has {len(r)} geometry elements but "
                f"{len(current_channels)} current channels"
            )
        elif current_key in group and len(r) == 1:
            # Scalar passive-loop signals such as ENDCROWN_L/U have no
            # explicit current_channel coordinate in FAIR-MAST.
            source_groups = [[geometry]]
        elif current_key in group:
            raise ValueError(
                f"{geometry} has current data without an unambiguous channel mapping"
            )
        else:
            source_groups = [[] for _ in r]
        for index, (r_value, z_value) in enumerate(zip(r, z)):
            sources = source_groups[index]
            if len(sources) == 1:
                effective_source = sources[0]
                reduction = "identity"
            elif len(sources) > 1:
                effective_source = f"{geometry}__sum"
                reduction = "sum"
            else:
                effective_source = f"{geometry}__zero"
                reduction = "zero"
            payload.append(
                {
                    "R": r_value,
                    "Z": z_value,
                    "dR": abs(width[index] if len(width) > 1 else width[0]),
                    "dZ": abs(height[index] if len(height) > 1 else height[0]),
                    "resistivity": PASSIVE_RESISTIVITY,
                    "element": geometry,
                    "name": str(channels[index]) if index < len(channels) else f"{geometry}_{index}",
                    "efitGroup": geometry,
                    "source_current_channel": effective_source,
                    "source_current_channels": sources,
                    "source_current_reduction": reduction,
                    "shapeAngle1": angle1[index] if len(angle1) > 1 else angle1[0],
                    "shapeAngle2": angle2[index] if len(angle2) > 1 else angle2[0],
                }
            )
    if not payload:
        raise ValueError("pf_passive contains no geometry groups")
    return payload


def _magnetic_probe_payload(root: Any) -> dict[str, list[dict[str, Any]]]:
    group = _group(root, "magnetics")
    channels = _values(group, "flux_loop_channel")
    geometry_channels = _values(group, "flux_loop_geometry_channel")
    r = _values(group, "flux_loop_r")
    z = _values(group, "flux_loop_z")

    if len(geometry_channels) != len(r) or len(geometry_channels) != len(z):
        raise ValueError(
            "Flux-loop geometry channel, R, and Z arrays have different lengths"
        )
    geometry_names = [str(name) for name in geometry_channels]
    if len(set(geometry_names)) != len(geometry_names):
        raise ValueError("flux_loop_geometry_channel contains duplicate names")

    # flux_loop_channel is the measured subset; flux_loop_geometry_channel is
    # the full geometry list. Join measured signals by normalized name, then
    # retain every remaining geometry as an explicitly named virtual diagnostic.
    # This gives FreeGSNKE all available MAST locations without allowing the
    # geometry-only probes to masquerade as experimental measurements.
    geometry_by_name = {
        str(name): np.array([r[index], z[index]])
        for index, name in enumerate(geometry_channels)
    }

    def flux_loop_geometry_name(channel: Any) -> str:
        return "FL_" + str(channel).replace("/", "_")

    flux_loops: list[dict[str, Any]] = []
    measured_geometry_names: set[str] = set()
    for channel in channels:
        signal_channel = str(channel)
        geometry_name = flux_loop_geometry_name(channel)
        if geometry_name not in geometry_by_name:
            raise ValueError(
                f"No flux-loop geometry {geometry_name!r} for measured channel {channel!r}"
            )
        if geometry_name in measured_geometry_names:
            raise ValueError(
                f"Multiple measured flux-loop channels map to geometry {geometry_name!r}"
            )
        measured_geometry_names.add(geometry_name)
        flux_loops.append(
            {
                "name": signal_channel,
                "geometry_name": geometry_name,
                "position": geometry_by_name[geometry_name].copy(),
                "measurement_status": "measured",
                "source_signal_channel": signal_channel,
            }
        )

    for geometry_name in geometry_names:
        if geometry_name in measured_geometry_names:
            continue
        flux_loops.append(
            {
                "name": f"VIRTUAL::{geometry_name}",
                "geometry_name": geometry_name,
                "position": geometry_by_name[geometry_name].copy(),
                "measurement_status": "virtual",
                "source_signal_channel": None,
            }
        )

    pickups: list[dict[str, Any]] = []
    pickup_specs = (
        ("CCBV", "b_field_pol_probe_ccbv", np.array([0.0, 0.0, 1.0])),
        # These directions match the sign convention of Level 2 *_field arrays.
        ("OBR", "b_field_pol_probe_obr", np.array([1.0, 0.0, 0.0])),
        ("OBV", "b_field_pol_probe_obv", np.array([0.0, 0.0, 1.0])),
    )
    for family, prefix, orientation in pickup_specs:
        channels = _values(group, f"{prefix}_channel")
        geometry_channels = _values(group, f"{prefix}_geometry_channel")
        r = _values(group, f"{prefix}_r")
        phi = _values(group, f"{prefix}_phi_1")
        z = _values(group, f"{prefix}_z")
        for index in range(len(r)):
            pickups.append(
                {
                    "name": str(channels[index]),
                    "geometry_name": str(geometry_channels[index]),
                    "family": family,
                    "position": np.array([r[index], phi[index], z[index]]),
                    "orientation_vector": orientation.copy(),
                }
            )
    return {"flux_loops": flux_loops, "pickups": pickups}


def _wall_payload(root: Any) -> list[dict[str, float]]:
    group = _group(root, "wall")
    r = _values(group, "limiter_r")
    z = _values(group, "limiter_z")
    if len(r) != len(z):
        raise ValueError("wall limiter_r and limiter_z have different lengths")
    return [{"R": r_value, "Z": z_value} for r_value, z_value in zip(r, z)]


def build_machine_payloads(zarr_path: str | Path) -> dict[str, Any]:
    try:
        import zarr
    except ImportError as exc:
        raise RuntimeError("Building machine configuration requires the 'zarr' package") from exc

    path = Path(zarr_path).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Shot Zarr not found: {path}")
    root = zarr.open_group(str(path), mode="r")
    wall = _wall_payload(root)
    return {
        "active_coils": _active_payload(root),
        "limiter": wall,
        "magnetic_probes": _magnetic_probe_payload(root),
        "passive_coils": _passive_payload(root),
        "wall": wall,
    }


def write_machine_pickles(
    zarr_path: str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    payloads = build_machine_payloads(zarr_path)
    output = Path(output_dir).expanduser().resolve()
    paths = {key: output / filename for key, filename in REQUIRED_OUTPUTS.items()}
    if not overwrite:
        existing = [str(path) for path in paths.values() if path.exists()]
        if existing:
            raise FileExistsError("Machine configuration already exists: " + ", ".join(existing))
    output.mkdir(parents=True, exist_ok=True)
    for key, path in paths.items():
        with path.open("wb") as handle:
            pickle.dump(payloads[key], handle, protocol=pickle.HIGHEST_PROTOCOL)
    return paths
