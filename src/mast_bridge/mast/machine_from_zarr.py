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

REQUIRED_OUTPUTS = {
    "active_coils": "MAST_active_coils.pickle",
    "limiter": "MAST_limiter.pickle",
    "magnetic_probes": "MAST_magentic_probes.pickle",
    "passive_coils": "MAST_passive_coilds.pickle",
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
            raise ValueError(f"pf_active/current_channel does not contain {channel!r}")
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
        for index, (r_value, z_value) in enumerate(zip(r, z)):
            source = (
                current_channels[index]
                if index < len(current_channels)
                else f"{geometry}_current_channel_{index}"
            )
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
                    "source_current_channel": str(source),
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
    flux_loops = [
        {
            "name": str(channels[index]) if index < len(channels) else f"flux_loop_channel_{index}",
            "geometry_name": str(geometry_channels[index]),
            "position": np.array([r[index], z[index]]),
        }
        for index in range(len(r))
    ]

    pickups: list[dict[str, Any]] = []
    pickup_specs = (
        ("CCBV", "b_field_pol_probe_ccbv", np.array([0.0, 0.0, 1.0])),
        ("OBR", "b_field_pol_probe_obr", np.array([-1.0, 0.0, 0.0])),
        ("OBV", "b_field_pol_probe_obv", np.array([0.0, 0.0, -1.0])),
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
