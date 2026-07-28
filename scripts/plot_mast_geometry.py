#!/usr/bin/env python3
"""Plot the raw MAST geometry extracted from one shot's Zarr store."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = SCRIPT_ROOT.parent
sys.path.insert(0, str(SCRIPT_ROOT / "src"))


def _active_leaves(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and {"R", "Z", "dR", "dZ"} <= payload.keys():
        return [payload]
    if isinstance(payload, dict):
        leaves: list[dict[str, Any]] = []
        for child in payload.values():
            leaves.extend(_active_leaves(child))
        return leaves
    return []


def _rectangle_points(
    radius: float, height: float, width: float, depth: float, angle_degrees: float = 0.0
) -> np.ndarray:
    corners = np.array(
        [[-width / 2, -depth / 2], [width / 2, -depth / 2],
         [width / 2, depth / 2], [-width / 2, depth / 2]],
        dtype=float,
    )
    angle = np.deg2rad(angle_degrees)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return corners @ rotation.T + np.array([radius, height])


def _draw_box(ax: Any, item: dict[str, Any], color: str, label: str, rotated: bool = False) -> None:
    radius = float(item["R"])
    height = float(item["Z"])
    width = abs(float(item["dR"]))
    depth = abs(float(item["dZ"]))
    angle = 0.0
    if rotated:
        angle = float(item.get("shapeAngle1", 0.0) or item.get("shapeAngle2", 0.0))
    points = _rectangle_points(radius, height, width, depth, angle)
    ax.fill(
        points[:, 0],
        points[:, 1],
        facecolor="none",
        edgecolor=color,
        linewidth=0.45,
        alpha=0.75,
        label=label,
    )


def plot_geometry(payloads: dict[str, Any], shot: str, output: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(9, 9))
    labelled: set[str] = set()

    def label_once(name: str) -> str | None:
        if name in labelled:
            return None
        labelled.add(name)
        return name

    for item in _active_leaves(payloads["active_coils"]):
        for radius, height in zip(np.atleast_1d(item["R"]), np.atleast_1d(item["Z"])):
            copy = dict(item, R=radius, Z=height)
            _draw_box(axis, copy, "tab:red", label_once("active coils"))

    for item in payloads["passive_coils"]:
        _draw_box(axis, item, "tab:gray", label_once("passive structures"), rotated=True)

    wall = payloads["wall"]
    wall_r = [float(item["R"]) for item in wall] + [float(wall[0]["R"])]
    wall_z = [float(item["Z"]) for item in wall] + [float(wall[0]["Z"])]
    axis.plot(wall_r, wall_z, color="black", linewidth=1.5, label="wall / limiter")

    magnetics = payloads["magnetic_probes"]
    flux_loops = magnetics["flux_loops"]
    if flux_loops:
        axis.scatter(
            [item["position"][0] for item in flux_loops],
            [item["position"][1] for item in flux_loops],
            marker="o", s=15, color="tab:blue", label="flux loops",
        )
    pickups = magnetics["pickups"]
    if pickups:
        axis.scatter(
            [item["position"][0] for item in pickups],
            [item["position"][2] for item in pickups],
            marker="x", s=18, color="tab:purple", label="pickup probes",
        )

    all_r = np.array(wall_r + [float(x["R"]) for x in payloads["passive_coils"]])
    all_z = np.array(wall_z + [float(x["Z"]) for x in payloads["passive_coils"]])
    margin = 0.12
    axis.set_xlim(all_r.min() - margin, all_r.max() + margin)
    axis.set_ylim(all_z.min() - margin, all_z.max() + margin)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("R [m]")
    axis.set_ylabel("Z [m]")
    axis.set_title(f"Raw MAST machine geometry from shot {shot}")
    axis.grid(True, alpha=0.2)
    axis.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        fontsize=8,
    )
    figure.tight_layout(rect=(0.0, 0.0, 0.78, 1.0))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot raw MAST geometry from a shot Zarr store.")
    parser.add_argument("--shot", required=True, help="Shot ID, for example 11766.")
    parser.add_argument(
        "--data-dir", type=Path, default=WORKSPACE_ROOT / "data" / "raw" / "mast",
        help="Directory containing <shot>.zarr.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="PNG path; default is data/processed/geometry/<shot>.png.",
    )
    args = parser.parse_args(argv)

    from mast_bridge.mast.machine_from_zarr import build_machine_payloads

    zarr_path = args.data_dir.expanduser().resolve() / f"{args.shot}.zarr"
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else WORKSPACE_ROOT / "data" / "processed" / "geometry" / f"{args.shot}.png"
    )
    payloads = build_machine_payloads(zarr_path)
    plot_geometry(payloads, str(args.shot), output)
    print(f"shot: {args.shot}")
    print(f"zarr: {zarr_path}")
    print(f"geometry_plot: {output}")
    print(f"active_elements: {len(_active_leaves(payloads['active_coils']))}")
    print(f"passive_elements: {len(payloads['passive_coils'])}")
    print(f"flux_loops: {len(payloads['magnetic_probes']['flux_loops'])}")
    print(f"pickup_probes: {len(payloads['magnetic_probes']['pickups'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
