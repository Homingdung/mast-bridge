#!/usr/bin/env python3
"""Run one time-specific MAST forward equilibrium with FreeGSNKE."""

from __future__ import annotations

import argparse
import atexit
import contextlib
import io
import json
import pickle
import re
import resource
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = SCRIPT_ROOT.parent
DEFAULT_FIT_PATH = (
    WORKSPACE_ROOT
    / "data"
    / "processed"
    / "real"
    / "lao_parameter_ensemble"
    / "all_zarr_lao_parameter_fits.npz"
)


def default_output_dir(workspace_root: Path, shot: str, target_time: float) -> Path:
    """Return a stable output directory for one shot/time solve."""
    time_label = f"{target_time:g}"
    return workspace_root / "data" / "processed" / "synthetic" / f"{shot}_t{time_label}"


def plot_path(output_dir: Path) -> Path:
    return output_dir / "equilibrium.png"


FORWARD_SOLVER_SUMMARY_RE = re.compile(
    r"Forward static solve (?P<status>SUCCESS|DID NOT CONVERGE)\. "
    r"Tolerance (?P<tolerance>[0-9.eE+-]+) "
    r"\(vs\. requested (?P<requested>[0-9.eE+-]+)\) reached in "
    r"(?P<iterations>\d+)/(?P<max_iterations>\d+) iterations\."
)


def parse_forward_solver_diagnostics(
    solver_output: str, requested_tolerance: float, max_iterations: int
) -> dict[str, Any]:
    """Parse FreeGSNKE's forward solver summary into stable metadata."""
    match = FORWARD_SOLVER_SUMMARY_RE.search(solver_output)
    if match is None:
        return {
            "solver_status": "unknown",
            "solver_converged": False,
            "solver_final_tolerance": None,
            "solver_requested_tolerance": float(requested_tolerance),
            "solver_iterations": None,
            "solver_max_iterations": int(max_iterations),
            "solver_output": solver_output,
        }

    status = match.group("status")
    return {
        "solver_status": "success" if status == "SUCCESS" else "non_converged",
        "solver_converged": status == "SUCCESS",
        "solver_final_tolerance": float(match.group("tolerance")),
        "solver_requested_tolerance": float(match.group("requested")),
        "solver_iterations": int(match.group("iterations")),
        "solver_max_iterations": int(match.group("max_iterations")),
        "solver_output": solver_output,
    }


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def equilibrium_topology_diagnostics(eq: Any) -> dict[str, Any]:
    """Return JSON-serializable topology diagnostics relevant to LCFS QC."""
    import numpy as np

    xpt = np.asarray(getattr(eq, "xpt", []))
    opt = np.asarray(getattr(eq, "opt", []))
    profiles = getattr(eq, "_profiles", None)
    primary_xpt_psi = None
    if xpt.ndim == 2 and xpt.shape[0] > 0 and xpt.shape[1] > 2:
        primary_xpt_psi = _safe_float(xpt[0, 2])

    return {
        "xpt_count": int(xpt.shape[0]) if xpt.ndim >= 2 else 0,
        "opt_count": int(opt.shape[0]) if opt.ndim >= 2 else 0,
        "flag_limiter": bool(getattr(profiles, "flag_limiter", getattr(eq, "flag_limiter", False))),
        "psi_axis": _safe_float(getattr(eq, "psi_axis", None)),
        "psi_bndry": _safe_float(getattr(eq, "psi_bndry", None)),
        "primary_xpt_psi": primary_xpt_psi,
    }


def solve_with_diagnostics(
    solver: Any,
    eq: Any,
    profiles: Any,
    requested_tolerance: float,
    max_iterations: int,
) -> dict[str, Any]:
    """Run the official FreeGSNKE solver and collect its printed diagnostics."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        solver.solve(
            eq=eq,
            profiles=profiles,
            constrain=None,
            target_relative_tolerance=requested_tolerance,
            max_solving_iterations=max_iterations,
        )
    solver_output = buffer.getvalue()
    print(solver_output, end="")
    return parse_forward_solver_diagnostics(
        solver_output, requested_tolerance, max_iterations
    )


def apply_lao85_perturbation(
    *,
    Ip: float,
    fvac: float,
    alpha: list[float],
    beta: list[float],
    ip_scale: float,
    fvac_scale: float,
    alpha_scale: float,
    beta_scale: float,
    alpha_offset: float,
    beta_offset: float,
) -> dict[str, Any]:
    """Apply scalar Lao85 perturbations to a fitted parameter row."""
    return {
        "Ip": float(Ip) * float(ip_scale),
        "fvac": float(fvac) * float(fvac_scale),
        "alpha": [float(value) * float(alpha_scale) + float(alpha_offset) for value in alpha],
        "beta": [float(value) * float(beta_scale) + float(beta_offset) for value in beta],
        "perturbation": {
            "ip_scale": float(ip_scale),
            "fvac_scale": float(fvac_scale),
            "alpha_scale": float(alpha_scale),
            "beta_scale": float(beta_scale),
            "alpha_offset": float(alpha_offset),
            "beta_offset": float(beta_offset),
        },
    }


def _draw_rectangles(ax: Any, payload: Any, color: str, label: str, alpha: float = 0.75) -> None:
    from matplotlib.patches import Rectangle
    import numpy as np

    first = True

    def visit(item: Any) -> None:
        nonlocal first
        if isinstance(item, dict) and {"R", "Z", "dR", "dZ"} <= item.keys():
            radii = np.atleast_1d(item["R"])
            heights = np.atleast_1d(item["Z"])
            width = abs(float(item["dR"]))
            height = abs(float(item["dZ"]))
            for radius, vertical in zip(radii, heights):
                ax.add_patch(
                    Rectangle(
                        (float(radius) - width / 2, float(vertical) - height / 2),
                        width,
                        height,
                        facecolor="none",
                        edgecolor=color,
                        linewidth=0.45,
                        alpha=alpha,
                        label=label if first else None,
                    )
                )
                first = False
        elif isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(payload)


def save_equilibrium_plot(
    eq: Any, machine_dir: Path, output_dir: Path, shot: str, target_time: float
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from freegs4e.plotting import plotEquilibrium
    from mast_bridge.mast.machine_config import MachineGeometry

    machine = MachineGeometry.load(machine_dir)
    with machine.files["active_coils"].open("rb") as handle:
        active = pickle.load(handle)
    with machine.files["passive_coils"].open("rb") as handle:
        passive = pickle.load(handle)
    figure, axis = plt.subplots(figsize=(8, 8))
    # Use FreeGS4E's official equilibrium renderer. FreeGSNKE exposes the
    # solved equilibrium, while the plotting helper lives in its FreeGS4E
    # dependency.
    plotEquilibrium(
        eq,
        axis=axis,
        xpoints=True,
        opoints=True,
        wall=True,
        limiter=True,
        legend=False,
        show=False,
    )
    _draw_rectangles(axis, active, "tab:red", "active coils")
    _draw_rectangles(axis, passive, "0.35", "passive structures", alpha=0.6)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(0.0, 2.05)
    axis.set_ylim(-2.1, 2.1)
    axis.set_xlabel("R [m]")
    axis.set_ylabel("Z [m]")
    axis.grid(True, alpha=0.2)
    axis.set_title(f"FreeGSNKE forward solve + MAST machine: {shot} at t={target_time:g} s")
    axis.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        fontsize=8,
    )
    figure.tight_layout(rect=(0.0, 0.0, 0.78, 1.0))
    output = plot_path(output_dir)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


def select_fit_row(fit: Any, shot: str, target_time: float) -> int:
    """Select the nearest fitted Lao row for a shot and target time."""
    import numpy as np

    shot_values = np.asarray(fit["shot"]).astype(str)
    rows = np.flatnonzero(shot_values == str(shot))
    if rows.size == 0:
        raise ValueError(f"No Lao fit row found for shot {shot!r}")
    times = np.asarray(fit["time"], dtype=float)
    return int(rows[np.argmin(np.abs(times[rows] - target_time))])


def at_time(times: Any, values: Any, target_time: float) -> Any:
    """Linearly interpolate a signal at one target time."""
    import numpy as np

    return np.interp(target_time, np.asarray(times), np.asarray(values))


def _array_values(values: Any) -> Any:
    if hasattr(values, "__getitem__") and not isinstance(values, (list, tuple, dict)):
        try:
            return values[:]
        except (TypeError, ValueError, AttributeError):
            pass
    return values


def equilibrium_grid_bounds(equilibrium_group: Any) -> dict[str, float]:
    """Return FreeGSNKE grid bounds from the real MAST EFIT grid arrays."""
    import numpy as np

    major_radius = np.asarray(_array_values(equilibrium_group["major_radius"]), dtype=float)
    vertical = np.asarray(_array_values(equilibrium_group["z"]), dtype=float)
    if major_radius.size == 0 or vertical.size == 0:
        raise ValueError("equilibrium major_radius/z arrays must not be empty")
    if not np.isfinite(major_radius).any() or not np.isfinite(vertical).any():
        raise ValueError("equilibrium major_radius/z arrays must contain finite values")
    return {
        "Rmin": float(np.nanmin(major_radius)),
        "Rmax": float(np.nanmax(major_radius)),
        "Zmin": float(np.nanmin(vertical)),
        "Zmax": float(np.nanmax(vertical)),
    }


def scale_current_dicts(
    currents: dict[str, dict[str, float]], scale: float
) -> dict[str, dict[str, float]]:
    """Scale active/passive current metadata with the same factor used in FreeGSNKE."""
    return {
        family: {
            name: float(value) * float(scale)
            for name, value in values.items()
        }
        for family, values in currents.items()
    }


def passive_source_currents_at_time(
    passive_group: Any,
    target_time: float,
    current_scale: float = 1.0,
) -> dict[str, float]:
    """Interpolate every measured passive-loop current at one time.

    FAIR-MAST stores most passive currents as ``[channel, time]`` arrays, but
    scalar loops such as ``endcrown_l``/``endcrown_u`` have no channel
    coordinate.  Treat those family names as stable source-channel keys.
    """
    import numpy as np

    times = np.asarray(passive_group["time"][:], dtype=float)
    currents: dict[str, float] = {}
    for current_key in sorted(passive_group.array_keys()):
        if not current_key.endswith("_current"):
            continue
        family = current_key.removesuffix("_current")
        values = np.asarray(passive_group[current_key][:], dtype=float)
        channel_key = f"{family}_current_channel"
        if channel_key in passive_group:
            channels = [str(value) for value in passive_group[channel_key][:]]
            if values.ndim == 1:
                rows = values[None, :]
            elif values.ndim == 2 and values.shape[1] == times.size:
                rows = values
            elif values.ndim == 2 and values.shape[0] == times.size:
                rows = values.T
            else:
                raise ValueError(
                    f"Passive current {current_key} shape {values.shape} "
                    f"does not align with time axis {times.size}"
                )
            if rows.shape[0] != len(channels):
                raise ValueError(
                    f"Passive current {current_key} has {rows.shape[0]} rows "
                    f"but {len(channels)} channels"
                )
        else:
            if values.ndim == 1:
                rows = values[None, :]
            elif values.ndim == 2 and values.shape[0] == 1:
                rows = values
            elif values.ndim == 2 and values.shape[1] == 1:
                rows = values.T
            else:
                raise ValueError(
                    f"Passive current {current_key} has no channel coordinate "
                    f"and ambiguous shape {values.shape}"
                )
            channels = [family]

        for channel, row in zip(channels, rows, strict=True):
            value = float(at_time(times, row, target_time)) * float(current_scale)
            if not np.isfinite(value):
                raise ValueError(
                    f"Passive current {channel!r} is non-finite at {target_time}"
                )
            currents[channel] = value
    return currents


def effective_passive_current(
    item: dict[str, Any], source_currents: dict[str, float]
) -> float:
    """Resolve one FreeGSNKE passive element's explicit current mapping."""
    sources = item.get("source_current_channels")
    if sources is None:
        sources = [item["source_current_channel"]]
    sources = [str(source) for source in sources]
    reduction = str(item.get("source_current_reduction", "identity"))
    if reduction == "zero":
        return 0.0
    missing = [source for source in sources if source not in source_currents]
    if missing:
        raise KeyError(f"Passive current sources are missing: {missing}")
    values = [float(source_currents[source]) for source in sources]
    if reduction == "identity" and len(values) == 1:
        return values[0]
    if reduction == "sum" and values:
        return float(sum(values))
    raise ValueError(
        f"Invalid passive current mapping reduction={reduction!r}, sources={sources!r}"
    )


def machine_geometry_policy(machine_dir: Path) -> dict[str, Any]:
    """Describe the lossy MAST-to-FreeGSNKE geometry/current conversion."""
    from mast_bridge.mast.machine_config import MachineGeometry

    machine = MachineGeometry.load(machine_dir)
    with machine.files["passive_coils"].open("rb") as handle:
        passive_payload = pickle.load(handle)
    with machine.files["magnetic_probes"].open("rb") as handle:
        magnetic_probe_payload = pickle.load(handle)
    reductions: dict[str, int] = {}
    many_to_one: list[dict[str, Any]] = []
    for item in passive_payload:
        reduction = str(item.get("source_current_reduction", "identity"))
        reductions[reduction] = reductions.get(reduction, 0) + 1
        sources = [str(value) for value in item.get("source_current_channels", [])]
        if reduction == "sum":
            many_to_one.append(
                {
                    "element": str(item.get("element", item.get("name", ""))),
                    "effective_channel": str(item["source_current_channel"]),
                    "source_channels": sources,
                    "status": "empirically_validated_approximation",
                }
            )
    flux_loop_status_counts: dict[str, int] = {}
    for item in magnetic_probe_payload.get("flux_loops", []):
        status = str(item.get("measurement_status", "unspecified"))
        flux_loop_status_counts[status] = flux_loop_status_counts.get(status, 0) + 1
    return {
        "device": "MAST",
        "source": "per-shot FAIR-MAST Level-2 MAST Zarr",
        "freegsnke_default_mast_u_geometry_used": False,
        "passive_current_reduction_counts": reductions,
        "passive_many_to_one": many_to_one,
        "passive_many_to_one_reduction": "sum",
        "passive_many_to_one_status": "empirically_validated_approximation",
        "passive_unmeasured_current_policy": "zero_unmeasured",
        "passive_shape_angles_stored": True,
        "passive_shape_angles_applied_by_freegsnke_scalar_geometry": False,
        "wall_policy": "MAST limiter contour reused because no independent vessel-wall contour is present",
        "flux_loop_policy": (
            "retain all MAST geometries; measured channels are name-joined and "
            "geometry-only locations are explicit virtual diagnostics"
        ),
        "flux_loop_status_counts": flux_loop_status_counts,
        "virtual_flux_loop_name_prefix": "VIRTUAL::",
    }


def memory_provenance() -> dict[str, Any]:
    """Return Linux process/cgroup memory facts without allocating large buffers."""
    result: dict[str, Any] = {
        "process_peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }
    cgroup = Path("/sys/fs/cgroup")
    for filename, key in (
        ("memory.max", "cgroup_memory_max_bytes"),
        ("memory.current", "cgroup_memory_current_bytes"),
    ):
        try:
            value = (cgroup / filename).read_text(encoding="utf-8").strip()
            result[key] = None if value == "max" else int(value)
        except (OSError, ValueError):
            result[key] = None
    try:
        result["cgroup_memory_events"] = {
            key: int(value)
            for key, value in (
                line.split(maxsplit=1)
                for line in (cgroup / "memory.events").read_text(encoding="utf-8").splitlines()
            )
        }
    except (OSError, ValueError):
        result["cgroup_memory_events"] = None
    return result


def _apply_currents(
    tokamak: Any,
    shot: Any,
    machine_dir: Path,
    target_time: float,
    current_scale: float = 1.0,
) -> dict[str, dict[str, float]]:
    import numpy as np

    active_group = shot["pf_active"]
    active_channels = [str(value) for value in active_group["current_channel"][:]]
    active_current = np.asarray(active_group["coil_current"][:])
    active_at_time = dict(
        zip(
            active_channels,
            [
                float(at_time(active_group["time"][:], row, target_time)) * float(current_scale)
                for row in active_current
            ],
        )
    )

    from mast_bridge.mast.machine_config import MachineGeometry

    machine = MachineGeometry.load(machine_dir)
    active_payload = pickle.load(machine.files["active_coils"].open("rb"))
    tokamak.set_all_coil_currents(np.zeros(tokamak.n_coils))
    for coil_name, payload in active_payload.items():
        leaf = payload
        while "source_channel" not in leaf:
            leaf = next(iter(leaf.values()))
        channel = leaf["source_channel"]
        if channel not in active_at_time:
            raise KeyError(f"Active coil channel {channel!r} is missing from the shot")
        tokamak.set_coil_current(coil_name, active_at_time[channel])

    passive_group = shot["pf_passive"]
    passive_source_at_time = passive_source_currents_at_time(
        passive_group, target_time, current_scale=current_scale
    )

    passive_payload = pickle.load(machine.files["passive_coils"].open("rb"))
    passive_effective_at_time: dict[str, float] = {}
    for item in passive_payload:
        channel = str(item["source_current_channel"])
        name = item["name"]
        value = effective_passive_current(item, passive_source_at_time)
        previous = passive_effective_at_time.get(channel)
        if previous is not None and not np.isclose(previous, value):
            raise ValueError(
                f"Effective passive channel {channel!r} has inconsistent values"
            )
        passive_effective_at_time[channel] = value
        if name in tokamak.coil_names:
            tokamak.set_coil_current(name, value)

    return {
        "active": {name: float(value) for name, value in active_at_time.items()},
        "passive": passive_effective_at_time,
        "passive_source": passive_source_at_time,
    }


def _copy_machine_with_positive_widths(source: Path) -> Path:
    destination = Path(tempfile.mkdtemp(prefix="mast-freegsnke-machine-"))
    for path in source.glob("*.pickle"):
        shutil.copy2(path, destination / path.name)

    from mast_bridge.mast.machine_config import MachineGeometry

    passive_path = MachineGeometry.load(destination).files["passive_coils"]
    with passive_path.open("rb") as handle:
        passive_payload = pickle.load(handle)
    for item in passive_payload:
        item["dR"] = abs(item["dR"])
        item["dZ"] = abs(item["dZ"])
    with passive_path.open("wb") as handle:
        pickle.dump(passive_payload, handle)
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one time-specific MAST forward equilibrium with FreeGSNKE."
    )
    parser.add_argument("--shot", required=True, help="Shot ID, for example 11766.")
    parser.add_argument(
        "--time", type=float, required=True, dest="target_time", help="Target time in seconds."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=WORKSPACE_ROOT / "data" / "raw" / "mast",
        help="Directory containing <shot>.zarr.",
    )
    parser.add_argument(
        "--machine-dir",
        type=Path,
        default=None,
        help="Machine pickle directory; default is data/raw/mast/machine/<shot>.",
    )
    parser.add_argument(
        "--fit-path",
        type=Path,
        default=DEFAULT_FIT_PATH,
        help="NPZ file containing EFIT/Lao fitted parameters.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory; default is data/processed/synthetic/<shot>_t<time>.",
    )
    parser.add_argument("--nx", type=int, default=65)
    parser.add_argument("--ny", type=int, default=65)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--ip-scale", type=float, default=1.0)
    parser.add_argument("--fvac-scale", type=float, default=1.0)
    parser.add_argument("--alpha-scale", type=float, default=1.0)
    parser.add_argument("--beta-scale", type=float, default=1.0)
    parser.add_argument("--alpha-offset", type=float, default=0.0)
    parser.add_argument("--beta-offset", type=float, default=0.0)
    parser.add_argument("--coil-current-scale", type=float, default=1.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    import numpy as np
    import zarr

    from mast_bridge.mast.machine_config import MachineGeometry
    from mast_bridge.simulation.freegsnke_runner import build_machine

    # FreeGSNKE must be imported before this pickle compatibility alias.
    from freegsnke import GSstaticsolver, equilibrium_update
    from freegsnke.jtor_update import Lao85

    import numpy.core
    import numpy.core.numeric

    sys.modules.setdefault("numpy._core", numpy.core)
    sys.modules.setdefault("numpy._core.numeric", numpy.core.numeric)

    data_dir = args.data_dir.expanduser().resolve()
    shot_path = data_dir / f"{args.shot}.zarr"
    machine_dir = (
        args.machine_dir
        or data_dir / "machine" / str(args.shot)
    ).expanduser().resolve()
    fit_path = args.fit_path.expanduser().resolve()
    output_dir = (
        args.output_dir
        or default_output_dir(WORKSPACE_ROOT, str(args.shot), args.target_time)
    ).expanduser().resolve()

    if not shot_path.is_dir():
        raise FileNotFoundError(f"MAST Zarr not found: {shot_path}")
    machine = MachineGeometry.load(machine_dir)
    if not fit_path.is_file():
        raise FileNotFoundError(f"Lao fit file not found: {fit_path}")

    solve_machine_dir = _copy_machine_with_positive_widths(machine_dir)
    cleanup_solve_machine = lambda: shutil.rmtree(  # noqa: E731
        solve_machine_dir, ignore_errors=True
    )
    atexit.register(cleanup_solve_machine)
    tokamak = build_machine(MachineGeometry.load(solve_machine_dir))
    shot = zarr.open_group(str(shot_path), mode="r")
    currents = _apply_currents(
        tokamak,
        shot,
        solve_machine_dir,
        args.target_time,
        current_scale=args.coil_current_scale,
    )

    fit = np.load(fit_path)
    fit_index = select_fit_row(fit, str(args.shot), args.target_time)
    Ip = float(fit["ip"][fit_index])
    fvac = float(fit["fvac"][fit_index])
    alpha = np.asarray(fit["freegsnke_alpha"][fit_index]).tolist()
    beta = np.asarray(fit["freegsnke_beta"][fit_index]).tolist()
    fitted_time = float(fit["time"][fit_index])
    lao85_parameters = apply_lao85_perturbation(
        Ip=Ip,
        fvac=fvac,
        alpha=alpha,
        beta=beta,
        ip_scale=args.ip_scale,
        fvac_scale=args.fvac_scale,
        alpha_scale=args.alpha_scale,
        beta_scale=args.beta_scale,
        alpha_offset=args.alpha_offset,
        beta_offset=args.beta_offset,
    )
    Ip = lao85_parameters["Ip"]
    fvac = lao85_parameters["fvac"]
    alpha = lao85_parameters["alpha"]
    beta = lao85_parameters["beta"]

    grid_bounds = equilibrium_grid_bounds(shot["equilibrium"])
    eq = equilibrium_update.Equilibrium(
        tokamak=tokamak,
        Rmin=grid_bounds["Rmin"],
        Rmax=grid_bounds["Rmax"],
        Zmin=grid_bounds["Zmin"],
        Zmax=grid_bounds["Zmax"],
        nx=args.nx,
        ny=args.ny,
    )
    profiles = Lao85(eq=eq, Ip=Ip, fvac=fvac, alpha=alpha, beta=beta)
    solver = GSstaticsolver.NKGSsolver(eq)
    solver_diagnostics = solve_with_diagnostics(
        solver=solver,
        eq=eq,
        profiles=profiles,
        requested_tolerance=args.tolerance,
        max_iterations=args.max_iterations,
    )
    topology_diagnostics = equilibrium_topology_diagnostics(eq)
    geometry_policy = machine_geometry_policy(machine_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "equilibrium.npz",
        schema_version=np.asarray(1, dtype=np.int16),
        psi=eq.psi(),
        R=eq.R,
        Z=eq.Z,
        psi_axis=eq.psi_axis,
        psi_bndry=eq.psi_bndry,
        axis_order=np.asarray(["R", "Z"]),
        psi_units=np.asarray("Wb/rad"),
        R_units=np.asarray("m"),
        Z_units=np.asarray("m"),
        psi_convention=np.asarray("FreeGS poloidal flux per radian; COCOS ID unspecified"),
    )
    image_path = save_equilibrium_plot(
        eq, solve_machine_dir, output_dir, str(args.shot), args.target_time
    )
    metadata = {
        "schema_version": 1,
        "source": "synthetic",
        "device": "MAST",
        "parent_shot": str(args.shot),
        "target_time": args.target_time,
        "fitted_time": fitted_time,
        "machine_geometry_source": str(machine_dir),
        "fit_path": str(fit_path),
        "coil_currents": currents,
        "Ip": Ip,
        "fvac": fvac,
        "alpha": alpha,
        "beta": beta,
        "lao85_perturbation": lao85_parameters["perturbation"],
        "coil_current_scale": args.coil_current_scale,
        "grid": {"nx": args.nx, "ny": args.ny, **grid_bounds},
        "equilibrium_array_schema": {
            "axis_order": ["R", "Z"],
            "psi_units": "Wb/rad",
            "R_units": "m",
            "Z_units": "m",
            "cocos": None,
            "psi_sign": "FreeGS convention; not assigned an authoritative COCOS ID",
        },
        "machine_geometry_policy": geometry_policy,
        "source_zarr_group_revisions": {
            name: shot[name].attrs.get("commit_url")
            for name in ("equilibrium", "magnetics", "pf_active", "pf_passive", "wall")
        },
        "memory": memory_provenance(),
        "target_relative_tolerance": args.tolerance,
        "max_solving_iterations": args.max_iterations,
        "plot_path": str(image_path),
    }
    metadata.update(solver_diagnostics)
    metadata.update(topology_diagnostics)
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    print(f"shot: {args.shot}")
    print(f"target_time: {args.target_time}")
    print(f"fitted_time: {fitted_time}")
    print(f"psi_shape: {eq.psi().shape}")
    print(f"solver_status: {metadata['solver_status']}")
    print(f"solver_final_tolerance: {metadata['solver_final_tolerance']}")
    print(f"xpt_count: {metadata['xpt_count']}")
    print(f"opt_count: {metadata['opt_count']}")
    print(f"flag_limiter: {metadata['flag_limiter']}")
    print(f"equilibrium: {output_dir / 'equilibrium.npz'}")
    print(f"metadata: {output_dir / 'metadata.json'}")
    print(f"plot: {image_path}")
    cleanup_solve_machine()
    atexit.unregister(cleanup_solve_machine)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
