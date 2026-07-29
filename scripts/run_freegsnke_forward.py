#!/usr/bin/env python3
"""Run one time-specific MAST forward equilibrium with FreeGSNKE."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import pickle
import re
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


def _apply_currents(
    tokamak: Any, shot: Any, machine_dir: Path, target_time: float
) -> dict[str, dict[str, float]]:
    import numpy as np

    active_group = shot["pf_active"]
    active_channels = [str(value) for value in active_group["current_channel"][:]]
    active_current = np.asarray(active_group["coil_current"][:])
    active_at_time = dict(
        zip(
            active_channels,
            [at_time(active_group["time"][:], row, target_time) for row in active_current],
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
    passive_at_time: dict[str, float] = {}
    for key in passive_group.array_keys():
        if not key.endswith("_current_channel"):
            continue
        family = key.removesuffix("_current_channel")
        current_key = f"{family}_current"
        if current_key not in passive_group:
            continue
        channels = [str(value) for value in passive_group[key][:]]
        values = np.asarray(passive_group[current_key][:])
        if values.ndim == 1:
            values = values[None, :]
        for channel, row in zip(channels, values):
            passive_at_time[channel] = float(
                at_time(passive_group["time"][:], row, target_time)
            )

    passive_payload = pickle.load(machine.files["passive_coils"].open("rb"))
    for item in passive_payload:
        channel = item["source_current_channel"]
        name = item["name"]
        if name in tokamak.coil_names and channel in passive_at_time:
            tokamak.set_coil_current(name, passive_at_time[channel])

    return {
        "active": {name: float(value) for name, value in active_at_time.items()},
        "passive": passive_at_time,
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
    tokamak = build_machine(MachineGeometry.load(solve_machine_dir))
    shot = zarr.open_group(str(shot_path), mode="r")
    currents = _apply_currents(tokamak, shot, solve_machine_dir, args.target_time)

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

    eq = equilibrium_update.Equilibrium(
        tokamak=tokamak,
        Rmin=0.1,
        Rmax=2.0,
        Zmin=-2.0,
        Zmax=2.0,
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

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "equilibrium.npz",
        psi=eq.psi(),
        R=eq.R,
        Z=eq.Z,
        psi_axis=eq.psi_axis,
        psi_bndry=eq.psi_bndry,
    )
    image_path = save_equilibrium_plot(
        eq, solve_machine_dir, output_dir, str(args.shot), args.target_time
    )
    metadata = {
        "source": "synthetic",
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
        "grid": {"nx": args.nx, "ny": args.ny},
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
