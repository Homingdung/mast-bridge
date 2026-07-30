#!/usr/bin/env python3
"""Solve one MAST equilibrium with FreeGSNKE and compare magnetic diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
import tempfile
from pathlib import Path
from typing import Any


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = SCRIPT_ROOT.parent
if str(SCRIPT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT / "scripts"))
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mast-bridge-matplotlib")
)

import run_freegsnke_forward as forward


DEFAULT_FIT_PATH = forward.DEFAULT_FIT_PATH


def default_output_dir(workspace_root: Path, shot: str, target_time: float) -> Path:
    time_label = f"{target_time:g}"
    return (
        workspace_root
        / "data"
        / "processed"
        / "diagnostic_comparisons"
        / f"{shot}_t{time_label}"
    )


def default_artifact_dir(workspace_root: Path, shot: str, target_time: float) -> Path:
    time_label = f"{target_time:g}"
    return (
        workspace_root
        / "artifacts"
        / "freegsnke_magnetic_diagnostics"
        / f"{shot}_t{time_label}"
    )


def write_comparison_csv(path: Path, rows: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["diagnostic_type", "channel", "model", "observed", "error", "abs_error"])
        for row in rows:
            writer.writerow(
                [
                    row["diagnostic_type"],
                    row["channel"],
                    f"{float(row['model']):.17g}",
                    f"{float(row['observed']):.17g}",
                    f"{float(row['error']):.17g}",
                    f"{float(row['abs_error']):.17g}",
                ]
            )


def save_observed_vs_model_plot(path: Path, rows: Any, shot: str, target_time: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(11, 5))
    for axis, diagnostic_type in zip(axes, ["flux_loop", "pickup"]):
        subset = rows[rows["diagnostic_type"] == diagnostic_type]
        axis.set_title(diagnostic_type.replace("_", " "))
        axis.set_xlabel("observed")
        axis.set_ylabel("model")
        axis.grid(True, alpha=0.25)
        if subset.size == 0:
            axis.text(0.5, 0.5, "no matched channels", ha="center", va="center")
            continue
        observed = subset["observed"].astype(float)
        model = subset["model"].astype(float)
        axis.scatter(observed, model, s=18, alpha=0.8)
        lower = float(np.nanmin([observed.min(), model.min()]))
        upper = float(np.nanmax([observed.max(), model.max()]))
        if np.isfinite(lower) and np.isfinite(upper) and lower != upper:
            axis.plot([lower, upper], [lower, upper], color="0.35", linewidth=1.0)
        axis.text(
            0.04,
            0.96,
            f"n={subset.size}",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=9,
        )
    figure.suptitle(f"FreeGSNKE magnetic diagnostics vs MAST: {shot} at t={target_time:g} s")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def save_current_constraint_plot(
    path: Path, comparison: dict[str, Any], shot: str, target_time: float
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    observed = float(comparison["observed"])
    model = float(comparison["model"])
    values = np.asarray([observed, model], dtype=float)
    labels = ["MAST magnetics/ip", "FreeGSNKE Lao85 Ip"]

    figure, axis = plt.subplots(figsize=(7, 4))
    colors = ["tab:blue", "tab:orange"]
    axis.bar(labels, values / 1e3, color=colors, alpha=0.85)
    axis.set_ylabel("plasma current [kA]")
    axis.set_title(f"Global plasma current constraint: {shot} at t={target_time:g} s")
    axis.grid(True, axis="y", alpha=0.25)
    for index, value in enumerate(values / 1e3):
        axis.text(index, value, f"{value:.1f}", ha="center", va="bottom", fontsize=9)

    error_ka = float(comparison["error"]) / 1e3
    rel = float(comparison["relative_error"]) * 100.0
    axis.text(
        0.02,
        0.95,
        f"model - observed = {error_ka:.2f} kA ({rel:.2f}%)",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def correct_magnetic_probes_in_machine_dir(machine_dir: Path, magnetics: Any) -> None:
    from mast_bridge.mast.machine_config import MachineGeometry
    from mast_bridge.simulation.magnetic_diagnostics import (
        correct_mast_level2_flux_loop_positions,
        correct_mast_level2_pickup_orientations,
    )

    probe_path = MachineGeometry.load(machine_dir).files["magnetic_probes"]
    with probe_path.open("rb") as handle:
        payload = pickle.load(handle)
    # Keep raw machine pickles immutable. Older generated pickles used index
    # based flux-loop positions and opposite OBR/OBV signs, so fix only the
    # temporary copy used by this comparison run.
    correct_mast_level2_flux_loop_positions(payload, magnetics)
    correct_mast_level2_pickup_orientations(payload)
    with probe_path.open("wb") as handle:
        pickle.dump(payload, handle)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare FreeGSNKE synthetic magnetic diagnostics with MAST measurements."
    )
    parser.add_argument("--shot", required=True, help="Shot ID, for example 11771.")
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
        help="Numeric output directory; default is data/processed/diagnostic_comparisons/<shot>_t<time>.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="Plot output directory; default is artifacts/freegsnke_magnetic_diagnostics/<shot>_t<time>.",
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
    from mast_bridge.simulation.magnetic_diagnostics import LEVEL2_FLUX_LOOP_SCALE

    parser.add_argument(
        "--flux-loop-scale",
        type=float,
        default=LEVEL2_FLUX_LOOP_SCALE,
        help="Scale applied to FreeGSNKE Wb/(2pi) flux-loop psi before comparing to Level 2 Wb flux_loop_flux.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    import numpy as np
    import zarr

    from mast_bridge.mast.machine_config import MachineGeometry
    from mast_bridge.simulation.freegsnke_runner import build_machine
    from mast_bridge.simulation.magnetic_diagnostics import (
        build_magnetic_diagnostic_comparison,
        current_constraint_comparison,
        summarize_comparisons,
    )

    from freegsnke import GSstaticsolver, equilibrium_update
    from freegsnke.jtor_update import Lao85

    import numpy.core
    import numpy.core.numeric

    sys.modules.setdefault("numpy._core", numpy.core)
    sys.modules.setdefault("numpy._core.numeric", numpy.core.numeric)

    data_dir = args.data_dir.expanduser().resolve()
    shot_path = data_dir / f"{args.shot}.zarr"
    machine_dir = (
        args.machine_dir or data_dir / "machine" / str(args.shot)
    ).expanduser().resolve()
    fit_path = args.fit_path.expanduser().resolve()
    output_dir = (
        args.output_dir
        or default_output_dir(WORKSPACE_ROOT, str(args.shot), args.target_time)
    ).expanduser().resolve()
    artifact_dir = (
        args.artifact_dir
        or default_artifact_dir(WORKSPACE_ROOT, str(args.shot), args.target_time)
    ).expanduser().resolve()

    if not shot_path.is_dir():
        raise FileNotFoundError(f"MAST Zarr not found: {shot_path}")
    if not fit_path.is_file():
        raise FileNotFoundError(f"Lao fit file not found: {fit_path}")
    MachineGeometry.load(machine_dir)

    solve_machine_dir = forward._copy_machine_with_positive_widths(machine_dir)
    shot = zarr.open_group(str(shot_path), mode="r")
    correct_magnetic_probes_in_machine_dir(solve_machine_dir, shot["magnetics"])
    tokamak = build_machine(MachineGeometry.load(solve_machine_dir))
    currents = forward._apply_currents(
        tokamak,
        shot,
        solve_machine_dir,
        args.target_time,
        current_scale=args.coil_current_scale,
    )

    fit = np.load(fit_path)
    fit_index = forward.select_fit_row(fit, str(args.shot), args.target_time)
    lao85_parameters = forward.apply_lao85_perturbation(
        Ip=float(fit["ip"][fit_index]),
        fvac=float(fit["fvac"][fit_index]),
        alpha=np.asarray(fit["freegsnke_alpha"][fit_index]).tolist(),
        beta=np.asarray(fit["freegsnke_beta"][fit_index]).tolist(),
        ip_scale=args.ip_scale,
        fvac_scale=args.fvac_scale,
        alpha_scale=args.alpha_scale,
        beta_scale=args.beta_scale,
        alpha_offset=args.alpha_offset,
        beta_offset=args.beta_offset,
    )

    grid_bounds = forward.equilibrium_grid_bounds(shot["equilibrium"])
    eq = equilibrium_update.Equilibrium(
        tokamak=tokamak,
        Rmin=grid_bounds["Rmin"],
        Rmax=grid_bounds["Rmax"],
        Zmin=grid_bounds["Zmin"],
        Zmax=grid_bounds["Zmax"],
        nx=args.nx,
        ny=args.ny,
    )
    profiles = Lao85(
        eq=eq,
        Ip=lao85_parameters["Ip"],
        fvac=lao85_parameters["fvac"],
        alpha=lao85_parameters["alpha"],
        beta=lao85_parameters["beta"],
    )
    solver = GSstaticsolver.NKGSsolver(eq)
    solver_diagnostics = forward.solve_with_diagnostics(
        solver=solver,
        eq=eq,
        profiles=profiles,
        requested_tolerance=args.tolerance,
        max_iterations=args.max_iterations,
    )
    topology_diagnostics = forward.equilibrium_topology_diagnostics(eq)
    tokamak.probes.initialise_setup(eq)

    rows = build_magnetic_diagnostic_comparison(
        tokamak=tokamak,
        eq=eq,
        magnetics=shot["magnetics"],
        target_time=args.target_time,
        flux_loop_scale=args.flux_loop_scale,
    )
    summary = summarize_comparisons(rows)
    current_constraint = current_constraint_comparison(
        model_ip=lao85_parameters["Ip"],
        magnetics=shot["magnetics"],
        target_time=args.target_time,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "freegsnke_equilibrium.npz",
        psi=eq.psi(),
        R=eq.R,
        Z=eq.Z,
        psi_axis=eq.psi_axis,
        psi_bndry=eq.psi_bndry,
    )
    np.savez_compressed(output_dir / "diagnostic_comparison.npz", rows=rows)
    csv_path = output_dir / "diagnostic_comparison.csv"
    write_comparison_csv(csv_path, rows)
    plot_path = artifact_dir / "diagnostic_observed_vs_model.png"
    save_observed_vs_model_plot(plot_path, rows, str(args.shot), args.target_time)
    current_plot_path = artifact_dir / "current_global_constraint.png"
    save_current_constraint_plot(
        current_plot_path, current_constraint, str(args.shot), args.target_time
    )

    metadata = {
        "shot": str(args.shot),
        "target_time": args.target_time,
        "fitted_time": float(fit["time"][fit_index]),
        "machine_geometry_source": str(machine_dir),
        "fit_path": str(fit_path),
        "coil_currents": currents,
        "Ip": lao85_parameters["Ip"],
        "fvac": lao85_parameters["fvac"],
        "alpha": lao85_parameters["alpha"],
        "beta": lao85_parameters["beta"],
        "lao85_perturbation": lao85_parameters["perturbation"],
        "coil_current_scale": args.coil_current_scale,
        "flux_loop_scale": args.flux_loop_scale,
        "grid": {"nx": args.nx, "ny": args.ny, **grid_bounds},
        "target_relative_tolerance": args.tolerance,
        "max_solving_iterations": args.max_iterations,
        "current_constraint": current_constraint,
        "comparison_summary": summary,
        "outputs": {
            "equilibrium_npz": str(output_dir / "freegsnke_equilibrium.npz"),
            "comparison_npz": str(output_dir / "diagnostic_comparison.npz"),
            "comparison_csv": str(csv_path),
            "plot": str(plot_path),
            "current_plot": str(current_plot_path),
        },
    }
    metadata.update(solver_diagnostics)
    metadata.update(topology_diagnostics)
    summary_path = output_dir / "diagnostic_summary.json"
    summary_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"shot: {args.shot}")
    print(f"target_time: {args.target_time}")
    print(f"solver_status: {metadata['solver_status']}")
    print(f"matched_diagnostics: {summary['total']['count']}")
    print(f"mean_abs_error: {summary['total']['mean_abs_error']}")
    print(f"comparison_csv: {csv_path}")
    print(f"summary: {summary_path}")
    print(f"plot: {plot_path}")
    print(f"current_plot: {current_plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
