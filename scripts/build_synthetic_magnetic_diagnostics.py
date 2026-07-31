#!/usr/bin/env python3
"""Backfill FreeGSNKE magnetic diagnostics without rerunning equilibrium solves."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import compare_freegsnke_magnetic_diagnostics as comparison  # noqa: E402
import run_freegsnke_forward as forward  # noqa: E402

from mast_bridge.simulation.synthetic_diagnostics import (  # noqa: E402
    synthetic_diagnostics_rejection_reason,
    write_synthetic_diagnostics,
)


STRICT_SOLVER_TOLERANCE = 1e-8


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate FreeGSNKE magnetic diagnostics from accepted saved equilibria "
            "without rerunning the Grad-Shafranov solver."
        )
    )
    parser.add_argument("--accepted-manifest", type=Path, required=True)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=WORKSPACE_ROOT / "data" / "raw" / "mast",
        help="Directory containing raw <shot>.zarr and machine/<shot>.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="JSONL report path; defaults beside the accepted manifest.",
    )
    parser.add_argument("--output-name", default="diagnostics.npz")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def reconstruct_plasma_psi(
    saved_total_psi: np.ndarray, coil_psi: np.ndarray
) -> np.ndarray:
    """Recover plasma flux from the saved total flux and current coil state."""
    total = np.asarray(saved_total_psi, dtype=float)
    coils = np.asarray(coil_psi, dtype=float)
    if total.shape != coils.shape:
        raise ValueError(
            f"Saved total psi shape {total.shape} does not match coil psi {coils.shape}"
        )
    if not np.isfinite(total).all() or not np.isfinite(coils).all():
        raise ValueError("Cannot reconstruct plasma psi from non-finite arrays")
    return total - coils


def _source_channel(payload: dict[str, Any]) -> str:
    leaf: Any = payload
    while isinstance(leaf, dict) and "source_channel" not in leaf:
        if not leaf:
            break
        leaf = next(iter(leaf.values()))
    if not isinstance(leaf, dict) or "source_channel" not in leaf:
        raise KeyError("Active coil payload is missing source_channel")
    return str(leaf["source_channel"])


def apply_saved_currents(
    tokamak: Any,
    *,
    active_payload: dict[str, Any],
    passive_payload: Iterable[dict[str, Any]],
    current_metadata: dict[str, Any],
) -> None:
    """Restore the exact active and passive currents stored with a solved sample."""
    active = {
        str(name): float(value)
        for name, value in (current_metadata.get("active") or {}).items()
    }
    passive = {
        str(name): float(value)
        for name, value in (current_metadata.get("passive") or {}).items()
    }
    tokamak.set_all_coil_currents(np.zeros(tokamak.n_coils))
    for coil_name, payload in active_payload.items():
        channel = _source_channel(payload)
        if channel not in active:
            raise KeyError(f"Saved active current is missing channel {channel!r}")
        tokamak.set_coil_current(str(coil_name), active[channel])
    for item in passive_payload:
        name = str(item["name"])
        channel = str(item["source_current_channel"])
        if name not in tokamak.coil_names:
            continue
        if channel not in passive:
            raise KeyError(f"Saved passive current is missing channel {channel!r}")
        tokamak.set_coil_current(name, passive[channel])


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.expanduser().resolve().open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    output = path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def _strictly_accepted(row: dict[str, Any]) -> bool:
    try:
        tolerance = float(row["solver_final_tolerance"])
    except (KeyError, TypeError, ValueError):
        return False
    return row.get("solver_converged") is True and tolerance <= STRICT_SOLVER_TOLERANCE


def _grid_from_equilibrium(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        total_psi = np.asarray(payload["psi"], dtype=float)
        radius = np.asarray(payload["R"], dtype=float)
        vertical = np.asarray(payload["Z"], dtype=float)
    if total_psi.shape != radius.shape or total_psi.shape != vertical.shape:
        raise ValueError("Saved psi, R, and Z arrays must have identical shapes")
    if total_psi.ndim != 2:
        raise ValueError("Saved equilibrium arrays must be two-dimensional")
    if not (
        np.allclose(radius, radius[:, :1])
        and np.allclose(vertical, vertical[:1, :])
    ):
        raise ValueError("Saved equilibrium grid must be rectilinear")
    return total_psi, radius, vertical


def _machine_payloads(machine_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from mast_bridge.mast.machine_config import MachineGeometry

    machine = MachineGeometry.load(machine_dir)
    with machine.files["active_coils"].open("rb") as handle:
        active_payload = pickle.load(handle)
    with machine.files["passive_coils"].open("rb") as handle:
        passive_payload = pickle.load(handle)
    return active_payload, passive_payload


def _reconstruct_equilibrium(
    row: dict[str, Any],
    tokamak: Any,
) -> Any:
    from scipy.interpolate import RectBivariateSpline
    from freegsnke import equilibrium_update
    from freegsnke.jtor_update import Lao85

    equilibrium_path = Path(row["equilibrium_path"]).expanduser().resolve()
    total_psi, radius, vertical = _grid_from_equilibrium(equilibrium_path)
    eq = equilibrium_update.Equilibrium(
        tokamak=tokamak,
        Rmin=float(radius[0, 0]),
        Rmax=float(radius[-1, 0]),
        Zmin=float(vertical[0, 0]),
        Zmax=float(vertical[0, -1]),
        nx=total_psi.shape[0],
        ny=total_psi.shape[1],
        psi=np.zeros_like(total_psi),
    )
    coil_psi = tokamak.getPsitokamak(eq._vgreen)
    eq.plasma_psi = reconstruct_plasma_psi(total_psi, coil_psi)
    eq.psi_func_interp = RectBivariateSpline(
        eq.R[:, 0], eq.Z[0, :], eq.plasma_psi
    )

    profiles = Lao85(
        eq=eq,
        Ip=float(row["Ip"]),
        fvac=float(row["fvac"]),
        alpha=np.asarray(row["alpha"], dtype=float).tolist(),
        beta=np.asarray(row["beta"], dtype=float).tolist(),
    )
    profiles.Jtor(
        eq.R,
        eq.Z,
        eq.psi(),
        psi_bndry=float(row["psi_bndry"]),
    )
    eq._profiles = profiles
    eq.psi_axis = float(row["psi_axis"])
    eq.psi_bndry = float(row["psi_bndry"])
    reconstruction_error = float(np.max(np.abs(eq.psi() - total_psi)))
    if reconstruction_error > 1e-11:
        raise ValueError(
            f"Reconstructed total psi differs from saved psi by {reconstruction_error:.3e}"
        )
    eq.synthetic_reconstruction_error = reconstruction_error
    return eq


def _write_row_diagnostics(
    row: dict[str, Any],
    *,
    tokamak: Any,
    active_payload: dict[str, Any],
    passive_payload: list[dict[str, Any]],
    output_name: str,
    initialise_probes: bool,
) -> tuple[Path, bool, float]:
    from mast_bridge.simulation.magnetic_diagnostics import (
        LEVEL2_FLUX_LOOP_SCALE,
        modeled_flux_loop_signals,
        modeled_pickup_signals,
    )

    apply_saved_currents(
        tokamak,
        active_payload=active_payload,
        passive_payload=passive_payload,
        current_metadata=row["coil_currents"],
    )
    eq = _reconstruct_equilibrium(row, tokamak)
    if initialise_probes:
        tokamak.probes.initialise_setup(eq)
    flux_loops = modeled_flux_loop_signals(
        tokamak, eq, scale=LEVEL2_FLUX_LOOP_SCALE
    )
    pickups = modeled_pickup_signals(tokamak, eq)
    if pickups.families is None:
        raise ValueError("FreeGSNKE pickup payload is missing probe family labels")
    output = Path(row["data_path"]).expanduser().resolve() / output_name
    write_synthetic_diagnostics(
        output,
        target_time=float(row["target_time"]),
        magnetics_ip=float(row["Ip"]),
        flux_loop_names=flux_loops.names,
        flux_loop_values=flux_loops.values,
        pickup_names=pickups.names,
        pickup_families=pickups.families,
        pickup_values=pickups.values,
        active_coil_currents=row["coil_currents"]["active"],
        flux_loop_scale=LEVEL2_FLUX_LOOP_SCALE,
    )
    return output, True, float(eq.synthetic_reconstruction_error)


def _default_report_path(manifest: Path) -> Path:
    resolved = manifest.expanduser().resolve()
    return resolved.with_name(f"{resolved.stem}_diagnostics_report.jsonl")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = args.accepted_manifest.expanduser().resolve()
    rows = [row for row in _read_jsonl(manifest) if _strictly_accepted(row)]
    if args.sample_id:
        selected = set(args.sample_id)
        rows = [row for row in rows if str(row.get("sample_id")) in selected]
    if args.limit is not None:
        rows = rows[: max(args.limit, 0)]
    report_path = (
        args.report.expanduser().resolve()
        if args.report is not None
        else _default_report_path(manifest)
    )

    print(f"accepted_rows_selected: {len(rows)}")
    print(f"output_name: {args.output_name}")
    print(f"report: {report_path}")
    if args.dry_run:
        for row in rows[:5]:
            print(f"[dry-run] {row['sample_id']}")
        return 0

    from mast_bridge.mast.machine_config import MachineGeometry
    from mast_bridge.simulation.freegsnke_runner import build_machine

    import zarr

    data_dir = args.data_dir.expanduser().resolve()
    report_rows: list[dict[str, Any]] = []
    generated = 0
    skipped = 0
    failed = 0

    rows_by_shot: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_shot.setdefault(str(row["parent_shot"]), []).append(row)

    for shot, shot_rows in rows_by_shot.items():
        raw_machine_dir = data_dir / "machine" / shot
        raw_shot_path = data_dir / f"{shot}.zarr"
        temporary_machine_dir: Path | None = None
        try:
            shot_group = zarr.open_group(str(raw_shot_path), mode="r")
            temporary_machine_dir = forward._copy_machine_with_positive_widths(
                raw_machine_dir
            )
            comparison.correct_magnetic_probes_in_machine_dir(
                temporary_machine_dir, shot_group["magnetics"]
            )
            machine = MachineGeometry.load(temporary_machine_dir)
            tokamak = build_machine(machine)
            active_payload, passive_payload = _machine_payloads(
                temporary_machine_dir
            )
            probes_initialised = False

            for row in shot_rows:
                sample_id = str(row["sample_id"])
                output = Path(row["data_path"]).expanduser().resolve() / args.output_name
                if (
                    not args.overwrite
                    and synthetic_diagnostics_rejection_reason(
                        output,
                        expected_target_time=float(row["target_time"]),
                    )
                    is None
                ):
                    skipped += 1
                    report_rows.append(
                        {
                            "sample_id": sample_id,
                            "status": "skipped_existing",
                            "diagnostics_path": str(output),
                        }
                    )
                    continue
                try:
                    output, _, reconstruction_error = _write_row_diagnostics(
                        row,
                        tokamak=tokamak,
                        active_payload=active_payload,
                        passive_payload=passive_payload,
                        output_name=args.output_name,
                        initialise_probes=not probes_initialised,
                    )
                    probes_initialised = True
                    generated += 1
                    report_rows.append(
                        {
                            "sample_id": sample_id,
                            "status": "generated",
                            "diagnostics_path": str(output),
                            "psi_reconstruction_max_abs_error": reconstruction_error,
                        }
                    )
                    print(f"[generated] {sample_id}")
                except Exception as exc:
                    failed += 1
                    report_rows.append(
                        {
                            "sample_id": sample_id,
                            "status": "failed",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
                    print(f"[failed] {sample_id}: {type(exc).__name__}: {exc}")
                _write_report(report_path, report_rows)
        except Exception as exc:
            failed += len(shot_rows)
            for row in shot_rows:
                report_rows.append(
                    {
                        "sample_id": str(row["sample_id"]),
                        "status": "failed_shot_setup",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
            _write_report(report_path, report_rows)
        finally:
            if temporary_machine_dir is not None:
                shutil.rmtree(temporary_machine_dir, ignore_errors=True)

    _write_report(report_path, report_rows)
    print(f"generated: {generated}")
    print(f"skipped_existing: {skipped}")
    print(f"failed: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
