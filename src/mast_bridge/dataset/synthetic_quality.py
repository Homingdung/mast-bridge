"""Post-hoc physical quality criteria for synthetic equilibria (§34).

Implements the criteria that can be evaluated after solving (criteria 3/4/6/7),
plus helpers that read solver-side instrumentation (criteria 2/5) from the
metadata written by ``run_freegsnke_forward.py``.

Criterion numbering (progress.md §34.1):
  1. GS nonlinear residual <= 1e-8            -> metadata (existing)
  2. |∫Jφ - Ip|/Ip < 1e-6                      -> solver-side: ip_integrated
  3. p(s) >= 0                                  -> analytic Lao85 integration
  4. F²(s) > 0                                  -> analytic Lao85 integration
  5. Jφ strictly inside limiter mask            -> solver-side: jtor_outside_limiter_fraction
  6. ψ_norm ∈ [0,1] inside mask                 -> equilibrium.npz + limiter mask
  7. topology (limited/diverted) matches EFIT   -> solve vs EFIT zarr x_point_r
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np

# EFIT stores non-finite X points as a sentinel far outside the machine
# (e.g. -9.99); any value with |R| > 5 m is treated as "no X point".
EFIT_XPOINT_SENTINEL_R = 5.0

# Limiter contour is drawn on the same (R, Z) grid as the equilibrium.
# psi_norm outside [0,1] inside the limiter indicates a non-physical solve.
PSI_NORM_LO, PSI_NORM_HI = 0.0, 1.0

# Relative tolerance for |∫Jφ - Ip| / Ip (criterion 2).
IP_INTEGRATED_REL_TOL = 1e-6

# Criterion 5: any Jφ current density outside the limiter mask (fraction of
# the total |Jφ| integrated over the grid) must stay below this threshold.
JTOR_OUTSIDE_LIMITER_MAX = 1e-12


def analytic_alpha(alpha: list[float] | np.ndarray) -> np.ndarray:
    """Return the extended Lao85 alpha coefficients.

    FreeGS4E's ``Lao85.initialize_profile`` appends ``-sum(alpha)`` so that
    p'(1) = 0 (alpha_logic).  This reproduces that extension for the analytic
    pressure integration used by criterion 3.
    """
    alpha = np.asarray(alpha, dtype=float)
    return np.concatenate((alpha, [-float(np.sum(alpha))]))


def analytic_beta(beta: list[float] | np.ndarray) -> np.ndarray:
    """Return the extended Lao85 beta coefficients (beta_logic, ff'(1)=0)."""
    beta = np.asarray(beta, dtype=float)
    return np.concatenate((beta, [-float(np.sum(beta))]))


def analytic_pressure(alpha: Iterable[float], s: np.ndarray) -> np.ndarray:
    """Integrate p'(s) analytically, mirroring FreeGS4E ``Lao85.pressure``.

    p(s) = Σ_i a_i/(i+1) * (1 - s^(i+1))     (ignoring L and (ψ_axis-ψ_bndry)
    scaling factors, which are strictly positive or strictly negative for all
    samples and therefore do not affect the sign of p(s)).

    Criterion 3 passes iff p(s) >= 0 for all s in [0, 1].
    """
    a = analytic_alpha(alpha)
    s = np.asarray(s, dtype=float)
    result = np.zeros_like(s, dtype=float)
    for i, coef in enumerate(a):
        result += coef / (i + 1) * (1.0 - s ** (i + 1))
    return result


def analytic_fpol_squared(
    fvac: float, beta: Iterable[float], s: np.ndarray, delta_psi: float = 1.0
) -> np.ndarray:
    """Compute F²(s) = fvac² + 2 Δψ ∫_s^1 ff'(u) du analytically.

    Mirrors FreeGS4E ``Lao85.fpol`` with L * Raxis set to 1 (both strictly
    positive; Raxis defaults to 1).  Δψ = ψ_axis - ψ_bndry carries the sign of
    the poloidal flux convention; without it the sign of F²(s) would be wrong
    for samples where ψ_axis < ψ_bndry.

    Criterion 4 passes iff F²(s) > 0 for all s in [0, 1].
    """
    b = analytic_beta(beta)
    s = np.asarray(s, dtype=float)
    integral = np.zeros_like(s, dtype=float)
    for i, coef in enumerate(b):
        integral += coef / (i + 1) * (1.0 - s ** (i + 1))
    return float(fvac) ** 2 + 2.0 * float(delta_psi) * integral


def psi_norm(psi: np.ndarray, psi_axis: float, psi_bndry: float) -> np.ndarray:
    """Normalised poloidal flux (psi - psi_axis) / (psi_bndry - psi_axis)."""
    denom = float(psi_bndry) - float(psi_axis)
    if abs(denom) < 1e-12:
        return np.full_like(psi, np.nan, dtype=float)
    return (np.asarray(psi, dtype=float) - float(psi_axis)) / denom


def limiter_mask_from_machine(
    R: np.ndarray, Z: np.ndarray, machine_dir: Path
) -> np.ndarray | None:
    """Rebuild the limiter mask on the equilibrium grid from the machine pickle.

    Mirrors FreeGSNKE's ``Limiter_handler.build_mask_inside_limiter`` using the
    limiter contour vertices stored in the machine pickle.
    """
    try:
        from matplotlib.path import Path as MplPath

        import pickle

        # Machine pickles use a MAST_ prefix (e.g. MAST_wall.pickle); accept
        # either naming convention.
        candidates = [
            Path(machine_dir) / "wall.pickle",
            Path(machine_dir) / "MAST_wall.pickle",
            Path(machine_dir) / "MAST_limiter.pickle",
        ]
        wall = None
        for candidate in candidates:
            if candidate.is_file():
                with candidate.open("rb") as handle:
                    wall = pickle.load(handle)
                break
        if wall is None:
            return None
        if isinstance(wall, dict):
            machine_payload = wall
            wall = wall.get("limiter") or wall.get("wall")
        # MAST wall pickles are a list of {R, Z} contour segments.
        if isinstance(wall, (list, tuple)) and wall and isinstance(wall[0], dict):
            r_parts, z_parts = [], []
            for segment in wall:
                seg_R = np.asarray(segment.get("R"), dtype=float)
                seg_Z = np.asarray(segment.get("Z"), dtype=float)
                if seg_R.ndim == 0:
                    seg_R = seg_R.reshape(1)
                    seg_Z = seg_Z.reshape(1)
                r_parts.append(seg_R)
                z_parts.append(seg_Z)
            limiter_R = np.concatenate(r_parts)
            limiter_Z = np.concatenate(z_parts)
        else:
            limiter_R = np.atleast_1d(np.asarray(getattr(wall, "R", None), dtype=float))
            limiter_Z = np.atleast_1d(np.asarray(getattr(wall, "Z", None), dtype=float))
        if limiter_R.size < 3 or limiter_R.size != limiter_Z.size:
            return None
        verts = np.stack((limiter_R, limiter_Z), axis=-1)
        path = MplPath(verts)
        points = np.stack(
            (
                np.broadcast_to(R, (R.shape[0], Z.shape[1])).reshape(-1),
                np.broadcast_to(Z, (R.shape[0], Z.shape[1])).reshape(-1),
            ),
            axis=-1,
        )
        inside = path.contains_points(points).reshape(R.shape[0], Z.shape[1])
        return inside.astype(bool)
    except Exception:  # noqa: BLE001 - optional post-hoc path
        return None


def psi_norm_range_in_mask(
    psi: np.ndarray,
    psi_axis: float,
    psi_bndry: float,
    mask: np.ndarray | None,
) -> tuple[float, float] | None:
    """Return (min, max) of ψ_norm on the masked grid points, or None."""
    pn = psi_norm(psi, psi_axis, psi_bndry)
    if mask is None or mask.shape != pn.shape:
        return None
    values = pn[mask]
    if values.size == 0 or not np.isfinite(values).all():
        return None
    return float(values.min()), float(values.max())


def plasma_mask_from_psi(
    R: np.ndarray, Z: np.ndarray, psi: np.ndarray, psi_axis: float, psi_bndry: float
) -> np.ndarray | None:
    """Rebuild the plasma (LCFS-internal) mask from the psi field.

    Uses freegs4e's critical-point machinery when available; falls back to a
    psi_norm ∈ [0, 1] threshold (which is the natural definition of the core).
    Returns None when freegs4e is unavailable and the threshold fallback is
    degenerate.
    """
    try:
        from freegs4e import critical
    except ImportError:
        critical = None
    try:
        R = np.asarray(R, dtype=float)
        Z = np.asarray(Z, dtype=float)
        psi = np.asarray(psi, dtype=float)
        if critical is not None:
            opt, xpt = critical.find_critical(R, Z, psi)
            if len(opt) == 0:
                return None
            mask = critical.inside_mask(
                R, Z, psi, opt, xpt, psi_bndry=float(psi_bndry)
            )
            if mask is not None and mask.shape == psi.shape:
                return np.asarray(mask, dtype=bool)
        pn = psi_norm(psi, psi_axis, psi_bndry)
        return (pn >= PSI_NORM_LO) & (pn <= PSI_NORM_HI)
    except Exception:  # noqa: BLE001 - optional post-hoc path
        return None


@lru_cache(maxsize=4096)
def _efit_xpoint_series(shot_zarr_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Cached EFIT x_point count per frame + frame times for one shot."""
    try:
        import zarr

        group = zarr.open_group(str(shot_zarr_dir / "equilibrium"), mode="r")
        x_point_r = np.asarray(group["x_point_r"][:], dtype=float)
        times = np.asarray(group["time"][:], dtype=float)
        if x_point_r.ndim != 2 or x_point_r.shape[0] == 0 or times.size == 0:
            return None
        valid = np.isfinite(x_point_r) & (np.abs(x_point_r) < EFIT_XPOINT_SENTINEL_R)
        return times, valid.sum(axis=0)
    except Exception:  # noqa: BLE001 - optional post-hoc path
        return None


def efit_topology_at_time(shot_zarr_dir: Path, target_time: float | None) -> str | None:
    """EFIT topology at the frame nearest to target_time.

    'diverted' if >=1 X point at that frame, 'limited' if 0.  Uses the exact
    per-frame x_point count (not a time-axis aggregate) so the anchor is the
    real topology at the sample's own time.
    """
    series = _efit_xpoint_series(shot_zarr_dir)
    if series is None or target_time is None:
        return None
    times, counts = series
    idx = int(np.argmin(np.abs(times - float(target_time))))
    if times[idx] != times[idx] and times.size == 0:  # pragma: no cover
        return None
    if not np.isfinite(times[idx]):
        return None
    return "diverted" if int(counts[idx]) >= 1 else "limited"


def efit_xpoint_count(shot_zarr_dir: Path) -> int:
    """Count physically meaningful EFIT X points (R > sentinel) at the closest
    available times; return the median count across the recorded time axis.

    The EFIT store uses -9.99 as a sentinel for "no X point"; any |R| >= 5 is
    treated as no X point.

    Note: kept for backward compatibility; use ``efit_topology_at_time`` for
    the target-time-anchored topology used by criterion 7.
    """
    series = _efit_xpoint_series(shot_zarr_dir)
    if series is None:
        return -1
    times, per_time = series
    if per_time.size == 0:
        return -1
    counts = per_time[per_time > 0]
    if counts.size == 0:
        return 0
    return int(np.median(counts))


def solve_topology(metadata: dict[str, Any]) -> str | None:
    """Topology label from solve metadata: 'diverted' / 'limited' / None."""
    flag_limiter = metadata.get("flag_limiter")
    if flag_limiter is None:
        return None
    return "limited" if bool(flag_limiter) else "diverted"


@lru_cache(maxsize=4096)
def efit_topology(shot_zarr_dir: Path) -> str | None:
    """EFIT topology (time-axis aggregate): 'diverted' if >=1 X point at the
    median frame, 'limited' otherwise.  Backward-compatible helper; criterion
    7 uses ``efit_topology_at_time``."""
    series = _efit_xpoint_series(shot_zarr_dir)
    if series is None:
        return None
    times, per_time = series
    if per_time.size == 0:
        return None
    counts = per_time[per_time > 0]
    median_count = int(np.median(counts)) if counts.size else 0
    return "diverted" if median_count >= 1 else "limited"


def quality_rejection_reasons(
    metadata: dict[str, Any],
    equilibrium_path: Path | None,
    machine_dir: Path | None,
    shot_zarr_dir: Path | None = None,
) -> list[tuple[str, float]]:
    """Return [(reason, value)] for violated post-hoc quality criteria.

    Only criteria with the required data are evaluated; a criterion that
    cannot be evaluated (missing data) is skipped, never rejected.
    """
    reasons: list[tuple[str, float]] = []

    # --- criterion 3/4: analytic Lao85 profiles ---------------------------
    # FreeGS4E computes p(s) = L * norm_pressure(s) * (psi_axis - psi_bndry)
    # and F²(s) = fvac² + 2 * integral(s) * (psi_axis - psi_bndry), where
    # L > 0 (Ip normalisation).  The sign of (psi_axis - psi_bndry) must
    # therefore be included.
    alpha = metadata.get("alpha")
    beta = metadata.get("beta")
    psi_axis = metadata.get("psi_axis")
    psi_bndry = metadata.get("psi_bndry")
    delta_psi = None
    if isinstance(psi_axis, (int, float)) and isinstance(psi_bndry, (int, float)):
        delta_psi = float(psi_axis) - float(psi_bndry)
    if isinstance(alpha, (list, np.ndarray)) and len(alpha) == 3 and delta_psi is not None:
        s = np.linspace(0.0, 1.0, 201)
        p = analytic_pressure(alpha, s) * delta_psi
        if not np.isfinite(p).all() or float(np.min(p)) < 0.0:
            reasons.append(("pressure_negative", float(np.min(p)) if np.isfinite(p).all() else float("nan")))
    if isinstance(beta, (list, np.ndarray)) and len(beta) == 3 and delta_psi is not None:
        fvac = metadata.get("fvac")
        if isinstance(fvac, (int, float)):
            s = np.linspace(0.0, 1.0, 201)
            f2 = analytic_fpol_squared(fvac, beta, s, delta_psi=delta_psi)
            if not np.isfinite(f2).all() or float(np.min(f2)) <= 0.0:
                reasons.append(("fpol_squared_nonpositive", float(np.min(f2)) if np.isfinite(f2).all() else float("nan")))

    # --- criterion 6: psi_norm in [0, 1] inside plasma mask ----------------
    # Solver-side instrumentation (eq.mask = LCFS-internal) takes precedence;
    # fall back to a post-hoc reconstruction when the fields are absent.
    psi_norm_span = None
    if isinstance(metadata.get("psi_norm_in_mask_min"), (int, float)) and isinstance(
        metadata.get("psi_norm_in_mask_max"), (int, float)
    ):
        psi_norm_span = (
            float(metadata["psi_norm_in_mask_min"]),
            float(metadata["psi_norm_in_mask_max"]),
        )
    elif equilibrium_path is not None:
        try:
            with np.load(equilibrium_path, allow_pickle=False) as payload:
                psi = np.asarray(payload["psi"], dtype=float)
                R = np.asarray(payload["R"], dtype=float)
                Z = np.asarray(payload["Z"], dtype=float)
                psi_axis = float(payload["psi_axis"])
                psi_bndry = float(payload["psi_bndry"])
            mask = plasma_mask_from_psi(R, Z, psi, psi_axis, psi_bndry)
            if mask is not None:
                psi_norm_span = psi_norm_range_in_mask(psi, psi_axis, psi_bndry, mask)
        except (OSError, KeyError, ValueError):
            pass
    if psi_norm_span is not None:
        lo, hi = psi_norm_span
        if not (PSI_NORM_LO <= lo <= PSI_NORM_HI and PSI_NORM_LO <= hi <= PSI_NORM_HI):
            reasons.append(("psi_norm_mask_out_of_range", float(max(abs(lo), abs(hi)))))

    # --- criterion 7: topology vs EFIT at target_time ---------------------
    # 位形锚定 = EFIT 在 target_time 处的真实位形：solve 与之匹配即通过，
    # 不匹配才拒（topology_mismatch）。不做任何人为位形筛选（例如不因
    # EFIT/solve 为 limited 而直接拒绝）。
    if shot_zarr_dir is not None:
        solve_topo = solve_topology(metadata)
        efit_topo = efit_topology_at_time(
            shot_zarr_dir, metadata.get("target_time", metadata.get("fitted_time"))
        )
        if solve_topo is not None and efit_topo is not None and solve_topo != efit_topo:
            reasons.append(("topology_mismatch", float(1.0)))

    return reasons


def solver_instrumented_rejection_reasons(metadata: dict[str, Any]) -> list[tuple[str, float]]:
    """Evaluate criteria 2 and 5 from solver-side instrumentation.

    These are only available for samples solved after the instrumentation was
    added; older samples without the fields are skipped.
    """
    reasons: list[tuple[str, float]] = []

    ip = metadata.get("Ip")
    ip_integrated = metadata.get("ip_integrated")
    if isinstance(ip, (int, float)) and isinstance(ip_integrated, (int, float)):
        if abs(float(ip)) > 0.0:
            rel = abs(float(ip_integrated) - float(ip)) / abs(float(ip))
            if not np.isfinite(rel) or rel >= IP_INTEGRATED_REL_TOL:
                reasons.append(("ip_mismatch", float(rel)))

    outside = metadata.get("jtor_outside_limiter_fraction")
    if isinstance(outside, (int, float)):
        if not np.isfinite(float(outside)) or float(outside) > JTOR_OUTSIDE_LIMITER_MAX:
            reasons.append(("jtor_outside_limiter", float(outside)))

    return reasons


def rejection_reasons_v34(
    metadata: dict[str, Any],
    equilibrium_path: Path | None = None,
    machine_dir: Path | None = None,
    shot_zarr_dir: Path | None = None,
) -> list[tuple[str, float]]:
    """Full 7-criterion quality gate (§34.1).

    Criteria 1 (solver convergence) is handled by the existing
    ``synthetic_manifest.rejection_reason``; this helper covers 2-7 and returns
    a list of (reason, value) pairs, empty for a fully accepted sample.
    """
    reasons: list[tuple[str, float]] = []
    reasons.extend(solver_instrumented_rejection_reasons(metadata))
    reasons.extend(
        quality_rejection_reasons(
            metadata,
            equilibrium_path=equilibrium_path,
            machine_dir=machine_dir,
            shot_zarr_dir=shot_zarr_dir,
        )
    )
    return reasons
