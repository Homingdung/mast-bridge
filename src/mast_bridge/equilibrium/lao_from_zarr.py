from __future__ import annotations

from pathlib import Path
from typing import Iterable, Any

import numpy as np


FIT_FILENAME = "all_zarr_lao_parameter_fits.npz"


def default_lao_fit_path(workspace_root: str | Path) -> Path:
    return (
        Path(workspace_root)
        / "data"
        / "processed"
        / "real"
        / "lao_parameter_ensemble"
        / FIT_FILENAME
    )


def _array(group: Any, name: str) -> np.ndarray:
    if name not in group:
        raise ValueError(f"Zarr group {group.name!r} is missing required array: {name}")
    return np.asarray(group[name][:])


def _profile_at_time(values: np.ndarray, time_index: int, time_count: int) -> np.ndarray:
    if values.ndim != 2:
        raise ValueError(f"Expected 2D profile array, got shape {values.shape}")
    if values.shape[1] == time_count:
        return np.asarray(values[:, time_index], dtype=float)
    if values.shape[0] == time_count:
        return np.asarray(values[time_index, :], dtype=float)
    raise ValueError(f"Profile array shape {values.shape} does not match {time_count} times")


def _fit_lao_coefficients(
    psi_norm: np.ndarray,
    pprime: np.ndarray,
    ffprime: np.ndarray,
    *,
    n_alpha: int,
    n_beta: int,
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(psi_norm) & np.isfinite(pprime) & np.isfinite(ffprime)
    if mask.sum() < max(n_alpha, n_beta) + 1:
        raise ValueError("Not enough finite profile points for Lao fit")

    pn = np.asarray(psi_norm[mask], dtype=float)
    p = np.asarray(pprime[mask], dtype=float)
    ff = np.asarray(ffprime[mask], dtype=float)
    order = np.argsort(pn)
    pn = pn[order]
    p = p[order]
    ff = ff[order]

    alpha_design = pn[:, None] ** np.arange(n_alpha)[None, :]
    alpha_design -= pn[:, None] ** n_alpha
    beta_design = pn[:, None] ** np.arange(n_beta)[None, :]
    beta_design -= pn[:, None] ** n_beta
    alpha = np.linalg.lstsq(alpha_design, p, rcond=None)[0]
    beta = np.linalg.lstsq(beta_design, ff, rcond=None)[0]
    return alpha.astype(float), beta.astype(float)


def _shot_rows(
    zarr_path: str | Path,
    *,
    n_alpha: int,
    n_beta: int,
    min_finite_points: int,
) -> list[dict[str, Any]]:
    try:
        import zarr
    except ImportError as exc:
        raise RuntimeError("Building Lao fit NPZ requires the 'zarr' package") from exc

    path = Path(zarr_path).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"MAST shot Zarr not found: {path}")

    root = zarr.open_group(str(path), mode="r")
    if "equilibrium" not in root:
        raise ValueError(f"Shot Zarr is missing equilibrium group: {path}")
    if "magnetics" not in root:
        raise ValueError(f"Shot Zarr is missing magnetics group: {path}")

    equilibrium = root["equilibrium"]
    magnetics = root["magnetics"]
    times = np.asarray(_array(equilibrium, "time"), dtype=float)
    psi_norm = np.asarray(_array(equilibrium, "psi_norm"), dtype=float)
    pprime = np.asarray(_array(equilibrium, "dpressure_dpsi"), dtype=float)
    ffprime = np.asarray(_array(equilibrium, "f_df_dpsi"), dtype=float)
    bvac = np.asarray(_array(equilibrium, "bvac_rmag"), dtype=float)
    ip_times = np.asarray(_array(magnetics, "time"), dtype=float)
    ip_values = np.asarray(_array(magnetics, "ip"), dtype=float)

    finite_ip = np.isfinite(ip_times) & np.isfinite(ip_values)
    if finite_ip.sum() < 2:
        raise ValueError(f"Shot {path.stem} has fewer than two finite magnetics/ip points")
    ip_times = ip_times[finite_ip]
    ip_values = ip_values[finite_ip]

    shot = path.stem.removesuffix(".zarr")
    rows: list[dict[str, Any]] = []
    for time_index, time_value in enumerate(times):
        p_at_time = _profile_at_time(pprime, time_index, len(times))
        ff_at_time = _profile_at_time(ffprime, time_index, len(times))
        finite_profile = np.isfinite(psi_norm) & np.isfinite(p_at_time) & np.isfinite(ff_at_time)
        if finite_profile.sum() < min_finite_points:
            continue
        if time_index >= len(bvac) or not np.isfinite(bvac[time_index]):
            continue
        alpha, beta = _fit_lao_coefficients(
            psi_norm,
            p_at_time,
            ff_at_time,
            n_alpha=n_alpha,
            n_beta=n_beta,
        )
        rows.append(
            {
                "shot": shot,
                "time": float(time_value),
                "ip": float(np.interp(time_value, ip_times, ip_values)),
                "fvac": abs(float(bvac[time_index])),
                "freegsnke_alpha": alpha,
                "freegsnke_beta": beta,
            }
        )
    return rows


def build_lao_fit_table(
    zarr_paths: Iterable[str | Path],
    *,
    n_alpha: int = 3,
    n_beta: int = 3,
    min_finite_points: int = 8,
) -> dict[str, np.ndarray]:
    rows: list[dict[str, Any]] = []
    for zarr_path in zarr_paths:
        rows.extend(
            _shot_rows(
                zarr_path,
                n_alpha=n_alpha,
                n_beta=n_beta,
                min_finite_points=min_finite_points,
            )
        )
    if not rows:
        raise ValueError("No valid Lao fit rows found")

    return {
        "shot": np.asarray([row["shot"] for row in rows], dtype=str),
        "time": np.asarray([row["time"] for row in rows], dtype=float),
        "ip": np.asarray([row["ip"] for row in rows], dtype=float),
        "fvac": np.asarray([row["fvac"] for row in rows], dtype=float),
        "freegsnke_alpha": np.vstack([row["freegsnke_alpha"] for row in rows]),
        "freegsnke_beta": np.vstack([row["freegsnke_beta"] for row in rows]),
    }


def write_lao_fit_npz(
    zarr_paths: Iterable[str | Path],
    output_path: str | Path,
    *,
    n_alpha: int = 3,
    n_beta: int = 3,
    min_finite_points: int = 8,
) -> dict[str, np.ndarray]:
    table = build_lao_fit_table(
        zarr_paths,
        n_alpha=n_alpha,
        n_beta=n_beta,
        min_finite_points=min_finite_points,
    )
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **table)
    return table
