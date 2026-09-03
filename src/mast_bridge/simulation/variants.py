from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

import numpy as np


def build_variant_rows(
    shots: Iterable[str],
    times: Iterable[float],
    variants_per_point: int,
    seed: int,
) -> list[dict[str, float | str]]:
    """Build deterministic bounded uniform perturbation rows.

    The rows encode perturbations applied later by ``run_freegsnke_forward.py``
    to each fitted Lao85 profile. To change sampling later, keep the public row
    fields stable so batch solve scripts continue to work.

    Perturbation ranges (v2, 2026-08-11): multiplicative ranges match the
    natural width of each parameter measured over the full 818,139-row fit
    ensemble (cleaned), i.e. scale = U(1 - CV, 1 + CV) with CV = sigma/median,
    truncated to the positive domain when CV >= 100%. fvac is a measured,
    near-constant quantity that is not part of the input feature schema, so it
    is left unperturbed (scale 1.0). Additive offsets are zeroed because a
    single offset cannot fit the a0 (~1e3) and a1/a2 (~5e5) magnitude scales.
    """
    if variants_per_point < 0:
        raise ValueError("variants_per_point must be non-negative")

    rng = random.Random(seed)
    rows: list[dict[str, float | str]] = []
    for shot in shots:
        for target_time in times:
            for index in range(variants_per_point):
                rows.append(
                    {
                        "shot": str(shot),
                        "target_time": float(target_time),
                        "variant_id": f"v{index:03d}",
                        "sampling_method": "uniform_random",
                        "ip_scale": rng.uniform(0.6, 1.4),
                        "fvac_scale": 1.0,
                        "alpha_scale": rng.uniform(0.5, 1.5),
                        "beta_scale": rng.uniform(0.5, 1.5),
                        "alpha_offset": 0.0,
                        "beta_offset": 0.0,
                        "coil_current_scale": rng.uniform(0.8, 1.2),
                    }
                )
    return rows


def rows_from_lao_fit_npz(
    fit_path: str | Path,
    *,
    variants_per_point: int,
    seed: int,
    min_time: float | None = None,
    max_time: float | None = None,
) -> list[dict[str, float | str]]:
    """Build perturbation rows for every fitted shot/time in a Lao fit NPZ."""
    with np.load(Path(fit_path).expanduser().resolve()) as fit:
        missing = {"shot", "time"} - set(fit.files)
        if missing:
            raise ValueError(f"Lao fit NPZ is missing required arrays: {sorted(missing)}")

        shots = [str(value) for value in fit["shot"].tolist()]
        times = [float(value) for value in fit["time"].tolist()]
    if len(shots) != len(times):
        raise ValueError("Lao fit NPZ shot/time arrays have different lengths")

    rows: list[dict[str, float | str]] = []
    for shot, target_time in zip(shots, times, strict=True):
        if min_time is not None and target_time < min_time:
            continue
        if max_time is not None and target_time > max_time:
            continue
        rows.extend(
            build_variant_rows(
                [shot],
                [target_time],
                variants_per_point=variants_per_point,
                seed=seed + len(rows),
            )
        )
    return rows
