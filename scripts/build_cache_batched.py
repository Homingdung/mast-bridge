#!/usr/bin/env python3
"""Batched dataset-cache builder for the PCA 01σ curve experiment.

Why batched: the row-wise builder (build_dataset_cache.py) re-opens the shot
zarr for every row (~6s/row on spinning/network FS), making a 213k-row real
manifest take ~7-8 h.  This version groups rows by shot, opens each zarr ONCE,
pre-loads the arrays needed for every target time of that shot, then extracts
all rows of the shot.  Numerically identical to the row-wise path (same
interpolation and psi logic, copied verbatim from tokamind_manifest).

Output npz: {sample_ids (N,), features (N,69) f32, psi (N,65,65) f32} in
manifest row order (rows with non-finite features/psi are dropped).

Usage:
  build_cache_batched.py --manifest FILE --output FILE [--workers 48]
      [--diagnostics-name diagnostics.npz]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

SCRIPT_ROOT = Path(__file__).resolve().parents[0]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))
sys.path.insert(0, str(SCRIPT_ROOT.parents[0] / "src"))

WORKSPACE_ROOT = SCRIPT_ROOT.parents[1]
SCALERS_PATH = WORKSPACE_ROOT / "runs" / "tokamind-large-real-scratch-100e" / "manifest_scalers.npz"
PICKUP_FAMILIES = (
    ("CCBV", "b_field_pol_probe_ccbv"),
    ("OBR", "b_field_pol_probe_obr"),
    ("OBV", "b_field_pol_probe_obv"),
)


def _nearest_index(times: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(np.asarray(times, dtype=float) - float(target))))


def _arr(group: object, name: str) -> np.ndarray:
    return np.asarray(group[name][:], dtype=float)


def _extract_shot(args: tuple[Path, Path, str, list[dict], float, list[str]]) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Extract (sample_id, features, psi) for all rows of one shot."""
    data_dir, fit_path, shot, shot_rows, ip_scale, feature_names = args
    out: list[tuple[str, np.ndarray, np.ndarray]] = []
    try:
        import zarr

        root = zarr.open_group(str(data_dir / f"{shot}.zarr"), mode="r")
        magnetics = root["magnetics"]
        m_times = _arr(magnetics, "time")
        m_ip = _arr(magnetics, "ip")
        fl_names = [str(v) for v in magnetics["flux_loop_channel"][:]]
        fl_flux = np.asarray(magnetics["flux_loop_flux"][:], dtype=float)
        # pickups
        pk_items: list[tuple[str, str, np.ndarray]] = []  # (family, name, field)
        for family, prefix in PICKUP_FAMILIES:
            channel_key, field_key = f"{prefix}_channel", f"{prefix}_field"
            if channel_key in magnetics and field_key in magnetics:
                names = [str(v) for v in magnetics[channel_key][:]]
                field = np.asarray(magnetics[field_key][:], dtype=float)
                pk_items.append((family, names, field))
        # active coils
        active = root["pf_active"]
        a_times = _arr(active, "time")
        a_channels = [str(v) for v in active["current_channel"][:]]
        a_current = np.asarray(active["coil_current"][:], dtype=float)
        # equilibrium psi
        eq = root["equilibrium"]
        eq_times = _arr(eq, "time")
        eq_psi = np.asarray(eq["psi"][:], dtype=np.float32)  # [Z, R, time]

        for row in shot_rows:
            sid = str(row["sample_id"])
            tt = float(row["target_time"])
            # --- features (magnetic diagnostics) ---
            mag_ip = float(np.interp(tt, m_times, m_ip))
            flux = {
                f"flux_loop_{name}": float(np.interp(tt, m_times, fl_flux[i]))
                for i, name in enumerate(fl_names)
            }
            pickups: dict[str, float] = {}
            for family, names, field in pk_items:
                for i, name in enumerate(names):
                    pickups[f"pickup_{family}_{name}"] = float(np.interp(tt, m_times, field[i]))
            active_vals = {
                ch: float(np.interp(tt, a_times, a_current[i]))
                for i, ch in enumerate(a_channels)
            }
            values = {
                "target_time": tt,
                "magnetics_ip": mag_ip,
                **flux,
                **pickups,
                **{f"coil_active_{k}": v for k, v in active_vals.items()},
            }
            entries = []
            ok = True
            for name in feature_names:
                if name.startswith("coil_active_"):
                    entries.append(0.0 if not np.isfinite(values.get(name, np.nan)) else float(values[name]))
                else:
                    entries.append(values.get(name, np.nan))
            fv = np.asarray(entries, dtype=np.float32)
            if not np.isfinite(fv).all():
                continue
            # --- psi label ---
            idx = _nearest_index(eq_times, tt)
            psi = np.ascontiguousarray(np.asarray(eq_psi[:, :, idx], dtype=np.float32).T)
            if psi.shape != (65, 65) or not np.isfinite(psi).all():
                continue
            out.append((sid, fv, psi))
    except Exception as exc:  # noqa: BLE001 - per-shot tolerance
        print(f"[warn] shot {shot}: {type(exc).__name__}: {exc}", flush=True)
    return out


def _extract_synth(args: tuple[dict, list[str], str]) -> tuple[str, np.ndarray, np.ndarray] | None:
    """Extract one synthetic row: features from diagnostics.npz, psi from equilibrium.npz."""
    row, feature_names, diagnostics_name = args
    try:
        data_dir = Path(row["data_path"]).expanduser().resolve()
        diag_path = row.get("diagnostics_path") or (data_dir / diagnostics_name)
        with np.load(diag_path, allow_pickle=True) as d:
            payload = {
                "target_time": float(d["target_time"]),
                "magnetics_ip": float(d["magnetics_ip"]),
                "flux_loop_names": [str(v) for v in d["flux_loop_names"]],
                "flux_loop_values": [float(v) for v in d["flux_loop_values"]],
                "pickup_names": [str(v) for v in d["pickup_names"]],
                "pickup_families": [str(v) for v in d["pickup_families"]],
                "pickup_values": [float(v) for v in d["pickup_values"]],
                "active_coil_names": [str(v) for v in d["active_coil_names"]],
                "active_coil_values": [float(v) for v in d["active_coil_values"]],
            }
        values: dict[str, float] = {
            "target_time": payload["target_time"],
            "magnetics_ip": payload["magnetics_ip"],
        }
        values.update(
            {f"flux_loop_{n}": v for n, v in zip(payload["flux_loop_names"], payload["flux_loop_values"])}
        )
        values.update(
            {
                f"pickup_{fam}_{n}": v
                for fam, n, v in zip(
                    payload["pickup_families"], payload["pickup_names"], payload["pickup_values"]
                )
            }
        )
        values.update(
            {
                f"coil_active_{n}": v
                for n, v in zip(payload["active_coil_names"], payload["active_coil_values"])
            }
        )
        entries = []
        for name in feature_names:
            if name.startswith("coil_active_"):
                entries.append(0.0 if not np.isfinite(values.get(name, np.nan)) else float(values[name]))
            else:
                entries.append(values.get(name, np.nan))
        fv = np.asarray(entries, dtype=np.float32)
        if not np.isfinite(fv).all():
            return None
        eq_path = row.get("equilibrium_path") or (data_dir / "equilibrium.npz")
        with np.load(eq_path) as eq:
            psi = np.asarray(eq["psi"], dtype=np.float32)
        if psi.shape != (65, 65) or not np.isfinite(psi).all():
            return None
        return str(row.get("sample_id")), fv, psi
    except Exception as exc:  # noqa: BLE001 - per-row tolerance
        print(f"[warn] {row.get('sample_id')}: {type(exc).__name__}: {exc}", flush=True)
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=WORKSPACE_ROOT / "data" / "raw" / "mast")
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument(
        "--diagnostics-name",
        type=str,
        default="diagnostics.npz",
        help="Diagnostics file name inside synthetic sample dirs (synth rows).",
    )
    args = parser.parse_args(argv)

    with np.load(SCALERS_PATH, allow_pickle=True) as data:
        feature_names = [str(v) for v in data["feature_names"].tolist()]

    rows = [json.loads(line) for line in args.manifest.expanduser().resolve().open(encoding="utf-8") if line.strip()]
    by_shot: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        key = str(row.get("shot_id") or row.get("parent_shot"))
        by_shot[key].append(row)
    print(f"rows={len(rows)} shots={len(by_shot)}", flush=True)

    data_dir = args.data_dir.expanduser().resolve()
    fit_path = WORKSPACE_ROOT / "data" / "processed" / "real" / "lao_parameter_ensemble" / "all_zarr_lao_parameter_fits.npz"

    t0 = time.time()
    results: list[tuple[str, np.ndarray, np.ndarray]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        if all(str(row.get("source")) == "real" for row in rows):
            tasks = [
                (data_dir, fit_path, shot, shot_rows, 1.0, feature_names)
                for shot, shot_rows in sorted(by_shot.items())
            ]
            for i, extracted in enumerate(pool.map(_extract_shot, tasks, chunksize=1)):
                results.extend(extracted)
                if (i + 1) % 200 == 0:
                    print(
                        f"[batched] shots {i + 1}/{len(tasks)} "
                        f"rows {len(results)}/{len(rows)} ({time.time() - t0:.0f}s)",
                        flush=True,
                    )
        else:
            tasks = [(row, feature_names, args.diagnostics_name) for row in rows]
            for i, extracted in enumerate(pool.map(_extract_synth, tasks, chunksize=64)):
                if extracted is not None:
                    results.append(extracted)
                if (i + 1) % 20000 == 0:
                    print(
                        f"[batched] rows {i + 1}/{len(rows)} ok {len(results)} "
                        f"({time.time() - t0:.0f}s)",
                        flush=True,
                    )

    order = {str(row.get("sample_id")): row for row in rows}
    good_rows = [order[sid] for sid, _, _ in results]
    features = np.stack([f for _, f, _ in results], axis=0).astype(np.float32)
    psi = np.stack([p for _, _, p in results], axis=0).astype(np.float32)
    sample_ids = np.asarray([sid for sid, _, _ in results], dtype=str)
    print(f"extracted {features.shape[0]}/{len(rows)} rows in {time.time() - t0:.1f}s", flush=True)

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, sample_ids=sample_ids, features=features, psi=psi)
    print(f"cache: {output} ({output.stat().st_size / 1e9:.2f} GB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
