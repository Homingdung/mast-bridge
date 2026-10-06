"""Compute measured/predicted plasma-current pairs for each Test slice.

Use raw shot Zarr (equilibrium/j_phi, equilibrium/time, magnetics/ip and time)
and archived-run predictions. Compute Delta*psi with second-order differences
on the inner 63x63 grid, J_phi = -Delta*psi/(mu_0 R), and integrate over the
EFIT j_phi > frame maximum * 1e-3 mask. Interpolate the Rogowski-coil current
at target_time. plot_jtor_ip_combined.py can then plot the saved pairs without Zarr.
Values are stored in amperes and converted to MA by the plotting script."""
import argparse
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import zarr

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from plot_paths import add_plot_paths

BASE = Path(__file__).resolve().parents[1]          # <archive>/random/plot
PREDS_DIR = BASE / "data" / "preds"
OUT = BASE / "data" / "jtor_ip_pairs.npz"
MANIFEST = Path(__file__).resolve().parents[2] / "test" / "split_test_real.jsonl"

R1 = np.linspace(0.06, 1.98, 65)
dR = R1[1] - R1[0]
dZ = 4.0 / 64
MU0 = 4e-7 * np.pi

RUNS = ["random-scratch-5pct-s54-lr1e4-ep500",
        "random-scratch-1pct-s54-lr1e4-ep500",
        "random-ft-clean-5pct-s54-lr1e4-ep150-warm10",
        "random-ft-clean-1pct-s54-lr1e4-ep150-warm10"]


def lapstar(psi):
    d2dR2 = (psi[:, 2:, 1:-1] - 2 * psi[:, 1:-1, 1:-1] + psi[:, :-2, 1:-1]) / dR**2
    d2dZ2 = (psi[:, 1:-1, 2:] - 2 * psi[:, 1:-1, 1:-1] + psi[:, 1:-1, :-2]) / dZ**2
    d1dR = (psi[:, 2:, 1:-1] - psi[:, :-2, 1:-1]) / (2 * dR)
    R = R1[1:-1][None, :, None]
    return (d2dR2 + d2dZ2 - d1dR / R).astype(np.float32)


def main():
    global PREDS_DIR, OUT, MANIFEST
    ap = argparse.ArgumentParser()
    add_plot_paths(ap, __file__, zarr=True, pairs=True)
    args = ap.parse_args()
    PREDS_DIR = args.preds_dir.expanduser().resolve()
    MANIFEST = args.manifest.expanduser().resolve()
    OUT = args.pairs.expanduser().resolve()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    ZARR = Path(args.zarr_root)

    @lru_cache(maxsize=64)
    def load_shot(shot):
        z = zarr.open_group(str(ZARR / f"{shot}.zarr"), mode="r")
        eq, mg = z["equilibrium"], z["magnetics"]
        return (np.asarray(eq["j_phi"]), np.asarray(eq["time"]),
                np.asarray(mg["ip"]), np.asarray(mg["time"]))

    rows = sorted((json.loads(l) for l in open(MANIFEST)), key=lambda r: r["shot_id"])
    preds_all = {run: np.load(PREDS_DIR / f"test_predictions_{run}.npz") for run in RUNS}
    idx = {run: {str(s): i for i, s in enumerate(p["sample_ids"])} for run, p in preds_all.items()}
    ls = {run: lapstar(np.asarray(p["pred_psi"], dtype=np.float32)) for run, p in preds_all.items()}
    R_inner = R1[1:-1]

    out = {run: {"meas_ma": [], "pred_ma": []} for run in RUNS}
    n_done = n_skip = 0
    for r in rows:
        sid, shot, tt = r["sample_id"], r["shot_id"], r["target_time"]
        if not all(sid in idx[run] for run in RUNS):
            n_skip += 1
            continue
        j_phi, t_eq, ip, t_ip = load_shot(shot)
        fi = int(np.argmin(np.abs(t_eq - tt)))
        jp = j_phi[:, :, fi].T
        m = float(np.nanmax(jp)) if np.isfinite(jp).all() else float("nan")
        if not np.isfinite(m):
            n_skip += 1
            continue
        mask = (jp > m * 1e-3)[1:-1, 1:-1]
        ip_meas = float(np.interp(tt, t_ip, ip))
        for run in RUNS:
            j = idx[run][sid]
            Jphi = -ls[run][j] / (MU0 * R_inner[:, None])
            out[run]["pred_ma"].append(float((Jphi * mask).sum() * dR * dZ))
            out[run]["meas_ma"].append(ip_meas)
        n_done += 1
        if n_done % 5000 == 0:
            print(f"{n_done} rows ...", flush=True)

    save = {run: {k: np.asarray(v) for k, v in out[run].items()} for run in RUNS}
    np.savez(OUT, **{f"{run}__{k}": v for run, d in save.items() for k, v in d.items()})
    print(f"saved {OUT}  rows={n_done} skipped={n_skip}")
    for run in RUNS:
        m = np.asarray(out[run]["meas_ma"]) / 1e6
        p = np.asarray(out[run]["pred_ma"]) / 1e6
        print(f"  {run}: n={len(m)} R={np.corrcoef(m, p)[0, 1]:.4f}")


if __name__ == "__main__":
    main()
