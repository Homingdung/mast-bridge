#!/usr/bin/env python3
"""Shape-OOD 步骤 [0]+[1]+[2]：Real/Synth LCFS 提取 + topology + δu/δl 计算。

Real pass:   zarr equilibrium/lcfs_r,lcfs_z（权威 EFIT label）+ x_point_r（topology）
Synthetic:   equilibrium.npz 的 psi/psi_axis/psi_bndry → 固定 level + axis-connected core
             （§16 已验证方法：O 点连通区 + skimage find_contours）
canonical:   去重 → CCW → max(R)起点 → 弧长 170 点（implicit closure）
δ:           R_geo/a + 局部二次插值极值精化（raw + refined）
输出:        data/processed/shape_ood/{real,synthetic}_shape_metadata.jsonl
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = SCRIPT_ROOT.parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))
if str(SCRIPT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT / "src"))

from scipy import ndimage
from skimage import measure

from mast_bridge.dataset.synthetic_quality import efit_topology_at_time  # noqa: E402
from mast_bridge.training.tokamind_manifest import load_manifest_rows  # noqa: E402

N_CANONICAL = 170
RAW_ZARR = WORKSPACE_ROOT / "data/raw/mast"
SYNTH_ROOT = WORKSPACE_ROOT / "data/processed/synthetic_pca_01sigma"
OUTDIR = WORKSPACE_ROOT / "data/processed/shape_ood"


def canonical_lcfs(r: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """去重 → CCW → max(R)起点 → 弧长均匀重采样 170 点（implicit closure）。"""
    r = np.asarray(r, float)
    z = np.asarray(z, float)
    if r.size != z.size or r.size < 10:
        return None
    mask = np.isfinite(r) & np.isfinite(z)
    if mask.sum() < 10:
        return None
    r, z = r[mask], z[mask]
    keep = np.ones(r.size, bool)
    keep[1:] = ~((np.abs(np.diff(r)) < 1e-12) & (np.abs(np.diff(z)) < 1e-12))
    r, z = r[keep], z[keep]
    if r.size < 10:
        return None
    if np.abs(r[0] - r[-1]) < 1e-12 and np.abs(z[0] - z[-1]) < 1e-12:
        r, z = r[:-1], z[:-1]
    area = 0.5 * np.sum(r[:-1] * z[1:] - r[1:] * z[:-1] + r[-1] * z[0] - r[0] * z[-1])
    if abs(area) < 1e-9:
        return None
    if area < 0:
        r, z = r[::-1], z[::-1]
    start = int(np.argmax(r))
    r = np.concatenate([r[start:], r[:start]])
    z = np.concatenate([z[start:], z[:start]])
    rc = np.concatenate([r, [r[0]]])
    zc = np.concatenate([z, [z[0]]])
    s = np.hypot(np.diff(rc), np.diff(zc))
    s = np.concatenate([[0.0], np.cumsum(s)])
    if s[-1] <= 0:
        return None
    pts = np.linspace(0, s[-1], N_CANONICAL)
    return np.interp(pts, s[:-1], rc[:-1]), np.interp(pts, s[:-1], zc[:-1])


def lcfs_delta(r: np.ndarray, z: np.ndarray) -> dict:
    """δu/δl（raw + refined），R_geo/a + 局部二次插值精化极值。"""
    r, z = np.asarray(r, float), np.asarray(z, float)
    out: dict = {"lcfs_valid": False, "failure_reason": None}
    if r.size != z.size or r.size < N_CANONICAL * 0.5:
        out["failure_reason"] = "too_few_points"
        return out
    rmax, rmin = float(r.max()), float(r.min())
    a = (rmax - rmin) / 2.0
    if a < 1e-4:
        out["failure_reason"] = "degenerate"
        return out
    rgeo = (rmax + rmin) / 2.0
    out.update(R_max=rmax, R_min=rmin, R_geo=rgeo, minor_radius_a=a)

    def refine_extremum(i: int, sign: int) -> tuple[float, float, float, float]:
        idxs = [(i + k) % N_CANONICAL for k in (-2, -1, 0, 1, 2)]
        s_loc = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        zs = z[idxs]
        az, bz, cz = np.polyfit(s_loc, zs, 2)
        s_star = -bz / (2.0 * az) if az != 0 else 2.0
        if sign * az > 0 and 0.0 <= s_star <= 4.0:
            z_ref = az * s_star**2 + bz * s_star + cz
            r_ref = float(np.polyval(np.polyfit(s_loc, r[idxs], 2), s_star))
            return float(z_ref), r_ref, float(zs[2]), float(r[i])
        return float(zs[2]), float(r[i]), float(zs[2]), float(r[i])

    i_top = int(np.argmax(z))
    z_u_ref, r_u_ref, z_u_raw, r_u_raw = refine_extremum(i_top, -1)
    i_bot = int(np.argmin(z))
    z_l_ref, r_l_ref, z_l_raw, r_l_raw = refine_extremum(i_bot, +1)

    out["Z_upper_raw"], out["R_upper_raw"] = z_u_raw, r_u_raw
    out["Z_lower_raw"], out["R_lower_raw"] = z_l_raw, r_l_raw
    out["Z_upper"], out["R_upper"] = z_u_ref, r_u_ref
    out["Z_lower"], out["R_lower"] = z_l_ref, r_l_ref
    out["delta_u_raw"] = (rgeo - r_u_raw) / a
    out["delta_l_raw"] = (rgeo - r_l_raw) / a
    out["delta_u"] = (rgeo - r_u_ref) / a
    out["delta_l"] = (rgeo - r_l_ref) / a
    out["delta_mean"] = (out["delta_u"] + out["delta_l"]) / 2.0
    out["delta_asym"] = out["delta_u"] - out["delta_l"]
    out["lcfs_valid"] = True
    return out


def real_row(row: dict) -> dict:
    shot = str(row["shot_id"])
    zdir = RAW_ZARR / f"{shot}.zarr"
    eq_path = str(row.get("equilibrium_path") or zdir / "equilibrium")
    out = {
        "sample_id": row["sample_id"], "shot_id": shot, "parent_shot": shot,
        "target_time": row["target_time"], "source": "real",
        "lcfs_source": "efit_zarr", "lcfs_method": "canonical_closed_lcfs_arclength_v2",
        "lcfs_valid": False, "failure_reason": None, "topology": None,
    }
    try:
        import zarr
        z = zarr.open(zdir, mode="r")
        eq = z["equilibrium"]
        times = np.asarray(eq["time"][:], float)
        t = float(row["target_time"])
        idx = np.flatnonzero(np.abs(times - t) < 1e-9)
        if idx.size == 0:
            out["failure_reason"] = "time_mismatch"
            return out
        i = int(idx[0])
        lcfs_r = np.asarray(eq["lcfs_r"][:], float)[:, i]
        lcfs_z = np.asarray(eq["lcfs_z"][:], float)[:, i]
        if not (np.isfinite(lcfs_r).any() and np.isfinite(lcfs_z).any()):
            out["failure_reason"] = "lcfs_nan"
            return out
        canon = canonical_lcfs(lcfs_r, lcfs_z)
        if canon is None:
            out["failure_reason"] = "canonical_failed"
            return out
        out["lcfs_r"], out["lcfs_z"] = [v.tolist() for v in canon]
        out.update(lcfs_delta(*canon))
        topo = efit_topology_at_time(zdir, t)
        out["topology"] = topo if topo in ("diverted", "limited") else None
    except Exception as exc:  # noqa: BLE001
        out["failure_reason"] = f"{type(exc).__name__}: {exc}"
    return out


def synth_row(row: dict) -> dict:
    out = {
        "sample_id": row["sample_id"], "shot_id": str(row["parent_shot"]),
        "parent_shot": str(row["parent_shot"]), "target_time": row["target_time"],
        "source": "synthetic",
        "lcfs_source": "solver_psi_bndry", "lcfs_method": "canonical_closed_lcfs_arclength_v2",
        "lcfs_valid": False, "failure_reason": None,
        "topology": "limited" if row.get("flag_limiter") else "diverted",
    }
    try:
        sdir = SYNTH_ROOT / str(row["sample_id"])
        eq = np.load(sdir / "equilibrium.npz")
        psi = np.asarray(eq["psi"], float)
        R1 = np.asarray(eq["R"], float)[:, 0]
        Z1 = np.asarray(eq["Z"], float)[0, :]
        psi_axis = float(eq["psi_axis"])
        psi_bndry = float(eq["psi_bndry"])
        if not (np.isfinite(psi).all() and np.isfinite(psi_axis) and np.isfinite(psi_bndry)):
            out["failure_reason"] = "nonfinite"
            return out
        if abs(psi_bndry - psi_axis) < 1e-12:
            out["failure_reason"] = "degenerate_level"
            return out
        psi_n = (psi - psi_axis) / (psi_bndry - psi_axis)
        oi, oj = np.unravel_index(np.argmax(psi), psi.shape)
        mask = psi_n <= 1.0 + 1e-6
        lbl, _ = ndimage.label(mask)
        lab = lbl[oi, oj]
        core = None
        if lab != 0:
            core = lbl == lab
            if core[0, :].any() or core[-1, :].any() or core[:, 0].any() or core[:, -1].any():
                core = None
        method = "fixed_psi_bndry"
        if core is None:
            # 回退：动态扫描（§16 已验证）——从 psi_max 向下扫 level，泄漏到网格边界前最大闭合面
            method = "fallback_dynamic_scan"
            best = None
            best_area = 0
            for frac in np.arange(0.05, 0.95, 0.01):
                lv = psi.max() - frac * (psi.max() - psi.min())
                blob = psi >= lv
                lbl2, _ = ndimage.label(blob)
                lab2 = lbl2[oi, oj]
                if lab2 == 0:
                    continue
                main = lbl2 == lab2
                if main[0, :].any() or main[-1, :].any() or main[:, 0].any() or main[:, -1].any():
                    break
                if main.sum() > best_area:
                    best_area = main.sum()
                    best = main
            if best is None:
                out["failure_reason"] = "no_core"
                return out
            core = best
        contours = measure.find_contours(core.astype(float), 0.5)
        if not contours:
            out["failure_reason"] = "no_contour"
            return out
        c = max(contours, key=len)
        r_pts = np.interp(c[:, 0], [0, 64], [R1[0], R1[-1]])
        z_pts = np.interp(c[:, 1], [0, 64], [Z1[0], Z1[-1]])
        canon = canonical_lcfs(r_pts, z_pts)
        if canon is None:
            out["failure_reason"] = "canonical_failed"
            return out
        out["lcfs_method"] = f"canonical_closed_lcfs_arclength_v2/{method}"
        out["lcfs_r"], out["lcfs_z"] = [v.tolist() for v in canon]
        out.update(lcfs_delta(*canon))
    except Exception as exc:  # noqa: BLE001
        out["failure_reason"] = f"{type(exc).__name__}: {exc}"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["real", "synthetic"], required=True)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    if args.phase == "real":
        manifest = WORKSPACE_ROOT / "data/manifests/training_pca_01sigma/real.jsonl"
        out_path = OUTDIR / "real_shape_metadata.jsonl"
        fn = real_row
    else:
        manifest = WORKSPACE_ROOT / "data/manifests/fullsolve_pca_01sigma_accepted.jsonl"
        out_path = OUTDIR / "synthetic_shape_metadata.jsonl"
        fn = synth_row

    src = "real" if args.phase == "real" else "synthetic"
    rows = [r for r in load_manifest_rows(manifest) if r.get("source") == src]
    if args.limit:
        rows = rows[: args.limit]
    print(f"processing {len(rows)} rows (workers={args.workers})", flush=True)

    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, res in enumerate(pool.map(fn, rows, chunksize=64), 1):
            results.append(res)
            if i % 20000 == 0:
                print(f"  {i}/{len(rows)}", flush=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_valid = sum(1 for r in results if r["lcfs_valid"])
    n_topo = sum(1 for r in results if r.get("topology"))
    print(f"done: {len(results)} rows, lcfs_valid={n_valid}, topology_known={n_topo} -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
