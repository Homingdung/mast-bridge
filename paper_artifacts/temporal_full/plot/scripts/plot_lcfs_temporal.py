"""单炮平衡场可视化——temporal split 双臂对比版（仿 plot_lcfs_ip.py 风格）。

用法（归档内自包含，需先按 README.md 生成 data/preds/ 预测）:
  python plot_lcfs_temporal.py --shot 29412 [--pct 5|1]
布局：上排 = A 臂 real scratch（4 帧 R-Z），下排 = C 臂 pretrain+ft（同 4 帧），
      第三排 = Ip(t)（共享，红虚线标记帧时刻）。
每个 R-Z 图：pred psi 等值线 + 机械元件（PF/passive/wall/limiter）+ EFIT LCFS（蓝实线）
vs predicted boundary（橙虚线，动态扫描 O 点连通区闭合面）。
temporal split：训练 M5-M7（shot 11766-25017）、val M8（25605-28346）、test = M9 未来实验（28631-30451）。
run 名 = 归档统一新名（旧名见 ../EXPERIMENTS.md 附录 A）。
输出：out/lcfs_pred_{shot}_combined_{pct}pct.png
"""
import argparse
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import zarr
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy import ndimage
from skimage import measure

BASE = Path(__file__).resolve().parents[1]  # <archive>/temporal（数据与输出根）
ZARR_ROOT = str(BASE / "data" / "raw" / "mast")
PREDS_DIR = BASE / "data" / "preds"
MANIFEST = Path(__file__).resolve().parents[2] / "test" / "split_test_real.jsonl"
OUT_DIR = BASE / "out"

R1 = np.linspace(0.06, 1.98, 65)
Z1 = np.linspace(-2, 2, 65)
Rg, Zg = np.meshgrid(R1, Z1, indexing="ij")

ARMS = {
    "5": [
        ("real scratch 5%", "temporal-scratch-5pct-s54-lr1e4-ep100"),
        ("pretrain + ft 5%", "temporal-ft-5pct-s54-lr1e4-ep100"),
    ],
    "1": [
        ("real scratch 1%", "temporal-scratch-1pct-s54-lr1e4-ep100"),
        ("pretrain + ft 1%", "temporal-ft-1pct-s54-lr1e4-ep100"),
    ],
}


def load_shot(shot):
    z = zarr.open(f"{ZARR_ROOT}/{shot}.zarr", mode="r")
    return (
        np.asarray(z["equilibrium/lcfs_r"]),
        np.asarray(z["equilibrium/lcfs_z"]),
        np.asarray(z["equilibrium/time"]),
        np.asarray(z["magnetics/ip"]),
        np.asarray(z["magnetics/time"]),
    )


def load_pickle(shot, name):
    with open(f"{ZARR_ROOT}/machine/{shot}/{name}.pickle", "rb") as f:
        return pickle.load(f)


def draw_rectangles(ax, payload, color):
    def visit(item):
        if isinstance(item, dict) and {"R", "Z", "dR", "dZ"} <= set(item.keys()):
            for radius, vertical in zip(np.atleast_1d(item["R"]), np.atleast_1d(item["Z"])):
                ax.add_patch(Rectangle((float(radius) - abs(float(item["dR"])) / 2,
                                        float(vertical) - abs(float(item["dZ"])) / 2),
                                       abs(float(item["dR"])), abs(float(item["dZ"])),
                                       facecolor="none", edgecolor=color, linewidth=0.45, alpha=0.7))
        elif isinstance(item, dict):
            for c in item.values():
                visit(c)
        elif isinstance(item, list):
            for c in item:
                visit(c)
    visit(payload)


def lcfs_extract(psi):
    oi, oj = np.unravel_index(np.argmax(psi), psi.shape)
    best, best_area = None, 0
    for frac in np.arange(0.05, 0.95, 0.01):
        lv = psi.max() - frac * (psi.max() - psi.min())
        lbl, n = ndimage.label(psi >= lv)
        if n == 0:
            continue
        lab = lbl[oi, oj]
        if lab == 0:
            continue
        main = lbl == lab
        if main[0, :].any() or main[-1, :].any() or main[:, 0].any() or main[:, -1].any():
            break
        if main.sum() > best_area:
            best_area, best = main.sum(), main.copy()
    if best is None:
        return None, None
    c = max(measure.find_contours(best.astype(float), 0.5), key=len)
    r = np.interp(c[:, 0], [0, 64], [R1[0], R1[-1]])
    zz = np.interp(c[:, 1], [0, 64], [Z1[0], Z1[-1]])
    return r, zz


def pad_psi(psi):
    dR = R1[1] - R1[0]
    dZ = Z1[1] - Z1[0]
    R_lim, Z_lim = (0.0, 2.05), (-2.1, 2.1)
    pl = int(round((R1[0] - R_lim[0]) / dR))
    pr = int(round((R_lim[1] - R1[-1]) / dR))
    pb = int(round((Z1[0] - Z_lim[0]) / dZ))
    pt = int(round((Z_lim[1] - Z1[-1]) / dZ))
    psi_p = np.pad(psi, ((pl, pr), (pb, pt)), mode="edge")
    Re = np.linspace(R_lim[0], R_lim[1], psi_p.shape[0])
    Ze = np.linspace(Z_lim[0], Z_lim[1], psi_p.shape[1])
    return psi_p, Re, Ze


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shot", default="29412")
    ap.add_argument("--pct", default="5", choices=sorted(ARMS))
    args = ap.parse_args()
    shot = args.shot

    arm_data = []
    for label, run in ARMS[args.pct]:
        preds = np.load(PREDS_DIR / f"test_predictions_{run}.npz")
        arm_data.append((label, run, preds["pred_psi"], {str(s): i for i, s in enumerate(preds["sample_ids"])}))

    rows = [json.loads(l) for l in open(MANIFEST)]
    sshot = sorted([r for r in rows if r["shot_id"] == shot], key=lambda r: r["target_time"])
    if not sshot:
        raise SystemExit(f"shot {shot} not in test manifest")

    lcfs_r, lcfs_z, t_eq, ip, t_ip = load_shot(shot)

    finite = []
    for r in sshot:
        fi = int(np.argmin(np.abs(t_eq - r["target_time"])))
        if np.isfinite(lcfs_r[:, fi]).all() and np.isfinite(lcfs_z[:, fi]).all():
            finite.append(r)
    pool = finite if len(finite) >= 4 else sshot
    idxs = np.unique(np.linspace(0, len(pool) - 1, 4).astype(int))
    frames = [pool[i] for i in idxs]

    active = load_pickle(shot, "MAST_active_coils")
    passive = load_pickle(shot, "MAST_passive_coils")
    wall = load_pickle(shot, "MAST_wall")
    limiter = load_pickle(shot, "MAST_limiter")

    fig = plt.figure(figsize=(14.5, 10.2), dpi=300)
    gs = fig.add_gridspec(4, 4, height_ratios=[1.8, 1.8, 0.14, 1.0], hspace=0.42, wspace=0.14)

    handles = [
        Line2D([0], [0], color="0.75", lw=0.6, label="psi contour"),
        Line2D([0], [0], color="#174ea6", lw=2.0, label="EFIT / ground truth"),
        Line2D([0], [0], color="#e07b00", lw=2.0, ls="--", label="predicted LCFS"),
    ]

    axes_rz = []
    im = None
    for row, (label, run, pred_psi, idx) in enumerate(arm_data):
        for i, r in enumerate(frames):
            ax = fig.add_subplot(gs[row, i])
            axes_rz.append(ax)

            psi = pred_psi[idx[r["sample_id"]]]
            psi_p, Re, Ze = pad_psi(psi)
            im = ax.pcolormesh(Re, Ze, psi_p.T, cmap="viridis", shading="auto", zorder=0)
            ax.contour(Rg, Zg, psi, levels=8, colors="white", linewidths=0.4, alpha=0.55, zorder=1)

            draw_rectangles(ax, active, "#1a5fb4")
            draw_rectangles(ax, passive, "#8a8a8a")
            wr = [p["R"] for p in wall]
            wz = [p["Z"] for p in wall]
            ax.plot(wr, wz, color="#e01b24", lw=1.2, zorder=3)
            lr = [p["R"] for p in limiter]
            lz = [p["Z"] for p in limiter]
            ax.plot(lr, lz, color="#666666", lw=0.8, zorder=3)

            fi = int(np.argmin(np.abs(t_eq - r["target_time"])))
            ax.plot(lcfs_r[:, fi], lcfs_z[:, fi], color="#174ea6", lw=2.0, zorder=4)
            pr, pz = lcfs_extract(psi)
            if pr is not None:
                ax.plot(pr, pz, color="#e07b00", lw=2.0, ls="--", zorder=4)

            ax.set_aspect("equal")
            ax.set_xlim(0.0, 2.05)
            ax.set_ylim(-2.1, 2.1)
            if row == 0:
                ax.set_title(f"t = {r['target_time']:.3g} s", fontsize=9)
            if i == 0:
                ax.set_ylabel(f"{label}\nZ (m)", fontsize=8.5)
                if row == 1:
                    ax.set_xlabel("R (m)")
            else:
                ax.set_xticks([])
                ax.set_yticks([])

    cbar = fig.colorbar(im, ax=axes_rz, location="right", shrink=0.85, pad=0.02, aspect=22)
    cbar.set_label("predicted psi (Wb)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax_leg = fig.add_subplot(gs[2, :])
    ax_leg.axis("off")
    ax_leg.legend(handles=handles, ncol=3, loc="center", frameon=False, fontsize=9,
                  bbox_to_anchor=(0.5, 0.5))

    ax_ip = fig.add_subplot(gs[3, :])
    ax_ip.plot(t_ip, ip / 1e6, color="#174ea6", lw=1.6)
    for r in frames:
        ax_ip.axvline(r["target_time"], color="#d62728", ls="--", lw=0.9, alpha=0.8)
        ax_ip.plot(r["target_time"], np.interp(r["target_time"], t_ip, ip) / 1e6, "o",
                   color="#d62728", ms=5, mec="white", mew=0.8)
    ax_ip.set_xlabel("time (s)")
    ax_ip.set_ylabel(r"$I_p$ (MA)")
    ax_ip.set_xlim(t_ip.min(), t_ip.max())
    ax_ip.grid(alpha=0.18)
    for s in ("top", "right"):
        ax_ip.spines[s].set_visible(False)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"lcfs_pred_{shot}_combined_{args.pct}pct.png"
    fig.savefig(str(out), dpi=300)
    plt.close(fig)
    print(f"saved {out} (frames at {[round(r['target_time'],3) for r in frames]})")


if __name__ == "__main__":
    main()
