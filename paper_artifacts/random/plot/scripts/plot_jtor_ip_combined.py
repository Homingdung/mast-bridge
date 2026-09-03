"""(Random split j_tor–Ip 散点绘图，风格规范 = paper_artifacts/PLOT_STYLE.md §j_tor–Ip)。

自包含画图（无 zarr 依赖）：读 data/jtor_ip_pairs.npz（由 compute_jtor_pairs.py 预计算，
或 data/ 内已备的归档产物）→ 2×2 hexbin 散点。布局：行 A/C、列 5%/1%；每点 = 一个 test slice 的
(Ip_meas, Ip_pred)，MA 单位；x = $I_p^{meas}$ (MA)、y = $I_p^{pred}$ (MA)。
输出：out/jtor_ip_combined_1_5pct.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parents[1]  # <archive>/random/plot
DATA = BASE / "data" / "jtor_ip_pairs.npz"
OUT_DIR = BASE / "out"

PANELS = [
    [("real scratch 5%", "random-scratch-5pct-s54-lr1e4-ep500"),
     ("real scratch 1%", "random-scratch-1pct-s54-lr1e4-ep500")],
    [("pretrain + ft 5%", "random-ft-clean-5pct-s54-lr1e4-ep150-warm10"),
     ("pretrain + ft 1%", "random-ft-clean-1pct-s54-lr1e4-ep150-warm10")],
]


def main():
    d = np.load(DATA)
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.6), dpi=300)
    info = []
    for row, cols in enumerate(PANELS):
        for col, (label, run) in enumerate(cols):
            ax = axes[row, col]
            meas = d[f"{run}__meas_ma"]
            pred = d[f"{run}__pred_ma"]
            corr = np.corrcoef(meas, pred)[0, 1]
            hb = ax.hexbin(meas, pred, gridsize=80, cmap="viridis", mincnt=1)
            lims = [min(meas.min(), pred.min()), max(meas.max(), pred.max())]
            ax.plot(lims, lims, "r--", lw=1.6, alpha=0.8)
            ax.set_aspect("equal")
            ax.text(0.04, 0.97, f"$R = {corr:.4f}$", transform=ax.transAxes,
                    fontsize=11, ha="left", va="top")
            ax.text(0.04, 0.88, label, transform=ax.transAxes,
                    fontsize=9, ha="left", va="top", color="0.25")
            if row == 1:
                ax.set_xlabel(r"$I_p^{\mathrm{meas}}$ (MA)")
            if col == 0:
                ax.set_ylabel(r"$I_p^{\mathrm{pred}}$ (MA)")
            ax.tick_params(labelsize=8)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            cb = fig.colorbar(hb, ax=ax, shrink=0.72, pad=0.02, aspect=18)
            cb.set_label("count", fontsize=8)
            cb.outline.set_visible(False)
            cb.ax.tick_params(labelsize=7)
            info.append((label, corr, len(meas)))
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "jtor_ip_combined_1_5pct.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    for label, corr, n in info:
        print(f"{label:20s} R={corr:.4f} n={n}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
