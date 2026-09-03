"""(Random split psi 网格点值采样散点，风格规范 = paper_artifacts/PLOT_STYLE.md §psi-scatter)。

A(scratch e500) vs C(ft w150 clean) × 5%/1%。口径 = progress2 §13 图 2 / plot_psi_scatter.py：
对每样本 65×65 网格**逐像素均匀采样**（rng=2026，概率 0.2% ≈ 18 万点）+ hexbin(gridsize=200)
+ 红虚线 y=x；R = 采样点像素值 corr（不是逐样本均值——均值版 = 同目录 plot_psi_scatter_combined.py）。
数据：data/preds/test_predictions_<run 新名>.npz（evaluate --save-predictions 生成，raw psi）
run 名 = 归档统一新名（旧名对照 = ../EXPERIMENTS.md 附录 A）。
输出：out/psi_scatter_points_combined_1_5pct.png（2×2：行 A/C、列 5%/1%）
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parents[1]  # <archive>/random/plot
PREDS_DIR = BASE / "data" / "preds"
OUT_DIR = BASE / "out"

# rows: A / C; cols: 5% / 1%
PANELS = [
    [("real scratch 5%", "random-scratch-5pct-s54-lr1e4-ep500"),
     ("real scratch 1%", "random-scratch-1pct-s54-lr1e4-ep500")],
    [("pretrain + ft 5%", "random-ft-clean-5pct-s54-lr1e4-ep150-warm10"),
     ("pretrain + ft 1%", "random-ft-clean-1pct-s54-lr1e4-ep150-warm10")],
]

FRAC = 0.002
SEED = 2026


def load(run):
    d = np.load(PREDS_DIR / f"test_predictions_{run}.npz")
    return d["true_psi"], d["pred_psi"]


def main():
    true, pred = load(PANELS[0][0][1])
    N = true.shape[0]
    rng = np.random.default_rng(SEED)
    mask = rng.random((N, 65 * 65)) < FRAC

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.6), dpi=300)
    info = []
    for row, cols in enumerate(PANELS):
        for col, (label, run) in enumerate(cols):
            ax = axes[row, col]
            t, p = load(run)
            tp = t.reshape(N, 65 * 65)[mask]
            pp = p.reshape(N, 65 * 65)[mask]
            corr = np.corrcoef(tp, pp)[0, 1]
            hb = ax.hexbin(tp, pp, gridsize=200, cmap="viridis", mincnt=1)
            lims = [min(tp.min(), pp.min()), max(tp.max(), pp.max())]
            ax.plot(lims, lims, "r--", lw=1.6, alpha=0.8)
            ax.set_aspect("equal")
            ax.text(0.04, 0.97, f"$R = {corr:.4f}$", transform=ax.transAxes,
                    fontsize=11, ha="left", va="top")
            ax.text(0.04, 0.88, label, transform=ax.transAxes,
                    fontsize=9, ha="left", va="top", color="0.25")
            if row == 1:
                ax.set_xlabel("EFIT psi / ground truth")
            if col == 0:
                ax.set_ylabel("psi pred")
            ax.tick_params(labelsize=8)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            cb = fig.colorbar(hb, ax=ax, shrink=0.72, pad=0.02, aspect=18)
            cb.set_label("count", fontsize=8)
            cb.outline.set_visible(False)
            cb.ax.tick_params(labelsize=7)
            info.append((label, corr, tp.size))
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "psi_scatter_points_combined_1_5pct.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    for label, corr, n in info:
        print(f"{label:20s} R={corr:.4f}  sampled_pts={n}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
