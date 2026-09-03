"""绘图公共配置：ROOT 路径定位、配色、风格（沿用 §18 规范）。"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

# 项目根（plots/script/ -> plots/ -> 项目根）
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "plots" / "data"))

PLOT_DIR = ROOT / "plots"

# 配色
COLOR_A = "#d62728"   # A 臂（scratch）
COLOR_C = "#2ca02c"   # C 臂（pretrain+finetune）
COLOR_10EP = "#9ecae1"  # 浅（10ep）
COLOR_40EP = "#2171b5"  # 深（40ep）
GRAY = "0.4"


def style_ax(ax):
    """去掉 top/right spine + 网格。"""
    ax.grid(alpha=0.18)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return ax


def add_colorbar(fig, mappable, ax, label="count"):
    """紧凑美观的 colorbar：缩短长度、去边框、小字号。"""
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.72, pad=0.02, aspect=18)
    cbar.set_label(label, fontsize=8)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=7)
    return cbar


def save(fig, name: str):
    out = PLOT_DIR / name
    fig.savefig(out, dpi=300)
    print(f"saved {out}")
    return out
