"""Shared plotting configuration: project paths, colors, and figure styles."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

# Project root: plots/script/ -> plots/ -> project root.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "plots" / "data"))

PLOT_DIR = ROOT / "plots"

# Plot colors.
COLOR_A = "#d62728"   # Scratch model.
COLOR_C = "#2ca02c"   # Pretrained and fine-tuned model.
COLOR_10EP = "#9ecae1"  # Light shade (10 epochs).
COLOR_40EP = "#2171b5"  # Dark shade (40 epochs).
GRAY = "0.4"


def style_ax(ax):
    """Remove the top and right spines and add a grid."""
    ax.grid(alpha=0.18)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return ax


def add_colorbar(fig, mappable, ax, label="count"):
    """Add a compact colorbar with a reduced length, no border, and small labels."""
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
