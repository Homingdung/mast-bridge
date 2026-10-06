"""Shared archive-relative plotting paths, colors, and styling."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

# Resolve the plotting archive independently of the working directory.
ROOT = Path(__file__).resolve().parents[4]
BASE = Path(__file__).resolve().parents[1]

PLOT_DIR = BASE / "out"

# Colors.
COLOR_A = "#d62728"   # A: scratch.
COLOR_C = "#2ca02c"   # C: pretraining and fine-tuning.
COLOR_10EP = "#9ecae1"  # Light shade (10 epochs).
COLOR_40EP = "#2171b5"  # Dark shade (40 epochs).
GRAY = "0.4"


def style_ax(ax):
    """Hide top/right spines and add a light grid."""
    ax.grid(alpha=0.18)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return ax


def add_colorbar(fig, mappable, ax, label="count"):
    """Add a compact colorbar with small labels and no outline."""
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.72, pad=0.02, aspect=18)
    cbar.set_label(label, fontsize=8)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=7)
    return cbar


def save(fig, name: str):
    out = PLOT_DIR / name
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    print(f"saved {out}")
    return out
