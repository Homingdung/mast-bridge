#!/usr/bin/env python
"""Recreate the three main-table nRMSE curves of the paper from the archived
`eval/*.json` data sources under repo/mast-bridge/paper_artifacts/.

For each experiment (Random split, Temporal split, Shape-OOD) we plot the
per-sample nRMSE (%) vs. the real-data fraction for two training recipes:
  * "Real scratch"  (A) : train on real data from random init
  * "Pre-train + ft" (C): synthetic pre-training, then real fine-tuning
The vertical gap between A and C is the sim-to-real gain.

Random split is shown as three side-by-side panels (Clean / Noisy 0.5 / Noisy 2
pre-training arms) that share the same y-axis and the same A scratch baseline,
so the arms are directly comparable. Temporal and Shape-OOD each get their own
figure (single arm). Lower nRMSE is better. All numbers are read from the eval
JSONs (the provenance of the paper tables in main.tex).

Styling: academic-clean — no grid, no top/right spines, outward ticks, vector
fonts. "Real scratch" is always the same neutral gray; "Pre-train + ft" uses a
distinct accent per arm/experiment (its own color read straight from the data).
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

ARTIFACTS = Path(__file__).resolve().parent
FRACTIONS = [1, 5, 25, 50, 100]
FRACTION_LABELS = ["1%", "5%", "25%", "50%", "100%"]

A_COLOR = "#6B7280"        # neutral gray: "Real scratch" baseline (all figs)
ARM_COLOR = {              # accent per pre-training arm ("Pre-train + ft")
    "clean": "#0072B2",    # blue
    "noisy05": "#D55E00",  # vermilion / orange
    "noisy20": "#009E73",  # bluish green
}
TEMP_SHAPE_C = "#C44E52"   # crimson accent for the single-arm experiments


def _mean_std(values):
    mu = statistics.fmean(values)
    sd = (statistics.pstdev(values) if len(values) > 1 else 0.0)
    return mu, sd


def _read(glob_pat):
    """nRMSE (%) list for all runs matching a glob pattern."""
    out = []
    for f in sorted(glob.glob(str(ARTIFACTS / glob_pat))):
        d = json.load(open(f))
        out.append(d[0]["nrmse_per_sample"] * 100.0)
    return out


def _temporal() -> dict[str, dict[int, tuple[float, float]]]:
    """method -> {fraction: (mean_nrmse_pct, std)}. 100/50/25 from per-run
    eval json; 5/1 are only archived as 3-seed aggregates."""
    res = {"scratch": {}, "ft": {}}
    for frac, tier in [(100, 100), (50, 50), (25, 25)]:
        res["scratch"][frac] = _mean_std(
            _read(f"temporal_full/eval/temporal-scratch-{tier}pct-s*-lr1e4-ep100.json"))
        res["ft"][frac] = _mean_std(
            _read(f"temporal_full/eval/temporal-ft-{tier}pct-s*-lr1e4-ep100.json"))
    for frac, pat, meth in [
        (5, "temporal-5pct-A-scratch-3seed.json", "scratch"),
        (5, "temporal-5pct-C-ft-3seed.json", "ft"),
    ]:
        res[meth][frac] = _mean_std(
            [x["nrmse_per_sample"] * 100
             for x in json.load(open(ARTIFACTS / "temporal_full" / "eval" / pat))])
    ac = json.load(open(ARTIFACTS / "temporal_full" / "eval" / "temporal-1pct-AC-3seed.json"))
    a, c = [], []
    for x in ac:
        (a if x["run"].startswith("temporal-scratch") else c).append(x["nrmse_per_sample"] * 100)
    res["scratch"][1] = _mean_std(a)
    res["ft"][1] = _mean_std(c)
    return res


def _random() -> tuple[dict[int, tuple[float, float]], dict[str, dict[int, tuple[float, float]]]]:
    """(scratch, {arm: {fraction: (mean,std)}}). Shared A baseline per tier."""
    scratch = {}
    for frac in FRACTIONS:
        # The final 1% baseline uses s54/s56/s58; s57 is an archived backup.
        seeds = "s5[468]" if frac == 1 else "s5[456]"
        scratch[frac] = _mean_std(
            _read(f"random/eval/random-scratch-{frac}pct-{seeds}-lr1e4-ep500.json"))
    arms = {}
    for arm in ("clean", "noisy05", "noisy20"):
        arms[arm] = {}
        for frac in FRACTIONS:
            arms[arm][frac] = _mean_std(_read(
                f"random/eval/random-ft-{arm}-{frac}pct-s*-lr1e4-ep150-warm10.json"))
    return scratch, arms


def _shape_ood() -> dict[str, dict[int, tuple[float, float]]]:
    summary = json.load(open(ARTIFACTS / "shape_ood" / "eval" / "shape-ood-summary-3seed.json"))
    res = {"scratch": {}, "ft": {}}
    for r in summary["rows"]:
        frac = r["fraction_percent"]
        res["scratch"][frac] = (r["A_test_ood_nrmse_mean"], r["A_test_ood_nrmse_std_sample"])
        res["ft"][frac] = (r["C_test_ood_nrmse_mean"], r["C_test_ood_nrmse_std_sample"])
    return res


def _lighten(color, f=0.55):
    """Mix a color toward white so error bars stay subtle."""
    rgb = mcolors.to_rgb(color)
    mix = tuple(c + (1.0 - c) * f for c in rgb)
    return mix


def _style_ax(ax):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#333333")
        ax.spines[spine].set_linewidth(0.9)
    ax.tick_params(axis="both", direction="out", length=3.5, width=0.9,
                   top=False, right=False, colors="#333333")


def _draw(ax, series):
    """series: list of (curve, color, marker, label)."""
    artists = []
    xs = list(range(len(FRACTIONS)))
    for curve, color, marker, label in series:
        y = [curve[f][0] for f in FRACTIONS]
        ye = [curve[f][1] for f in FRACTIONS]
        ln = ax.errorbar(xs, y, yerr=ye, color=color, ls="-", lw=2.1,
                         marker=marker, ms=6.0, mfc=color, mec="white", mew=1.0,
                         ecolor=_lighten(color), elinewidth=1.0, capsize=2.5,
                         capthick=1.0, label=label, zorder=3)
        artists.append(ln)
    return artists


def _plot_axis(ax, scratch, ft, c_color, legend=False):
    series = [
        (scratch, A_COLOR, "o", "Real scratch"),
        (ft, c_color, "s", "Pre-train + ft"),
    ]
    _draw(ax, series)
    ax.set_xticks(list(range(len(FRACTIONS))))
    ax.set_xticklabels(FRACTION_LABELS)
    if legend:
        ax.legend(frameon=False, loc="upper right",
                  handletextpad=0.3, labelspacing=0.4)
    _style_ax(ax)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path,
                    default=ARTIFACTS / "plot" / "out",
                    help="Directory for generated figures (default: paper_artifacts/plot/out).")
    args = ap.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10, "axes.labelsize": 11.5,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
        "legend.fontsize": 9, "axes.linewidth": 0.9,
        "axes.labelcolor": "#1a1a1a", "xtick.color": "#333333",
        "ytick.color": "#333333", "text.color": "#1a1a1a",
        "axes.unicode_minus": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    # ---- Random split: three side-by-side arms on a shared y-axis ----
    scratch, arms = _random()
    names = {"clean": "Clean", "noisy05": "Noisy 0.5", "noisy20": "Noisy 2"}
    fig, axs = plt.subplots(1, 3, figsize=(11.8, 3.9), sharey=True)
    for ax, (arm, ft) in zip(axs, arms.items()):
        _plot_axis(ax, scratch, ft, ARM_COLOR[arm], legend=True)
        ax.set_title(names[arm], fontsize=10.5, pad=4)
        ax.set_xlabel("Real data fraction")
        if ax is axs[0]:
            ax.set_ylabel("nRMSE (%)")
        else:
            ax.tick_params(labelleft=False)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"random_split_nrmse_curves.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)

    # ---- Temporal split ----
    res = _temporal()
    fig, ax = plt.subplots(figsize=(5.2, 3.7))
    _plot_axis(ax, res["scratch"], res["ft"], TEMP_SHAPE_C, legend=True)
    ax.set_xlabel("Real data fraction")
    ax.set_ylabel("nRMSE (%)")
    for ext in ("png", "pdf"):
        fig.savefig(out / f"temporal_split_nrmse_curves.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)

    # ---- Shape-OOD ----
    res = _shape_ood()
    fig, ax = plt.subplots(figsize=(5.2, 3.7))
    _plot_axis(ax, res["scratch"], res["ft"], TEMP_SHAPE_C, legend=True)
    ax.set_xlabel("Real data fraction")
    ax.set_ylabel("nRMSE (%)")
    for ext in ("png", "pdf"):
        fig.savefig(out / f"shape_ood_nrmse_curves.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)

    print("Wrote figures to:", out)
    for f in sorted(out.glob("*nrmse_curves.*")):
        print("  ", f)


if __name__ == "__main__":
    main()
