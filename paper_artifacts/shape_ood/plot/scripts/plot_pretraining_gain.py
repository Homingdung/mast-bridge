#!/usr/bin/env python
"""Plot Test-OOD gains from mean A/C errors; positive is better for C."""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from common import axes_style, finish, load_summary, style

parser=argparse.ArgumentParser();parser.add_argument("--output-dir",type=Path,required=True);args=parser.parse_args()
style();rows=load_summary();x=[r["fraction_percent"] for r in rows]
fig,ax=plt.subplots(figsize=(5.4,3.8))
# Gains use ratios of seed means, so paired per-seed gain error bars do not apply.
ax.plot(x,[r["nrmse_gain_percent"] for r in rows],marker="o",lw=1.8,color="#0072b2",label="nRMSE gain")
ax.plot(x,[r["lcfs_gain_percent"] for r in rows],marker="s",lw=1.8,color="#cc79a7",label="LCFS-distance gain")
ax.axhline(0,color="0.25",lw=.9);ax.set_xscale("symlog",linthresh=1);ax.set_xticks(x);ax.set_xticklabels(["1%","5%","25%","50%","100%"])
ax.set_xlim(min(x)*0.8,max(x)*1.25)
ax.set_xlabel("Real training fraction (%)");ax.set_ylabel("C gain over A (%)")
axes_style(ax);ax.legend(frameon=False,loc="upper right");finish(fig,args.output_dir,"shape_ood_pretraining_gain");plt.close(fig)
