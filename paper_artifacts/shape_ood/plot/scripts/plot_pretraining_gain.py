#!/usr/bin/env python
"""Plot paired A-to-C Test-OOD gains; positive is better for C."""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from common import axes_style, finish, load_summary, style

parser=argparse.ArgumentParser();parser.add_argument("--output-dir",type=Path,required=True);args=parser.parse_args()
style();rows=load_summary();x=[r["fraction_percent"] for r in rows]
fig,ax=plt.subplots(figsize=(5.4,3.8))
ax.errorbar(x,[r["paired_nrmse_gain_percent_mean"] for r in rows],yerr=[r["paired_nrmse_gain_percent_std_sample"] for r in rows],marker="o",lw=1.8,capsize=3,color="#0072b2",label="nRMSE gain")
ax.errorbar(x,[r["paired_lcfs_gain_percent_mean"] for r in rows],yerr=[r["paired_lcfs_gain_percent_std_sample"] for r in rows],marker="s",lw=1.8,capsize=3,color="#cc79a7",label="LCFS-distance gain")
ax.axhline(0,color="0.25",lw=.9);ax.set_xscale("symlog",linthresh=1);ax.set_xticks(x);ax.set_xticklabels(["1%","5%","25%","50%","100%"])
ax.set_xlabel("Real training fraction (%)");ax.set_ylabel("Paired C gain over A (%)")
axes_style(ax);ax.legend(frameon=False,loc="upper right");finish(fig,args.output_dir,"shape_ood_pretraining_gain");plt.close(fig)
