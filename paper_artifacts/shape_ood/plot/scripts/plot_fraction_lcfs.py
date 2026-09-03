#!/usr/bin/env python
"""Plot Test-OOD mean LCFS boundary distance by Real-data fraction."""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from common import axes_style, finish, load_summary, style

parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", type=Path, required=True); args = parser.parse_args()
style(); rows=load_summary(); x=[r["fraction_percent"] for r in rows]
fig, ax=plt.subplots(figsize=(5.4,3.8))
for label,color,mean,std in [("Real scratch (A)","#d55e00","A_lcfs_m_mean","A_lcfs_m_std_sample"),("Synthetic-pretrained + fine-tuned (C)","#009e73","C_lcfs_m_mean","C_lcfs_m_std_sample")]:
    ax.errorbar(x,[r[mean] for r in rows],yerr=[r[std] for r in rows],marker="o",lw=1.8,capsize=3,color=color,label=label)
ax.set_xscale("symlog",linthresh=1); ax.set_xticks(x); ax.set_xticklabels(["1%","5%","25%","50%","100%"])
ax.set_xlabel("Real training fraction (%)"); ax.set_ylabel("Test-OOD LCFS distance (m)")
axes_style(ax);ax.legend(frameon=False,loc="upper right");finish(fig,args.output_dir,"shape_ood_fraction_lcfs");plt.close(fig)
