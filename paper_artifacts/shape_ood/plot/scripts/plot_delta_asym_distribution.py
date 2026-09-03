#!/usr/bin/env python
"""Plot Real delta_asym distribution and the frozen Shape-OOD threshold."""
from __future__ import annotations
import argparse,csv
from pathlib import Path
import matplotlib.pyplot as plt
from common import ROOT, axes_style, finish, style

parser=argparse.ArgumentParser();parser.add_argument("--output-dir",type=Path,required=True);args=parser.parse_args()
style();all_real=[];ood=[]
with (ROOT/"dataset_split.csv").open() as f:
    for r in csv.DictReader(f):
        all_real.append(float(r["delta_asym"]))
        if r["phase"]=="test" and r["is_OOD"]=="true":ood.append(float(r["delta_asym"]))
fig,ax=plt.subplots(figsize=(5.4,3.8));bins=80
ax.hist(all_real,bins=bins,density=True,color="#4c78a8",alpha=.62,label="All frozen Real")
ax.hist(ood,bins=bins,density=True,histtype="step",lw=1.7,color="#d55e00",label="Test-OOD-only")
ax.axvspan(min(all_real),-0.214,color="#f4a582",alpha=.22,label="Frozen Shape-OOD region");ax.axvline(-.214,color="#b2182b",lw=1.4,ls="--")
ax.set_xlabel(r"$\delta_{asym}=\delta_u-\delta_l$");ax.set_ylabel("Density")
axes_style(ax);ax.legend(frameon=False,loc="upper right");finish(fig,args.output_dir,"shape_ood_delta_asym_distribution");plt.close(fig)
