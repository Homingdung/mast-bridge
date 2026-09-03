#!/usr/bin/env python
"""Plot the frozen Train/Val/Test Shape-OOD distribution in shape space."""
from __future__ import annotations
import argparse, csv
from pathlib import Path
import matplotlib.pyplot as plt
from common import ROOT, axes_style, finish, style

parser=argparse.ArgumentParser();parser.add_argument("--output-dir",type=Path,required=True);parser.add_argument("--max-points",type=int,default=30000);args=parser.parse_args()
style();groups={"train":[],"val":[],"test":[],"test_ood":[]}
with (ROOT/"dataset_split.csv").open() as f:
    for row in csv.DictReader(f):
        phase=row["phase"]; point=(float(row["delta_u"]),float(row["delta_l"]))
        if phase=="train": groups["train"].append(point)
        elif phase=="val": groups["val"].append(point)
        else:
            groups["test"].append(point)
            if row["is_OOD"]=="true": groups["test_ood"].append(point)
fig,ax=plt.subplots(figsize=(5.4,4.5))
for key,color,label,size,alpha,z in [("train","#8da0cb","Train",2,.14,1),("val","#fc8d62","Validation",5,.35,2),("test","#66c2a5","Test-all",2,.12,3),("test_ood","#d73027","Test-OOD-only",5,.45,4)]:
    pts=groups[key]; stride=max(1,len(pts)//args.max_points);pts=pts[::stride]
    ax.scatter([p[0] for p in pts],[p[1] for p in pts],s=size,c=color,alpha=alpha,label=label,linewidths=0,zorder=z)
xs=ax.get_xlim();ax.plot(xs,[x+0.214 for x in xs],"k--",lw=1,label=r"$\delta_{asym}=-0.214$")
ax.set_xlabel(r"Upper triangularity, $\delta_u$");ax.set_ylabel(r"Lower triangularity, $\delta_l$")
axes_style(ax);ax.legend(frameon=False,loc="best",markerscale=2);finish(fig,args.output_dir,"shape_ood_shape_distribution");plt.close(fig)
