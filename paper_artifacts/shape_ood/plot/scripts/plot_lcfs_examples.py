#!/usr/bin/env python
"""Plot external GT/A/C LCFS CSVs without committing prediction arrays."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import matplotlib.pyplot as plt
from common import axes_style, finish, style

def load(path):
    with path.open() as f:return {r['sample_id']:r for r in csv.DictReader(f)}
parser=argparse.ArgumentParser();parser.add_argument('--ground-truth',type=Path,required=True);parser.add_argument('--scratch',type=Path,required=True);parser.add_argument('--pretrained',type=Path,required=True);parser.add_argument('--cases',type=Path,required=True,help='CSV containing case_id,sample_id');parser.add_argument('--output-dir',type=Path,required=True);args=parser.parse_args()
style();gt,a,c=load(args.ground_truth),load(args.scratch),load(args.pretrained)
cases=list(csv.DictReader(args.cases.open()))[:4];fig,axs=plt.subplots(1,len(cases),figsize=(3.0*len(cases),3.3),squeeze=False)
for ax,case in zip(axs[0],cases):
    sid=case['sample_id']
    for row,color,ls,label in [(gt[sid],'black','-', 'EFIT / ground truth'),(a[sid],'#d55e00','--','Real-only'),(c[sid],'#009e73','-.','Synthetic-pretrained + fine-tuned')]:
        r=json.loads(row['lcfs_r']);z=json.loads(row['lcfs_z']);ax.plot(r,z,color=color,ls=ls,lw=1.8,label=label)
    ax.set_aspect('equal',adjustable='box');ax.set_title(f"Case {case['case_id']}",fontsize=10);axes_style(ax)
    ax.set_xlabel('R (m)');
axs[0].set_ylabel('Z (m)');handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=3,frameon=False,loc='upper center',bbox_to_anchor=(.5,1.06));fig.tight_layout();finish(fig,args.output_dir,'shape_ood_lcfs_examples');plt.close(fig)
