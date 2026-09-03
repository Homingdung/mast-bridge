# Shape-OOD plotting scripts

The scripts in `scripts/` read only the small split/evaluation artifacts in this directory and write PNG/PDF files to an explicitly chosen output directory. Generated figures, prediction arrays, and caches are intentionally not part of the Git artifact.

Examples (run from the repository root):

```bash
python paper_artifacts/shape_ood/plot/scripts/plot_fraction_nrmse.py --output-dir /tmp/shape-ood-figures
python paper_artifacts/shape_ood/plot/scripts/plot_fraction_lcfs.py --output-dir /tmp/shape-ood-figures
python paper_artifacts/shape_ood/plot/scripts/plot_pretraining_gain.py --output-dir /tmp/shape-ood-figures
python paper_artifacts/shape_ood/plot/scripts/plot_shape_distribution.py --output-dir /tmp/shape-ood-figures
python paper_artifacts/shape_ood/plot/scripts/plot_delta_asym_distribution.py --output-dir /tmp/shape-ood-figures
```

`plot_lcfs_examples.py` additionally requires externally generated prediction/LCFS CSV files with columns `sample_id`, `lcfs_r`, and `lcfs_z`; `lcfs_r`/`lcfs_z` are JSON arrays. This keeps large predictions out of Git while retaining a reproducible paper plotting interface.
