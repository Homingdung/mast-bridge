# mast-bridge — MAST synthetic-data equilibrium reconstruction (paper repository)

Paper repository: real MAST data + FreeGSNKE synthetic equilibria (PCA ±0.1σ, EFIT-anchored
filtering) train a TokaMind-style network mapping 69 magnetic-diagnostic channels →
65×65 flux map ψ. **Main research claim (fully converged protocol, 3 seeds)**:
synthetic pre-training provides a transferable equilibrium-physics prior — its value is
concentrated in scarce-data settings under a matched (random-split) distribution and is
present at **all** data fractions under distribution shift (temporal split, M9
extrapolation). Results are summarized in `progress2.md` §59 (paper tables with LaTeX
source) and §58.3 (generalization evidence); quick overview: `paper_artifacts/EXPERIMENTS.md`.

## Repository layout

```
├── README.md / progress2.md / paper_artifacts/EXPERIMENTS.md / PLOT_STYLE.md
├── configs/           # diagnostic feature schemas, noise profiles, machine geometry
├── scripts/           # data generation, filtering, caching, training & evaluation
├── src/               # mast_bridge python package (datasets, solver wrappers)
├── pyproject.toml
└── paper_artifacts/   # paper figures + per-run evaluation results (curated, see below)
    ├── random/                 # random-split experiment (tab:random_table)
    │   ├── README.md           #   run naming, configs, data sources
    │   ├── eval/<run>.json     #   per-run metrics on test 21,350 slices / 730 shots
    │   ├── dataset_split*.csv  #   slice/shot-level split lists (213,924 rows)
    │   ├── test/split_test_real.jsonl
    │   └── plot/               # canonical figures (out/) + self-contained plot scripts
    ├── temporal_full/          # temporal-split experiment (tab:temporal_table)
    │   ├── README.md / eval/<run>.json / dataset_split* / test/split_test_real.jsonl
    │   └── plot/               # LCFS prediction figures (out/) + scripts
    └── generalization/         # feature-drift attribution figures (§58)
```

## Key numbers (per-sample nRMSE %, mean ± std over 3 seeds)

| scenario | tier | real-only | pre-train + fine-tune | gain |
|---|---|---|---|---|
| Random (test 21,350) | 1% | 1.483 ± 0.039 | 1.076 ± 0.010 | **+37.8%** (A−C)/C |
| Random | 5% | 1.107 ± 0.022 | 0.974 ± 0.012 | **+13.7%** |
| Temporal (M9 test 17,498) | 1% | 6.42 ± 0.04 | 2.31 ± 0.17 | **+64.0%** (A−C)/A |
| Temporal | 100% | 2.489 ± 0.167 | 2.204 ± 0.129 | **+11.4%** |

Full 5-tier tables (random × clean/noisy0.5/noisy2, temporal A/C): `progress2.md` §59.

## Environment & data

- Two python venvs are used and **not committed**: `.tokamind-train-env` (training/
  evaluation) and `.freegsnke-solve-env` (FreeGSNKE solving/feature extraction).
  Reproduce: `python -m venv` + install `pyproject.toml` deps; solver env additionally
  needs freegsnke + the tokamak equilibrium input files (see `configs/`).
- Raw MAST zarr data come from FAIR-MAST (S3); training caches (npz, ~37 GB), model
  checkpoints (97 runs, 157 MB) and test caches (npz, ~576 MB) are **not committed**
  (GitHub 100 MB/file limit). They live in the original collab workspace
  (`data/`, `runs/`, `paper_artifacts/{random,temporal_full}/checkpoints|test/*.npz`)
  and can be re-created with `scripts/build_cache_batched.py` etc.; `eval/*.json` here
  are fully sufficient to reproduce every number in the paper tables, and the
  checkpoints/test caches reproduce them by re-running
  `scripts/evaluate_tokamind_testset.py`.
- Full authoritative experiment log: `progress2.md` (start §53 for status, §59 for paper
  tables, §58.3 for generalization conclusions, §57 for the 1%-tier seed handling).

## External dependency: tokamind (MMT model zoo) — required for training/eval

The training and evaluation scripts (`scripts/train_tokamind_manifest.py`,
`scripts/evaluate_tokamind_testset.py`) import the `mmt` package (MultiModal
Transformer) from the sibling checkout **`external/tokamind`**:

```python
# both scripts resolve the workspace root as two levels above the script dir and,
# if present, prepend it to sys.path (no pip install of mmt itself required):
TOKAMIND_SRC = WORKSPACE_ROOT / "external" / "tokamind" / "src"
```

Setup:

```bash
# 1. from the repository root, clone tokamind as a sibling directory
git clone https://github.com/UKAEA-IBM-STFC-Fusion-FMs/tokamind.git external/tokamind

# 2. (optional but recommended) pin the commit used for the paper results
cd external/tokamind && git checkout 0b67cf56cd945b883fd3b0c9050cdfc560f98533 && cd ../..

# 3. make sure mmt's runtime dependencies are installed in the training venv
#    (torch, torchvision, einops, ... — see external/tokamind/pyproject.toml);
#    `pip install -e external/tokamind` works as well and makes `import mmt`
#    resolvable even outside the scripts' sys.path hook.
```

- The scripts still need the MAST workspace layout: they must be launched from a root
  that has `external/tokamind/` next to `mast-bridge/` (or the equivalent
  `scripts/`/`configs/` layout, as in this repo).
- If `external/tokamind` is absent, both scripts **fail at import time** with
  `ModuleNotFoundError: No module named 'mmt'` — this is expected; follow the steps
  above. A copy of the pinned tokamind checkout is included in the original collab
  workspace (`external/tokamind/`, commit `0b67cf56`) if you need the exact tree.

## For collaborators

Branch off `main` for new studies (e.g., Shape-OOD). Follow the naming/config conventions
in `paper_artifacts/EXPERIMENTS.md` §2-3; keep per-run eval jsons in
`paper_artifacts/<study>/eval/` and figures in `.../plot/out/`. Data/split conventions for
Shape-OOD (LCFS canonicalization, δu/δl, OOD definition) are documented in
`progress2.md` §20 and the workflow doc `Plasma_Shape_OOD_Workflow_v2_Detailed_LCFS.md`
(available in the original workspace).
