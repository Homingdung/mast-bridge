# mast-bridge — MAST synthetic-data equilibrium reconstruction (paper repository)

Real MAST data + FreeGSNKE synthetic equilibria (PCA ±0.1σ, EFIT-anchored filtering)
train a TokaMind-style network mapping 69 magnetic-diagnostic channels → 65×65 flux map ψ.
This repository holds the latest code together with per-run evaluation results and
dataset-split metadata for the two main studies (random split and temporal split).
Experiment logs and result tables: `progress2.md` (§58.3, §59) and
`paper_artifacts/EXPERIMENTS.md`.

## Repository layout

```
├── README.md / progress2.md / paper_artifacts/EXPERIMENTS.md / PLOT_STYLE.md
├── configs/           # diagnostic feature schemas, noise profiles, machine geometry
├── scripts/           # data generation, filtering, caching, training & evaluation
├── src/               # mast_bridge python package (datasets, solver wrappers)
├── pyproject.toml
└── paper_artifacts/   # per-run evaluation results + dataset splits (curated, see below)
    ├── random/                 # random-split experiment
    │   ├── README.md           #   run naming, configs, data sources
    │   ├── eval/<run>.json     #   per-run metrics on test 21,350 slices / 730 shots
    │   ├── dataset_split*.csv  #   slice/shot-level split lists (213,924 rows)
    │   ├── test/split_test_real.jsonl
    │   └── plot/               # figure scripts + README (outputs not committed)
    ├── temporal_full/          # temporal-split experiment
    │   ├── README.md / eval/<run>.json / dataset_split* / test/split_test_real.jsonl
    │   └── plot/               # figure scripts + README (outputs not committed)
    └── PLOT_STYLE.md           # figure style conventions
```

## Environment & data

- Two python venvs are used and **not committed**: `.tokamind-train-env` (training/
  evaluation) and `.freegsnke-solve-env` (FreeGSNKE solving/feature extraction).
  Reproduce: `python -m venv` + install `pyproject.toml` deps; solver env additionally
  needs freegsnke + the tokamak equilibrium input files (see `configs/`).
- Raw MAST zarr data come from FAIR-MAST (S3); training caches (npz), model checkpoints
  and test caches (npz) are **not committed** (GitHub 100 MB/file limit). They live in
  the original collab workspace (`data/`, `runs/`,
  `paper_artifacts/{random,temporal_full}/checkpoints|test/*.npz`) and can be re-created
  with `scripts/build_cache_batched.py` etc.; the `eval/*.json` here are sufficient to
  reproduce every number in the paper tables, and the checkpoints/test caches reproduce
  them by re-running `scripts/evaluate_tokamind_testset.py`.
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
`paper_artifacts/<study>/eval/` and figure-regeneration scripts in `.../plot/`. Data/split
conventions for Shape-OOD (LCFS canonicalization, δu/δl, OOD definition) are documented in
`progress2.md` §20 and the workflow doc
`Plasma_Shape_OOD_Workflow_v2_Detailed_LCFS.md` (available in the original workspace).
