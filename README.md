# ![MAST Bridge](docs/assets/title-banner.svg)

Learn in simulation, reconstruct real tokamak equilibria.

![Synthetic pretraining on FreeGSNKE data, fine-tuning on MAST data with EFIT targets, and tokamak equilibrium reconstruction.](docs/assets/pretraining-finetuning.png)


## Experiments

The experiments compare training on real data from scratch with synthetic
pretraining followed by fine-tuning, using 1–100% of the real training data.

| Study | Evaluation setting | Artifacts |
|---|---|---|
| Random split | Random partition of shots into training, validation, and test sets | [random](paper_artifacts/random/README.md) |
| Temporal split | Training on earlier campaigns and testing on a later campaign | [temporal_full](paper_artifacts/temporal_full/README.md) |
| Shape-OOD | Generalization to held-out diverted plasma geometries | [shape_ood](paper_artifacts/shape_ood/README.md) |

Each study provides evaluation results, dataset split metadata, test manifests,
and figure scripts under `paper_artifacts/`.

## Repository structure

```text
configs/          Diagnostic schemas and experiment configuration
scripts/          Data preparation, synthetic generation, training, and evaluation
src/mast_bridge/  Data readers, equilibrium solvers, and model utilities
paper_artifacts/  Results, dataset splits, and plotting scripts for each study
```

## Installation

Use Python 3.10–3.13. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
mkdir -p ../external
git clone https://github.com/UKAEA-IBM-STFC-Fusion-FMs/tokamind.git ../external/tokamind
git -C ../external/tokamind checkout 0b67cf56cd945b883fd3b0c9050cdfc560f98533
python -m pip install -e . -e ../external/tokamind numpy zarr
```

The pinned TokaMind revision supplies the `mmt` package used for training and
evaluation. Synthetic equilibrium generation additionally requires FreeGSNKE
and MAST machine geometry inputs.

## Data and reproduction

Raw MAST data are sourced from FAIR-MAST and must be obtained separately.
Training and test caches are not included. Checkpoints and normalization scalers
are included for the random and temporal studies; Shape-OOD provides evaluation
summaries without checkpoints.

To rerun an evaluation, use the corresponding checkpoint and test manifest.
Update the manifest's `data_path` entries to your local MAST data locations, or
place the matching test cache at
`<output-directory>/test_cache_<manifest-stem>.npz`.

For example, with the temporal test data prepared:

```bash
mkdir -p outputs/temporal
python scripts/evaluate_tokamind_testset.py \
  --manifest paper_artifacts/temporal_full/test/split_test_real.jsonl \
  --run-dir paper_artifacts/temporal_full/checkpoints/temporal-ft-5pct-s54-lr1e4-ep100 \
  --output-json outputs/temporal/result.json
```

Compare the output with the corresponding JSON file in the study's `eval/`
directory. Study-specific configurations and figure reproduction instructions
are documented in the linked artifact directories.
