# Shape-OOD — 69D Sim-to-Real Equilibrium Reconstruction

## Purpose

This paper artifact tests whether topology-aligned Synthetic pretraining improves reconstruction of **unseen diverted plasma geometry**, especially with sparse Real training data. It is a frozen Shape-OOD benchmark, distinct from the random and temporal split experiments.

## Frozen benchmark

- Input: `mast_level2_common_69` (69D magnetic diagnostics); target: raw `psi[65,65]`.
- Shape metadata: `delta_asym = delta_u - delta_l`; an OOD slice satisfies **`delta_asym < -0.214`**.
- Only diverted samples are included. `delta_u`, `delta_l`, `delta_mean`, and `delta_asym` are metadata/evaluation fields only; none is appended to the model input.
- The shot-level split is frozen: Train has low `r_OOD` (`<= 0.10`), Validation intermediate `r_OOD` (`(0.10, 0.50]`), and Test high `r_OOD` (`> 0.50`). Train/Val/Test parent shots are disjoint.
- Test-all contains 20,761 slices. Test-OOD-only is its frozen 19,009-slice subset (`phase=test`, `is_OOD=true`).
- C Synthetic samples are topology-aligned diverted samples with exact `(parent_shot, target_time)` matches from Train only. The archive audit confirms zero Validation/Test parent leakage.

## Real fractions and training arms

Historical authoritative fractions are 5% (287 shots / 8,952 slices), 25% (1,436 / 44,019), 50% (2,872 / 88,044), and 100% (5,743 / 176,319). They are independent historical sampler draws and are **not strictly nested**. NEW 1% (57 / 1,575) is a separately frozen low-data ablation generated with the recovered historical sampler logic; it is not claimed to be historical or nested.

- **A — scratch:** 69D TokaMind MMT from random initialization.
- **C — Synthetic pretrain + Real fine-tune:** same-seed (`s54`, `s55`, `s56`) best checkpoint from independently initialized, 50-epoch topology-aligned Synthetic pretraining, then Real fine-tuning.
- Synthetic stage: AdamW, batch 64, lr `2e-4`, weight decay `1e-4`, cosine schedule, validation-best selection.
- Real stage: AdamW, batch 64, lr `1e-4`, weight decay `1e-4`, cosine schedule, validation-best selection. A uses fully converged max300/patience30 for NEW 1%, Historical 5%, and Historical 25%; completed Historical 50%/100% A runs retain max100. C Real fine-tuning uses max50 with patience10. Test never selects a checkpoint.

## Current completed Test-OOD-only results

Values are three-seed mean plus sample standard deviation. Gains use the paired definition `(A - C) / A × 100%` within each seed. This table is generated from `eval/shape-ood-summary-3seed.json`.

| Real fraction | A nRMSE (%) | C nRMSE (%) | paired nRMSE gain | A LCFS distance (m) | C LCFS distance (m) | paired LCFS gain |
|---|---:|---:|---:|---:|---:|---:|
| 1% | 5.383 ± 0.019 | 2.285 ± 0.202 | 57.57 ± 3.60% | 0.1372 ± 0.0008 | 0.0317 ± 0.0035 | 76.87 ± 2.55% |
| 5% | 2.447 ± 0.126 | 1.706 ± 0.142 | 29.99 ± 9.09% | 0.0428 ± 0.0030 | 0.0250 ± 0.0014 | 41.31 ± 7.06% |
| 25% | 2.071 ± 0.094 | 1.721 ± 0.107 | 16.68 ± 8.22% | 0.0359 ± 0.0010 | 0.0271 ± 0.0014 | 24.62 ± 2.02% |
| 50% | 2.033 ± 0.098 | 1.674 ± 0.206 | 17.27 ± 13.28% | 0.0332 ± 0.0013 | 0.0273 ± 0.0018 | 17.40 ± 8.55% |
| 100% | 2.007 ± 0.079 | 1.858 ± 0.273 | 7.66 ± 11.05% | 0.0336 ± 0.0007 | 0.0285 ± 0.0028 | 15.39 ± 8.11% |

The small per-run JSON files in `eval/` are the authority for the table.

## Contents and reproduction

- `dataset_split.csv`: slice-level frozen Real split. `train_fraction` is a membership list because historical fraction draws are non-nested.
- `dataset_split_shots.csv`: shot-level OOD occupancy and fraction membership.
- `dataset_split_info.json`: definition, provenance hashes, and integrity checks.
- `test/`: frozen Test-all and Test-OOD-only manifests only; no cache is committed.
- `eval/`: 30 small per-run summaries and one three-seed aggregate; no checkpoint, predictions, scaler, or cache is committed.
- `plot/scripts/`: reproducible plotting source only. Generated PNG/PDF files remain untracked.

To reproduce an evaluation, obtain the corresponding external checkpoint/cache and run the project evaluator against `test/split_test_real.jsonl` or `test/split_test_ood_only_real.jsonl`. Every archived eval JSON records source-result provenance and SHA256.

## Known limits

- Raw MAST Zarr, training/test caches, checkpoints (`.pt`), scalers, prediction arrays (`.npz`), and generated figures are intentionally excluded from Git.
- Historical 50%/100% A rows retain max100 Real-stage results. Do not describe cross-fraction trends as a strictly nested controlled ablation or as a universal A-300 convergence study.
