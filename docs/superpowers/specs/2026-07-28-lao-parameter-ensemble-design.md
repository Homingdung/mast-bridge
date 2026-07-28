# Lao Parameter Ensemble Data Augmentation Design

## Status

Design only. Do not implement this spec until the user explicitly approves implementation.

## Goal

Add a script that fits Lao profile parameters from real MAST `p'` and `ff'` profiles, then samples many new MAST-like Lao parameter rows for data augmentation.

This script produces a parameter ensemble only. It does not run FreeGSNKE and does not write synthetic equilibrium solves.

The generated NPZ should be field-compatible with the Lao parameter data currently consumed by forward tooling, but the current `scripts/run_freegsnke_forward.py` selects rows by `shot/time`. Consuming every generated sample will require either a future sample-index-aware runner or one exported NPZ per selected sample. That consumer is outside this spec.

## Non-Goals

- Do not batch-run FreeGSNKE.
- Do not create `equilibrium.npz` synthetic solve outputs.
- Do not alter real MAST Zarr stores.
- Do not change the existing `scripts/run_freegsnke_forward.py` behavior in this phase.
- Do not add the future sample-index-aware parameter consumer in this phase.

## Existing Context

The repository already has:

- `src/mast_bridge/equilibrium/lao_fit.py`, currently a lightweight config container.
- `scripts/run_freegsnke_forward.py`, which consumes an NPZ with `ip`, `fvac`, `freegsnke_alpha`, and `freegsnke_beta`.
- `configs/simulation/lao_custom.example.yaml`, which documents user-provided Lao coefficient ranges.
- Historical report outputs under `../data_analysis_report/efit_lao_freegsnke_forward/lao_parameter_ensemble/`, including:
  - `all_zarr_lao_parameter_fits.npz`
  - `random_lao_sources_n512.npz`

The historical report documents the FreeGSNKE conversion:

```text
freegsnke_alpha = axis_r * pprime_coeff
freegsnke_beta = ffprime_coeff / axis_r
```

`Lao85(..., Ip=...)` handles the current normalization later, so the parameter ensemble must not fold `Ip` into `alpha` or `beta`.

## Proposed CLI

Add a script:

```bash
python scripts/build_lao_parameter_ensemble.py \
  --data-dir ../data/raw/mast \
  --shot 11766 \
  --shot 11767 \
  --n-samples 512 \
  --seed 42 \
  --output ../data_analysis_report/efit_lao_freegsnke_forward/lao_parameter_ensemble/random_lao_sources_n512.npz
```

Useful options:

- `--shot`: may be repeated. If omitted, discover `*.zarr` in `--data-dir`.
- `--time-min`, `--time-max`: optional time window.
- `--profile-grid-size`: default `201`.
- `--degree`: default `2`, producing 3 Lao coefficients per profile.
- `--percentile-low`, `--percentile-high`: defaults `2.5` and `97.5`.
- `--method`: default `multivariate_normal`, alternatives `bootstrap_jitter` and `independent_uniform`.
- `--fits-output`: optional NPZ path for the real fitted rows.
- `--output`: required NPZ path for sampled rows.
- `--anchor-strategy`: default `resample_real_time`, used to choose `shot/time` anchors for later machine-current context.

## Real Data Inputs

For each shot Zarr, read:

- `equilibrium/psi_norm`: normalized flux grid for profiles.
- `equilibrium/dpressure_dpsi`: real `p'` profile samples.
- `equilibrium/f_df_dpsi`: real `ff'` profile samples.
- `equilibrium/time`: EFIT time axis.
- `equilibrium/magnetic_axis_r`: major radius at magnetic axis, meters.
- `equilibrium/bvac_rmag`: `fvac`, the vacuum toroidal field radius product.
- Plasma current from the most reliable available source:
  - Prefer `summary/ip` interpolated to the EFIT time.
  - Fall back to an equilibrium current field if a suitable field exists.
  - Skip rows when no finite `Ip` can be recovered.

The script must skip rows where required profile values, `axis_r`, `Ip`, or `fvac` are missing, non-finite, or physically unusable.

## Lao Profile Fit

Fit each profile as a polynomial in normalized flux:

```text
profile(psi_norm) = c0 + c1 * psi_norm + c2 * psi_norm^2
```

For both `p'` and `ff'`:

1. Select finite points only.
2. Require enough valid points for the requested degree.
3. Fit coefficients in ascending order `[c0, c1, c2]`.
4. Evaluate the fit on the original profile grid.
5. Compute RMSE and relative RMSE.
6. Optionally reject rows above a configurable relative RMSE threshold.

The fit function should be solver-neutral. It should operate on arrays and return physical coefficients before FreeGSNKE conversion.

## Unit Conversion Boundary

This is the most important correctness boundary.

Zarr profiles and fitted coefficients are physical EFIT-like profile values:

- `pprime_coeff` fits `equilibrium/dpressure_dpsi`.
- `ffprime_coeff` fits `equilibrium/f_df_dpsi`.

FreeGSNKE `Lao85` expects coefficients converted using magnetic-axis major radius:

```text
freegsnke_alpha = axis_r * pprime_coeff
freegsnke_beta = ffprime_coeff / axis_r
```

Rules:

- Preserve both physical and FreeGSNKE coefficients in outputs.
- Do not multiply or divide by `Ip` in this script.
- Do not apply the `axis_r` conversion twice.
- Require `axis_r > 0`.
- Record conversion metadata in the JSON sidecar and preserve both coefficient arrays in the NPZ.

## Sampling Design

The default sampler should preserve correlations across real fitted parameters.

Build a matrix with columns:

```text
pprime_coeff[0:3], ffprime_coeff[0:3], ip, fvac, axis_r
```

For `multivariate_normal`:

1. Filter real rows to finite values.
2. Compute percentile clipping bounds from real rows.
3. Estimate mean and covariance from the clipped matrix.
4. Draw candidate rows from the multivariate normal.
5. Reject candidates outside percentile bounds or with invalid `Ip`, `fvac`, or `axis_r`.
6. Assign each accepted parameter row an anchor real `shot/time` row by resampling fitted rows. The anchor supplies later machine-current context only; it is not used in the Lao coefficient conversion.
7. Continue until `n_samples` accepted or a maximum attempt limit is reached.

For `bootstrap_jitter`:

1. Sample real fitted rows with replacement.
2. Add configurable Gaussian jitter in standardized parameter space.
3. Clip or reject outside percentile bounds.

For `independent_uniform`:

1. Independently sample each parameter between percentile bounds.
2. Mark metadata with a warning that correlations are not preserved.

## Outputs

The sampled parameter NPZ should contain:

- `shot`: anchor shot, shaped for compatibility with existing row-selection conventions.
- `time`: anchor time, shaped for compatibility with existing row-selection conventions.
- `profile_grid`: common profile grid, shape `(profile_grid_size,)`.
- `pprime_coeff`: physical coefficients, shape `(n_samples, degree + 1)`.
- `ffprime_coeff`: physical coefficients, shape `(n_samples, degree + 1)`.
- `freegsnke_alpha`: converted FreeGSNKE coefficients.
- `freegsnke_beta`: converted FreeGSNKE coefficients.
- `ip`: sampled plasma current.
- `fvac`: sampled vacuum toroidal field radius product.
- `axis_r`: sampled magnetic-axis major radius.
- `pprime`: sampled physical profile values on `profile_grid`.
- `ffprime`: sampled physical profile values on `profile_grid`.
- `sample_id`: stable synthetic parameter IDs.
- `anchor_shot`: real shot selected for later machine-current context.
- `anchor_time`: real time selected for later machine-current context.
- `parent_shot`: source shot for bootstrap rows; otherwise the same as `anchor_shot`.
- `parent_time`: source time for bootstrap rows; otherwise the same as `anchor_time`.
- `random_seed`: scalar seed.
- `parameter_names`: ordered parameter names.
- `parameter_range_low`: fitted percentile lower bounds.
- `parameter_range_high`: fitted percentile upper bounds.
- `parameter_range_percentiles`: `[low, high]`.

Field compatibility note: the current forward script can read the scalar and coefficient field names, but it cannot distinguish multiple synthetic rows that share the same `shot/time`. A future runner should select by `sample_id` or explicit row index.

If `--fits-output` is provided, also write real fitted rows with:

- `shot`
- `shot_path`
- `time`
- `ip`
- `fvac`
- `axis_r`
- `pprime_coeff`
- `ffprime_coeff`
- `freegsnke_alpha`
- `freegsnke_beta`
- `pprime_relative_rmse`
- `ffprime_relative_rmse`

Write a JSON sidecar next to each NPZ containing readable metadata:

- input paths and shots
- time filters
- polynomial degree
- sampling method
- anchor strategy
- sample count
- rejection count
- random seed
- unit conversion formula
- profile field names
- skip/rejection summary

## Data Flow

1. Discover shot Zarr stores.
2. Read equilibrium and summary fields.
3. Fit physical Lao coefficients from real `p'` and `ff'` profiles.
4. Convert physical coefficients to FreeGSNKE `alpha` and `beta`.
5. Save optional real fitted-row NPZ.
6. Sample new physical coefficients and scalar parameters.
7. Recompute sampled `alpha` and `beta` from sampled `axis_r`.
8. Assign anchor `shot/time` rows for later machine-current context.
9. Evaluate sampled `pprime` and `ffprime` profiles on the common grid.
10. Save sampled NPZ and JSON sidecar.

## Error Handling

- Missing required Zarr fields: fail the shot with a clear warning and continue other shots.
- No usable fitted rows: fail the command.
- Too few usable fitted rows for covariance sampling: fail unless `--method bootstrap_jitter` or `--method independent_uniform` is selected.
- Singular covariance: add a small diagonal regularizer and record it in metadata.
- Acceptance sampler exhaustion: fail with diagnostics showing bounds and accepted count.
- Existing output path: overwrite only with `--overwrite`.

## Testing Plan

Use test-first implementation when this spec is approved.

Core unit tests:

- Polynomial fit returns ascending-order coefficients.
- Fit ignores NaNs and rejects too-few-points profiles.
- Conversion uses exactly `alpha = axis_r * pprime_coeff`.
- Conversion uses exactly `beta = ffprime_coeff / axis_r`.
- Conversion rejects zero or negative `axis_r`.
- Sampler output shapes match `n_samples` and `degree + 1`.
- Sampler preserves required scalar and coefficient fields used by existing forward tooling.
- Sampler writes `shot/time`, `anchor_shot/anchor_time`, and stable `sample_id` fields.
- CLI refuses to overwrite output without `--overwrite`.

Light integration tests:

- Build a temporary minimal Zarr-like store or in-memory arrays and verify the script writes an NPZ plus JSON sidecar.
- Verify a tiny generated NPZ can be consumed by `select_fit_row`-style logic or a small compatibility helper without importing FreeGSNKE.

## Open Decisions

- Exact `Ip` fallback source if `summary/ip` is absent should be confirmed during implementation by inspecting available MAST equilibrium fields across more shots.
- Default relative RMSE rejection threshold should be chosen after measuring current real fitted rows. A conservative starting value is `0.05`, but it should be data-driven.
- Whether `fvac` should always be absolute-valued. Historical code used `abs(bvac_rmag)` in places; this should be verified against current MAST conventions before final implementation.
