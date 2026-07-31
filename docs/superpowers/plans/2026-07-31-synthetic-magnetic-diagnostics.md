# Synthetic Magnetic Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate FreeGSNKE magnetic diagnostics for accepted synthetic equilibria and build real, synthetic, and mixed TokaMind manifests with diagnostics as model input.

**Architecture:** A focused serialization module owns the `diagnostics.npz` contract. A batch script reconstructs a solved FreeGSNKE state from saved total `psi`, coil currents, and Lao85 parameters, then uses the official probe calculators. Manifest and training loaders share the same validation and loading functions.

**Tech Stack:** Python 3.12, NumPy, Zarr, FreeGSNKE, unittest, JSONL manifests.

## Global Constraints

- Only synthetic samples already accepted at solver tolerance `<= 1e-8` are eligible.
- Do not modify downloaded raw MAST Zarr or machine pickle files.
- Flux loops are stored in MAST Level 2 `Wb` using scale `2*pi`.
- Pickups use corrected MAST Level 2 OBR/OBV orientations.
- Synthetic diagnostics generation does not rerun the Grad-Shafranov solver.

---

### Task 1: Diagnostics NPZ contract

**Files:**
- Create: `src/mast_bridge/simulation/synthetic_diagnostics.py`
- Create: `tests/test_synthetic_diagnostics.py`

**Interfaces:**
- Produces: `write_synthetic_diagnostics(path, ...)`, `load_synthetic_diagnostic_values(path)`, and `synthetic_diagnostics_rejection_reason(path)`.

- [ ] Write tests for a valid round trip and malformed/non-finite payload rejection.
- [ ] Run the focused tests and confirm they fail because the module is absent.
- [ ] Implement the minimal typed NPZ writer, loader, and validator.
- [ ] Run the focused tests and confirm they pass.

### Task 2: Existing equilibrium reconstruction and batch generation

**Files:**
- Create: `scripts/build_synthetic_magnetic_diagnostics.py`
- Create: `tests/test_build_synthetic_magnetic_diagnostics_script.py`
- Modify: `scripts/run_freegsnke_forward.py`
- Modify: `tests/test_forward_script.py`

**Interfaces:**
- Consumes: accepted synthetic manifest rows and their `equilibrium.npz`/`metadata.json`.
- Produces: one `diagnostics.npz` per successful row and a JSONL batch report.

- [ ] Test saved-current application, total-psi reconstruction, CLI defaults, skip behavior, and report rows.
- [ ] Run focused tests and confirm expected failures.
- [ ] Implement state reconstruction and official FreeGSNKE probe calculation.
- [ ] Keep diagnostics generation as an explicit accepted-manifest backfill step so equilibrium solving and diagnostics QC remain separately resumable.
- [ ] Run focused tests and confirm they pass.

### Task 3: Diagnostics-aware manifests and training loader

**Files:**
- Modify: `scripts/build_experiment_manifests.py`
- Modify: `tests/test_build_experiment_manifests_script.py`
- Modify: `src/mast_bridge/training/tokamind_manifest.py`
- Modify: `tests/test_tokamind_manifest_training.py`

**Interfaces:**
- Consumes: real Zarr magnetics or synthetic `diagnostics.npz`.
- Produces: identical named feature vectors for both sources and three diagnostics-ready manifests.

- [ ] Test that invalid/missing synthetic diagnostics are excluded.
- [ ] Test that valid paths are written into synthetic and mixed rows.
- [ ] Test synthetic diagnostic feature loading with literal names and values.
- [ ] Run focused tests and confirm expected failures.
- [ ] Implement manifest filtering and source-aware diagnostics loading.
- [ ] Run focused and regression tests.

### Task 4: Documentation and end-to-end verification

**Files:**
- Modify: `README.md`

**Interfaces:**
- Documents: generation, validation, manifest construction, dry-run, and training commands.

- [ ] Add commands using the current `tokamark_lao85_uniform_small_iter500` paths.
- [ ] Generate diagnostics for one accepted sample and inspect its NPZ contract.
- [ ] Build diagnostics-ready manifests and run TokaMind diagnostics dry-run.
- [ ] Run all affected unit tests and Python compilation checks.
