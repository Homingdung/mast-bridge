# Machine Configuration From Shot Zarr Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate the five FreeGSNKE-compatible machine pickle files from each downloaded shot Zarr.

**Architecture:** Keep Zarr-to-machine conversion in a small library module and expose it through a workspace-aware CLI. The converter consumes only geometry arrays in the shot Zarr and writes all five outputs under `data/raw/mast/machine/`.

**Tech Stack:** Python 3.10+, zarr, NumPy, pickle, unittest.

## Global Constraints

- Never read pre-generated machine files from `external/freegsnke`.
- Preserve the existing five output filenames expected by `MachineGeometry`.
- Keep download and machine generation as explicit separate steps.

### Task 1: Zarr conversion contracts

**Files:**
- Create: `tests/test_machine_from_zarr.py`
- Modify: `src/mast_bridge/mast/__init__.py`

**Interfaces:**
- Test the public `build_machine_payloads(root)` function and the five output keys.

- [ ] Write fixture tests for active coils, passives, probes, and wall/limiter payloads.
- [ ] Run `python -m unittest tests.test_machine_from_zarr -v` and observe import failure.
- [ ] Export the future converter function from the package without changing reader behavior.

### Task 2: Implement converter

**Files:**
- Create: `src/mast_bridge/mast/machine_from_zarr.py`

**Interfaces:**
- `build_machine_payloads(zarr_path: str | Path) -> dict[str, object]`
- `write_machine_pickles(zarr_path: str | Path, output_dir: str | Path, overwrite: bool = False) -> dict[str, Path]`

- [ ] Implement explicit active-channel and geometry-group mappings.
- [ ] Implement passive-group array expansion with scalar broadcasting and metadata fields.
- [ ] Implement magnetic probe payloads from flux-loop and pickup geometry arrays.
- [ ] Implement limiter/wall payloads from `wall/limiter_r` and `wall/limiter_z`.
- [ ] Run focused tests and confirm they pass.

### Task 3: CLI integration

**Files:**
- Create: `scripts/build_machine_from_zarr.py`
- Modify: `README.md`

- [ ] Add `--shot`, `--data-dir`, `--output-dir`, and `--overwrite`.
- [ ] Use workspace defaults so commands run from `mast-bridge` while data remains in the sibling workspace `data/` directory.
- [ ] Document the required generation command before `inspect_shot.py`.
- [ ] Run the CLI on shot 11766.

### Task 4: Verification

**Files:**
- Modify: `tests/test_mast_reader.py` only if integration coverage requires it.

- [ ] Load generated files through `MachineGeometry`.
- [ ] Run `python scripts/inspect_shot.py --shot 11766`.
- [ ] Run `python -m unittest discover -s tests -q`.
- [ ] Run `python -m py_compile scripts/build_machine_from_zarr.py`.
