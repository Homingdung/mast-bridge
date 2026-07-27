# MAST Real and Synthetic Data Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a tested foundation for selected MAST shot downloads, machine-aware shot loading, and real/synthetic dataset manifests.

**Architecture:** `mast_bridge` owns thin orchestration and data contracts. The external `LARGE_MODEL_FUSION` downloader remains the source of MAST acquisition, while FreeGSNKE will later consume the normalized `ShotRecord.machine` and Lao profile inputs. Tests use filesystem fixtures and avoid importing optional external scientific packages.

**Tech Stack:** Python 3.10+, standard library, optional `zarr`, `numpy`, `xarray`, and FreeGSNKE at runtime; `unittest` for tests; JSON/JSONL for portable metadata.

## Global Constraints

- Do not execute data downloads or FreeGSNKE solves during implementation.
- Keep `Lao85.ipynb` as an exploration draft; production interfaces live under `src/mast_bridge/`.
- Do not silently substitute missing machine geometry.
- Keep downloaded data, processed data, runs, and artifacts outside git-tracked source files.
- Preserve user changes in existing files.

---

### Task 1: Data contracts and machine configuration loader

**Files:**
- Create: `src/mast_bridge/data/__init__.py`
- Create: `src/mast_bridge/data/schema.py`
- Create: `src/mast_bridge/mast/__init__.py`
- Create: `src/mast_bridge/mast/machine_config.py`
- Test: `tests/test_data_contracts.py`

**Interfaces:**
- `MachineGeometry.files: dict[str, Path]`
- `MachineGeometry.load(directory: str | Path) -> MachineGeometry`
- `ShotRecord(shot_id: str, zarr_path: Path, signals: dict, equilibrium: dict, machine: MachineGeometry, metadata: dict)`

- [ ] Write tests for all five required filenames, missing-file errors, and JSON-serializable record metadata.
- [ ] Run `python3 -m unittest tests/test_data_contracts.py`; expect failure because modules do not exist.
- [ ] Implement frozen dataclasses and strict machine-file discovery.
- [ ] Run the focused test and the full suite.

### Task 2: MAST shot reader and download wrapper

**Files:**
- Create: `src/mast_bridge/mast/reader.py`
- Create: `src/mast_bridge/mast/downloader.py`
- Create: `scripts/download_mast_shots.py`
- Test: `tests/test_mast_reader.py`

**Interfaces:**
- `ShotReader.read(shot_id: str | int) -> ShotRecord`
- `build_download_command(script_path: Path, shot_ids: Sequence[str], data_dir: Path) -> list[list[str]]`
- CLI accepts repeatable `--shot`, optional `--data-dir`, and `--external-root`.

- [ ] Write fixture-based tests for shot existence, machine directory loading, and one command per selected shot.
- [ ] Run the focused test; expect failure.
- [ ] Implement lazy optional Zarr loading and subprocess wrapper without downloading.
- [ ] Run focused and full tests.

### Task 3: Dataset manifests and simulation input boundary

**Files:**
- Create: `src/mast_bridge/dataset/__init__.py`
- Create: `src/mast_bridge/dataset/manifest.py`
- Create: `src/mast_bridge/equilibrium/__init__.py`
- Create: `src/mast_bridge/equilibrium/lao_fit.py`
- Test: `tests/test_manifest_and_lao.py`

**Interfaces:**
- `ManifestEntry.to_dict() -> dict`
- `write_manifest(entries: Iterable[ManifestEntry], output_path: str | Path) -> None`
- `LaoProfile.from_dict(data: dict) -> LaoProfile`
- `LaoProfile.to_dict() -> dict`

- [ ] Write tests for real/synthetic manifest fields and fixed/range Lao parameter serialization.
- [ ] Run the focused test; expect failure.
- [ ] Implement JSONL manifest persistence and a solver-neutral Lao parameter contract, without fitting or solving.
- [ ] Run focused and full tests.

### Task 4: Operational documentation

**Files:**
- Modify: `README.md`
- Create: `configs/simulation/lao_custom.example.yaml`

- [ ] Document environment setup, selected-shot download, machine-file validation, reading, output layout, and future FreeGSNKE flow.
- [ ] Document that `--limit` is not an exact shot selector and that no download is run by tests.
- [ ] Verify all documented source paths exist and scan for placeholders.

### Task 5: Final verification

- [ ] Run `python3 -m unittest discover -s tests`.
- [ ] Run `python3 -m py_compile` on all new Python files.
- [ ] Run CLI help for the new download script.
- [ ] Inspect `git status --short` and report ignored/generated files separately.
