# FreeGSNKE Environment Configuration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task with verification checkpoints.

**Goal:** Make the existing workspace bootstrap install `external/freegsnke` and its runtime dependencies into the active `.mast-bridge-env`, with clear reader documentation and import verification.

**Architecture:** Keep FreeGSNKE outside the repository. Extend the existing workspace command generation and README instructions; do not add a hard-coded local path dependency to the package metadata. Use the existing doctor import check as the user-facing verification.

**Tech Stack:** Python 3.10+, setuptools, pip editable installs, unittest/pytest.

## Global Constraints

- FreeGSNKE remains at `external/freegsnke`.
- Full setup uses the active interpreter via `python -m pip`.
- MAST-only setup must not require FreeGSNKE.
- Do not modify external repository source files.

### Task 1: Cover complete editable-install command generation

**Files:**
- Modify: `tests/test_workspace_bootstrap.py`
- Modify: `src/mast_bridge/workspace.py`

**Interfaces:**
- Preserve `editable_install_commands(layout, python, with_deps)` and its existing four-repository order.
- Ensure the FreeGSNKE command targets `layout.freegsnke_root` and includes dependency installation only when `with_deps=True`.

- [ ] **Step 1: Write the failing test** asserting the FreeGSNKE command uses the active Python, editable mode, the discovered external path, and omits `--no-deps` for full setup.
- [ ] **Step 2: Run `pytest tests/test_workspace_bootstrap.py -q` and verify the new assertion fails for the current behavior.
- [ ] **Step 3: Implement the minimal command-generation adjustment.
- [ ] **Step 4: Run the focused test and verify it passes.

### Task 2: Document the one-command full environment setup

**Files:**
- Modify: `README.md`

**Interfaces:**
- The documented command is `python scripts/bootstrap_workspace.py --install-editable --with-deps` after activation.
- The documented verification is `python -c "import freegsnke; print(freegsnke.__file__)"` and `python scripts/doctor.py`.

- [ ] **Step 1: Update the full-environment section to state that the command installs FreeGSNKE from `external/freegsnke` and its declared runtime dependencies.
- [ ] **Step 2: Add an explicit import verification and explain that `doctor.py` may still report missing workspace data paths independently of Python import status.
- [ ] **Step 3: Run `git diff --check`.

### Task 3: Verify the active environment

**Files:**
- No source changes.

- [ ] **Step 1: Run the complete test suite with `pytest -q`.
- [ ] **Step 2: Run `.mast-bridge-env/bin/python -m pip install -e ../external/freegsnke` to install FreeGSNKE into the project environment.
- [ ] **Step 3: Run `.mast-bridge-env/bin/python -c 'import freegsnke; print(freegsnke.__file__)'` and verify the import succeeds.
- [ ] **Step 4: Run `.mast-bridge-env/bin/python scripts/doctor.py` and record the import result and any unrelated missing workspace paths.
