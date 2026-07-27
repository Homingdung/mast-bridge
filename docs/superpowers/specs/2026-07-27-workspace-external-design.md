# Workspace External Dependencies Design

## Goal

Make `mast_bridge` the main project repository while keeping `tokamark`, `tokamind`, `freegsnke`, and `LARGE_MODEL_FUSION` as workspace-level external repositories.

## Reader Workflow

The first user step is to create and enter a workspace directory:

```bash
mkdir fusion-workspace
cd fusion-workspace
git clone <mast_bridge-repo-url> mast_bridge
```

Then the user runs an `mast_bridge` bootstrap script that checks or prepares this layout:

```text
fusion-workspace/
├── mast_bridge/
├── external/
│   ├── tokamark/
│   ├── tokamind/
│   ├── freegsnke/
│   └── LARGE_MODEL_FUSION/
├── data/
├── runs/
└── artifacts/
```

## Architecture

`mast_bridge` owns all production logic for MAST data orchestration, EFIT profile reading, Lao85 fitting, FreeGSNKE synthetic generation, dataset manifests, and training entrypoints. External repositories are never copied into `mast_bridge/src`; they are imported through small adapter modules and configured through local path files.

The first implementation only needs a minimal bootstrap and doctor layer:

- discover external repositories under `fusion-workspace/external/`
- tolerate the current development layout where repositories are direct workspace siblings
- write `configs/paths.local.yaml` from discovered absolute paths
- keep `configs/paths.local.yaml` out of git
- provide editable install commands for the main project and external Python packages
- run import checks for `mast_bridge`, `tokamark`, `mmt`, and `freegsnke`

## Configuration

The committed template is `configs/paths.example.yaml`. The generated local file is `configs/paths.local.yaml`.

`paths.local.yaml` contains absolute paths:

```yaml
workspace_root: /path/to/fusion-workspace
external_root: /path/to/fusion-workspace/external
tokamark_root: /path/to/fusion-workspace/external/tokamark
tokamind_root: /path/to/fusion-workspace/external/tokamind
freegsnke_root: /path/to/fusion-workspace/external/freegsnke
large_model_fusion_root: /path/to/fusion-workspace/external/LARGE_MODEL_FUSION
mast_data_dir: /path/to/fusion-workspace/external/LARGE_MODEL_FUSION/mast_data
data_root: /path/to/fusion-workspace/data
runs_root: /path/to/fusion-workspace/runs
artifacts_root: /path/to/fusion-workspace/artifacts
```

## Notebook Policy

`Lao85.ipynb` is an exploration draft only. Production code must live under `src/mast_bridge/`, and notebooks may only import that production code for inspection or demonstration.

## Validation

The bootstrap behavior is covered by unit tests that create temporary workspace layouts, verify repository discovery, verify local YAML generation, and verify doctor checks report missing and available dependencies clearly.
