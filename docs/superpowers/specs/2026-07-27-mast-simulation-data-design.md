# MAST Real and Synthetic Data Design

## Goal

Provide a small, explicit mast_bridge workflow that lets readers select a limited set of MAST shots, load each shot's Zarr data and five machine-configuration files, and preserve a common representation for future FreeGSNKE simulation and AI training.

## Scope

This first implementation provides the download wrapper, shot reader, machine-configuration loader, `ShotRecord` model, manifest writer, and documentation. The Lao profile fitter and FreeGSNKE solver receive stable input/output boundaries but are not reimplemented from the notebook draft or copied from external repositories.

## Data Flow

```text
LARGE_MODEL_FUSION downloader
        -> data/raw/mast/<shot>.zarr
        -> ShotReader
        -> ShotRecord(signals, equilibrium, machine, metadata)
        -> processed real manifest
        -> future lao_fit / FreeGSNKE adapters
        -> processed synthetic manifest
```

## Machine Configuration

The reader requires these files for a complete machine description, using the exact filenames present in the source data and allowing explicit aliases for known spelling variations:

- `MAST_active_coils.pickle`
- `MAST_limiter.pickle`
- `MAST_magentic_probes.pickle`
- `MAST_passive_coilds.pickle`
- `MAST_wall.pickle`

The loader returns a `MachineGeometry` object. It does not silently replace missing files with a different machine, because geometry mismatches would invalidate real-versus-synthetic comparisons.

## Storage

```text
data/raw/mast/<shot>.zarr
data/processed/real/<shot>/equilibrium/lao_fit.json
data/processed/real/<shot>/machine/
data/processed/synthetic/<shot>_variant_<id>/metadata.json
data/manifests/*.jsonl
```

`lao_fit.json` remains the fitted real-shot baseline. User-defined Lao parameter ranges belong in a separate simulation configuration and are recorded in each synthetic sample's metadata.

## Training Boundary

Real and synthetic samples share manifest fields and are split by parent shot, never by individual time window or synthetic variant. This prevents synthetic variants of a validation shot from leaking into training.

## Error Handling

- Download failures are returned by the external downloader and surfaced by the wrapper.
- Missing shot Zarr paths raise `FileNotFoundError`.
- Missing machine files raise a single error listing all missing names.
- Unsupported file formats are reported as explicit reader errors.

## Acceptance Criteria

- A reader can run `python scripts/download_mast_shots.py --shot 11766` without modifying the external repository.
- A local fixture can be loaded into a `ShotRecord` without requiring Zarr or FreeGSNKE at test collection time.
- All five machine configuration paths are represented and validated.
- A JSONL manifest can describe both real and future synthetic samples.
- The README documents setup, download, inspection, reading, and future solver inputs.
