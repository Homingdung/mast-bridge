#!/usr/bin/env python
"""Extract (features, psi) matrices from a manifest into a dataset cache npz.

The cache format matches ``_load_dataset_cache``: npz with sample_ids,
features (N x D) and psi (N x 65 x 65) in manifest row order. Rows with
non-finite features or psi are dropped with a warning.

Usage:
  build_dataset_cache.py --manifest FILE --output FILE [--workers 32]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

SCRIPT_ROOT = Path(__file__).resolve().parents[0]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

sys.path.insert(0, str(SCRIPT_ROOT.parents[0] / "src"))

from mast_bridge.training.tokamind_manifest import (  # noqa: E402
    INPUT_MAGNETIC_DIAGNOSTICS,
    TARGET_RAW_PSI,
    _feature_vector,
    _psi_for_row,
)

WORKSPACE_ROOT = SCRIPT_ROOT.parents[1]
SCALERS_PATH = WORKSPACE_ROOT / "runs" / "tokamind-large-real-scratch-100e" / "manifest_scalers.npz"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a dataset cache npz from a manifest.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument(
        "--diagnostics-name",
        type=str,
        default="diagnostics.npz",
        help="Diagnostics file name inside synthetic sample dirs (e.g. diagnostics_noisy.npz).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    with np.load(SCALERS_PATH, allow_pickle=True) as data:
        feature_names = [str(v) for v in data["feature_names"].tolist()]

    rows = [json.loads(line) for line in args.manifest.expanduser().resolve().open(encoding="utf-8") if line.strip()]
    if args.diagnostics_name != "diagnostics.npz":
        for row in rows:
            row["diagnostics_path"] = str(
                Path(row["data_path"]).expanduser().resolve() / args.diagnostics_name
            )

    def extract(row: dict) -> tuple[dict, np.ndarray, np.ndarray] | Exception | None:
        try:
            fv = _feature_vector(row, feature_names, INPUT_MAGNETIC_DIAGNOSTICS)
            pv2d = _psi_for_row(row, TARGET_RAW_PSI)
            pv = pv2d.reshape(-1)
            if not (np.isfinite(fv).all() and np.isfinite(pv).all()):
                return None
            return row, fv, pv2d
        except Exception as exc:  # noqa: BLE001 - per-row tolerance
            return exc

    t0 = time.time()
    good_rows: list[dict] = []
    feature_list: list[np.ndarray] = []
    psi_list: list[np.ndarray] = []
    bad: list[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(extract, rows, chunksize=32):
            if result is None:
                continue
            if isinstance(result, Exception):
                bad.append(str(result))
                continue
            good_rows.append(result[0])
            feature_list.append(result[1])
            psi_list.append(result[2])
    if bad:
        print(f"WARNING: dropped {len(bad)} rows, first: {bad[0]}", flush=True)
    if not feature_list:
        raise ValueError("No rows survived feature extraction")

    features = np.stack(feature_list, axis=0).astype(np.float32)
    psi = np.stack(psi_list, axis=0).astype(np.float32)
    sample_ids = np.asarray([str(r.get("sample_id")) for r in good_rows], dtype=str)
    print(
        f"extracted {features.shape[0]}/{len(rows)} rows in {time.time() - t0:.1f}s "
        f"(features {features.shape[1]} dims, psi {psi.shape[1]}x{psi.shape[2]})",
        flush=True,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, sample_ids=sample_ids, features=features, psi=psi)
    print(f"cache: {output} ({output.stat().st_size / 1e9:.2f} GB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
