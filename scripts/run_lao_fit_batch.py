#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = SCRIPT_ROOT.parent
sys.path.insert(0, str(SCRIPT_ROOT / "src"))

from mast_bridge.equilibrium.lao_from_zarr import (
    default_lao_fit_path,
    _shot_rows,
)
from mast_bridge.workspace import discover_workspace  # noqa: E402

import numpy as np  # noqa: E402


def load_shot_list(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def process_one(shot: str, args: argparse.Namespace, data_dir: Path, cache_dir: Path) -> dict:
    cache = cache_dir / f"{shot}.json"
    if not args.force and cache.is_file():
        return {"shot": shot, "status": "skipped"}
    try:
        rows = _shot_rows(
            data_dir / f"{shot}.zarr",
            n_alpha=args.n_alpha,
            n_beta=args.n_beta,
            min_finite_points=args.min_finite_points,
        )
        payload = {
            "rows": [
                {
                    "shot": row["shot"],
                    "time": float(row["time"]),
                    "ip": float(row["ip"]),
                    "fvac": float(row["fvac"]),
                    "freegsnke_alpha": row["freegsnke_alpha"].tolist(),
                    "freegsnke_beta": row["freegsnke_beta"].tolist(),
                }
                for row in rows
            ]
        }
        status = "ok"
    except Exception as exc:  # noqa: BLE001 - one bad shot must not abort the batch
        payload = {"error": f"{type(exc).__name__}: {exc}"}
        status = "failed"
    tmp = cache.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(cache)
    return {"shot": shot, "status": status, "n_rows": len(rows) if status == "ok" else 0}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Batch Lao85 fit with per-shot fault tolerance and resume."
    )
    parser.add_argument("--shot-list", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--n-alpha", type=int, default=3)
    parser.add_argument("--n-beta", type=int, default=3)
    parser.add_argument("--min-finite-points", type=int, default=8)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--assemble-only", action="store_true", help="Only assemble NPZ from cache.")
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    layout = discover_workspace(SCRIPT_ROOT)
    data_dir = (args.data_dir or layout.data_root / "raw" / "mast").expanduser().resolve()
    output = (args.output or default_lao_fit_path(WORKSPACE_ROOT)).expanduser().resolve()
    cache_dir = (args.cache_dir or output.parent / "lao_fit_rows").expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    shots = load_shot_list(args.shot_list.expanduser().resolve())
    report = args.report or data_dir / ".lao_fit_report.jsonl"

    if args.assemble_only:
        pass
    else:
        if not args.force:
            shots = [s for s in shots if not (cache_dir / f"{s}.json").is_file()]
        print(f"[lao-fit] shots={len(shots)} workers={args.workers}", flush=True)
        if args.dry_run:
            return 0
        counts = {"ok": 0, "failed": 0, "skipped": 0}
        with report.open("a", encoding="utf-8") as rf:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(process_one, s, args, data_dir, cache_dir) for s in shots]
                for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
                    item = future.result()
                    item["time"] = datetime.now(timezone.utc).isoformat()
                    rf.write(json.dumps(item) + "\n")
                    rf.flush()
                    counts[item["status"]] += 1
                    print(
                        f"[lao-fit] {i}/{len(shots)} {item['shot']} -> {item['status']} rows={item.get('n_rows', 0)}",
                        flush=True,
                    )
        print(f"[lao-fit] done: {counts}", flush=True)

    rows: list[dict] = []
    errors: list[str] = []
    for shot in load_shot_list(args.shot_list.expanduser().resolve()):
        cache = cache_dir / f"{shot}.json"
        if not cache.is_file():
            continue
        payload = json.loads(cache.read_text(encoding="utf-8"))
        if "error" in payload:
            errors.append(shot)
        else:
            rows.extend(payload["rows"])
    if not rows:
        print(f"[lao-fit] no rows to assemble (errors={len(errors)})", flush=True)
        return 1
    table = {
        "shot": np.asarray([row["shot"] for row in rows], dtype=str),
        "time": np.asarray([row["time"] for row in rows], dtype=float),
        "ip": np.asarray([row["ip"] for row in rows], dtype=float),
        "fvac": np.asarray([row["fvac"] for row in rows], dtype=float),
        "freegsnke_alpha": np.vstack([row["freegsnke_alpha"] for row in rows]),
        "freegsnke_beta": np.vstack([row["freegsnke_beta"] for row in rows]),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **table)
    shots_with_rows = sorted({row["shot"] for row in rows})
    print(f"[lao-fit] assembled: rows={len(rows)} shots={len(shots_with_rows)} errors={len(errors)}", flush=True)
    print(f"[lao-fit] output: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
