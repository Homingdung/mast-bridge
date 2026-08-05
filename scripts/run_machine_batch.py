#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_ROOT / "src"))

from mast_bridge.workspace import discover_workspace

MACHINE_PICKLES = (
    "MAST_active_coils.pickle",
    "MAST_limiter.pickle",
    "MAST_magentic_probes.pickle",
    "MAST_passive_coils.pickle",
    "MAST_wall.pickle",
)


def machine_done(machine_dir: Path) -> bool:
    return machine_dir.is_dir() and all(
        (machine_dir / name).is_file() for name in MACHINE_PICKLES
    )


def run_one(shot: str, args: argparse.Namespace, data_dir: Path) -> dict:
    machine_dir = data_dir / "machine" / shot
    if not args.force and machine_done(machine_dir):
        return {"shot": shot, "status": "skipped"}
    command = [
        args.python,
        str(SCRIPT_ROOT / "scripts" / "build_machine_from_zarr.py"),
        "--data-dir",
        str(data_dir),
        "--output-dir",
        str(machine_dir),
        "--shot",
        shot,
        "--overwrite",
    ]
    result = subprocess.run(
        command, capture_output=True, text=True, timeout=args.timeout
    )
    if result.returncode != 0:
        return {
            "shot": shot,
            "status": "failed",
            "error": result.stderr.strip().splitlines()[-1] if result.stderr else "unknown",
        }
    if not machine_done(machine_dir):
        return {"shot": shot, "status": "failed", "error": "pickles missing after success"}
    return {"shot": shot, "status": "ok"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Batch build machine pickles for many shots with resume support."
    )
    parser.add_argument("--shot-list", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--python", default=sys.executable, help="Python interpreter for the CLI.")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--timeout", type=int, default=600, help="Per-shot timeout seconds.")
    parser.add_argument("--force", action="store_true", help="Rebuild even if pickles exist.")
    parser.add_argument("--report", type=Path, default=None, help="Append-only jsonl report.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    layout = discover_workspace(SCRIPT_ROOT)
    data_dir = (args.data_dir or layout.data_root / "raw" / "mast").expanduser().resolve()
    shots = [
        line.strip()
        for line in args.shot_list.read_text().splitlines()
        if line.strip()
    ]
    report = args.report or data_dir / ".machine_batch_report.jsonl"
    if not args.force:
        done = set()
        if report.is_file():
            for line in report.read_text().splitlines():
                try:
                    item = json.loads(line)
                    if item.get("status") == "ok":
                        done.add(item["shot"])
                except json.JSONDecodeError:
                    continue
        shots = [s for s in shots if s not in done]

    print(f"[machine] shots={len(shots)} workers={args.workers} force={args.force}", flush=True)
    if args.dry_run:
        print(f"[machine] first: {shots[:5]} ...", flush=True)
        return 0

    counts = {"ok": 0, "failed": 0, "skipped": 0}
    with report.open("a", encoding="utf-8") as rf:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_one, s, args, data_dir) for s in shots]
            for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
                item = future.result()
                item["time"] = datetime.now(timezone.utc).isoformat()
                rf.write(json.dumps(item) + "\n")
                rf.flush()
                counts[item["status"]] += 1
                print(
                    f"[machine] {i}/{len(shots)} {item['shot']} -> {item['status']}"
                    + (f" ({item['error']})" if item["status"] == "failed" else ""),
                    flush=True,
                )
    print(f"[machine] done: {counts}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
