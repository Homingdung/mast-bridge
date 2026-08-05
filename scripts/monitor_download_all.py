#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_ROOT / "src"))

from mast_bridge.mast.downloader import REQUIRED_SHOT_GROUPS


def count_shot_states(shots: list[str], data_dir: Path) -> tuple[int, int]:
    complete = 0
    started = 0
    for shot in shots:
        shot_path = data_dir / f"{shot}.zarr"
        if not (shot_path / "zarr.json").is_file():
            continue
        started += 1
        if all(
            (shot_path / group / "zarr.json").is_file()
            for group in REQUIRED_SHOT_GROUPS
        ):
            complete += 1
    return complete, started


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def shard_count() -> str:
    try:
        result = subprocess.run(
            ["pgrep", "-f", "download_all_mast_shots", "-c"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip() or "?"
    except (subprocess.SubprocessError, OSError):
        return "?"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Periodic download progress snapshots.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--pid", type=int, required=True, help="Main downloader process PID.")
    parser.add_argument("--interval", type=int, default=1800, help="Snapshot interval in seconds.")
    parser.add_argument("--output", type=Path, required=True, help="Append-only progress log.")
    args = parser.parse_args(argv)

    data_dir = args.data_dir.resolve()
    shot_list = data_dir / ".all_level2_shots.txt"
    shots = (
        [line.strip() for line in shot_list.read_text().splitlines() if line.strip()]
        if shot_list.is_file()
        else []
    )

    def snapshot() -> str:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            counts = pool.submit(count_shot_states, shots, data_dir)
            procs = pool.submit(shard_count)
            complete, started = counts.result()
        return (
            f"{stamp} | started={started}/{len(shots) if shots else '?'} "
            f"complete={complete} | download_procs={procs.result()}"
        )

    with args.output.open("a", encoding="utf-8") as f:
        f.write(
            "# monitor start "
            + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            + "\n"
        )
        f.flush()
        while True:
            time.sleep(args.interval)
            line = snapshot()
            print(line, flush=True)
            f.write(line + "\n")
            f.flush()
            if not alive(args.pid):
                f.write(
                    "# monitor done (downloader exited) "
                    + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
                    + "\n"
                )
                f.flush()
                return 0


if __name__ == "__main__":
    raise SystemExit(main())
