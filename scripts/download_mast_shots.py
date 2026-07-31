#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_ROOT / "src"))

from mast_bridge.mast.downloader import (
    build_download_command,
    download_complete_marker,
    downloaded_shot_has_required_groups,
    downloaded_shot_is_complete,
)
from mast_bridge.workspace import discover_workspace


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download explicitly selected MAST Level 2 shots from STFC Echo."
    )
    parser.add_argument("--shot", action="append", required=True, help="Shot ID; repeat for multiple shots.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Output directory; default is workspace/data/raw/mast.",
    )
    parser.add_argument("--s5cmd", default=None, help="s5cmd executable; default searches PATH.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without downloading.")
    args = parser.parse_args(argv)

    layout = discover_workspace(SCRIPT_ROOT)
    data_dir = (args.data_dir or layout.data_root / "raw" / "mast").expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    executable = args.s5cmd or shutil.which("s5cmd")
    if executable is None and not args.dry_run:
        parser.error("s5cmd was not found on PATH; install it before downloading")
    commands = build_download_command(args.shot, data_dir, executable or "s5cmd")
    for shot, command in zip(args.shot, commands, strict=True):
        shot_path = data_dir / f"{shot}.zarr"
        print(" ".join(command))
        if args.dry_run:
            continue
        if downloaded_shot_is_complete(shot_path):
            print(f"Skipping complete shot: {shot_path}")
            continue
        subprocess.run(command, check=True)
        if not downloaded_shot_has_required_groups(shot_path):
            raise RuntimeError(
                "s5cmd exited successfully but the downloaded shot is missing "
                f"required Zarr groups: {shot_path}"
            )
        download_complete_marker(shot_path).write_text("complete\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
