#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_ROOT / "src"))

from mast_bridge.mast.downloader import build_download_command
from mast_bridge.workspace import discover_workspace


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download explicitly selected MAST shots through LARGE_MODEL_FUSION."
    )
    parser.add_argument("--shot", action="append", required=True, help="Shot ID; repeat for multiple shots.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Output directory; default is workspace/data/raw/mast.",
    )
    parser.add_argument(
        "--external-root",
        type=Path,
        default=None,
        help="Workspace external directory or LARGE_MODEL_FUSION root.",
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable for the external script.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without downloading.")
    args = parser.parse_args(argv)

    layout = discover_workspace(SCRIPT_ROOT)
    external_root = args.external_root or layout.large_model_fusion_root
    if external_root.name != "LARGE_MODEL_FUSION" and external_root.name != "LARGE_MODEL_FUSION-master":
        external_root = external_root / "LARGE_MODEL_FUSION"
        if not external_root.exists():
            external_root = external_root.with_name("LARGE_MODEL_FUSION-master")

    script_path = external_root / "scripts" / "download" / "download_data_v2.py"
    if not script_path.is_file():
        parser.error(f"LARGE_MODEL_FUSION downloader not found: {script_path}")

    data_dir = args.data_dir or layout.data_root / "raw" / "mast"
    commands = build_download_command(script_path, args.shot, data_dir, args.python)
    for command in commands:
        print(" ".join(command))
        if not args.dry_run:
            subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
