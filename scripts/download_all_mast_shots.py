#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_ROOT / "src"))

from mast_bridge.mast.downloader import (
    MAST_LEVEL2_ENDPOINT,
    REQUIRED_SHOT_GROUPS,
    build_download_command,
    download_complete_marker,
)
from mast_bridge.workspace import discover_workspace


def parse_shots_from_s5cmd_ls(s5cmd: str, endpoint: str) -> list[str]:
    command = [
        s5cmd,
        "--no-sign-request",
        "--endpoint-url",
        endpoint,
        "ls",
        "s3://mast/level2/shots/",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    shots = []
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0].upper() == "DIR" and parts[1].endswith(".zarr/"):
            shots.append(parts[1][: -len(".zarr/")])
    return sorted(shots, key=int)


def load_shot_list(args: argparse.Namespace) -> list[str]:
    if args.shot_list:
        return [
            line.strip()
            for line in Path(args.shot_list).read_text().splitlines()
            if line.strip()
        ]
    return parse_shots_from_s5cmd_ls(args.s5cmd or "s5cmd", args.endpoint_url)


def build_shard_files(shots: list[str], data_dir: Path, shards: int) -> list[Path]:
    shard_dir = data_dir / ".download_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    size = max(1, (len(shots) + shards - 1) // shards)
    for i in range(0, len(shots), size):
        chunk = shots[i : i + size]
        path = shard_dir / f"shard_{len(paths):04d}.txt"
        lines = [
            " ".join(
                [
                    "sync",
                    f"s3://mast/level2/shots/{shot}.zarr/**",
                    str(data_dir / f"{shot}.zarr") + "/",
                ]
            )
            for shot in chunk
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        paths.append(path)
    return paths


def run_shard(
    shard_file: Path,
    args: argparse.Namespace,
    attempt: int,
) -> tuple[Path, bool]:
    command = [
        args.s5cmd or "s5cmd",
        "--no-sign-request",
        "--endpoint-url",
        args.endpoint_url,
        "--numworkers",
        str(args.numworkers),
        "run",
        str(shard_file),
    ]
    print(f"[shard-start] {shard_file.name} attempt={attempt + 1}", flush=True)
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(
            f"[shard-fail] {shard_file.name} attempt={attempt + 1} rc={result.returncode}",
            flush=True,
        )
        for line in result.stderr.splitlines()[-20:]:
            print(f"  {line}", flush=True)
        return shard_file, False
    print(f"[shard-done] {shard_file.name} attempt={attempt + 1}", flush=True)
    return shard_file, True


def verify_shots(shots: list[str], data_dir: Path) -> tuple[list[str], list[str]]:
    complete = []
    incomplete = []
    for shot in shots:
        shot_path = data_dir / f"{shot}.zarr"
        if shot_path.is_dir() and all(
            (shot_path / group / "zarr.json").is_file() for group in REQUIRED_SHOT_GROUPS
        ):
            complete.append(shot)
        else:
            incomplete.append(shot)
    return complete, incomplete


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download every MAST Level 2 shot from STFC Echo with resumable parallel sync."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Output directory; default is workspace/data/raw/mast.",
    )
    parser.add_argument("--shot-list", type=Path, default=None, help="Optional shot list file.")
    parser.add_argument(
        "--endpoint-url",
        default=MAST_LEVEL2_ENDPOINT,
        help="S3 endpoint; default is STFC Echo.",
    )
    parser.add_argument("--s5cmd", default=None, help="s5cmd executable; default searches PATH.")
    parser.add_argument("--shards", type=int, default=8, help="Parallel s5cmd run processes.")
    parser.add_argument("--numworkers", type=int, default=128, help="s5cmd workers per process.")
    parser.add_argument("--max-retries", type=int, default=5, help="Retries per failed shard.")
    parser.add_argument("--dry-run", action="store_true", help="Print plans without downloading.")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing downloads.")
    args = parser.parse_args(argv)

    layout = discover_workspace(SCRIPT_ROOT)
    data_dir = (args.data_dir or layout.data_root / "raw" / "mast").expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    executable = args.s5cmd or shutil.which("s5cmd")
    if executable is None and not args.dry_run:
        parser.error("s5cmd was not found on PATH; install it before downloading")

    print(f"[shots] enumerating shots from endpoint", flush=True)
    all_shots = load_shot_list(args)
    print(f"[shots] total on remote: {len(all_shots)}", flush=True)

    complete, incomplete = verify_shots(all_shots, data_dir)
    pending = [s for s in all_shots if s not in set(complete)]
    print(f"[shots] complete locally: {len(complete)}; pending: {len(pending)}", flush=True)

    (data_dir / ".all_level2_shots.txt").write_text(
        "\n".join(all_shots) + "\n", encoding="utf-8"
    )

    if args.dry_run or args.verify_only:
        for shot in pending[:5]:
            print(" ".join(build_download_command([shot], data_dir, executable)[0]))
        if len(pending) > 5:
            print(f"  ... and {len(pending) - 5} more shots")
        print(f"[verify] complete={len(complete)} incomplete={len(incomplete)}", flush=True)
        return 0

    shard_files = build_shard_files(pending, data_dir, args.shards)
    print(f"[shards] {len(shard_files)} shards, {args.numworkers} workers each", flush=True)

    attempt = 0
    remaining = shard_files
    while remaining and attempt < args.max_retries:
        print(f"[round] attempt={attempt + 1} shards_left={len(remaining)}", flush=True)
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(remaining)) as pool:
            futures = [pool.submit(run_shard, f, args, attempt) for f in remaining]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
        remaining = [f for f, ok in results if not ok]
        attempt += 1
    if remaining:
        print(f"[abort] {len(remaining)} shards still failing after {attempt} attempts", flush=True)
        for f in remaining:
            print(f"  rerun: s5cmd --no-sign-request --endpoint-url {args.endpoint_url} --numworkers {args.numworkers} run {f}", flush=True)
        return 1

    complete, incomplete = verify_shots(all_shots, data_dir)
    for shot in complete:
        download_complete_marker(data_dir / f"{shot}.zarr").write_text(
            "complete\n", encoding="utf-8"
        )
    (data_dir / ".all_level2_complete.txt").write_text(
        "\n".join(complete) + "\n", encoding="utf-8"
    )
    (data_dir / ".all_level2_incomplete.txt").write_text(
        "\n".join(incomplete) + "\n", encoding="utf-8"
    )
    print(f"[verify] complete={len(complete)} incomplete={len(incomplete)}", flush=True)
    for shot in incomplete:
        print(f"  incomplete: {shot}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
