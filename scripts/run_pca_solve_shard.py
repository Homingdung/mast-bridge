#!/usr/bin/env python3
"""PCA 0.1σ 全量求解分片 worker。

每个 shard 处理 manifest 的一行区间，按炮分组炮内时间序 warm start，
逐片调用 run_freegsnke_forward.py（绝对参数注入）。断点续跑（已有
metadata.json 的跳过）。进度写入 jsonl 日志。
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = SCRIPT_ROOT.parent
FW = SCRIPT_ROOT / "scripts" / "run_freegsnke_forward.py"
FIT = WORKSPACE_ROOT / "data" / "processed" / "real" / "lao_parameter_ensemble" / "all_zarr_lao_parameter_fits.npz"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()

    rows = list(csv.DictReader(args.csv.open()))[args.start : args.end]
    print(f"[shard] rows {args.start}:{args.end} = {len(rows)}", flush=True)

    # 按炮分组、炮内时间序
    by_shot: dict[str, list[dict]] = {}
    for r in rows:
        by_shot.setdefault(r["shot"], []).append(r)
    for s in by_shot:
        by_shot[s].sort(key=lambda r: float(r["target_time"]))

    out_root = args.out_root.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    done = 0
    for shot, shot_rows in by_shot.items():
        prev: Path | None = None
        for row in shot_rows:
            tt = float(row["target_time"])
            sample = f"{shot}_t{tt:g}_v000"
            out_dir = out_root / sample
            if (out_dir / "metadata.json").is_file():
                prev = out_dir / "equilibrium.npz"
                done += 1
                continue
            coil = json.loads(row["coil_abs"])
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "coil_abs.json").write_text(json.dumps(coil), encoding="utf-8")
            cmd = [
                str(args.python), str(FW),
                "--shot", shot, "--time", f"{tt:.17g}",
                "--data-dir", str(WORKSPACE_ROOT / "data/raw/mast"),
                "--machine-dir", str(WORKSPACE_ROOT / f"data/raw/mast/machine/{shot}"),
                "--fit-path", str(FIT),
                "--output-dir", str(out_dir),
                "--nx", "65", "--ny", "65",
                "--tolerance", "1e-8", "--max-iterations", "500",
                f"--alpha-abs={row['alpha_abs']}",
                f"--beta-abs={row['beta_abs']}",
                "--coil-currents-json", str(out_dir / "coil_abs.json"),
            ]
            if prev is not None:
                cmd += ["--warm-start", str(prev)]
            t0 = time.time()
            try:
                rc = subprocess.run(cmd, timeout=args.timeout, check=False).returncode
            except subprocess.TimeoutExpired:
                rc = 124
            dt = time.time() - t0
            entry = {
                "shot": shot, "time": tt, "sample": sample,
                "mode": "warm" if prev is not None else "cold",
                "rc": rc, "elapsed_s": round(dt, 2),
            }
            with args.log.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
            if rc == 0 and (out_dir / "equilibrium.npz").is_file():
                prev = out_dir / "equilibrium.npz"
                done += 1
            else:
                prev = None
            if done % 20 == 0:
                print(f"[shard] {done} done (rc=0) {shot}", flush=True)
    print(f"[shard] FINISHED {args.start}:{args.end} done={done}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
