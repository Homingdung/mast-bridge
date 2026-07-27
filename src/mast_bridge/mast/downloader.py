from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence


def build_download_command(
    script_path: str | Path,
    shot_ids: Sequence[str | int],
    data_dir: str | Path,
    python: str | Path = sys.executable,
) -> list[list[str]]:
    """Build one external download command per explicitly selected shot."""
    script = Path(script_path).expanduser().resolve()
    output = Path(data_dir).expanduser().resolve()
    return [
        [
            str(python),
            str(script),
            "--data-dir",
            str(output),
            "--shot",
            str(shot_id),
        ]
        for shot_id in shot_ids
    ]
