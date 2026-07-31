from __future__ import annotations

from pathlib import Path
from typing import Sequence


MAST_LEVEL2_ENDPOINT = "https://s3.echo.stfc.ac.uk"
REQUIRED_SHOT_GROUPS = (
    "equilibrium",
    "magnetics",
    "pf_active",
    "pf_passive",
    "wall",
)


def download_complete_marker(path: str | Path) -> Path:
    shot_path = Path(path).expanduser().resolve()
    return shot_path.parent / f".{shot_path.name}.mast-bridge-complete"


def downloaded_shot_has_required_groups(path: str | Path) -> bool:
    shot_path = Path(path).expanduser().resolve()
    return shot_path.is_dir() and all(
        (shot_path / group / "zarr.json").is_file() for group in REQUIRED_SHOT_GROUPS
    )


def build_download_command(
    shot_ids: Sequence[str | int],
    data_dir: str | Path,
    s5cmd: str | Path = "s5cmd",
) -> list[list[str]]:
    """Build one anonymous STFC Echo download command per selected MAST shot."""
    output = Path(data_dir).expanduser().resolve()
    return [
        [
            str(s5cmd),
            "--no-sign-request",
            "--endpoint-url",
            MAST_LEVEL2_ENDPOINT,
            "cp",
            f"s3://mast/level2/shots/{shot_id}.zarr/**",
            str(output / f"{shot_id}.zarr"),
        ]
        for shot_id in shot_ids
    ]


def downloaded_shot_is_complete(path: str | Path) -> bool:
    """Return true after a successful command left the required Zarr groups."""
    return (
        downloaded_shot_has_required_groups(path)
        and download_complete_marker(path).is_file()
    )
