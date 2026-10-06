"""Resolve portable manifest paths independently of the working directory."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LEGACY_WORKSPACE_PREFIXES = (
    "/inspire/qb-ilm/project/ai-for-fusion/public/fusion-workspace/",
    "/inspire/qb-ilm/project/ai-for-fusion/public/collab_package/",
    "/root/autodl-tmp/fusion-workspace/collab_package/extracted/collab_package/",
)
MANIFEST_PATH_FIELDS = (
    "data_path", "fit_path", "label_path", "equilibrium_path",
    "diagnostics_path", "machine_config_path",
)


def workspace_root() -> Path:
    """Use MAST_WORKSPACE_ROOT, or the parent of the repository by default."""
    return Path(os.environ.get("MAST_WORKSPACE_ROOT", REPOSITORY_ROOT.parent)).expanduser().resolve()


def portable_path(value: str) -> str:
    """Remove historical machine prefixes while preserving the path suffix."""
    for prefix in LEGACY_WORKSPACE_PREFIXES:
        if value.startswith(prefix):
            return value[len(prefix):]
    return value


def resolve_path(value: str | Path, root: str | Path | None = None) -> Path:
    """Resolve archived workspace paths; explicit user absolute paths are kept."""
    path = Path(os.path.expandvars(portable_path(str(value)))).expanduser()
    if path.is_absolute():
        return path.resolve()
    if path.parts and path.parts[0] in {"paper_artifacts", "configs", "scripts", "src"}:
        return (REPOSITORY_ROOT / path).resolve()
    base = Path(root).expanduser().resolve() if root is not None else workspace_root()
    if path.parts and path.parts[0] == "data" and os.environ.get("MAST_DATA_ROOT"):
        return (Path(os.environ["MAST_DATA_ROOT"]).expanduser() / Path(*path.parts[1:])).resolve()
    return (base / path).resolve()


def resolve_manifest_row(row: dict[str, Any], root: str | Path | None = None) -> dict[str, Any]:
    """Resolve path fields without changing sample IDs, labels, or split metadata."""
    result = dict(row)
    for field in MANIFEST_PATH_FIELDS:
        if result.get(field):
            result[field] = str(resolve_path(result[field], root))
    return result
