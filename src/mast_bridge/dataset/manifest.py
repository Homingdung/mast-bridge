from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass
class ManifestEntry:
    """Portable metadata for one real or synthetic training sample."""

    sample_id: str
    source: str
    shot_id: str
    data_path: Path
    machine_config_path: Path | None = None
    equilibrium_path: Path | None = None
    label_path: Path | None = None
    parent_shot: str | None = None
    solver_status: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.source not in {"real", "synthetic"}:
            raise ValueError("source must be 'real' or 'synthetic'")

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "sample_id": self.sample_id,
            "source": self.source,
            "shot_id": self.shot_id,
            "data_path": str(self.data_path),
            "machine_config_path": _stringify_path(self.machine_config_path),
            "equilibrium_path": _stringify_path(self.equilibrium_path),
            "label_path": _stringify_path(self.label_path),
            "parent_shot": self.parent_shot,
            "solver_status": self.solver_status,
        }
        result.update(self.metadata)
        return result


def _stringify_path(path: Path | None) -> str | None:
    return str(path) if path is not None else None


def write_manifest(entries: Iterable[ManifestEntry], output_path: str | Path) -> None:
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry.to_dict(), sort_keys=True) + "\n")
