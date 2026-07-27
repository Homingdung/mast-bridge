from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mast_bridge.mast.machine_config import MachineGeometry


@dataclass
class ShotRecord:
    """Normalized container shared by real MAST and future synthetic shots."""

    shot_id: str
    zarr_path: Path
    signals: dict[str, Any]
    equilibrium: dict[str, Any]
    machine: MachineGeometry
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return metadata and lightweight references, not array payloads."""
        return {
            "shot_id": self.shot_id,
            "zarr_path": str(self.zarr_path),
            "machine_files": {
                key: str(path) for key, path in self.machine.files.items()
            },
            **self.metadata,
        }
