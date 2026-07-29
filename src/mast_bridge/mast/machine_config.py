from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REQUIRED_MACHINE_FILES = {
    "active_coils": "MAST_active_coils.pickle",
    "limiter": "MAST_limiter.pickle",
    "magnetic_probes": "MAST_magentic_probes.pickle",
    "passive_coils": "MAST_passive_coils.pickle",
    "wall": "MAST_wall.pickle",
}

LEGACY_MACHINE_FILES = {
    "passive_coils": ("MAST_passive_coilds.pickle",),
}


class MachineConfigurationError(FileNotFoundError):
    """Raised when a complete machine description cannot be assembled."""


@dataclass(frozen=True)
class MachineGeometry:
    """Paths to the machine geometry required by plotting and FreeGSNKE."""

    files: dict[str, Path]

    @classmethod
    def load(cls, directory: str | Path) -> "MachineGeometry":
        root = Path(directory).expanduser().resolve()
        files: dict[str, Path] = {}
        for key, filename in REQUIRED_MACHINE_FILES.items():
            path = root / filename
            if not path.is_file():
                for legacy_filename in LEGACY_MACHINE_FILES.get(key, ()):
                    legacy_path = root / legacy_filename
                    if legacy_path.is_file():
                        path = legacy_path
                        break
            files[key] = path
        missing = [path.name for path in files.values() if not path.is_file()]
        if missing:
            raise MachineConfigurationError(
                f"Missing machine configuration files in {root}: "
                + ", ".join(missing)
            )
        return cls(files=files)

    def load_pickles(self) -> dict[str, Any]:
        """Load trusted local pickle payloads for plotting or FreeGSNKE."""
        payloads: dict[str, Any] = {}
        for key, path in self.files.items():
            with path.open("rb") as handle:
                payloads[key] = pickle.load(handle)
        return payloads

    def to_dict(self) -> dict[str, str]:
        return {key: str(path) for key, path in self.files.items()}
