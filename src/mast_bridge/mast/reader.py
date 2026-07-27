from __future__ import annotations

from pathlib import Path
from typing import Any

from mast_bridge.data.schema import ShotRecord

from .machine_config import MachineGeometry, REQUIRED_MACHINE_FILES


class ShotReader:
    """Read one MAST shot and its machine geometry from the local workspace."""

    def __init__(self, data_dir: str | Path, machine_dir: str | Path | None = None):
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.machine_dir = (
            Path(machine_dir).expanduser().resolve() if machine_dir is not None else None
        )

    def read(self, shot_id: str | int) -> ShotRecord:
        shot_name = str(shot_id)
        zarr_path = (self.data_dir / f"{shot_name}.zarr").resolve()
        if not zarr_path.is_dir():
            raise FileNotFoundError(f"MAST shot Zarr not found: {zarr_path}")

        geometry = MachineGeometry.load(self._find_machine_dir(zarr_path))
        signals, equilibrium = self._read_zarr_groups(zarr_path)
        return ShotRecord(
            shot_id=shot_name,
            zarr_path=zarr_path,
            signals=signals,
            equilibrium=equilibrium,
            machine=geometry,
            metadata={"source": "real", "shot_id": shot_name},
        )

    def _find_machine_dir(self, zarr_path: Path) -> Path:
        if self.machine_dir is not None:
            return self.machine_dir

        candidates = [
            zarr_path,
            zarr_path / "machine",
            self.data_dir / "machine" / zarr_path.stem.removesuffix(".zarr"),
        ]
        for candidate in candidates:
            if self._contains_all_machine_files(candidate):
                return candidate

        for path in zarr_path.rglob("*"):
            if path.is_dir() and self._contains_all_machine_files(path):
                return path
        return zarr_path

    @staticmethod
    def _contains_all_machine_files(directory: Path) -> bool:
        return directory.is_dir() and all(
            (directory / filename).is_file()
            for filename in REQUIRED_MACHINE_FILES.values()
        )

    @staticmethod
    def _read_zarr_groups(zarr_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
        """Open top-level Zarr groups lazily when the optional dependency is present."""
        if not ((zarr_path / "zarr.json").exists() or (zarr_path / ".zgroup").exists()):
            return {}, {}
        try:
            import zarr
        except ImportError as exc:
            raise RuntimeError(
                "Reading MAST Zarr data requires the optional 'zarr' package."
            ) from exc

        root = zarr.open_group(str(zarr_path), mode="r")
        groups = {name: root[name] for name in root.group_keys()}
        equilibrium = {
            name: groups[name]
            for name in groups
            if "equilibrium" in name.lower() or "efit" in name.lower()
        }
        signals = {
            name: group
            for name, group in groups.items()
            if name not in equilibrium
        }
        return signals, equilibrium
