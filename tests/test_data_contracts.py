import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from mast_bridge.data.schema import ShotRecord
from mast_bridge.mast.machine_config import (
    REQUIRED_MACHINE_FILES,
    MachineConfigurationError,
    MachineGeometry,
)


class MachineGeometryTests(unittest.TestCase):
    def _machine_dir(self, root: Path) -> Path:
        directory = root / "machine"
        directory.mkdir()
        for filename in REQUIRED_MACHINE_FILES.values():
            (directory / filename).write_bytes(b"fixture")
        return directory

    def test_loads_all_required_machine_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            geometry = MachineGeometry.load(self._machine_dir(Path(temp_dir)))

        self.assertEqual(set(geometry.files), set(REQUIRED_MACHINE_FILES))
        self.assertTrue(all(path.name for path in geometry.files.values()))

    def test_reports_all_missing_machine_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(MachineConfigurationError) as context:
                MachineGeometry.load(Path(temp_dir))

        message = str(context.exception)
        for filename in REQUIRED_MACHINE_FILES.values():
            self.assertIn(filename, message)


class ShotRecordTests(unittest.TestCase):
    def test_metadata_is_json_serializable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            machine_dir = root / "machine"
            machine_dir.mkdir()
            for filename in REQUIRED_MACHINE_FILES.values():
                (machine_dir / filename).write_bytes(b"fixture")
            record = ShotRecord(
                shot_id="11766",
                zarr_path=root / "11766.zarr",
                signals={"magnetics": {"name": "fixture"}},
                equilibrium={"source": "EFIT"},
                machine=MachineGeometry.load(machine_dir),
                metadata={"source": "real", "shot_id": "11766"},
            )

        json.dumps(record.to_dict())
        self.assertEqual(record.to_dict()["shot_id"], "11766")
        self.assertEqual(record.to_dict()["source"], "real")
