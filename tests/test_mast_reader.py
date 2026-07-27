import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from mast_bridge.mast.downloader import build_download_command
from mast_bridge.mast.machine_config import REQUIRED_MACHINE_FILES, MachineConfigurationError
from mast_bridge.mast.reader import ShotReader


class ShotReaderTests(unittest.TestCase):
    def _shot_fixture(self, root: Path, shot_id: str = "11766") -> Path:
        shot_path = root / f"{shot_id}.zarr"
        shot_path.mkdir()
        for filename in REQUIRED_MACHINE_FILES.values():
            (shot_path / filename).write_bytes(b"fixture")
        return shot_path

    def test_reads_shot_and_machine_configuration(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            shot_path = self._shot_fixture(root)
            record = ShotReader(root).read("11766")

        self.assertEqual(record.shot_id, "11766")
        self.assertEqual(record.zarr_path, shot_path.resolve())
        self.assertEqual(set(record.machine.files), set(REQUIRED_MACHINE_FILES))

    def test_reports_missing_shot(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(FileNotFoundError):
                ShotReader(temp_dir).read("99999")

    def test_reports_incomplete_machine_configuration(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "11766.zarr").mkdir()
            with self.assertRaises(MachineConfigurationError):
                ShotReader(root).read("11766")

    def test_finds_shot_specific_machine_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "11766.zarr").mkdir()
            machine_dir = root / "machine" / "11766"
            machine_dir.mkdir(parents=True)
            for filename in REQUIRED_MACHINE_FILES.values():
                (machine_dir / filename).write_bytes(b"fixture")

            record = ShotReader(root).read("11766")

        self.assertEqual(record.machine.files["wall"].parent, machine_dir.resolve())

    def test_does_not_use_machine_configuration_from_another_shot(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "11766.zarr").mkdir()
            machine_dir = root / "machine" / "11767"
            machine_dir.mkdir(parents=True)
            for filename in REQUIRED_MACHINE_FILES.values():
                (machine_dir / filename).write_bytes(b"fixture")

            with self.assertRaises(MachineConfigurationError):
                ShotReader(root).read("11766")


class DownloadCommandTests(unittest.TestCase):
    def test_builds_one_command_per_selected_shot(self):
        commands = build_download_command(
            script_path=Path("external/LARGE_MODEL_FUSION-master/scripts/download/download_data_v2.py"),
            shot_ids=["11766", "11767"],
            data_dir=Path("data/raw/mast"),
            python="python3",
        )

        self.assertEqual(len(commands), 2)
        self.assertEqual(commands[0][-2:], ["--shot", "11766"])
        self.assertTrue(commands[1][3].endswith("data/raw/mast"))
