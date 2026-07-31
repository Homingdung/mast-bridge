import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from mast_bridge.mast.downloader import (
    REQUIRED_SHOT_GROUPS,
    build_download_command,
    download_complete_marker,
    downloaded_shot_is_complete,
)
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
            shot_ids=["11766", "11767"],
            data_dir=Path("data/raw/mast"),
            s5cmd="s5cmd",
        )

        self.assertEqual(len(commands), 2)
        self.assertEqual(
            commands[0],
            [
                "s5cmd",
                "--no-sign-request",
                "--endpoint-url",
                "https://s3.echo.stfc.ac.uk",
                "cp",
                "s3://mast/level2/shots/11766.zarr/**",
                str((Path("data/raw/mast") / "11766.zarr").resolve()),
            ],
        )
        self.assertTrue(commands[1][-1].endswith("data/raw/mast/11767.zarr"))

    def test_download_is_complete_only_when_required_zarr_groups_exist(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            missing = root / "missing.zarr"
            empty = root / "empty.zarr"
            partial = root / "partial.zarr"
            complete = root / "complete.zarr"
            empty.mkdir()
            (partial / "equilibrium").mkdir(parents=True)
            for group in REQUIRED_SHOT_GROUPS:
                (complete / group).mkdir(parents=True)
                (complete / group / "zarr.json").write_text("{}")

            self.assertFalse(downloaded_shot_is_complete(missing))
            self.assertFalse(downloaded_shot_is_complete(empty))
            self.assertFalse(downloaded_shot_is_complete(partial))
            self.assertFalse(downloaded_shot_is_complete(complete))
            download_complete_marker(complete).write_text("complete\n")
            self.assertTrue(downloaded_shot_is_complete(complete))
