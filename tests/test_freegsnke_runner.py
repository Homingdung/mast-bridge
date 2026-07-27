import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from mast_bridge.mast.machine_config import REQUIRED_MACHINE_FILES, MachineGeometry
from mast_bridge.simulation.freegsnke_runner import machine_build_kwargs


class FreeGSNKEAdapterTests(unittest.TestCase):
    def test_maps_machine_geometry_to_freegsnke_keyword_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            machine_dir = Path(temp_dir)
            for filename in REQUIRED_MACHINE_FILES.values():
                (machine_dir / filename).write_bytes(b"fixture")
            geometry = MachineGeometry.load(machine_dir)

        kwargs = machine_build_kwargs(geometry)

        self.assertEqual(kwargs["active_coils_path"].name, "MAST_active_coils.pickle")
        self.assertEqual(kwargs["magnetic_probe_path"].name, "MAST_magentic_probes.pickle")
        self.assertEqual(kwargs["passive_coils_path"].name, "MAST_passive_coilds.pickle")
