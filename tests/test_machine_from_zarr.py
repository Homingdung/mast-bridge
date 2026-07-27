import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from mast_bridge.mast.machine_from_zarr import (
    ACTIVE_GROUPS,
    REQUIRED_OUTPUTS,
    build_machine_payloads,
    write_machine_pickles,
)


class MachineFromZarrTests(unittest.TestCase):
    def _fixture(self, root: Path) -> Path:
        shot = root / "11766.zarr"
        z = zarr.open_group(str(shot), mode="w")

        active = z.create_group("pf_active")
        active.create_array("current_channel", data=np.array([value[0] for value in ACTIVE_GROUPS.values()], dtype="U12"))
        for index, (_, geometry) in enumerate(ACTIVE_GROUPS.values()):
            values = [float(index + 1)]
            active.create_array(f"{geometry}_r", data=np.array(values))
            active.create_array(f"{geometry}_z", data=np.array(values))
            active.create_array(f"{geometry}_width", data=np.array([0.02]))
            active.create_array(f"{geometry}_height", data=np.array([0.03]))
        active["sol_r"][:] = np.array([1.0])
        active["sol_z"][:] = np.array([-1.0])
        active["sol_width"][:] = np.array([0.02])
        active["sol_height"][:] = np.array([0.03])
        active["p2_inner_lower_r"][:] = np.array([0.5])
        active["p2_inner_lower_z"][:] = np.array([-0.5])
        active["p2_inner_lower_width"][:] = np.array([0.04])
        active["p2_inner_lower_height"][:] = np.array([0.05])

        passive = z.create_group("pf_passive")
        passive.create_array("ring_r", data=np.array([1.2, 1.3]))
        passive.create_array("ring_z", data=np.array([0.1, 0.2]))
        passive.create_array("ring_width", data=np.array([0.06, 0.06]))
        passive.create_array("ring_height", data=np.array([0.07, 0.07]))
        passive.create_array("ring_shapeAngle1", data=np.array([0.0, 0.0]))
        passive.create_array("ring_shapeAngle2", data=np.array([10.0, 20.0]))
        passive.create_array("ring_geometry_channel", data=np.array(["r1", "r2"], dtype="U2"))

        wall = z.create_group("wall")
        wall.create_array("limiter_r", data=np.array([1.0, 1.1]))
        wall.create_array("limiter_z", data=np.array([-1.0, 1.0]))

        magnetics = z.create_group("magnetics")
        magnetics.create_array("flux_loop_channel", data=np.array(["FL1"], dtype="U3"))
        magnetics.create_array("flux_loop_geometry_channel", data=np.array(["geo1"], dtype="U4"))
        magnetics.create_array("flux_loop_r", data=np.array([0.7]))
        magnetics.create_array("flux_loop_z", data=np.array([0.8]))
        magnetics.create_array("b_field_pol_probe_ccbv_channel", data=np.array(["P1"], dtype="U2"))
        magnetics.create_array("b_field_pol_probe_ccbv_geometry_channel", data=np.array(["p1"], dtype="U2"))
        magnetics.create_array("b_field_pol_probe_ccbv_r", data=np.array([0.9]))
        magnetics.create_array("b_field_pol_probe_ccbv_phi_1", data=np.array([0.0]))
        magnetics.create_array("b_field_pol_probe_ccbv_phi", data=np.array([0.0]))
        magnetics.create_array("b_field_pol_probe_ccbv_z", data=np.array([0.1]))
        for prefix, family in (("obr", "O"), ("obv", "V")):
            magnetics.create_array(f"b_field_pol_probe_{prefix}_channel", data=np.array([f"{family}1"], dtype="U2"))
            magnetics.create_array(f"b_field_pol_probe_{prefix}_geometry_channel", data=np.array([f"{prefix}1"], dtype="U4"))
            magnetics.create_array(f"b_field_pol_probe_{prefix}_r", data=np.array([0.9]))
            magnetics.create_array(f"b_field_pol_probe_{prefix}_phi_1", data=np.array([0.0]))
            magnetics.create_array(f"b_field_pol_probe_{prefix}_z", data=np.array([0.1]))

        return shot

    def test_builds_five_freegsnke_payloads_from_zarr(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            payloads = build_machine_payloads(self._fixture(Path(temp_dir)))

        self.assertEqual(set(payloads), set(REQUIRED_OUTPUTS))
        self.assertEqual(payloads["active_coils"]["Solenoid"]["R"], [1.0])
        self.assertEqual(payloads["active_coils"]["P2IL"]["source_channel"], "P2IL FEED")
        self.assertEqual(len(payloads["passive_coils"]), 2)
        self.assertEqual(payloads["limiter"], [{"R": 1.0, "Z": -1.0}, {"R": 1.1, "Z": 1.0}])
        self.assertEqual(payloads["magnetic_probes"]["flux_loops"][0]["name"], "FL1")
        self.assertEqual(payloads["magnetic_probes"]["pickups"][0]["family"], "CCBV")

    def test_writes_expected_filenames_and_refuses_existing_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = write_machine_pickles(self._fixture(root), root / "machine")
            self.assertEqual(set(p.name for p in paths.values()), set(REQUIRED_OUTPUTS.values()))
            with self.assertRaises(FileExistsError):
                write_machine_pickles(root / "11766.zarr", root / "machine")
            with paths["active_coils"].open("rb") as handle:
                self.assertIn("Solenoid", pickle.load(handle))


if __name__ == "__main__":
    unittest.main()
