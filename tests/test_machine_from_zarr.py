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
        magnetics.create_array("flux_loop_channel", data=np.array(["CC03", "P3U/1"], dtype="U5"))
        magnetics.create_array(
            "flux_loop_geometry_channel",
            data=np.array(["FL_P2U_1", "FL_CC03", "FL_P3U_1"], dtype="U8"),
        )
        magnetics.create_array("flux_loop_r", data=np.array([0.7, 0.18, 1.16]))
        magnetics.create_array("flux_loop_z", data=np.array([0.8, 0.62, 1.08]))
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
        self.assertEqual(payloads["magnetic_probes"]["flux_loops"][0]["name"], "CC03")
        self.assertEqual(
            payloads["magnetic_probes"]["flux_loops"][0]["measurement_status"],
            "measured",
        )
        self.assertEqual(
            payloads["magnetic_probes"]["flux_loops"][0]["source_signal_channel"],
            "CC03",
        )
        np.testing.assert_allclose(
            payloads["magnetic_probes"]["flux_loops"][0]["position"], [0.18, 0.62]
        )
        self.assertEqual(payloads["magnetic_probes"]["flux_loops"][1]["name"], "P3U/1")
        np.testing.assert_allclose(
            payloads["magnetic_probes"]["flux_loops"][1]["position"], [1.16, 1.08]
        )
        self.assertEqual(len(payloads["magnetic_probes"]["flux_loops"]), 3)
        virtual = payloads["magnetic_probes"]["flux_loops"][2]
        self.assertEqual(virtual["name"], "VIRTUAL::FL_P2U_1")
        self.assertEqual(virtual["geometry_name"], "FL_P2U_1")
        self.assertEqual(virtual["measurement_status"], "virtual")
        self.assertIsNone(virtual["source_signal_channel"])
        np.testing.assert_allclose(virtual["position"], [0.7, 0.8])
        self.assertEqual(payloads["magnetic_probes"]["pickups"][0]["family"], "CCBV")
        pickups = payloads["magnetic_probes"]["pickups"]
        obr = next(item for item in pickups if item["family"] == "OBR")
        obv = next(item for item in pickups if item["family"] == "OBV")
        np.testing.assert_allclose(obr["orientation_vector"], [1.0, 0.0, 0.0])
        np.testing.assert_allclose(obv["orientation_vector"], [0.0, 0.0, 1.0])

    def test_normalizes_passive_widths_for_freegsnke(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            shot = self._fixture(Path(temp_dir))
            passive = zarr.open_group(str(shot), mode="a")["pf_passive"]
            passive["ring_width"][:] = np.array([-0.06, 0.06])

            payloads = build_machine_payloads(shot)

        self.assertEqual([item["dR"] for item in payloads["passive_coils"]], [0.06, 0.06])

    def test_maps_scalar_and_many_to_one_passive_currents_explicitly(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            shot = self._fixture(Path(temp_dir))
            passive = zarr.open_group(str(shot), mode="a")["pf_passive"]
            passive.create_array("botcol_r", data=np.array([0.23]))
            passive.create_array("botcol_z", data=np.array([-2.02]))
            passive.create_array("botcol_width", data=np.array([0.05]))
            passive.create_array("botcol_height", data=np.array([0.30]))
            passive.create_array("botcol_geometry_channel", data=np.array(["botcol"], dtype="U6"))
            passive.create_array("botcol_current_channel", data=np.array(["BOTCOL1", "BOTCOL2"], dtype="U7"))
            passive.create_array("botcol_current", data=np.zeros((2, 3)))
            passive.create_array("endcrown_l_r", data=np.array([0.16]))
            passive.create_array("endcrown_l_z", data=np.array([-2.46]))
            passive.create_array("endcrown_l_width", data=np.array([0.23]))
            passive.create_array("endcrown_l_height", data=np.array([0.08]))
            passive.create_array(
                "endcrown_l_geometry_channel", data=np.array(["endcrown_l"], dtype="U10")
            )
            passive.create_array("endcrown_l_current", data=np.zeros(3))

            items = build_machine_payloads(shot)["passive_coils"]

        botcol = next(item for item in items if item["element"] == "botcol")
        self.assertEqual(botcol["source_current_channels"], ["BOTCOL1", "BOTCOL2"])
        self.assertEqual(botcol["source_current_channel"], "botcol__sum")
        self.assertEqual(botcol["source_current_reduction"], "sum")
        endcrown = next(item for item in items if item["element"] == "endcrown_l")
        self.assertEqual(endcrown["source_current_channels"], ["endcrown_l"])
        self.assertEqual(endcrown["source_current_reduction"], "identity")

    def test_rejects_unverified_many_to_one_passive_current_mapping(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            shot = self._fixture(Path(temp_dir))
            passive = zarr.open_group(str(shot), mode="a")["pf_passive"]
            passive.create_array("mystery_r", data=np.array([0.23]))
            passive.create_array("mystery_z", data=np.array([-2.02]))
            passive.create_array("mystery_width", data=np.array([0.05]))
            passive.create_array("mystery_height", data=np.array([0.30]))
            passive.create_array(
                "mystery_geometry_channel", data=np.array(["mystery"], dtype="U7")
            )
            passive.create_array(
                "mystery_current_channel",
                data=np.array(["MYSTERY1", "MYSTERY2"], dtype="U8"),
            )
            passive.create_array("mystery_current", data=np.zeros((2, 3)))

            with self.assertRaisesRegex(
                ValueError, "mystery has 1 geometry elements but 2 current channels"
            ):
                build_machine_payloads(shot)

    def test_writes_expected_filenames_and_refuses_existing_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = write_machine_pickles(self._fixture(root), root / "machine")
            self.assertEqual(set(p.name for p in paths.values()), set(REQUIRED_OUTPUTS.values()))
            self.assertIn("MAST_passive_coils.pickle", {p.name for p in paths.values()})
            self.assertNotIn("MAST_passive_coilds.pickle", {p.name for p in paths.values()})
            with self.assertRaises(FileExistsError):
                write_machine_pickles(root / "11766.zarr", root / "machine")
            with paths["active_coils"].open("rb") as handle:
                self.assertIn("Solenoid", pickle.load(handle))


if __name__ == "__main__":
    unittest.main()
