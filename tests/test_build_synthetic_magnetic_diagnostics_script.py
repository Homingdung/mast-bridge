import importlib.util
import unittest
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "build_synthetic_magnetic_diagnostics.py"
)
SPEC = importlib.util.spec_from_file_location(
    "build_synthetic_magnetic_diagnostics", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FakeTokamak:
    def __init__(self):
        self.n_coils = 3
        self.coil_names = ["P2U", "P2L", "PASSIVE_MID1"]
        self.currents = {}

    def set_all_coil_currents(self, values):
        self.currents = {
            name: float(value)
            for name, value in zip(self.coil_names, np.asarray(values, dtype=float))
        }

    def set_coil_current(self, name, value):
        self.currents[name] = float(value)


class BuildSyntheticMagneticDiagnosticsTests(unittest.TestCase):
    def test_reconstruct_plasma_psi_preserves_saved_total_psi(self):
        saved_total = np.asarray([[3.0, 4.0], [5.0, 6.0]])
        coil_psi = np.asarray([[1.0, 1.5], [2.0, 2.5]])

        plasma_psi = MODULE.reconstruct_plasma_psi(saved_total, coil_psi)

        np.testing.assert_allclose(plasma_psi, [[2.0, 2.5], [3.0, 3.5]])
        np.testing.assert_allclose(plasma_psi + coil_psi, saved_total)

    def test_apply_saved_currents_maps_source_channels_to_machine_coils(self):
        tokamak = FakeTokamak()
        active_payload = {
            "P2U": {"upper": {"source_channel": "P2U FEED"}},
            "P2L": {"source_channel": "P2L FEED"},
        }
        passive_payload = [
            {"name": "PASSIVE_MID1", "source_current_channel": "MID1"},
            {"name": "NOT_IN_MACHINE", "source_current_channel": "MID2"},
        ]
        current_metadata = {
            "active": {"P2U FEED": 120.0, "P2L FEED": -80.0},
            "passive": {"MID1": 4.5, "MID2": 9.0},
        }

        MODULE.apply_saved_currents(
            tokamak,
            active_payload=active_payload,
            passive_payload=passive_payload,
            current_metadata=current_metadata,
        )

        self.assertEqual(
            tokamak.currents,
            {"P2U": 120.0, "P2L": -80.0, "PASSIVE_MID1": 4.5},
        )

    def test_apply_saved_currents_rejects_missing_passive_channel(self):
        tokamak = FakeTokamak()

        with self.assertRaisesRegex(KeyError, "MID1"):
            MODULE.apply_saved_currents(
                tokamak,
                active_payload={
                    "P2U": {"source_channel": "P2U FEED"},
                    "P2L": {"source_channel": "P2L FEED"},
                },
                passive_payload=[
                    {
                        "name": "PASSIVE_MID1",
                        "source_current_channel": "MID1",
                    }
                ],
                current_metadata={
                    "active": {"P2U FEED": 1.0, "P2L FEED": 2.0},
                    "passive": {},
                },
            )

    def test_parser_defaults_to_diagnostics_npz_and_resume_mode(self):
        args = MODULE.build_parser().parse_args(
            ["--accepted-manifest", "accepted.jsonl"]
        )

        self.assertEqual(args.output_name, "diagnostics.npz")
        self.assertFalse(args.overwrite)
        self.assertIsNone(args.limit)


if __name__ == "__main__":
    unittest.main()
