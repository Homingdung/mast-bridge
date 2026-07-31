import tempfile
import unittest
from pathlib import Path

import numpy as np

from mast_bridge.simulation.synthetic_diagnostics import (
    load_synthetic_diagnostic_values,
    synthetic_diagnostics_rejection_reason,
    write_synthetic_diagnostics,
)


class SyntheticDiagnosticsTests(unittest.TestCase):
    def test_round_trip_returns_named_training_features(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "diagnostics.npz"
            write_synthetic_diagnostics(
                path,
                target_time=0.155,
                magnetics_ip=487_500.0,
                flux_loop_names=["CC03", "P3U/1"],
                flux_loop_values=[0.12, -0.08],
                pickup_names=["CCBV01", "OBR01", "OBV01"],
                pickup_families=["CCBV", "OBR", "OBV"],
                pickup_values=[0.01, -0.02, 0.03],
                active_coil_currents={"P2": 123.0, "SOL": -456.0},
                flux_loop_scale=2.0 * np.pi,
            )

            values = load_synthetic_diagnostic_values(path)
            rejection_reason = synthetic_diagnostics_rejection_reason(path)

        self.assertEqual(values["target_time"], 0.155)
        self.assertEqual(values["magnetics_ip"], 487_500.0)
        self.assertEqual(values["flux_loop_CC03"], 0.12)
        self.assertEqual(values["flux_loop_P3U/1"], -0.08)
        self.assertEqual(values["pickup_CCBV_CCBV01"], 0.01)
        self.assertEqual(values["pickup_OBR_OBR01"], -0.02)
        self.assertEqual(values["pickup_OBV_OBV01"], 0.03)
        self.assertEqual(values["coil_active_P2"], 123.0)
        self.assertEqual(values["coil_active_SOL"], -456.0)
        self.assertIsNone(rejection_reason)

    def test_rejects_nonfinite_diagnostic_values(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "diagnostics.npz"
            np.savez_compressed(
                path,
                schema_version=np.asarray(1, dtype=np.int16),
                target_time=np.asarray(0.155),
                magnetics_ip=np.asarray(487_500.0),
                flux_loop_names=np.asarray(["CC03"], dtype="U4"),
                flux_loop_values=np.asarray([np.nan]),
                pickup_names=np.asarray(["CCBV01"], dtype="U6"),
                pickup_families=np.asarray(["CCBV"], dtype="U4"),
                pickup_values=np.asarray([0.01]),
                active_coil_names=np.asarray(["SOL"], dtype="U3"),
                active_coil_values=np.asarray([-456.0]),
                flux_loop_scale=np.asarray(2.0 * np.pi),
            )

            reason = synthetic_diagnostics_rejection_reason(path)

        self.assertEqual(reason, "diagnostics_nonfinite")

    def test_rejects_mismatched_name_and_value_lengths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "diagnostics.npz"
            np.savez_compressed(
                path,
                schema_version=np.asarray(1, dtype=np.int16),
                target_time=np.asarray(0.155),
                magnetics_ip=np.asarray(487_500.0),
                flux_loop_names=np.asarray(["CC03", "P3U/1"], dtype="U5"),
                flux_loop_values=np.asarray([0.12]),
                pickup_names=np.asarray(["CCBV01"], dtype="U6"),
                pickup_families=np.asarray(["CCBV"], dtype="U4"),
                pickup_values=np.asarray([0.01]),
                active_coil_names=np.asarray(["SOL"], dtype="U3"),
                active_coil_values=np.asarray([-456.0]),
                flux_loop_scale=np.asarray(2.0 * np.pi),
            )

            reason = synthetic_diagnostics_rejection_reason(path)

        self.assertEqual(reason, "diagnostics_shape_mismatch")

    def test_rejects_duplicate_channel_names(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "diagnostics.npz"
            np.savez_compressed(
                path,
                schema_version=np.asarray(1, dtype=np.int16),
                target_time=np.asarray(0.155),
                magnetics_ip=np.asarray(487_500.0),
                flux_loop_names=np.asarray(["CC03", "CC03"], dtype="U4"),
                flux_loop_values=np.asarray([0.12, 0.13]),
                pickup_names=np.asarray(["CCBV01"], dtype="U6"),
                pickup_families=np.asarray(["CCBV"], dtype="U4"),
                pickup_values=np.asarray([0.01]),
                active_coil_names=np.asarray(["SOL"], dtype="U3"),
                active_coil_values=np.asarray([-456.0]),
                flux_loop_scale=np.asarray(2.0 * np.pi),
            )

            reason = synthetic_diagnostics_rejection_reason(path)

        self.assertEqual(reason, "diagnostics_duplicate_channels")

    def test_rejects_wrong_flux_loop_scale_and_target_time(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "diagnostics.npz"
            np.savez_compressed(
                path,
                schema_version=np.asarray(1, dtype=np.int16),
                target_time=np.asarray(0.155),
                magnetics_ip=np.asarray(487_500.0),
                flux_loop_names=np.asarray(["CC03"], dtype="U4"),
                flux_loop_values=np.asarray([0.12]),
                pickup_names=np.asarray(["OBR01"], dtype="U5"),
                pickup_families=np.asarray(["OBR"], dtype="U3"),
                pickup_values=np.asarray([0.01]),
                active_coil_names=np.asarray(["SOL"], dtype="U3"),
                active_coil_values=np.asarray([-456.0]),
                flux_loop_scale=np.asarray(1.0),
            )

            scale_reason = synthetic_diagnostics_rejection_reason(path)
            time_reason = synthetic_diagnostics_rejection_reason(
                path, expected_target_time=0.16
            )

        self.assertEqual(scale_reason, "diagnostics_flux_loop_scale_invalid")
        self.assertEqual(time_reason, "diagnostics_target_time_mismatch")

    def test_rejects_empty_or_unsupported_channel_schema(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "diagnostics.npz"
            np.savez_compressed(
                path,
                schema_version=np.asarray(1, dtype=np.int16),
                target_time=np.asarray(0.155),
                magnetics_ip=np.asarray(487_500.0),
                flux_loop_names=np.asarray([""], dtype="U1"),
                flux_loop_values=np.asarray([0.12]),
                pickup_names=np.asarray(["P1"], dtype="U2"),
                pickup_families=np.asarray(["OTHER"], dtype="U5"),
                pickup_values=np.asarray([0.01]),
                active_coil_names=np.asarray(["SOL"], dtype="U3"),
                active_coil_values=np.asarray([-456.0]),
                flux_loop_scale=np.asarray(2.0 * np.pi),
            )

            reason = synthetic_diagnostics_rejection_reason(path)

        self.assertEqual(reason, "diagnostics_channel_schema_invalid")

    def test_failed_overwrite_preserves_existing_valid_payload(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "diagnostics.npz"
            write_synthetic_diagnostics(
                path,
                target_time=0.155,
                magnetics_ip=487_500.0,
                flux_loop_names=["CC03"],
                flux_loop_values=[0.12],
                pickup_names=["OBR01"],
                pickup_families=["OBR"],
                pickup_values=[0.01],
                active_coil_currents={"SOL": -456.0},
                flux_loop_scale=2.0 * np.pi,
            )

            with self.assertRaises(ValueError):
                write_synthetic_diagnostics(
                    path,
                    target_time=0.155,
                    magnetics_ip=1.0,
                    flux_loop_names=["CC03", "P3U/1"],
                    flux_loop_values=[0.12],
                    pickup_names=["OBR01"],
                    pickup_families=["OBR"],
                    pickup_values=[0.01],
                    active_coil_currents={"SOL": -456.0},
                    flux_loop_scale=2.0 * np.pi,
                )
            values = load_synthetic_diagnostic_values(path)

        self.assertEqual(values["magnetics_ip"], 487_500.0)


if __name__ == "__main__":
    unittest.main()
