import unittest

import numpy as np

from mast_bridge.simulation.magnetic_diagnostics import (
    LEVEL2_FLUX_LOOP_SCALE,
    compare_named_signals,
    correct_mast_level2_flux_loop_positions,
    correct_mast_level2_pickup_orientations,
    current_constraint_comparison,
    observed_plasma_current,
    observed_flux_loop_signals,
    observed_pickup_signals,
    summarize_comparisons,
)


class MagneticDiagnosticsTests(unittest.TestCase):
    def test_observed_flux_loop_signals_interpolate_each_channel(self):
        magnetics = {
            "time": np.array([0.0, 0.5, 1.0]),
            "flux_loop_channel": np.array(["FL1", "FL2"]),
            "flux_loop_flux": np.array([[0.0, 2.0, 4.0], [10.0, 14.0, 18.0]]),
        }

        signals = observed_flux_loop_signals(magnetics, 0.25)

        self.assertEqual(signals.names, ["FL1", "FL2"])
        np.testing.assert_allclose(signals.values, [1.0, 12.0])

    def test_observed_pickup_signals_concatenate_supported_probe_families(self):
        magnetics = {
            "time": np.array([0.0, 1.0]),
            "b_field_pol_probe_ccbv_channel": np.array(["CCBV01"]),
            "b_field_pol_probe_ccbv_field": np.array([[1.0, 3.0]]),
            "b_field_pol_probe_obr_channel": np.array(["OBR01", "OBR02"]),
            "b_field_pol_probe_obr_field": np.array([[10.0, 14.0], [20.0, 26.0]]),
        }

        signals = observed_pickup_signals(magnetics, 0.5)

        self.assertEqual(signals.names, ["CCBV01", "OBR01", "OBR02"])
        self.assertEqual(signals.families, ["CCBV", "OBR", "OBR"])
        np.testing.assert_allclose(signals.values, [2.0, 12.0, 23.0])

    def test_observed_plasma_current_interpolates_magnetics_ip(self):
        magnetics = {
            "time": np.array([0.0, 0.5, 1.0]),
            "ip": np.array([100.0, 200.0, 400.0]),
        }

        self.assertEqual(observed_plasma_current(magnetics, 0.25), 150.0)

    def test_current_constraint_comparison_reports_global_current_error(self):
        comparison = current_constraint_comparison(
            model_ip=175.0,
            magnetics={"time": np.array([0.0, 1.0]), "ip": np.array([100.0, 200.0])},
            target_time=0.5,
        )

        self.assertEqual(comparison["diagnostic_type"], "plasma_current")
        self.assertEqual(comparison["channel"], "ip")
        self.assertEqual(comparison["model"], 175.0)
        self.assertEqual(comparison["observed"], 150.0)
        self.assertEqual(comparison["error"], 25.0)
        self.assertAlmostEqual(comparison["relative_error"], 25.0 / 150.0)

    def test_compare_named_signals_inner_joins_and_computes_errors(self):
        comparison = compare_named_signals(
            diagnostic_type="flux_loop",
            model_names=["FL2", "FL1", "MODEL_ONLY"],
            model_values=np.array([11.0, 1.5, 99.0]),
            observed_names=["FL1", "FL2", "OBS_ONLY"],
            observed_values=np.array([1.0, 10.0, -5.0]),
        )

        self.assertEqual(comparison["channel"].tolist(), ["FL2", "FL1"])
        self.assertEqual(comparison["diagnostic_type"].tolist(), ["flux_loop", "flux_loop"])
        np.testing.assert_allclose(comparison["model"], [11.0, 1.5])
        np.testing.assert_allclose(comparison["observed"], [10.0, 1.0])
        np.testing.assert_allclose(comparison["error"], [1.0, 0.5])
        np.testing.assert_allclose(comparison["abs_error"], [1.0, 0.5])

    def test_level2_flux_loop_scale_converts_psi_to_webers(self):
        self.assertAlmostEqual(LEVEL2_FLUX_LOOP_SCALE, 2.0 * np.pi)

    def test_summarize_comparisons_reports_counts_and_error_metrics_by_type(self):
        rows = np.array(
            [
                ("flux_loop", "FL1", 2.0, 1.0, 1.0, 1.0),
                ("flux_loop", "FL2", 4.0, 2.0, 2.0, 2.0),
                ("pickup", "P1", 1.0, 2.0, -1.0, 1.0),
            ],
            dtype=[
                ("diagnostic_type", "U16"),
                ("channel", "U16"),
                ("model", "f8"),
                ("observed", "f8"),
                ("error", "f8"),
                ("abs_error", "f8"),
            ],
        )

        summary = summarize_comparisons(rows)

        self.assertEqual(summary["total"]["count"], 3)
        self.assertAlmostEqual(summary["total"]["mean_abs_error"], 4.0 / 3.0)
        self.assertAlmostEqual(summary["total"]["rmse"], np.sqrt(2.0))
        self.assertEqual(summary["by_type"]["flux_loop"]["count"], 2)
        self.assertAlmostEqual(summary["by_type"]["flux_loop"]["mean_error"], 1.5)
        self.assertEqual(summary["by_type"]["pickup"]["count"], 1)

    def test_correct_mast_level2_pickup_orientations_updates_outer_probe_signs(self):
        payload = {
            "pickups": [
                {"family": "CCBV", "orientation_vector": np.array([0.0, 0.0, 1.0])},
                {"family": "OBR", "orientation_vector": np.array([-1.0, 0.0, 0.0])},
                {"family": "OBV", "orientation_vector": np.array([0.0, 0.0, -1.0])},
            ]
        }

        correct_mast_level2_pickup_orientations(payload)

        np.testing.assert_allclose(payload["pickups"][0]["orientation_vector"], [0.0, 0.0, 1.0])
        np.testing.assert_allclose(payload["pickups"][1]["orientation_vector"], [1.0, 0.0, 0.0])
        np.testing.assert_allclose(payload["pickups"][2]["orientation_vector"], [0.0, 0.0, 1.0])

    def test_correct_mast_level2_flux_loop_positions_maps_channels_to_geometry(self):
        payload = {
            "flux_loops": [
                {"name": "CC03", "geometry_name": "FL_P2U_1", "position": np.array([0.7, 0.8])},
                {"name": "P3U/1", "geometry_name": "FL_P2L_3", "position": np.array([0.9, -0.8])},
            ]
        }
        magnetics = {
            "flux_loop_geometry_channel": np.array(["FL_P2U_1", "FL_CC03", "FL_P3U_1"]),
            "flux_loop_r": np.array([0.7, 0.18, 1.16]),
            "flux_loop_z": np.array([0.8, 0.62, 1.08]),
        }

        correct_mast_level2_flux_loop_positions(payload, magnetics)

        self.assertEqual(payload["flux_loops"][0]["geometry_name"], "FL_CC03")
        np.testing.assert_allclose(payload["flux_loops"][0]["position"], [0.18, 0.62])
        self.assertEqual(payload["flux_loops"][1]["geometry_name"], "FL_P3U_1")
        np.testing.assert_allclose(payload["flux_loops"][1]["position"], [1.16, 1.08])


if __name__ == "__main__":
    unittest.main()
