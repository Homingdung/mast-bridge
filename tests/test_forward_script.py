import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_freegsnke_forward.py"
SPEC = importlib.util.spec_from_file_location("run_freegsnke_forward", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ForwardScriptTests(unittest.TestCase):
    def test_select_fit_row_matches_shot_and_nearest_time(self):
        fit = {
            "shot": np.array(["11766", "11767", "11766"]),
            "time": np.array([0.10, 0.18, 0.21]),
        }

        index = MODULE.select_fit_row(fit, "11766", 0.20)

        self.assertEqual(index, 2)

    def test_default_output_dir_is_shot_and_time_specific(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = MODULE.default_output_dir(Path(temp_dir), "11766", 0.18)

        self.assertEqual(
            output,
            Path(temp_dir) / "data" / "processed" / "synthetic" / "11766_t0.18",
        )

    def test_default_fit_path_lives_under_data_processed_real(self):
        self.assertEqual(
            MODULE.DEFAULT_FIT_PATH,
            MODULE.WORKSPACE_ROOT
            / "data"
            / "processed"
            / "real"
            / "lao_parameter_ensemble"
            / "all_zarr_lao_parameter_fits.npz",
        )

    def test_plot_path_uses_equilibrium_png(self):
        self.assertEqual(
            MODULE.plot_path(Path("results")),
            Path("results") / "equilibrium.png",
        )

    def test_default_solver_tolerance_is_strict(self):
        args = MODULE.build_parser().parse_args(["--shot", "11766", "--time", "0.18"])

        self.assertEqual(args.tolerance, 1e-8)

    def test_parser_accepts_coil_current_scale(self):
        args = MODULE.build_parser().parse_args(
            ["--shot", "11766", "--time", "0.18", "--coil-current-scale", "1.02"]
        )

        self.assertEqual(args.coil_current_scale, 1.02)

    def test_lao85_perturbation_scales_and_offsets_parameters(self):
        alpha = [1.0, -2.0, 3.0]
        beta = [0.5, -0.25, 0.75]

        result = MODULE.apply_lao85_perturbation(
            Ip=100.0,
            fvac=2.0,
            alpha=alpha,
            beta=beta,
            ip_scale=1.1,
            fvac_scale=0.9,
            alpha_scale=1.2,
            beta_scale=0.8,
            alpha_offset=0.5,
            beta_offset=-0.1,
        )

        self.assertAlmostEqual(result["Ip"], 110.0)
        self.assertAlmostEqual(result["fvac"], 1.8)
        np.testing.assert_allclose(result["alpha"], [1.7, -1.9, 4.1])
        np.testing.assert_allclose(result["beta"], [0.3, -0.3, 0.5])

    def test_scale_current_dicts_applies_factor_to_active_and_passive_metadata(self):
        currents = {
            "active": {"P2": 10.0, "SOL": -2.0},
            "passive": {"MID1": 0.5},
        }

        scaled = MODULE.scale_current_dicts(currents, 1.5)

        self.assertEqual(scaled["active"], {"P2": 15.0, "SOL": -3.0})
        self.assertEqual(scaled["passive"], {"MID1": 0.75})

    def test_equilibrium_grid_bounds_use_real_mast_grid_arrays(self):
        equilibrium = {
            "major_radius": np.asarray([0.06, 0.54, 1.02, 1.50, 1.98]),
            "z": np.asarray([-1.92, -0.96, 0.0, 0.96, 1.92]),
        }

        bounds = MODULE.equilibrium_grid_bounds(equilibrium)

        self.assertEqual(
            bounds,
            {"Rmin": 0.06, "Rmax": 1.98, "Zmin": -1.92, "Zmax": 1.92},
        )

    def test_parse_forward_solver_success_diagnostics(self):
        text = (
            "Forward static solve SUCCESS. Tolerance 4.20e-04 "
            "(vs. requested 1.00e-03) reached in 17/100 iterations."
        )

        diagnostics = MODULE.parse_forward_solver_diagnostics(text, 1e-3, 100)

        self.assertEqual(diagnostics["solver_status"], "success")
        self.assertTrue(diagnostics["solver_converged"])
        self.assertAlmostEqual(diagnostics["solver_final_tolerance"], 4.20e-04)
        self.assertEqual(diagnostics["solver_iterations"], 17)
        self.assertEqual(diagnostics["solver_max_iterations"], 100)

    def test_parse_forward_solver_non_convergence_diagnostics(self):
        text = (
            "Forward static solve DID NOT CONVERGE. Tolerance 2.60e-01 "
            "(vs. requested 1.00e-03) reached in 100/100 iterations."
        )

        diagnostics = MODULE.parse_forward_solver_diagnostics(text, 1e-3, 100)

        self.assertEqual(diagnostics["solver_status"], "non_converged")
        self.assertFalse(diagnostics["solver_converged"])
        self.assertAlmostEqual(diagnostics["solver_final_tolerance"], 2.60e-01)
        self.assertEqual(diagnostics["solver_iterations"], 100)

    def test_parse_forward_solver_falls_back_to_unknown(self):
        diagnostics = MODULE.parse_forward_solver_diagnostics("", 1e-3, 100)

        self.assertEqual(diagnostics["solver_status"], "unknown")
        self.assertFalse(diagnostics["solver_converged"])
        self.assertIsNone(diagnostics["solver_final_tolerance"])
        self.assertEqual(diagnostics["solver_max_iterations"], 100)

    def test_equilibrium_topology_diagnostics_are_json_ready(self):
        class Profiles:
            flag_limiter = True

        class Equilibrium:
            xpt = np.array([[1.1, -0.5, 0.7], [1.2, 0.5, 0.8]])
            opt = np.array([[0.9, 0.0, 1.4]])
            psi_axis = np.float64(1.4)
            psi_bndry = np.float64(0.7)
            _profiles = Profiles()

        diagnostics = MODULE.equilibrium_topology_diagnostics(Equilibrium())

        self.assertEqual(diagnostics["xpt_count"], 2)
        self.assertEqual(diagnostics["opt_count"], 1)
        self.assertTrue(diagnostics["flag_limiter"])
        self.assertEqual(diagnostics["psi_axis"], 1.4)
        self.assertEqual(diagnostics["psi_bndry"], 0.7)
        self.assertEqual(diagnostics["primary_xpt_psi"], 0.7)


if __name__ == "__main__":
    unittest.main()
