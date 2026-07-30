import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "compare_freegsnke_magnetic_diagnostics.py"
SPEC = importlib.util.spec_from_file_location(
    "compare_freegsnke_magnetic_diagnostics", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class CompareFreegsnkeMagneticDiagnosticsScriptTests(unittest.TestCase):
    def test_default_output_dir_is_shot_and_time_specific(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = MODULE.default_output_dir(Path(temp_dir), "11771", 0.18)

        self.assertEqual(
            output,
            Path(temp_dir)
            / "data"
            / "processed"
            / "diagnostic_comparisons"
            / "11771_t0.18",
        )

    def test_default_artifact_dir_lives_under_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = MODULE.default_artifact_dir(Path(temp_dir), "11771", 0.18)

        self.assertEqual(
            output,
            Path(temp_dir)
            / "artifacts"
            / "freegsnke_magnetic_diagnostics"
            / "11771_t0.18",
        )

    def test_parser_defaults_to_artifact_plot_directory(self):
        args = MODULE.build_parser().parse_args(["--shot", "11771", "--time", "0.18"])

        self.assertIsNone(args.artifact_dir)
        self.assertEqual(args.nx, 65)
        self.assertEqual(args.ny, 65)
        self.assertAlmostEqual(args.flux_loop_scale, 2.0 * np.pi)


if __name__ == "__main__":
    unittest.main()
