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

    def test_plot_path_uses_equilibrium_png(self):
        self.assertEqual(
            MODULE.plot_path(Path("results")),
            Path("results") / "equilibrium.png",
        )


if __name__ == "__main__":
    unittest.main()
