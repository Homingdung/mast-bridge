import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "build_lao85_variant_rows.py"
SPEC = importlib.util.spec_from_file_location("build_lao85_variant_rows", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class BuildLao85VariantRowsScriptTests(unittest.TestCase):
    def test_writes_uniform_random_variant_csv_from_lao_fit_npz(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fit_path = root / "fits.npz"
            output = root / "variants.csv"
            np.savez_compressed(
                fit_path,
                shot=np.asarray(["11771"]),
                time=np.asarray([0.16]),
                ip=np.asarray([1.0]),
                fvac=np.asarray([0.4]),
                freegsnke_alpha=np.zeros((1, 3)),
                freegsnke_beta=np.zeros((1, 3)),
            )

            exit_code = MODULE.main(
                [
                    "--fit-path",
                    str(fit_path),
                    "--variants-per-point",
                    "3",
                    "--seed",
                    "123",
                    "--output",
                    str(output),
                ]
            )

            with output.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["shot"], "11771")
        self.assertEqual(rows[0]["target_time"], "0.16")
        self.assertEqual(rows[0]["sampling_method"], "uniform_random")
        self.assertIn("alpha_offset", rows[0])

    def test_filters_variant_csv_by_time_window(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fit_path = root / "fits.npz"
            output = root / "variants.csv"
            np.savez_compressed(
                fit_path,
                shot=np.asarray(["11771", "11771"]),
                time=np.asarray([0.03, 0.16]),
                ip=np.asarray([1.0, 2.0]),
                fvac=np.asarray([0.4, 0.5]),
                freegsnke_alpha=np.zeros((2, 3)),
                freegsnke_beta=np.zeros((2, 3)),
            )

            exit_code = MODULE.main(
                [
                    "--fit-path",
                    str(fit_path),
                    "--variants-per-point",
                    "1",
                    "--seed",
                    "123",
                    "--min-time",
                    "0.1",
                    "--output",
                    str(output),
                ]
            )

            with output.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["target_time"], "0.16")


if __name__ == "__main__":
    unittest.main()
