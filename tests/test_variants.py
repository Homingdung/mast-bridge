import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from mast_bridge.simulation.variants import build_variant_rows, rows_from_lao_fit_npz


class VariantTests(unittest.TestCase):
    def test_builds_deterministic_rows(self):
        first = build_variant_rows(["11771"], [0.16], variants_per_point=2, seed=123)
        second = build_variant_rows(["11771"], [0.16], variants_per_point=2, seed=123)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)
        self.assertEqual(first[0]["variant_id"], "v000")

    def test_scales_are_bounded(self):
        rows = build_variant_rows(["11771"], [0.16], variants_per_point=20, seed=123)

        for row in rows:
            self.assertGreaterEqual(row["ip_scale"], 0.95)
            self.assertLessEqual(row["ip_scale"], 1.05)
            self.assertGreaterEqual(row["fvac_scale"], 0.99)
            self.assertLessEqual(row["fvac_scale"], 1.01)
            self.assertGreaterEqual(row["alpha_scale"], 0.98)
            self.assertLessEqual(row["alpha_scale"], 1.02)
            self.assertGreaterEqual(row["beta_scale"], 0.98)
            self.assertLessEqual(row["beta_scale"], 1.02)
            self.assertGreaterEqual(row["alpha_offset"], -0.01)
            self.assertLessEqual(row["alpha_offset"], 0.01)
            self.assertGreaterEqual(row["beta_offset"], -0.01)
            self.assertLessEqual(row["beta_offset"], 0.01)
            self.assertGreaterEqual(row["coil_current_scale"], 0.97)
            self.assertLessEqual(row["coil_current_scale"], 1.03)

    def test_uses_original_uniform_random_sampling(self):
        rows = build_variant_rows(["11771"], [0.16], variants_per_point=8, seed=123)

        self.assertTrue(all(row["sampling_method"] == "uniform_random" for row in rows))
        self.assertEqual(len({row["ip_scale"] for row in rows}), 8)

    def test_builds_rows_from_lao_fit_npz(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fit_path = Path(temp_dir) / "fits.npz"
            np.savez_compressed(
                fit_path,
                shot=np.asarray(["11771", "11772"]),
                time=np.asarray([0.16, 0.18]),
                ip=np.asarray([1.0, 2.0]),
                fvac=np.asarray([0.4, 0.5]),
                freegsnke_alpha=np.zeros((2, 3)),
                freegsnke_beta=np.zeros((2, 3)),
            )

            rows = rows_from_lao_fit_npz(fit_path, variants_per_point=2, seed=123)

        self.assertEqual(len(rows), 4)
        self.assertEqual([row["shot"] for row in rows], ["11771", "11771", "11772", "11772"])
        self.assertEqual([row["target_time"] for row in rows], [0.16, 0.16, 0.18, 0.18])
        self.assertEqual([row["variant_id"] for row in rows], ["v000", "v001", "v000", "v001"])

    def test_filters_lao_fit_rows_by_time_window(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fit_path = Path(temp_dir) / "fits.npz"
            np.savez_compressed(
                fit_path,
                shot=np.asarray(["11771", "11771", "11771"]),
                time=np.asarray([0.03, 0.10, 0.16]),
                ip=np.asarray([1.0, 2.0, 3.0]),
                fvac=np.asarray([0.4, 0.5, 0.6]),
                freegsnke_alpha=np.zeros((3, 3)),
                freegsnke_beta=np.zeros((3, 3)),
            )

            rows = rows_from_lao_fit_npz(
                fit_path,
                variants_per_point=1,
                seed=123,
                min_time=0.09,
                max_time=0.12,
            )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["target_time"], 0.10)


if __name__ == "__main__":
    unittest.main()
