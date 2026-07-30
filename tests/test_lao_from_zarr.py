import tempfile
import unittest
from pathlib import Path

import numpy as np
import zarr

from mast_bridge.equilibrium.lao_from_zarr import build_lao_fit_table


class LaoFromZarrTests(unittest.TestCase):
    def test_fvac_is_vacuum_field_radius_product(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            shot_path = Path(temp_dir) / "12345.zarr"
            root = zarr.open_group(str(shot_path), mode="w")
            equilibrium = root.create_group("equilibrium")
            equilibrium.create_array("time", data=np.asarray([0.1], dtype=float))
            equilibrium.create_array("psi_norm", data=np.asarray([0.0, 0.5, 1.0, 1.5], dtype=float))
            equilibrium.create_array(
                "dpressure_dpsi",
                data=np.asarray([[1.0], [0.5], [0.1], [0.0]], dtype=float),
            )
            equilibrium.create_array(
                "f_df_dpsi",
                data=np.asarray([[0.8], [0.4], [0.2], [0.0]], dtype=float),
            )
            equilibrium.create_array("bvac_rmag", data=np.asarray([-0.5], dtype=float))
            equilibrium.create_array("magnetic_axis_r", data=np.asarray([0.8], dtype=float))

            magnetics = root.create_group("magnetics")
            magnetics.create_array("time", data=np.asarray([0.09, 0.11], dtype=float))
            magnetics.create_array("ip", data=np.asarray([100.0, 120.0], dtype=float))

            table = build_lao_fit_table(
                [shot_path],
                n_alpha=2,
                n_beta=2,
                min_finite_points=3,
            )

        self.assertEqual(table["shot"].tolist(), ["12345"])
        self.assertAlmostEqual(float(table["fvac"][0]), 0.4)


if __name__ == "__main__":
    unittest.main()
