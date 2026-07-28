import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from mast_bridge.equilibrium.lao_from_zarr import (
    build_lao_fit_table,
    write_lao_fit_npz,
)


class LaoFitFromZarrTests(unittest.TestCase):
    def _fixture(self, root: Path, shot_id: str = "11766") -> Path:
        shot = root / f"{shot_id}.zarr"
        z = zarr.open_group(str(shot), mode="w")

        eq = z.create_group("equilibrium")
        psi_norm = np.linspace(0.0, 1.0, 5)
        times = np.array([0.10, 0.20, 0.30])
        pprime = np.column_stack(
            [
                np.full_like(psi_norm, np.nan),
                -2.0 + 2.0 * psi_norm,
                -4.0 + 4.0 * psi_norm,
            ]
        )
        ffprime = np.column_stack(
            [
                np.full_like(psi_norm, np.nan),
                3.0 - 3.0 * psi_norm,
                6.0 - 6.0 * psi_norm,
            ]
        )
        eq.create_array("time", data=times)
        eq.create_array("psi_norm", data=psi_norm)
        eq.create_array("dpressure_dpsi", data=pprime)
        eq.create_array("f_df_dpsi", data=ffprime)
        eq.create_array("bvac_rmag", data=np.array([np.nan, -0.45, -0.50]))

        magnetics = z.create_group("magnetics")
        magnetics.create_array("time", data=np.array([0.0, 0.2, 0.4]))
        magnetics.create_array("ip", data=np.array([100_000.0, 200_000.0, 300_000.0]))
        return shot

    def test_builds_lao_fit_rows_for_valid_equilibrium_times(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            table = build_lao_fit_table(
                [self._fixture(Path(temp_dir))],
                min_finite_points=4,
            )

        self.assertEqual(table["shot"].tolist(), ["11766", "11766"])
        self.assertEqual(table["time"].tolist(), [0.20, 0.30])
        np.testing.assert_allclose(table["ip"], [200_000.0, 250_000.0])
        np.testing.assert_allclose(table["fvac"], [0.45, 0.50])
        self.assertEqual(table["freegsnke_alpha"].shape, (2, 3))
        self.assertEqual(table["freegsnke_beta"].shape, (2, 3))

    def test_writes_npz_with_forward_solver_keys(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = root / "fits" / "all_zarr_lao_parameter_fits.npz"
            write_lao_fit_npz([self._fixture(root)], output, min_finite_points=4)
            fit = np.load(output)

        self.assertEqual(set(fit.files), {"shot", "time", "ip", "fvac", "freegsnke_alpha", "freegsnke_beta"})
        self.assertEqual(fit["shot"].tolist(), ["11766", "11766"])


if __name__ == "__main__":
    unittest.main()
