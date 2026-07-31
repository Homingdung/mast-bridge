import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from mast_bridge.dataset.synthetic_manifest import rejection_reason, synthetic_entries


class SyntheticManifestTests(unittest.TestCase):
    def test_scans_completed_synthetic_sample(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sample = root / "11771_t0.16_v000"
            sample.mkdir()
            np.savez_compressed(
                sample / "equilibrium.npz",
                psi=np.zeros((65, 65)),
                R=np.zeros((65, 65)),
                Z=np.zeros((65, 65)),
                psi_axis=0.0,
                psi_bndry=1.0,
            )
            (sample / "metadata.json").write_text(
                json.dumps(
                    {
                        "source": "synthetic",
                        "parent_shot": "11771",
                        "target_time": 0.16,
                        "solver_status": "success",
                        "solver_converged": True,
                        "solver_final_tolerance": 9e-9,
                    }
                )
            )

            rows = synthetic_entries(root, task="task_1-3")

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].sample_id, "11771_t0.16_v000")
        self.assertEqual(rows[0].parent_shot, "11771")
        self.assertEqual(rows[0].metadata["task"], "task_1-3")

    def test_rejects_nonfinite_psi(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sample = root / "11771_t0.16_v000"
            sample.mkdir()
            np.savez_compressed(
                sample / "equilibrium.npz",
                psi=np.array([[float("nan")]]),
                R=np.zeros((1, 1)),
                Z=np.zeros((1, 1)),
                psi_axis=0.0,
                psi_bndry=1.0,
            )
            (sample / "metadata.json").write_text(
                json.dumps(
                    {
                        "source": "synthetic",
                        "parent_shot": "11771",
                        "target_time": 0.16,
                        "solver_status": "success",
                        "solver_converged": True,
                        "solver_final_tolerance": 9e-9,
                    }
                )
            )

            rows = synthetic_entries(root, task="task_1-3")

        self.assertEqual(rows, [])

    def test_rejects_non_converged_sample(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sample = root / "11771_t0.16_v000"
            sample.mkdir()
            np.savez_compressed(
                sample / "equilibrium.npz",
                psi=np.zeros((65, 65)),
                R=np.zeros((65, 65)),
                Z=np.zeros((65, 65)),
                psi_axis=0.0,
                psi_bndry=1.0,
            )
            (sample / "metadata.json").write_text(
                json.dumps(
                    {
                        "source": "synthetic",
                        "parent_shot": "11771",
                        "target_time": 0.16,
                        "solver_status": "non_converged",
                        "solver_converged": False,
                        "solver_final_tolerance": 2e-8,
                    }
                )
            )

            rows = synthetic_entries(root, task="task_1-3")

        self.assertEqual(rows, [])

    def test_rejects_converged_sample_above_strict_tolerance(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sample = root / "11771_t0.16_v000"
            sample.mkdir()
            np.savez_compressed(
                sample / "equilibrium.npz",
                psi=np.zeros((65, 65)),
                R=np.zeros((65, 65)),
                Z=np.zeros((65, 65)),
                psi_axis=0.0,
                psi_bndry=1.0,
            )
            (sample / "metadata.json").write_text(
                json.dumps(
                    {
                        "source": "synthetic",
                        "parent_shot": "11771",
                        "target_time": 0.16,
                        "solver_status": "success",
                        "solver_converged": True,
                        "solver_final_tolerance": 1.1e-8,
                    }
                )
            )

            rows = synthetic_entries(root, task="task_1-3")

        self.assertEqual(rows, [])

    def test_rejects_nonfinite_solver_tolerance(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            equilibrium_path = Path(temp_dir) / "equilibrium.npz"
            np.savez_compressed(
                equilibrium_path,
                psi=np.zeros((65, 65)),
            )

            reason = rejection_reason(
                {
                    "solver_converged": True,
                    "solver_final_tolerance": float("nan"),
                },
                equilibrium_path,
            )

        self.assertEqual(reason, "solver_tolerance_nonfinite")

    def test_rejects_finite_psi_with_wrong_grid_shape(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            equilibrium_path = Path(temp_dir) / "equilibrium.npz"
            np.savez_compressed(
                equilibrium_path,
                psi=np.zeros((64, 65)),
            )

            reason = rejection_reason(
                {
                    "solver_converged": True,
                    "solver_final_tolerance": 9e-9,
                },
                equilibrium_path,
            )

        self.assertEqual(reason, "invalid_equilibrium")


if __name__ == "__main__":
    unittest.main()
