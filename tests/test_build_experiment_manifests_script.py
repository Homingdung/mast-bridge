import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "build_experiment_manifests.py"
SPEC = importlib.util.spec_from_file_location("build_experiment_manifests", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class BuildExperimentManifestsScriptTests(unittest.TestCase):
    def test_builds_real_synthetic_and_mixed_experiment_manifests(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw = root / "raw" / "mast"
            machine = raw / "machine"
            (raw / "11772.zarr").mkdir(parents=True)
            (machine / "11772").mkdir(parents=True)
            accepted = root / "accepted.jsonl"
            accepted.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "sample_id": "11772_t0.10_v000",
                                "source": "synthetic",
                                "parent_shot": "11772",
                                "target_time": 0.10,
                                "data_path": "synthetic/11772_t0.10_v000",
                                "equilibrium_path": "synthetic/11772_t0.10_v000/equilibrium.npz",
                                "solver_final_tolerance": 2e-9,
                            }
                        ),
                        json.dumps(
                            {
                                "sample_id": "11772_t0.10_v001",
                                "source": "synthetic",
                                "parent_shot": "11772",
                                "target_time": 0.10,
                                "data_path": "synthetic/11772_t0.10_v001",
                                "equilibrium_path": "synthetic/11772_t0.10_v001/equilibrium.npz",
                                "solver_final_tolerance": 3e-9,
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            output_dir = root / "manifests"

            exit_code = MODULE.main(
                [
                    "--accepted-synthetic",
                    str(accepted),
                    "--raw-data-dir",
                    str(raw),
                    "--fit-path",
                    str(root / "fits.npz"),
                    "--output-dir",
                    str(output_dir),
                    "--prefix",
                    "small",
                    "--task",
                    "task_1-3",
                ]
            )

            real_rows = [
                json.loads(line)
                for line in (output_dir / "small_real_only.jsonl").read_text().splitlines()
            ]
            synthetic_rows = [
                json.loads(line)
                for line in (output_dir / "small_synthetic_only.jsonl").read_text().splitlines()
            ]
            mixed_rows = [
                json.loads(line)
                for line in (output_dir / "small_real_plus_synthetic.jsonl").read_text().splitlines()
            ]

        self.assertEqual(exit_code, 0)
        self.assertEqual([row["sample_id"] for row in real_rows], ["11772_t0.1_real"])
        self.assertEqual(real_rows[0]["source"], "real")
        self.assertEqual(real_rows[0]["comparison_group"], "real_only")
        self.assertIsNone(real_rows[0]["label_path"])
        self.assertEqual(real_rows[0]["label_source"], "zarr_equilibrium_psi")
        self.assertEqual(real_rows[0]["profile_parameter_source"], "lao_fit_npz")
        self.assertEqual(len(synthetic_rows), 2)
        self.assertTrue(all(row["comparison_group"] == "synthetic_only" for row in synthetic_rows))
        self.assertEqual(len(mixed_rows), 3)
        self.assertEqual(
            [row["comparison_group"] for row in mixed_rows],
            ["real_plus_synthetic", "real_plus_synthetic", "real_plus_synthetic"],
        )


if __name__ == "__main__":
    unittest.main()
