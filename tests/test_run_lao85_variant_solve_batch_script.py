import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_lao85_variant_solve_batch.py"
SPEC = importlib.util.spec_from_file_location("run_lao85_variant_solve_batch", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class RunLao85VariantSolveBatchScriptTests(unittest.TestCase):
    def _write_variant_csv(self, path: Path) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=MODULE.VARIANT_FIELDNAMES)
            writer.writeheader()
            writer.writerow(
                {
                    "shot": "11771",
                    "target_time": "0.16",
                    "variant_id": "v000",
                    "sampling_method": "uniform_random",
                    "ip_scale": "1.0001",
                    "fvac_scale": "0.9999",
                    "alpha_scale": "1.0002",
                    "beta_scale": "0.9998",
                    "alpha_offset": "0.0001",
                    "beta_offset": "-0.0001",
                    "coil_current_scale": "1.0",
                }
            )

    def test_builds_forward_command_for_one_variant(self):
        row = {
            "shot": "11771",
            "target_time": "0.16",
            "variant_id": "v000",
            "ip_scale": "1.0001",
            "fvac_scale": "0.9999",
            "alpha_scale": "1.0002",
            "beta_scale": "0.9998",
            "alpha_offset": "0.0001",
            "beta_offset": "-0.0001",
        }
        output_dir = Path("/tmp/synthetic/11771_t0.16_v000").resolve()

        command = MODULE.build_forward_command(
            row,
            python_executable=Path("/env/bin/python"),
            data_dir=Path("/data/raw/mast"),
            fit_path=Path("/data/fits.npz"),
            synthetic_root=Path("/tmp/synthetic"),
            nx=65,
            ny=65,
            tolerance=1e-8,
            max_iterations=100,
        )

        self.assertEqual(command[0], "/env/bin/python")
        self.assertIn("--shot", command)
        self.assertIn("11771", command)
        self.assertIn("--output-dir", command)
        self.assertIn(str(output_dir), command)
        self.assertIn("--ip-scale", command)
        self.assertIn("1.0001", command)

    def test_runs_solver_rows_and_filters_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            variant_csv = root / "variants.csv"
            synthetic_root = root / "synthetic"
            manifest_dir = root / "manifests"
            self._write_variant_csv(variant_csv)

            calls = []

            def fake_run(command):
                calls.append(command)
                output_dir = Path(command[command.index("--output-dir") + 1])
                output_dir.mkdir(parents=True)
                np.savez_compressed(
                    output_dir / "equilibrium.npz",
                    psi=np.zeros((65, 65)),
                    R=np.zeros((65, 65)),
                    Z=np.zeros((65, 65)),
                    psi_axis=0.0,
                    psi_bndry=1.0,
                )
                (output_dir / "metadata.json").write_text(
                    json.dumps(
                        {
                            "source": "synthetic",
                            "parent_shot": "11771",
                            "target_time": 0.16,
                            "solver_converged": True,
                            "solver_final_tolerance": 9e-9,
                        }
                    ),
                    encoding="utf-8",
                )
                return 0

            exit_code = MODULE.main(
                [
                    "--variant-csv",
                    str(variant_csv),
                    "--data-dir",
                    str(root / "raw"),
                    "--fit-path",
                    str(root / "fits.npz"),
                    "--synthetic-root",
                    str(synthetic_root),
                    "--manifest-dir",
                    str(manifest_dir),
                    "--prefix",
                    "small",
                    "--python",
                    "/env/bin/python",
                ],
                run_command=fake_run,
            )

            accepted = manifest_dir / "small_synthetic_accepted.jsonl"
            rejected = manifest_dir / "small_synthetic_rejected.jsonl"
            accepted_rows = [json.loads(line) for line in accepted.read_text().splitlines()]
            rejected_exists = rejected.is_file()

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(calls), 1)
        self.assertTrue(rejected_exists)
        self.assertEqual(accepted_rows[0]["sample_id"], "11771_t0.16_v000")


if __name__ == "__main__":
    unittest.main()
