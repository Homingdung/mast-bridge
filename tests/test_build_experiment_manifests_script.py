import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

import numpy as np

from mast_bridge.simulation.synthetic_diagnostics import write_synthetic_diagnostics


SCRIPT = Path(__file__).parents[1] / "scripts" / "build_experiment_manifests.py"
SPEC = importlib.util.spec_from_file_location("build_experiment_manifests", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def write_valid_synthetic_row(
    root: Path,
    sample_id: str,
    shot: str = "11772",
    target_time: float = 0.10,
) -> dict:
    sample = root / "synthetic" / sample_id
    sample.mkdir(parents=True)
    np.savez_compressed(
        sample / "equilibrium.npz",
        psi=np.zeros((65, 65), dtype=np.float32),
    )
    write_synthetic_diagnostics(
        sample / "diagnostics.npz",
        target_time=target_time,
        magnetics_ip=100_000.0,
        flux_loop_names=["FL1"],
        flux_loop_values=[0.1],
        pickup_names=["OBR01"],
        pickup_families=["OBR"],
        pickup_values=[0.2],
        active_coil_currents={"SOL": 10.0},
        flux_loop_scale=2.0 * np.pi,
    )
    return {
        "sample_id": sample_id,
        "source": "synthetic",
        "parent_shot": shot,
        "target_time": target_time,
        "data_path": str(sample),
        "equilibrium_path": str(sample / "equilibrium.npz"),
        "solver_converged": True,
        "solver_final_tolerance": 2e-9,
    }


class BuildExperimentManifestsScriptTests(unittest.TestCase):
    def test_one_synthetic_per_parent_is_balanced_and_order_independent(self):
        rows = [
            {
                "sample_id": "11772_t0.10_v001",
                "parent_shot": "11772",
                "target_time": 0.10,
            },
            {
                "sample_id": "11773_t0.12_v000",
                "parent_shot": "11773",
                "target_time": 0.12,
            },
            {
                "sample_id": "11772_t0.10_v000",
                "parent_shot": "11772",
                "target_time": 0.10,
            },
        ]

        selected = MODULE.select_one_synthetic_per_parent(rows, seed=20260731)
        selected_reversed = MODULE.select_one_synthetic_per_parent(
            list(reversed(rows)),
            seed=20260731,
        )

        self.assertEqual(len(selected), 2)
        self.assertEqual(
            [row["sample_id"] for row in selected],
            [row["sample_id"] for row in selected_reversed],
        )
        self.assertEqual(
            {
                (row["parent_shot"], row["target_time"])
                for row in selected
            },
            {("11772", 0.10), ("11773", 0.12)},
        )
        self.assertTrue(
            all(row["variant_selection_method"] == "seeded_hash" for row in selected)
        )
        self.assertTrue(
            all(row["variant_selection_seed"] == 20260731 for row in selected)
        )

    def test_one_synthetic_per_parent_rejects_duplicate_sample_ids(self):
        rows = [
            {
                "sample_id": "11772_t0.10_v000",
                "parent_shot": "11772",
                "target_time": 0.10,
                "data_path": "first",
            },
            {
                "sample_id": "11772_t0.10_v000",
                "parent_shot": "11772",
                "target_time": 0.10,
                "data_path": "second",
            },
        ]

        with self.assertRaisesRegex(ValueError, "duplicate sample_id"):
            MODULE.select_one_synthetic_per_parent(rows, seed=20260731)

    def test_paired_mode_requires_diagnostics_validation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            accepted = root / "accepted.jsonl"
            accepted.write_text("", encoding="utf-8")
            stderr = io.StringIO()

            with redirect_stderr(stderr), self.assertRaises(SystemExit):
                MODULE.main(
                    [
                        "--accepted-synthetic",
                        str(accepted),
                        "--raw-data-dir",
                        str(root),
                        "--fit-path",
                        str(root / "fits.npz"),
                        "--output-dir",
                        str(root / "manifests"),
                        "--one-synthetic-per-parent",
                    ]
                )

        self.assertIn(
            "--one-synthetic-per-parent requires --require-synthetic-diagnostics",
            stderr.getvalue(),
        )

    def test_paired_mode_rejects_missing_real_parent_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            accepted = root / "accepted.jsonl"
            row = write_valid_synthetic_row(root, "11772_t0.10_v000")
            accepted.write_text(json.dumps(row) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "paired parent mismatch"):
                MODULE.main(
                    [
                        "--accepted-synthetic",
                        str(accepted),
                        "--raw-data-dir",
                        str(root / "raw" / "mast"),
                        "--fit-path",
                        str(root / "fits.npz"),
                        "--output-dir",
                        str(root / "manifests"),
                        "--one-synthetic-per-parent",
                        "--require-synthetic-diagnostics",
                    ]
                )

    def test_cli_writes_one_synthetic_row_for_each_real_parent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw = root / "raw" / "mast"
            (raw / "11772.zarr").mkdir(parents=True)
            (raw / "machine" / "11772").mkdir(parents=True)
            accepted = root / "accepted.jsonl"
            rows = [
                write_valid_synthetic_row(
                    root,
                    f"11772_t0.10_v00{variant}",
                )
                for variant in range(2)
            ]
            accepted.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
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
                    "paired",
                    "--one-synthetic-per-parent",
                    "--selection-seed",
                    "20260731",
                    "--require-synthetic-diagnostics",
                ]
            )
            real_rows = [
                json.loads(line)
                for line in (
                    output_dir / "paired_real_only.jsonl"
                ).read_text().splitlines()
            ]
            synthetic_rows = [
                json.loads(line)
                for line in (
                    output_dir / "paired_synthetic_only.jsonl"
                ).read_text().splitlines()
            ]
            mixed_rows = [
                json.loads(line)
                for line in (
                    output_dir / "paired_real_plus_synthetic.jsonl"
                ).read_text().splitlines()
            ]

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(real_rows), 1)
        self.assertEqual(len(synthetic_rows), 1)
        self.assertEqual(len(mixed_rows), 2)
        self.assertEqual(synthetic_rows[0]["variant_selection_seed"], 20260731)

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

    def test_require_diagnostics_excludes_missing_payloads_and_writes_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw = root / "raw" / "mast"
            (raw / "11772.zarr").mkdir(parents=True)
            (raw / "machine" / "11772").mkdir(parents=True)
            valid_sample = root / "synthetic" / "11772_t0.10_v000"
            missing_sample = root / "synthetic" / "11772_t0.10_v001"
            valid_sample.mkdir(parents=True)
            missing_sample.mkdir(parents=True)
            for sample in (valid_sample, missing_sample):
                np.savez_compressed(
                    sample / "equilibrium.npz",
                    psi=np.zeros((65, 65), dtype=np.float32),
                )
            write_synthetic_diagnostics(
                valid_sample / "diagnostics.npz",
                target_time=0.10,
                magnetics_ip=100_000.0,
                flux_loop_names=["FL1"],
                flux_loop_values=[0.1],
                pickup_names=["OBR01"],
                pickup_families=["OBR"],
                pickup_values=[0.2],
                active_coil_currents={"SOL": 10.0},
                flux_loop_scale=2.0 * np.pi,
            )
            accepted = root / "accepted.jsonl"
            accepted.write_text(
                "\n".join(
                    json.dumps(
                        {
                            "sample_id": sample.name,
                            "source": "synthetic",
                            "parent_shot": "11772",
                            "target_time": 0.10,
                            "data_path": str(sample),
                            "equilibrium_path": str(sample / "equilibrium.npz"),
                            "solver_converged": True,
                            "solver_final_tolerance": 2e-9,
                        }
                    )
                    for sample in (valid_sample, missing_sample)
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
                    "diagnostics",
                    "--require-synthetic-diagnostics",
                ]
            )
            synthetic_rows = [
                json.loads(line)
                for line in (
                    output_dir / "diagnostics_synthetic_only.jsonl"
                ).read_text().splitlines()
            ]
            mixed_rows = [
                json.loads(line)
                for line in (
                    output_dir / "diagnostics_real_plus_synthetic.jsonl"
                ).read_text().splitlines()
            ]

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            [row["sample_id"] for row in synthetic_rows],
            ["11772_t0.10_v000"],
        )
        diagnostics_path = Path(synthetic_rows[0]["diagnostics_path"])
        self.assertEqual(diagnostics_path.name, "diagnostics.npz")
        self.assertEqual(diagnostics_path.parent.name, "11772_t0.10_v000")
        self.assertTrue(diagnostics_path.is_absolute())
        self.assertEqual(len(mixed_rows), 2)

    def test_require_diagnostics_revalidates_strict_solver_acceptance(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            sample = Path(temp_dir) / "11772_t0.10_v000"
            sample.mkdir()
            np.savez_compressed(
                sample / "equilibrium.npz",
                psi=np.zeros((65, 65), dtype=np.float32),
            )
            write_synthetic_diagnostics(
                sample / "diagnostics.npz",
                target_time=0.10,
                magnetics_ip=100_000.0,
                flux_loop_names=["FL1"],
                flux_loop_values=[0.1],
                pickup_names=["OBR01"],
                pickup_families=["OBR"],
                pickup_values=[0.2],
                active_coil_currents={"SOL": 10.0},
                flux_loop_scale=2.0 * np.pi,
            )
            row = {
                "sample_id": sample.name,
                "source": "synthetic",
                "parent_shot": "11772",
                "target_time": 0.10,
                "data_path": str(sample),
                "equilibrium_path": str(sample / "equilibrium.npz"),
                "solver_converged": False,
                "solver_final_tolerance": 2e-9,
            }

            accepted, excluded = MODULE.rows_with_valid_diagnostics([row])

        self.assertEqual(accepted, [])
        self.assertEqual(excluded[0]["reason"], "solver_not_converged")


if __name__ == "__main__":
    unittest.main()
