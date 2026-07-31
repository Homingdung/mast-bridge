from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "evaluate_tokamind_manifest.py"
SPEC = importlib.util.spec_from_file_location("evaluate_tokamind_manifest", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class EvaluateTokamindManifestScriptTests(unittest.TestCase):
    def test_select_real_validation_rows_filters_source_and_parent_shots(self):
        rows = [
            {"sample_id": "11768_real", "source": "real", "shot_id": "11768"},
            {"sample_id": "11768_syn", "source": "synthetic", "parent_shot": "11768"},
            {"sample_id": "11775_real", "source": "real", "shot_id": "11775"},
            {"sample_id": "11780_real", "source": "real", "shot_id": "11780"},
            {"sample_id": "11766_real", "source": "real", "shot_id": "11766"},
        ]

        selected = MODULE.select_real_validation_rows(rows, ["11768", "11775", "11780"])

        self.assertEqual(
            [row["sample_id"] for row in selected],
            ["11768_real", "11775_real", "11780_real"],
        )

    def test_compute_metrics_reports_standardized_and_raw_errors(self):
        prediction_std = np.asarray([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
        target_std = np.asarray([[0.0, 2.0], [1.0, 3.0]], dtype=np.float32)
        output_mean = np.asarray([10.0, 20.0], dtype=np.float32)
        output_std = np.asarray([2.0, 4.0], dtype=np.float32)

        metrics = MODULE.compute_metrics(
            prediction_std=prediction_std,
            target_std=target_std,
            output_mean=output_mean,
            output_std=output_std,
        )

        self.assertEqual(metrics["samples"], 2)
        self.assertAlmostEqual(metrics["standardized_mse"], 0.5)
        self.assertAlmostEqual(metrics["raw_mse"], 5.0)
        self.assertAlmostEqual(metrics["raw_rmse"], np.sqrt(5.0))
        self.assertAlmostEqual(metrics["raw_mae"], 1.5)

    def test_help_runs_when_script_is_executed_directly(self):
        result = subprocess.run(
            [sys.executable, "scripts/evaluate_tokamind_manifest.py", "--help"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Evaluate one or more TokaMind manifest runs", result.stdout)

    def test_generic_entry_requires_explicit_manifest_and_run_dir(self):
        result = subprocess.run(
            [sys.executable, "scripts/evaluate_tokamind_manifest.py"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("--manifest", result.stderr)
        self.assertIn("--run-dir", result.stderr)

    def test_resolve_target_mode_requires_scaler_and_summary_to_match(self):
        self.assertEqual(
            MODULE.resolve_target_mode(
                {"target_mode": "raw-psi"},
                {"target_mode": "raw-psi"},
            ),
            "raw-psi",
        )
        with self.assertRaisesRegex(ValueError, "target_mode mismatch"):
            MODULE.resolve_target_mode(
                {"target_mode": "raw-psi"},
                {"target_mode": "psi-norm"},
            )

    def test_load_scalers_preserves_magnetic_diagnostics_input_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            np.savez_compressed(
                run_dir / "manifest_scalers.npz",
                feature_names=np.asarray(["magnetics.ip"], dtype=object),
                input_mean=np.asarray([0.0], dtype=np.float32),
                input_std=np.asarray([1.0], dtype=np.float32),
                output_mean=np.asarray([0.0], dtype=np.float32),
                output_std=np.asarray([1.0], dtype=np.float32),
                input_mode=np.asarray("magnetic-diagnostics"),
                target_mode=np.asarray("raw-psi"),
            )

            scalers = MODULE._load_scalers(run_dir)

        self.assertEqual(scalers["input_mode"], "magnetic-diagnostics")

    def test_model_config_comes_from_training_summary(self):
        summary = {
            "model_config": {
                "d_model": 64,
                "n_layers": 2,
                "n_heads": 4,
                "dim_ff": 128,
                "dropout": 0.05,
            }
        }

        config = MODULE.resolve_model_config(summary)

        self.assertEqual(config["d_model"], 64)
        self.assertEqual(config["n_layers"], 2)
        self.assertEqual(config["n_heads"], 4)

    def test_validation_split_rejects_training_shot_leakage(self):
        summary = {
            "train_shots": ["11766", "11768"],
            "val_shots": ["11775", "11780"],
        }
        rows = [{"shot_id": "11768"}, {"shot_id": "11775"}]

        with self.assertRaisesRegex(ValueError, "training split"):
            MODULE.validate_evaluation_split(summary, rows)

    def test_missing_checkpoint_is_rejected(self):
        with self.assertRaisesRegex(FileNotFoundError, "checkpoint"):
            MODULE.require_loaded_checkpoint(-1, Path("missing-run"))


if __name__ == "__main__":
    unittest.main()
