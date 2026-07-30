from __future__ import annotations

import importlib.util
import subprocess
import sys
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

    def test_resolve_target_mode_prefers_cli_then_scaler_then_raw_default(self):
        self.assertEqual(MODULE.resolve_target_mode("psi-norm", {"target_mode": "raw-psi"}), "psi-norm")
        self.assertEqual(MODULE.resolve_target_mode(None, {"target_mode": "psi-norm"}), "psi-norm")
        self.assertEqual(MODULE.resolve_target_mode(None, {}), "raw-psi")


if __name__ == "__main__":
    unittest.main()
