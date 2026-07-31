from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

import scripts.evaluate_tokamind_diagnostics as diagnostics_script


class EvaluateTokamindDiagnosticsScriptTests(unittest.TestCase):
    def test_defaults_point_to_current_diagnostics_experiment(self):
        args = diagnostics_script.build_parser().parse_args([])

        self.assertEqual(
            args.manifest.name,
            "tokamark_lao85_uniform_small_iter500_diagnostics_real_only.jsonl",
        )
        self.assertEqual(
            [path.name for path in diagnostics_script.DEFAULT_RUN_DIRS],
            [
                "tokamind-diagnostics-real-only",
                "tokamind-diagnostics-synthetic-only",
                "tokamind-diagnostics-real-plus-synthetic",
            ],
        )
        self.assertIsNone(args.val_shot)
        self.assertEqual(
            diagnostics_script.DEFAULT_VAL_SHOTS,
            ["11768", "11775", "11780"],
        )

    def test_forwards_all_diagnostics_runs_and_outputs(self):
        args = diagnostics_script.build_parser().parse_args([])

        forwarded = diagnostics_script._manifest_args(args)

        self.assertEqual(forwarded.count("--run-dir"), 3)
        self.assertIn(
            "tokamind_diagnostics_real_val_metrics.json",
            [Path(value).name for value in forwarded],
        )
        self.assertIn(
            "tokamind_diagnostics_real_val_metrics.csv",
            [Path(value).name for value in forwarded],
        )

    def test_help_identifies_fixed_real_validation_evaluation(self):
        result = subprocess.run(
            [sys.executable, "scripts/evaluate_tokamind_diagnostics.py", "--help"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("fixed real EFIT validation set", result.stdout)

    def test_explicit_validation_shots_replace_defaults(self):
        args = diagnostics_script.build_parser().parse_args(
            ["--val-shot", "12000", "--val-shot", "12001"]
        )

        forwarded = diagnostics_script._manifest_args(args)

        self.assertNotIn("11768", forwarded)
        self.assertNotIn("11775", forwarded)
        self.assertNotIn("11780", forwarded)
        self.assertIn("12000", forwarded)
        self.assertIn("12001", forwarded)


if __name__ == "__main__":
    unittest.main()
