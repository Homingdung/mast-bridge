from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

import scripts.train_tokamind_diagnostics as diagnostics_script


class TrainTokamindDiagnosticsScriptTests(unittest.TestCase):
    def test_default_manifest_points_to_real_only_manifest(self):
        self.assertEqual(
            diagnostics_script.DEFAULT_MANIFEST.name,
            "tokamark_lao85_uniform_small_iter500_diagnostics_real_only.jsonl",
        )
        self.assertEqual(
            diagnostics_script.DEFAULT_FEATURE_SCHEMA.name,
            "mast_level2_common_94.json",
        )

    def test_help_mentions_magnetic_diagnostics(self):
        result = subprocess.run(
            [sys.executable, "scripts/train_tokamind_diagnostics.py", "--help"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("magnetic diagnostics", result.stdout)

    def test_forwards_common_feature_reference_manifest(self):
        args = diagnostics_script.build_parser().parse_args(
            [
                "--manifest",
                "synthetic.jsonl",
                "--feature-reference-manifest",
                "mixed.jsonl",
            ]
        )

        forwarded = diagnostics_script._manifest_args(args)

        self.assertIn("--feature-reference-manifest", forwarded)
        index = forwarded.index("--feature-reference-manifest")
        self.assertEqual(forwarded[index + 1], "mixed.jsonl")

    def test_forwards_default_versioned_feature_schema(self):
        args = diagnostics_script.build_parser().parse_args([])

        forwarded = diagnostics_script._manifest_args(args)

        self.assertIn("--feature-schema", forwarded)
        index = forwarded.index("--feature-schema")
        self.assertEqual(
            Path(forwarded[index + 1]).name,
            "mast_level2_common_94.json",
        )

    def test_forwards_fixed_validation_shots(self):
        args = diagnostics_script.build_parser().parse_args([])

        forwarded = diagnostics_script._manifest_args(args)

        self.assertEqual(
            [
                forwarded[index + 1]
                for index, value in enumerate(forwarded)
                if value == "--val-shot"
            ],
            ["11768", "11775", "11780"],
        )


if __name__ == "__main__":
    unittest.main()
