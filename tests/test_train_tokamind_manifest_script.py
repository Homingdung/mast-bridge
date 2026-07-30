from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

import scripts.train_tokamind_manifest as train_script


class TrainTokamindManifestScriptTests(unittest.TestCase):
    def test_default_manifest_uses_current_uniform_iter500_mixed_manifest(self):
        self.assertEqual(
            train_script.DEFAULT_MANIFEST.name,
            "tokamark_lao85_uniform_iter500_real_plus_synthetic.jsonl",
        )

    def test_dry_run_reports_manifest_dataset_summary_without_torch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for index in range(3):
                sample = root / f"s{index}"
                sample.mkdir()
                np.savez_compressed(
                    sample / "equilibrium.npz",
                    psi=np.full((65, 65), index, dtype=np.float32),
                    psi_axis=np.float32(index - 1.0),
                    psi_bndry=np.float32(index + 1.0),
                )
                rows.append(
                    {
                        "sample_id": f"s{index}",
                        "source": "synthetic",
                        "shot_id": f"s{index}",
                        "parent_shot": f"s{index}",
                        "target_time": 0.1,
                        "data_path": str(sample),
                        "equilibrium_path": str(sample / "equilibrium.npz"),
                        "Ip": 1.0,
                        "fvac": 0.4,
                        "alpha": [1.0, 2.0, 3.0],
                        "beta": [4.0, 5.0, 6.0],
                        "coil_currents": {"active": {"SOL": float(index)}},
                    }
                )

            manifest = root / "manifest.jsonl"
            manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/train_tokamind_manifest.py",
                    "--manifest",
                    str(manifest),
                    "--target-mode",
                    "psi-norm",
                    "--dry-run",
                    "--val-fraction",
                    "0.34",
                ],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("rows: 3", result.stdout)
            self.assertIn("train_windows: 2", result.stdout)
            self.assertIn("val_windows: 1", result.stdout)
            self.assertIn("target_mode: psi-norm", result.stdout)
            self.assertIn("input_signal: fusion-state", result.stdout)
            self.assertIn("output_signal: equilibrium-psi", result.stdout)


if __name__ == "__main__":
    unittest.main()
