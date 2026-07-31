from __future__ import annotations

import json
import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

import scripts.train_tokamind_manifest as train_script
from mast_bridge.training.tokamind_manifest import load_feature_schema


class TrainTokamindManifestScriptTests(unittest.TestCase):
    def test_generic_entry_requires_explicit_manifest_and_run_dir(self):
        result = subprocess.run(
            [
                sys.executable,
                "scripts/train_tokamind_manifest.py",
                "--manifest",
                "manifest.jsonl",
                "--run-dir",
                "run",
            ],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("required: --input-mode", result.stderr)

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
                    "--run-dir",
                    str(root / "run"),
                    "--input-mode",
                    "lao-params",
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
            self.assertIn("input_mode: lao-params", result.stdout)
            self.assertIn("target_mode: psi-norm", result.stdout)
            self.assertIn("input_signal: fusion-state", result.stdout)
            self.assertIn("output_signal: equilibrium-psi", result.stdout)

    def test_load_feature_schema_verifies_digest_and_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "features.json"
            names = ["target_time", "magnetics_ip", "flux_loop_FL1"]
            digest = hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "feature_count": 3,
                        "feature_names_sha256": digest,
                        "feature_names": names,
                    }
                ),
                encoding="utf-8",
            )

            loaded = load_feature_schema(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["feature_names"][2] = "flux_loop_WRONG"
            path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "digest"):
                load_feature_schema(path)

        self.assertEqual(loaded, names)


if __name__ == "__main__":
    unittest.main()
