import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from mast_bridge.dataset.manifest import ManifestEntry, write_manifest
from mast_bridge.equilibrium.lao_fit import LaoProfile


class LaoProfileTests(unittest.TestCase):
    def test_round_trips_fixed_profile(self):
        profile = LaoProfile.from_dict(
            {
                "model": "lao85",
                "pprime_coefficients": [1.2, -0.3],
                "ffprime_coefficients": [0.9, 0.1],
            }
        )

        self.assertEqual(profile.to_dict()["model"], "lao85")
        self.assertEqual(profile.to_dict()["pprime_coefficients"], [1.2, -0.3])

    def test_round_trips_sampling_ranges(self):
        profile = LaoProfile.from_dict(
            {
                "model": "lao85",
                "sampling": {"n_samples": 3, "seed": 42},
                "pprime_coefficients": [{"min": 0.8, "max": 1.5}],
                "ffprime_coefficients": [{"min": 0.0, "max": 0.2}],
            }
        )

        self.assertEqual(profile.sampling["n_samples"], 3)
        self.assertEqual(profile.to_dict()["ffprime_coefficients"][0]["max"], 0.2)


class ManifestTests(unittest.TestCase):
    def test_writes_real_and_synthetic_entries_as_jsonl(self):
        entries = [
            ManifestEntry(
                sample_id="11766",
                source="real",
                shot_id="11766",
                data_path=Path("data/raw/mast/11766.zarr"),
            ),
            ManifestEntry(
                sample_id="11766_variant_0001",
                source="synthetic",
                shot_id="11766",
                data_path=Path("data/processed/synthetic/11766_variant_0001.zarr"),
                parent_shot="11766",
                solver_status="success",
            ),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "manifest.jsonl"
            write_manifest(entries, output)
            rows = [json.loads(line) for line in output.read_text().splitlines()]

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["source"], "real")
        self.assertEqual(rows[1]["solver_status"], "success")
        self.assertEqual(rows[1]["parent_shot"], "11766")
