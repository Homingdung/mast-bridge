from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import zarr

from mast_bridge.training.tokamind_manifest import (
    ManifestWindowDataset,
    build_manifest_datasets,
    load_manifest_rows,
    normalize_psi,
)


class TokamindManifestTrainingTests(unittest.TestCase):
    def _write_real_shot(self, root: Path, shot: str) -> Path:
        shot_path = root / f"{shot}.zarr"
        z = zarr.open_group(str(shot_path), mode="w")
        eq = z.create_group("equilibrium")
        eq.create_array("time", data=np.asarray([0.10, 0.11], dtype=np.float64))
        psi = np.zeros((65, 65, 2), dtype=np.float32)
        psi[:, :, 0] = 1.0
        psi[:, :, 1] = 2.0
        eq.create_array("psi", data=psi)

        active = z.create_group("pf_active")
        active.create_array("time", data=np.asarray([0.10, 0.11], dtype=np.float64))
        active.create_array("current_channel", data=np.asarray(["SOL", "P2"], dtype="<U3"))
        active.create_array(
            "coil_current",
            data=np.asarray([[10.0, 12.0], [20.0, 22.0]], dtype=np.float32),
        )
        return shot_path

    def _write_fit(self, root: Path, shot: str) -> Path:
        fit_path = root / "fits.npz"
        np.savez(
            fit_path,
            shot=np.asarray([shot, shot]),
            time=np.asarray([0.10, 0.11], dtype=np.float64),
            ip=np.asarray([100.0, 110.0], dtype=np.float64),
            fvac=np.asarray([0.4, 0.5], dtype=np.float64),
            freegsnke_alpha=np.asarray([[1.0, 2.0, 3.0], [1.1, 2.1, 3.1]]),
            freegsnke_beta=np.asarray([[4.0, 5.0, 6.0], [4.1, 5.1, 6.1]]),
        )
        return fit_path

    def _write_synthetic_sample(self, root: Path) -> Path:
        sample = root / "11772_t0.10_v000"
        sample.mkdir()
        np.savez_compressed(
            sample / "equilibrium.npz",
            psi=np.full((65, 65), 3.0, dtype=np.float32),
            psi_axis=np.float32(1.0),
            psi_bndry=np.float32(5.0),
        )
        return sample

    def test_normalize_psi_uses_axis_and_boundary_flux(self):
        psi = np.asarray([[1.0, 3.0], [5.0, 7.0]], dtype=np.float32)

        normalized = normalize_psi(psi, psi_axis=1.0, psi_bndry=5.0)

        np.testing.assert_allclose(normalized, [[0.0, 0.5], [1.0, 1.5]])

    def test_manifest_dataset_builds_tokamind_window_dicts_from_real_and_synthetic_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            real_zarr = self._write_real_shot(root, "11772")
            fit_path = self._write_fit(root, "11772")
            synthetic_dir = self._write_synthetic_sample(root)

            manifest = root / "manifest.jsonl"
            rows = [
                {
                    "sample_id": "11772_t0.11_real",
                    "source": "real",
                    "shot_id": "11772",
                    "data_path": str(real_zarr),
                    "fit_path": str(fit_path),
                    "target_time": 0.11,
                },
                {
                    "sample_id": "11772_t0.10_v000",
                    "source": "synthetic",
                    "shot_id": "11772_t0.10_v000",
                    "parent_shot": "11772",
                    "data_path": str(synthetic_dir),
                    "equilibrium_path": str(synthetic_dir / "equilibrium.npz"),
                    "target_time": 0.10,
                    "Ip": 120.0,
                    "fvac": 0.6,
                    "alpha": [1.2, 2.2, 3.2],
                    "beta": [4.2, 5.2, 6.2],
                    "coil_currents": {"active": {"SOL": 30.0, "P2": 40.0}},
                },
            ]
            manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            loaded = load_manifest_rows(manifest)
            dataset = ManifestWindowDataset.from_rows(loaded)

            self.assertEqual(len(dataset), 2)
            self.assertIn("coil_active_SOL", dataset.feature_names)
            self.assertIn("coil_active_P2", dataset.feature_names)

            real_window = dataset[0]
            synthetic_window = dataset[1]

            for window in (real_window, synthetic_window):
                self.assertEqual(window["emb_chunks"][0].shape, (len(dataset.feature_names),))
                self.assertEqual(window["output_emb"][1].shape, (65 * 65,))
                self.assertEqual(window["output_names"][1], "equilibrium-psi")
                self.assertEqual(window["pos"].tolist(), [0])
                self.assertEqual(window["id"].tolist(), [0])

            self.assertTrue(np.isfinite(real_window["output_emb"][1]).all())
            self.assertTrue(np.isfinite(synthetic_window["output_emb"][1]).all())

    def test_manifest_dataset_can_use_synthetic_psi_norm_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            synthetic_dir = self._write_synthetic_sample(root)
            rows = [
                {
                    "sample_id": "11772_t0.10_v000",
                    "source": "synthetic",
                    "shot_id": "11772_t0.10_v000",
                    "parent_shot": "11772",
                    "data_path": str(synthetic_dir),
                    "equilibrium_path": str(synthetic_dir / "equilibrium.npz"),
                    "target_time": 0.10,
                    "Ip": 120.0,
                    "fvac": 0.6,
                    "alpha": [1.2, 2.2, 3.2],
                    "beta": [4.2, 5.2, 6.2],
                    "coil_currents": {"active": {"SOL": 30.0}},
                }
            ]

            dataset = ManifestWindowDataset.from_rows(rows, target_mode="psi-norm")

            np.testing.assert_allclose(dataset.output_mean, np.full(65 * 65, 0.5))
            np.testing.assert_allclose(dataset[0]["output_emb"][1], np.zeros(65 * 65))

    def test_build_manifest_datasets_uses_deterministic_train_val_split(self):
        rows = [
            {
                "sample_id": f"s{i}",
                "source": "synthetic",
                "shot_id": f"s{i}",
                "parent_shot": f"s{i}",
                "target_time": 0.1,
                "Ip": 1.0,
                "fvac": 0.4,
                "alpha": [1.0, 2.0, 3.0],
                "beta": [4.0, 5.0, 6.0],
                "coil_currents": {"active": {"SOL": float(i)}},
            }
            for i in range(5)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i, row in enumerate(rows):
                sample = root / f"s{i}"
                sample.mkdir()
                np.savez_compressed(sample / "equilibrium.npz", psi=np.full((65, 65), i, dtype=np.float32))
                row["data_path"] = str(sample)
                row["equilibrium_path"] = str(sample / "equilibrium.npz")

            train, val = build_manifest_datasets(rows, val_fraction=0.4, seed=7)

            self.assertEqual(len(train) + len(val), 5)
            self.assertEqual(len(val), 2)
            self.assertEqual(train.feature_names, val.feature_names)
            self.assertEqual(train.output_mean.shape, (65 * 65,))
            self.assertEqual(train.output_std.shape, (65 * 65,))

    def test_build_manifest_datasets_keeps_parent_shots_in_one_split(self):
        rows = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for parent in ("11772", "11773", "11774"):
                for variant in range(3):
                    sample = root / f"{parent}_v{variant}"
                    sample.mkdir()
                    np.savez_compressed(
                        sample / "equilibrium.npz",
                        psi=np.full((65, 65), variant, dtype=np.float32),
                    )
                    rows.append(
                        {
                            "sample_id": f"{parent}_v{variant}",
                            "source": "synthetic",
                            "shot_id": f"{parent}_v{variant}",
                            "parent_shot": parent,
                            "target_time": 0.1,
                            "data_path": str(sample),
                            "equilibrium_path": str(sample / "equilibrium.npz"),
                            "Ip": 1.0,
                            "fvac": 0.4,
                            "alpha": [1.0, 2.0, 3.0],
                            "beta": [4.0, 5.0, 6.0],
                            "coil_currents": {"active": {"SOL": float(variant)}},
                        }
                    )

            train, val = build_manifest_datasets(rows, val_fraction=0.34, seed=3)

            train_parents = {row["parent_shot"] for row in train.rows}
            val_parents = {row["parent_shot"] for row in val.rows}
            self.assertFalse(train_parents & val_parents)
            self.assertEqual(len(train) + len(val), len(rows))


if __name__ == "__main__":
    unittest.main()
