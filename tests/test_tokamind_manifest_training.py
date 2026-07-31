from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import zarr

from mast_bridge.training.tokamind_manifest import (
    INPUT_MAGNETIC_DIAGNOSTICS,
    ManifestWindowDataset,
    _real_psi,
    build_manifest_datasets,
    diagnostic_feature_names,
    diagnostic_feature_vector,
    load_manifest_rows,
    normalize_psi,
)
from mast_bridge.simulation.synthetic_diagnostics import write_synthetic_diagnostics


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

        magnetics = z.create_group("magnetics")
        magnetics.create_array("time", data=np.asarray([0.10, 0.11], dtype=np.float64))
        magnetics.create_array("ip", data=np.asarray([1000.0, 1100.0], dtype=np.float32))
        magnetics.create_array("flux_loop_channel", data=np.asarray(["FL1", "FL2"], dtype="<U3"))
        magnetics.create_array(
            "flux_loop_flux",
            data=np.asarray([[1.0, 1.2], [2.0, 2.2]], dtype=np.float32),
        )
        magnetics.create_array("b_field_pol_probe_ccbv_channel", data=np.asarray(["CCBV01"], dtype="<U6"))
        magnetics.create_array(
            "b_field_pol_probe_ccbv_field",
            data=np.asarray([[3.0, 3.2]], dtype=np.float32),
        )
        magnetics.create_array("b_field_pol_probe_obr_channel", data=np.asarray(["OBR01"], dtype="<U5"))
        magnetics.create_array(
            "b_field_pol_probe_obr_field",
            data=np.asarray([[4.0, 4.2]], dtype=np.float32),
        )
        magnetics.create_array("b_field_pol_probe_obv_channel", data=np.asarray(["OBV01"], dtype="<U5"))
        magnetics.create_array(
            "b_field_pol_probe_obv_field",
            data=np.asarray([[5.0, 5.2]], dtype=np.float32),
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

    def test_real_psi_transposes_mast_zr_storage_to_canonical_rz(self):
        with tempfile.TemporaryDirectory() as tmp:
            shot_path = self._write_real_shot(Path(tmp), "11772")
            root = zarr.open_group(str(shot_path), mode="a")
            z_index = np.arange(65, dtype=np.float32)[:, None]
            r_index = np.arange(65, dtype=np.float32)[None, :]
            stored_zr = 100.0 * z_index + r_index
            root["equilibrium"]["psi"][:, :, 1] = stored_zr
            row = {
                "sample_id": "11772_t0.11_real",
                "source": "real",
                "shot_id": "11772",
                "data_path": str(shot_path),
                "target_time": 0.11,
            }

            psi_rz = _real_psi(row)

        np.testing.assert_array_equal(psi_rz, stored_zr.T)

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

    def test_diagnostic_feature_vector_reads_real_magnetic_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            real_zarr = self._write_real_shot(root, "11772")
            row = {
                "sample_id": "11772_t0.11_real",
                "source": "real",
                "shot_id": "11772",
                "data_path": str(real_zarr),
                "target_time": 0.11,
            }

            names = diagnostic_feature_names([row])
            values = diagnostic_feature_vector(row, names)

            self.assertEqual(
                names,
                [
                    "target_time",
                    "magnetics_ip",
                    "flux_loop_FL1",
                    "flux_loop_FL2",
                    "pickup_CCBV_CCBV01",
                    "pickup_OBR_OBR01",
                    "pickup_OBV_OBV01",
                    "coil_active_P2",
                    "coil_active_SOL",
                ],
            )
            np.testing.assert_allclose(values, [0.11, 1100.0, 1.2, 2.2, 3.2, 4.2, 5.2, 22.0, 12.0])

    def test_diagnostic_feature_names_drop_nonfinite_channels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            real_zarr = self._write_real_shot(root, "11772")
            z = zarr.open_group(str(real_zarr), mode="a")
            z["magnetics"]["flux_loop_flux"][0, 1] = np.nan
            rows = [
                {
                    "sample_id": "11772_t0.11_real",
                    "source": "real",
                    "shot_id": "11772",
                    "data_path": str(real_zarr),
                    "target_time": 0.11,
                }
            ]

            names = diagnostic_feature_names(rows)

            self.assertNotIn("flux_loop_FL1", names)
            self.assertIn("flux_loop_FL2", names)

    def test_real_active_coil_currents_are_interpolated_at_target_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            real_zarr = self._write_real_shot(Path(tmp), "11772")
            row = {
                "sample_id": "11772_t0.105_real",
                "source": "real",
                "shot_id": "11772",
                "data_path": str(real_zarr),
                "target_time": 0.105,
            }

            values = diagnostic_feature_vector(
                row,
                ["coil_active_SOL", "coil_active_P2"],
            )

        np.testing.assert_allclose(values, [11.0, 21.0])

    def test_manifest_dataset_can_use_magnetic_diagnostic_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            real_zarr = self._write_real_shot(root, "11772")
            fit_path = self._write_fit(root, "11772")
            rows = [
                {
                    "sample_id": "11772_t0.11_real",
                    "source": "real",
                    "shot_id": "11772",
                    "data_path": str(real_zarr),
                    "fit_path": str(fit_path),
                    "target_time": 0.11,
                }
            ]

            dataset = ManifestWindowDataset.from_rows(rows, input_mode=INPUT_MAGNETIC_DIAGNOSTICS)

            self.assertIn("flux_loop_FL1", dataset.feature_names)
            self.assertIn("pickup_OBR_OBR01", dataset.feature_names)
            self.assertNotIn("alpha_0", dataset.feature_names)
            self.assertEqual(dataset[0]["emb_chunks"][0].shape, (len(dataset.feature_names),))

    def test_diagnostic_feature_vector_reads_synthetic_magnetic_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            synthetic_dir = self._write_synthetic_sample(root)
            diagnostics_path = write_synthetic_diagnostics(
                synthetic_dir / "diagnostics.npz",
                target_time=0.10,
                magnetics_ip=120_000.0,
                flux_loop_names=["FL1", "FL2"],
                flux_loop_values=[1.5, 2.5],
                pickup_names=["CCBV01", "OBR01", "OBV01"],
                pickup_families=["CCBV", "OBR", "OBV"],
                pickup_values=[3.5, 4.5, 5.5],
                active_coil_currents={"SOL": 30.0, "P2": 40.0},
                flux_loop_scale=2.0 * np.pi,
            )
            row = {
                "sample_id": "11772_t0.10_v000",
                "source": "synthetic",
                "shot_id": "11772_t0.10_v000",
                "parent_shot": "11772",
                "data_path": str(synthetic_dir),
                "equilibrium_path": str(synthetic_dir / "equilibrium.npz"),
                "diagnostics_path": str(diagnostics_path),
                "target_time": 0.10,
            }

            names = diagnostic_feature_names([row])
            values = diagnostic_feature_vector(row, names)

        self.assertEqual(
            names,
            [
                "target_time",
                "magnetics_ip",
                "flux_loop_FL1",
                "flux_loop_FL2",
                "pickup_CCBV_CCBV01",
                "pickup_OBR_OBR01",
                "pickup_OBV_OBV01",
                "coil_active_P2",
                "coil_active_SOL",
            ],
        )
        np.testing.assert_allclose(
            values,
            [0.10, 120_000.0, 1.5, 2.5, 3.5, 4.5, 5.5, 40.0, 30.0],
        )

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

    def test_build_manifest_datasets_honors_explicit_validation_shots(self):
        rows = [
            {
                "sample_id": f"{shot}_sample",
                "source": "synthetic",
                "shot_id": f"{shot}_sample",
                "parent_shot": shot,
                "target_time": 0.1,
                "Ip": 1.0,
                "fvac": 0.4,
                "alpha": [1.0, 2.0, 3.0],
                "beta": [4.0, 5.0, 6.0],
                "coil_currents": {"active": {"SOL": float(index)}},
            }
            for index, shot in enumerate(["11766", "11768", "11775", "11780"])
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index, row in enumerate(rows):
                sample = root / row["sample_id"]
                sample.mkdir()
                np.savez_compressed(
                    sample / "equilibrium.npz",
                    psi=np.full((65, 65), index, dtype=np.float32),
                )
                row["data_path"] = str(sample)
                row["equilibrium_path"] = str(sample / "equilibrium.npz")

            train, val = build_manifest_datasets(
                rows,
                val_shots=["11768", "11775", "11780"],
            )

        self.assertEqual([row["parent_shot"] for row in train.rows], ["11766"])
        self.assertEqual(
            [row["parent_shot"] for row in val.rows],
            ["11768", "11775", "11780"],
        )

    def test_build_manifest_datasets_uses_explicit_common_feature_names(self):
        rows = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index in range(3):
                case_root = root / f"case{index}"
                case_root.mkdir()
                sample = self._write_synthetic_sample(case_root)
                diagnostics_path = write_synthetic_diagnostics(
                    sample / "diagnostics.npz",
                    target_time=0.10,
                    magnetics_ip=100_000.0 + index,
                    flux_loop_names=["FL1", "FL2"],
                    flux_loop_values=[1.0 + index, 2.0 + index],
                    pickup_names=["OBR01"],
                    pickup_families=["OBR"],
                    pickup_values=[3.0 + index],
                    active_coil_currents={"SOL": 10.0 + index},
                    flux_loop_scale=2.0 * np.pi,
                )
                rows.append(
                    {
                        "sample_id": f"s{index}",
                        "source": "synthetic",
                        "shot_id": f"s{index}",
                        "parent_shot": f"s{index}",
                        "target_time": 0.10,
                        "data_path": str(sample),
                        "equilibrium_path": str(sample / "equilibrium.npz"),
                        "diagnostics_path": str(diagnostics_path),
                    }
                )

            common_names = [
                "target_time",
                "magnetics_ip",
                "flux_loop_FL1",
            ]
            train, val = build_manifest_datasets(
                rows,
                val_fraction=0.34,
                seed=7,
                input_mode=INPUT_MAGNETIC_DIAGNOSTICS,
                feature_names=common_names,
            )

        self.assertEqual(train.feature_names, common_names)
        self.assertEqual(val.feature_names, common_names)

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
