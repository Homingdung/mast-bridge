from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "plot_tokamind_diagnostics_losses.py"
SPEC = importlib.util.spec_from_file_location("plot_tokamind_diagnostics_losses", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PlotTokamindDiagnosticsLossesTests(unittest.TestCase):
    def test_load_loss_rows_reads_training_history(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            run_dir.mkdir()
            summary = {
                "history": {
                    "stages": {
                        "manifest_scratch": [
                            {"epoch_global": 1, "train_loss": 1.0, "val_loss": 1.2},
                            {"epoch_global": 2, "train_loss": 0.5, "val_loss": 0.7},
                        ]
                    }
                }
            }
            (run_dir / "manifest_training_summary.json").write_text(
                json.dumps(summary),
                encoding="utf-8",
            )

            rows = MODULE.load_loss_rows(run_dir, "Real only")

        self.assertEqual(
            rows,
            [
                {
                    "experiment": "Real only",
                    "epoch": 1,
                    "train_loss": 1.0,
                    "val_loss": 1.2,
                },
                {
                    "experiment": "Real only",
                    "epoch": 2,
                    "train_loss": 0.5,
                    "val_loss": 0.7,
                },
            ],
        )

    def test_load_loss_rows_rejects_missing_stage_history(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            run_dir.mkdir()
            (run_dir / "manifest_training_summary.json").write_text(
                json.dumps({"history": {"stages": {}}}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "manifest_scratch"):
                MODULE.load_loss_rows(run_dir, "Real only")

    def test_load_loss_rows_rejects_nonfinite_losses(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            run_dir.mkdir()
            summary = {
                "history": {
                    "stages": {
                        "manifest_scratch": [
                            {
                                "epoch_global": 1,
                                "train_loss": float("nan"),
                                "val_loss": 1.0,
                            }
                        ]
                    }
                }
            }
            (run_dir / "manifest_training_summary.json").write_text(
                json.dumps(summary),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "non-finite"):
                MODULE.load_loss_rows(run_dir, "Real only")


if __name__ == "__main__":
    unittest.main()
