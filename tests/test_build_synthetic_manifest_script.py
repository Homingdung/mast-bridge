import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "build_synthetic_manifest.py"
SPEC = importlib.util.spec_from_file_location("build_synthetic_manifest", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class BuildSyntheticManifestScriptTests(unittest.TestCase):
    def _write_sample(self, root: Path, name: str, tolerance: float, converged: bool) -> None:
        sample = root / name
        sample.mkdir()
        np.savez_compressed(
            sample / "equilibrium.npz",
            psi=np.zeros((65, 65)),
            R=np.zeros((65, 65)),
            Z=np.zeros((65, 65)),
            psi_axis=0.0,
            psi_bndry=1.0,
        )
        (sample / "metadata.json").write_text(
            json.dumps(
                {
                    "source": "synthetic",
                    "parent_shot": name.split("_", maxsplit=1)[0],
                    "target_time": 0.10,
                    "solver_status": "success" if converged else "non_converged",
                    "solver_converged": converged,
                    "solver_final_tolerance": tolerance,
                }
            ),
            encoding="utf-8",
        )

    def test_writes_accepted_manifest_and_rejected_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_sample(root, "11772_t0.10_v000", 9e-9, True)
            self._write_sample(root, "11772_t0.10_v001", 1.1e-8, True)
            manifest = root / "accepted.jsonl"
            rejected = root / "rejected.jsonl"

            exit_code = MODULE.main(
                [
                    "--synthetic-root",
                    str(root),
                    "--output",
                    str(manifest),
                    "--rejected-output",
                    str(rejected),
                    "--task",
                    "task_1-3",
                    "--max-solver-tolerance",
                    "1e-8",
                ]
            )

            accepted_rows = [json.loads(line) for line in manifest.read_text().splitlines()]
            rejected_rows = [json.loads(line) for line in rejected.read_text().splitlines()]

        self.assertEqual(exit_code, 0)
        self.assertEqual([row["sample_id"] for row in accepted_rows], ["11772_t0.10_v000"])
        self.assertEqual(rejected_rows[0]["sample_id"], "11772_t0.10_v001")
        self.assertEqual(rejected_rows[0]["reason"], "solver_tolerance_above_threshold")


if __name__ == "__main__":
    unittest.main()
