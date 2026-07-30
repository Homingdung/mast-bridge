import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from mast_bridge.dataset.splits import assign_parent_shot_splits, split_for_row


class SplitTests(unittest.TestCase):
    def test_synthetic_uses_parent_shot_split(self):
        rows = [
            {"sample_id": "11771", "source": "real", "shot_id": "11771"},
            {
                "sample_id": "11771_t0.16_v000",
                "source": "synthetic",
                "shot_id": "11771_t0.16_v000",
                "parent_shot": "11771",
            },
            {"sample_id": "11772", "source": "real", "shot_id": "11772"},
            {"sample_id": "11773", "source": "real", "shot_id": "11773"},
        ]
        assignments = assign_parent_shot_splits(
            rows, train_fraction=0.67, val_fraction=0.0, seed=7
        )

        self.assertEqual(
            split_for_row(rows[1], assignments),
            split_for_row(rows[0], assignments),
        )

    def test_requires_parent_for_synthetic(self):
        rows = [{"sample_id": "bad", "source": "synthetic", "shot_id": "bad"}]

        with self.assertRaisesRegex(ValueError, "parent_shot"):
            assign_parent_shot_splits(
                rows, train_fraction=0.8, val_fraction=0.1, seed=1
            )

    def test_keeps_validation_split_for_two_parent_shots(self):
        rows = [
            {"sample_id": "11772", "source": "real", "shot_id": "11772"},
            {"sample_id": "11773", "source": "real", "shot_id": "11773"},
        ]

        assignments = assign_parent_shot_splits(
            rows, train_fraction=0.8, val_fraction=0.2, seed=1
        )

        self.assertEqual(set(assignments.values()), {"train", "val"})

    def test_validation_prefers_smaller_parent_group(self):
        rows = [
            {"sample_id": f"11772_{index}", "source": "real", "shot_id": "11772"}
            for index in range(8)
        ]
        rows.extend(
            {"sample_id": f"11773_{index}", "source": "real", "shot_id": "11773"}
            for index in range(2)
        )

        assignments = assign_parent_shot_splits(
            rows, train_fraction=0.8, val_fraction=0.2, seed=1
        )

        self.assertEqual(assignments["11773"], "val")
        self.assertEqual(assignments["11772"], "train")


if __name__ == "__main__":
    unittest.main()
