"""Boundary contracts: test labels, duplicate groups and real resource ledgers."""

from pathlib import Path
from contextlib import redirect_stdout
import gzip
import hashlib
import io
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import experiment as ex
from fetch import signature


class Contracts(unittest.TestCase):
    def test_identical_features_share_group_despite_signed_zero_or_nan(self):
        self.assertEqual(
            signature([0.0, np.nan, 3]), signature([-0.0, float("nan"), 3])
        )
        self.assertNotEqual(signature([0.0, np.nan, 3]), signature([0.0, np.nan, 4]))

    def test_candidate_pool_is_finite_shared_and_repeatable(self):
        a = ex.candidate_pool(44140, 110)
        self.assertEqual(a, ex.candidate_pool(44140, 110))
        self.assertEqual(len(a), 18)
        self.assertEqual(len({c["id"] for c in a}), 18)
        self.assertTrue(
            {c["id"] for c in a}.issubset({c["id"] for c in ex.parameters()})
        )
        self.assertEqual(len(ex.parameters()), 36)

    def test_promotion_ties_use_candidate_identity(self):
        rows = [
            {"candidate": "b", "mse": 1},
            {"candidate": "a", "mse": 1},
            {"candidate": "c", "mse": 2},
        ]
        self.assertEqual([x["candidate"] for x in ex.promote(rows, 2)], ["a", "b"])

    def test_same_feature_group_never_changes_partition_with_target(self):
        group = signature([2, 5, 7])
        for seed in ex.SEEDS:
            self.assertEqual(
                ex.group_partition(group, seed),
                ex.group_partition(signature([2, 5, 7]), seed),
            )

    def fixture(self, root, change_test=False):
        frame = pd.DataFrame(
            {
                "source_row": range(2000),
                "x": np.arange(1, 2001, dtype=float),
                "target": np.arange(2000, dtype=float),
            }
        )
        frame.insert(1, "group_id", [signature([x]) for x in frame.x])
        test = frame.group_id.map(lambda g: ex.group_partition(g, 110)) == "test"
        if change_test:
            frame.loc[test, "target"] += 1e9
        path = root / "data/panels/fixture.csv.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(gzip.compress(frame.to_csv(index=False).encode(), mtime=0))
        return {
            "id": 44140,
            "role": "development",
            "target": "target",
            "features": ["x"],
            "panel_file": "data/panels/fixture.csv.gz",
            "panel_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "quality": {"selected_rows": len(frame)},
        }

    @staticmethod
    def model_spy(config, iterations, x, y, target, seed):
        assert max(y) < 1e8, "A held test target entered fitting"
        pred = np.full(len(target), float(np.mean(y)) + config["learning_rate"])
        return pred, {
            "target_center": float(np.mean(y)),
            "target_scale": 1.0,
            "actual_iterations": iterations,
            "elapsed_seconds": 0.0,
        }

    def test_test_labels_cannot_change_selection_or_resource_ledger(self):
        selections = []
        for change in [False, True]:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                record = self.fixture(root, change)
                with patch.object(ex, "ROOT", root), patch.object(
                    ex, "fit_predict", self.model_spy
                ), redirect_stdout(io.StringIO()):
                    ex.run_search(record, 110)
                folder = root / "results/runs/44140/110"
                selection = json.loads((folder / "selection.json").read_text())
                selections.append(selection["selections"])
                ledger = pd.read_csv(folder / "trials.csv")
                self.assertEqual(
                    ledger.groupby("strategy").requested_iterations.sum().to_dict(),
                    {"audit": 1620, "halving": 540, "random": 540},
                )
                self.assertEqual(
                    ledger.groupby("strategy").size().to_dict(),
                    {"audit": 18, "halving": 26, "random": 6},
                )
                with patch.object(ex, "ROOT", root), self.assertRaisesRegex(
                    RuntimeError, "already exists"
                ):
                    ex.run_search(record, 110)
        self.assertEqual(selections[0], selections[1])

    def test_failed_fit_is_retained_and_stops_unmatched_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = self.fixture(root)
            with patch.object(ex, "ROOT", root), patch.object(
                ex, "fit_predict", side_effect=ValueError("fixture failure")
            ), self.assertRaisesRegex(ValueError, "fixture failure"):
                ex.run_search(record, 110)
            folder = root / "results/runs/44140/110"
            self.assertTrue((folder / "failure.json").exists())
            self.assertFalse((folder / "selection.json").exists())
            ledger = pd.read_csv(folder / "trials.csv")
            self.assertEqual(len(ledger), 1)
            self.assertEqual(ledger.iloc[0]["status"], "failure")


if __name__ == "__main__":
    unittest.main()
