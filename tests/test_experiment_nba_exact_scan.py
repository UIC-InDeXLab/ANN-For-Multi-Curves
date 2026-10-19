import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from fresh.e1 import write_json
from fresh.experiment_nba_exact_scan import run_exact_scan, validate_exact_rows


TEST_ROOT = Path(__file__).resolve().parents[1] / ".test-work"


class ExactScanTests(unittest.TestCase):
    def test_tie_aware_validation_accepts_alternative_optimal_parent(self):
        rows = [dict(id=1, distance=0.0, timeout=False, seconds=0.1, candidates=2, verified=2,
                     probes=0, postings=0, found=True, exact_nn=True, ratio=1.0)]
        validated = validate_exact_rows(rows, np.array([0]), np.array([0.0]))
        self.assertTrue(validated[0]["tie_aware_exact"])
        self.assertFalse(validated[0]["strict_ground_truth_id"])

    def test_small_fixture_scan_writes_summary_and_repetition_logs(self):
        TEST_ROOT.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            root = Path(temporary)
            data_root = root / "data"
            output = root / "output"
            data_root.mkdir()
            X = np.array([
                [[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], [[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]],
                [[[0.0, 0.0], [1.0, 0.2], [2.0, 0.4]], [[0.0, 1.0], [1.0, 1.2], [2.0, 1.4]]],
                [[[3.0, 0.0], [4.0, 0.0], [5.0, 0.0]], [[3.0, 1.0], [4.0, 1.0], [5.0, 1.0]]],
            ], dtype=np.float64)
            Qtest = np.array([X[1], X[2]], dtype=np.float64)
            Qtune = np.array([X[0], X[1]], dtype=np.float64)
            np.savez(data_root / "database.npz", curves=X)
            np.savez(data_root / "test.npz", curves=Qtest)
            np.savez(data_root / "tune.npz", curves=Qtune)
            np.savez(data_root / "test-ground-truth.npz", ids=np.array([1, 2], dtype=np.int64),
                     distances=np.array([0.0, 0.0], dtype=np.float64))
            config = {
                "experiment": "unit-exact-scan",
                "data_directory": str(data_root),
                "warmup_data_directory": str(data_root),
                "database_file": "database.npz",
                "test_file": "test.npz",
                "ground_truth_file": "test-ground-truth.npz",
                "warmup_file": "tune.npz",
                "expected_database_size": 3,
                "expected_test_queries": 2,
                "k": 2,
                "m": 3,
                "d": 2,
                "warmup_queries": 2,
                "warmup_source": "unit-test tuning queries",
                "verification_chunk": 2,
                "query_deadline_seconds": 60.0,
                "agreement_tolerance": 1e-8,
                "repetitions": 2,
            }
            output.mkdir()
            summary = run_exact_scan(config, output)
            self.assertEqual(summary["exact_agreement_count"], 4)
            self.assertEqual(summary["strict_ground_truth_id_count"], 4)
            self.assertEqual(summary["repetitions"], 2)
            self.assertEqual(summary["queries_per_repetition"], 2)
            self.assertEqual(summary["warmup_source"], "unit-test tuning queries")
            self.assertTrue((output / "repetition-001.json").is_file())
            self.assertTrue((output / "repetition-002.json").is_file())
            first = json.loads((output / "repetition-001.json").read_text())
            self.assertEqual(len(first["rows"]), 2)
            self.assertTrue(all(row["tie_aware_exact"] for row in first["rows"]))


if __name__ == "__main__":
    unittest.main()
