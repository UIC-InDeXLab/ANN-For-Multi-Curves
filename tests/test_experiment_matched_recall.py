import csv
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
import uuid

import numpy as np

from fresh.e1 import write_json
from fresh.experiment_matched_recall import (matched_recall_rows, require_clean_tree,
                                             validate_l_values, worker)
from fresh.poc import Joint, PerCurve, dataset
from fresh.regimes import ground_truth


PROJECT = Path(__file__).resolve().parents[1]


def small_config(reference_root, output_root):
    return dict(dataset_seed=2026101217, m=8, d=2, rho=0.5, relevance_radius=1.5,
                tuning_queries=2, test_queries=4, warmup_queries=1, planted_per_query=1,
                plant_anchor_n=24, query_deadline_seconds=60, build_deadline_seconds=60,
                probe_chunk=1000, verification_chunk=1000, index_budget_bytes=10**9,
                peak_ram_cap_bytes=10**9, L_values=(1, 2), reference_results_root=str(reference_root),
                output_root=str(output_root))


def selection_rows(X, seed):
    blocking = Joint(X, 64.0, 2, seed, (2,), False)
    probe_all = Joint(X, 64.0, 2, seed, (1, 1), False)
    intersection = PerCurve(X, 64.0, 2, seed)
    return {
        "blocking": dict(config_id="blocking-u", method="blocking-family", delta=64.0,
                           blocks=[2], swapped=False, stored=2, probed=1, L=2,
                           index_bytes=blocking.index_bytes),
        "probe_all": dict(config_id="probe-u", method="blocking-family", delta=64.0,
                            blocks=[1, 1], swapped=False, stored=1, probed=2, L=2,
                            index_bytes=probe_all.index_bytes),
        "per_curve-intersection": dict(config_id="pc-u", method="per_curve-intersection", delta=64.0,
                                        blocks=None, swapped=False, stored=None, probed=2, L=2,
                                        index_bytes=intersection.index_bytes)
    }


class MatchedRecallExperimentTests(unittest.TestCase):
    def setUp(self):
        self.root = PROJECT / (".matched-recall-test-" + uuid.uuid4().hex)
        self.root.mkdir()
        self.reference_root = self.root / "reference"
        self.reference_root.mkdir()
        self.output_root = self.root / "output"
        self.output_root.mkdir()
        write_json(self.output_root / "manifest.json", dict(commit="unit", config_sha256="unit"))

    def tearDown(self):
        shutil.rmtree(self.root)

    def make_fixture(self):
        config = small_config(self.reference_root, self.output_root)
        case = dict(id="unit-k2-seed1701", case_id="unit-k2", kind="conjunctive", n=24, k=2,
                    frequency=0.18, index_seed=1701)
        X, Q, info = dataset(config, case)
        ids, distances, _ = ground_truth(X, Q, config["relevance_radius"])
        folder = self.reference_root / case["id"]
        folder.mkdir()
        write_json(folder / "dataset.json", dict(info, **case))
        np.savez(folder / "ground-truth.npz", queries=Q, ids=ids, distances=distances)
        write_json(folder / "frozen-main-selections.json", selection_rows(X, case["index_seed"]))
        return config, case

    def test_worker_runs_tiny_fixture_and_grows_index_with_L(self):
        config, case = self.make_fixture()
        folder = self.output_root / case["id"]
        folder.mkdir()
        worker(config, case, folder)
        rows = json.loads((folder / "main-results.json").read_text())
        self.assertEqual(len(rows), 6)
        self.assertTrue(all(row["status"] == "complete" for row in rows))
        by_method = {}
        for row in rows:
            by_method.setdefault(row["display_method"], []).append(row)
        for method, group in by_method.items():
            ordered = sorted(group, key=lambda row: row["target_L"])
            self.assertEqual([row["target_L"] for row in ordered], [1, 2])
            self.assertLessEqual(ordered[0]["index_bytes"], ordered[1]["index_bytes"], method)
            self.assertTrue(0.0 <= ordered[0]["recall_at_1"] <= 1.0)
        with (folder / "test-queries.csv").open() as handle:
            queries = list(csv.DictReader(handle))
        self.assertEqual(len(queries), 6 * config["test_queries"])

    def test_matched_recall_rule_uses_smallest_qualifying_L(self):
        aggregates = [
            dict(case_id="c", display_method="blocking", target_L=1, mean_recall_at_1=0.88,
                 median_seed_median_ms=1.0, median_seed_p95_ms=2.0, median_index_mb=3.0),
            dict(case_id="c", display_method="blocking", target_L=2, mean_recall_at_1=0.95,
                 median_seed_median_ms=1.5, median_seed_p95_ms=2.5, median_index_mb=4.0),
            dict(case_id="c", display_method="blocking", target_L=4, mean_recall_at_1=0.99,
                 median_seed_median_ms=2.0, median_seed_p95_ms=3.0, median_index_mb=5.0),
        ]
        matched = matched_recall_rows(aggregates, (0.9, 0.95, 0.999))
        self.assertEqual(matched[0]["target_L"], 2)
        self.assertEqual(matched[1]["target_L"], 2)
        self.assertEqual(matched[2]["status"], "not reached")

    def test_validation_and_dirty_tree_guard(self):
        self.assertEqual(validate_l_values([1, 2, 4]), (1, 2, 4))
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            validate_l_values([1, 1, 2])
        with patch("subprocess.check_output", return_value=" M changed\n"):
            with self.assertRaisesRegex(RuntimeError, "clean frozen source tree"):
                require_clean_tree()


if __name__ == "__main__":
    unittest.main()
