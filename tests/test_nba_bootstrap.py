import unittest
from pathlib import Path

import numpy as np

from fresh.nba_bootstrap import EXPECTED_RECALL, build_summary, game_bootstrap, load_audited_matrices


class NBABootstrapTests(unittest.TestCase):
    def test_game_bootstrap_resamples_whole_game_blocks(self):
        values = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        games = np.array(["A", "A", "B", "B"])
        samples = np.array([[0, 0], [1, 1], [0, 1]])
        mean, interval = game_bootstrap(values, games, samples=samples)
        self.assertAlmostEqual(mean, 0.5)
        self.assertEqual(interval, [0.5, 0.5])

    def test_audited_point_estimates_match_replicated_summary(self):
        root = Path(__file__).resolve().parents[1]
        results = root / "results" / "nba-full-20261012"
        metadata = root / "data" / "nba-full-20261012" / "test-metadata.json"
        if not (results / "replicated-summary.json").is_file() or not metadata.is_file():
            self.skipTest("Reproduce the full NBA benchmark to test corpus-level bootstrap estimates")
        matrices, game_ids, seeds, summary = load_audited_matrices(results, metadata)
        self.assertEqual(len(set(game_ids)), 56)
        self.assertEqual(seeds, [2301, 2302, 2303])
        self.assertTrue(set(EXPECTED_RECALL).issubset(
            {row["method"] for row in summary["main"] if row["status"] == "complete"}))
        for method, expected in EXPECTED_RECALL.items():
            self.assertAlmostEqual(float(matrices[method]["recall"].mean()), expected)

    def test_summary_contains_expected_methods_and_paired_differences(self):
        root = Path(__file__).resolve().parents[1]
        results = root / "results" / "nba-full-20261012"
        metadata = root / "data" / "nba-full-20261012" / "test-metadata.json"
        if not (results / "replicated-summary.json").is_file() or not metadata.is_file():
            self.skipTest("Reproduce the full NBA benchmark to test corpus-level bootstrap estimates")
        summary = build_summary(results, metadata, bootstrap_repetitions=128, bootstrap_seed=7)
        self.assertEqual(summary["game_count"], 56)
        self.assertEqual(summary["query_count"], 1000)
        self.assertEqual([row["method"] for row in summary["methods"]],
                         ["blocking", "probe_all", "store_all", "per_curve-intersection"])
        self.assertEqual(len(summary["paired_recall_differences"]), 3)
        labels = {row["comparison"] for row in summary["paired_recall_differences"]}
        self.assertEqual(labels, {"Intersection minus Blocking", "Store-all minus Blocking", "Probe-all minus Blocking"})


if __name__ == "__main__":
    unittest.main()
