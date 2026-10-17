import csv
import json
from itertools import permutations
import shutil
from pathlib import Path
import unittest

import numpy as np

from fresh.experiment_nba_selectivity import query_curve_distances, query_curve_statistics, query_play_counts, run


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "nba_selectivity_small"
ARTIFACTS = ROOT / "tests" / "_artifacts"


def reference_frechet(A, B):
    dist = np.linalg.norm(A[:, None] - B[None, :], axis=2)
    best = np.full(dist.shape, np.inf)
    for i in range(dist.shape[0]):
        for j in range(dist.shape[1]):
            if i == 0 and j == 0:
                prev = 0.0
            else:
                prev = min(
                    best[i - 1, j] if i else np.inf,
                    best[i, j - 1] if j else np.inf,
                    best[i - 1, j - 1] if i and j else np.inf,
                )
            best[i, j] = max(prev, dist[i, j])
    return best[-1, -1]


def reference_play_hits(distances, threshold):
    hits = 0
    for play in range(distances.shape[1]):
        D = distances[:, play, :]
        ok = any(all(D[i, perm[i]] <= threshold for i in range(D.shape[0])) for perm in permutations(range(D.shape[1])))
        hits += int(ok)
    return hits


class NBASelectivityTests(unittest.TestCase):
    def setUp(self):
        ARTIFACTS.mkdir(exist_ok=True)

    def tearDown(self):
        path = ARTIFACTS / "nba-selectivity-small-output"
        if path.exists():
            shutil.rmtree(path)

    def test_distance_and_count_kernels_match_bruteforce(self):
        with np.load(FIXTURE / "database.npz", allow_pickle=False) as source:
            database = source["curves"]
        with np.load(FIXTURE / "test.npz", allow_pickle=False) as source:
            query = source["curves"][0]
        distances = query_curve_distances(query, database)
        reference = np.empty_like(distances)
        for i in range(query.shape[0]):
            for play in range(database.shape[0]):
                for slot in range(database.shape[1]):
                    reference[i, play, slot] = reference_frechet(query[i], database[play, slot])
        np.testing.assert_allclose(distances, reference)
        occ_r, occ_15, occ_2, occ_fixed, curve_fraction = query_curve_statistics(distances, 0.25, 0.375, 0.5, 5.0)
        self.assertAlmostEqual(occ_r[0], 2 / 3)
        self.assertAlmostEqual(curve_fraction[1], 1 / 6)
        hits_r, hits_15 = query_play_counts(distances, 0.25, 0.375)
        self.assertEqual(hits_r, reference_play_hits(reference, 0.25))
        self.assertEqual(hits_15, reference_play_hits(reference, 0.375))
        np.testing.assert_allclose(occ_fixed, np.array([2 / 3, 2 / 3]))

    def test_small_fixture_end_to_end_summary(self):
        config = ROOT / "tests" / "fixtures" / "nba-selectivity-small.json"
        output = ARTIFACTS / "nba-selectivity-small-output"
        summary = run(config, output)
        self.assertEqual(summary["query_count"], 2)
        self.assertEqual(summary["query_curve_count"], 4)
        with (output / "summary.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 4)
        loaded = json.loads((output / "summary.json").read_text())
        self.assertEqual(loaded["play_count_distributions"]["within_bottleneck_r"]["min"], 1)


if __name__ == "__main__":
    unittest.main()
