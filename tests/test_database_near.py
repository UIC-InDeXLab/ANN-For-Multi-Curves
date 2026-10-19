import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from fresh.e1 import arrays_hash
from fresh.experiment_database_near import count_near, dataset_digest_header
from fresh.poc import dataset, implicit_dataset
from fresh.tuple_quality import exact_distance


class DatabaseNearTests(unittest.TestCase):
    def test_exact_counts_agree_with_full_distances_including_boundary(self):
        rng = np.random.default_rng(5571)
        for k, d in ((2, 2), (3, 2), (3, 3)):
            X = rng.normal(size=(17, k, 5, d))
            queries = rng.normal(size=(4, k, 5, d))
            radii = np.array([exact_distance(X[q], query) for q, query in enumerate(queries)])
            expected = [sum(exact_distance(P, query) <= radius for P in X)
                        for query, radius in zip(queries, radii)]
            self.assertEqual(count_near(X, queries, radii).tolist(), expected)

    def test_endpoint_coverage_is_not_enough_to_count_a_neighbor(self):
        X = np.zeros((1, 3, 3, 2))
        X[0, :, :, 0] = np.array([0, 20, 20])[:, None]
        queries = np.zeros_like(X)
        queries[0, :, :, 0] = np.array([0, 0, 20])[:, None]
        self.assertEqual(count_near(X, queries, np.array([1.0])).tolist(), [0])
        X = np.zeros((1, 1, 3, 2))
        X[0, 0, 1, 0] = 100
        self.assertEqual(count_near(X, np.zeros_like(X), np.array([1.0])).tolist(), [0])

    def test_inclusive_squared_endpoint_boundary_and_invalid_radii(self):
        X = np.ones((1, 1, 2, 3))
        queries = np.zeros_like(X)
        self.assertEqual(count_near(X, queries, np.array([np.sqrt(3)])).tolist(), [1])
        with self.assertRaises(ValueError):
            count_near(X, queries, np.array([np.nan]))
        with self.assertRaises(ValueError):
            count_near(X, queries, np.empty(0))

    def test_streamed_generation_and_hash_match_materialized_frozen_contract(self):
        config = dict(dataset_seed=55, m=5, d=2, rho=.5, tuning_queries=2,
                      test_queries=4, warmup_queries=1, plant_anchor_n=10, planted_per_query=1)
        case = dict(kind="conjunctive", n=23, k=3, frequency=.18)
        X, queries, info = dataset(config, case)
        implicit, other_queries, _ = implicit_dataset(config, case)
        hasher = dataset_digest_header(implicit)
        chunks = []
        for start in range(0, len(X), 4):
            chunk = implicit.materialize(np.arange(start, min(len(X), start + 4)))
            hasher.update(chunk.tobytes())
            chunks.append(chunk)
        np.testing.assert_array_equal(X, np.concatenate(chunks))
        np.testing.assert_array_equal(queries, other_queries)
        self.assertEqual(hasher.hexdigest(), arrays_hash(X))
        self.assertEqual(hasher.hexdigest(), info["dataset_sha256"])

    def test_full_and_threshold_references_must_agree(self):
        from fresh.render_fh_blocking_paper import completed_database_reference
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            full, near, case = root / "full", root / "near", "case"
            summary = dict(case=case, complete=True, queries=1000, evaluated_queries=1000,
                           evaluated_database_entries="all", dataset_sha256="same",
                           queries_sha256="same", mean_database_within_125_fraction=.001)
            for source in (full, near):
                (source / case).mkdir(parents=True)
                (source / case / "summary.json").write_text(json.dumps(summary))
            self.assertEqual(completed_database_reference(full, near, case), summary)
            summary["mean_database_within_125_fraction"] = .002
            (near / case / "summary.json").write_text(json.dumps(summary))
            with self.assertRaises(ValueError):
                completed_database_reference(full, near, case)


if __name__ == "__main__":
    unittest.main()
