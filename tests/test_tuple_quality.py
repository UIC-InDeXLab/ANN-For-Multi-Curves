import unittest

import numpy as np

from curveset import core
from fresh.tuple_quality import all_tuple_statistics, exact_distance, frechet_squared


class TupleQualityTests(unittest.TestCase):
    def test_squared_frechet_matches_original(self):
        rng = np.random.default_rng(808)
        for _ in range(200):
            d = int(rng.integers(1, 4))
            A = rng.normal(size=(int(rng.integers(1, 9)), d))
            B = rng.normal(size=(int(rng.integers(1, 9)), d))
            self.assertAlmostEqual(np.sqrt(frechet_squared(A, B)), core.frechet(A, B), places=12)

    def test_exact_matching_including_alternative_to_endpoint_match(self):
        P = np.array([[[0, 0], [100, 0], [0, 0]], [[10, 0], [0, 0], [10, 0]]], dtype=float)
        Q = np.array([[[0, 0], [0, 0], [0, 0]], [[10, 0], [100, 0], [10, 0]]], dtype=float)
        self.assertEqual(exact_distance(P, Q), 10)
        rng = np.random.default_rng(809)
        for _ in range(200):
            k = int(rng.integers(1, 6))
            d = int(rng.integers(2, 4))
            P = rng.normal(size=(k, 8, d))
            Q = rng.normal(size=(k, 8, d))
            self.assertAlmostEqual(exact_distance(P, Q), core.dmatch(P, Q), places=12)

    def test_means_use_every_tuple_including_far_ones(self):
        X = np.zeros((3, 2, 4, 2))
        X[1] += 10
        X[2] += 1000
        Q = X[:1].copy()
        stats = all_tuple_statistics(X, Q, 1)
        reference = np.array([core.dmatch(P, Q[0]) for P in X])
        self.assertAlmostEqual(stats[0, 0], reference.mean())
        self.assertEqual(stats[0, 1], reference.min())
        self.assertEqual(stats[0, 2], reference.max())
        self.assertEqual(stats[0, 3], 1)


if __name__ == "__main__":
    unittest.main()
