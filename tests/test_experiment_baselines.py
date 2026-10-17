import shutil
from pathlib import Path
import unittest

import numpy as np

from fresh.experiment_baselines import (PerCurveHnsw, RandomHyperplaneSketch, SketchHnsw,
                                        make_ann_index, select_sketch_trial, select_smallest_qualifying)
from fresh.regimes import timed_queries


def small_config():
    return dict(
        tuning_acceptance_recall=0.95,
        query_deadline_seconds=60.0,
        verification_chunk=4,
        probe_chunk=4,
        hnsw_build_threads=1,
        hnsw_M=16,
        hnsw_ef_construction=200,
        hnsw_query_ef_floor=4,
        sketch_projection_dim=16,
        sketch_space="ip",
        allow_bruteforce_fallback=True,
    )


def make_curve(value, m=32, d=2):
    curve = np.zeros((m, d), float)
    curve[:, 0] = value
    return curve


class BaselineExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1] / "tests" / "_baseline_scratch"
        if cls.root.exists():
            shutil.rmtree(cls.root)
        cls.root.mkdir(parents=True)

    @classmethod
    def tearDownClass(cls):
        if cls.root.exists():
            shutil.rmtree(cls.root)

    def workdir(self, name):
        path = self.root / name
        if path.exists():
            shutil.rmtree(path)
        path.mkdir()
        return path

    def test_per_curve_fallback_backend_and_empty_intersection(self):
        cfg = small_config()
        X = np.stack([
            np.stack([make_curve(0), make_curve(100), make_curve(200)]),
            np.stack([make_curve(10), make_curve(110), make_curve(210)]),
            np.stack([make_curve(20), make_curve(120), make_curve(220)]),
        ])
        factory = lambda space, dim, max_elements, seed, build_threads, M, ef_construction, allow_fallback=False: (
            make_ann_index(space, dim, max_elements, seed, build_threads, M, ef_construction,
                           allow_fallback=True, hnswlib_module=None)
        )
        index = PerCurveHnsw(X, cfg, 7, self.workdir("per-curve-empty"), backend_factory=factory,
                             allow_fallback=True)
        self.assertEqual(index.backend_name, "brute-force")
        query = np.stack([make_curve(0), make_curve(110), make_curve(220)])
        row = index.query(X, query, cfg, top_k=1)
        self.assertTrue(row["empty_answer"])
        self.assertFalse(row["timeout"])
        self.assertEqual(row["candidates"], 0)
        timed, complete = timed_queries(lambda Q: index.query(X, Q, cfg, top_k=1),
                                        query[None, ...], np.array([0.0]), cfg)
        self.assertTrue(complete)
        self.assertFalse(timed[0]["found"])
        self.assertFalse(timed[0]["exact_nn"])

    def test_tuning_selection_reports_not_reached(self):
        rows = [
            dict(config_id="k1", top_k=1, complete=True, tuning_recall=0.20, tuning_median_seconds=0.001),
            dict(config_id="k5", top_k=5, complete=True, tuning_recall=0.60, tuning_median_seconds=0.002),
            dict(config_id="k20", top_k=20, complete=True, tuning_recall=0.90, tuning_median_seconds=0.003),
        ]
        chosen = select_smallest_qualifying(rows, "top_k", 0.95, 20)
        self.assertEqual(chosen["status"], "not reached")
        self.assertEqual(chosen["choice"]["top_k"], 20)
        self.assertEqual(chosen["best_tuning_recall"], 0.90)

    def test_sketch_selection_prefers_smallest_qualifying_c(self):
        rows = [
            dict(config_id="b2-c100", bucket_bits=2, top_c=100, complete=True,
                 tuning_recall=0.95, tuning_median_seconds=0.010),
            dict(config_id="b4-c20", bucket_bits=4, top_c=20, complete=True,
                 tuning_recall=0.95, tuning_median_seconds=0.012),
            dict(config_id="b3-c20", bucket_bits=3, top_c=20, complete=True,
                 tuning_recall=0.95, tuning_median_seconds=0.011),
        ]
        chosen = select_sketch_trial(rows, 0.95, 100)
        self.assertEqual(chosen["status"], "qualified")
        self.assertEqual((chosen["choice"]["bucket_bits"], chosen["choice"]["top_c"]), (3, 20))

    def test_sketch_is_permutation_invariant(self):
        multicurve = np.stack([make_curve(1), make_curve(3), make_curve(5)])
        sketcher = RandomHyperplaneSketch(curve_dim=64, bucket_bits=3, projection_dim=16, seed=17)
        first = sketcher.encode_one(multicurve)
        second = sketcher.encode_one(multicurve[[2, 0, 1]])
        np.testing.assert_allclose(first, second)

    def test_deadline_path_marks_timeout(self):
        cfg = small_config()
        X = np.stack([
            np.stack([make_curve(0), make_curve(1), make_curve(2)]),
            np.stack([make_curve(50), make_curve(51), make_curve(52)]),
        ])
        factory = lambda space, dim, max_elements, seed, build_threads, M, ef_construction, allow_fallback=False: (
            make_ann_index(space, dim, max_elements, seed, build_threads, M, ef_construction,
                           allow_fallback=True, hnswlib_module=None)
        )
        index = PerCurveHnsw(X, cfg, 3, self.workdir("deadline"), backend_factory=factory,
                             allow_fallback=True)
        timed_cfg = dict(cfg, query_deadline_seconds=0.0)
        row = index.query(X, X[0], timed_cfg, top_k=1)
        self.assertTrue(row["timeout"])
        timed, complete = timed_queries(lambda Q: index.query(X, Q, timed_cfg, top_k=1),
                                        X[:1], np.array([0.0]), timed_cfg)
        self.assertFalse(complete)
        self.assertTrue(timed[0]["timeout"])

    def test_sketch_baseline_returns_exact_verified_candidate_on_fixture(self):
        cfg = small_config()
        X = np.stack([
            np.stack([make_curve(0), make_curve(10), make_curve(20)]),
            np.stack([make_curve(100), make_curve(110), make_curve(120)]),
            np.stack([make_curve(200), make_curve(210), make_curve(220)]),
        ])
        factory = lambda space, dim, max_elements, seed, build_threads, M, ef_construction, allow_fallback=False: (
            make_ann_index(space, dim, max_elements, seed, build_threads, M, ef_construction,
                           allow_fallback=True, hnswlib_module=None)
        )
        index = SketchHnsw(X, cfg, 19, 2, self.workdir("sketch"), backend_factory=factory,
                           allow_fallback=True)
        row = index.query(X, X[1], cfg, top_c=2)
        self.assertEqual(row["id"], 1)
        self.assertEqual(row["distance"], 0.0)
        self.assertGreaterEqual(row["verified"], 1)


if __name__ == "__main__":
    unittest.main()
