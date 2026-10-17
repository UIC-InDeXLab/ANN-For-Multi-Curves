from itertools import product
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from curveset import core
from curveset.index import make_index
from curveset.orders import partitions
from fresh.e1 import build_index
from fresh.regimes import (PerCurve, dataset, family_first_hits, ground_truth, joint_query,
                           timed_queries, worker, approximation_summary)
from fresh.e1 import write_json


def small_config():
    return dict(query_deadline_seconds=60, probe_chunk=4, verification_chunk=4)


class RegimeTests(unittest.TestCase):
    def test_family_tuning_matches_actual_index_for_both_roles(self):
        rng = np.random.default_rng(231)
        for k in (2, 3, 4):
            NN = rng.normal(size=(5, k, 6, 2))
            Q = np.stack([P[rng.permutation(k)] + rng.normal(0, .04, P.shape) for P in NN])
            shifts = np.random.default_rng(17).uniform(0, 2, (3, k, 2))
            for sizes, swapped in product(partitions(k), (False, True)):
                hits = family_first_hits(Q, NN, shifts, 2, np.array(sizes), swapped)
                for q in range(len(Q)):
                    index = build_index(make_index("blocking", k, 2, 2, 3, sizes=sizes,
                                                   swapped=swapped, seed=17), NN[q:q+1])
                    self.assertEqual(hits[q] <= 3,
                                     joint_query(index, NN[q:q+1], Q[q], small_config())["id"] == 0)

    def test_family_tuner_cache_reloads_safely_in_new_process(self):
        code = (
            "import numpy as np; from fresh.regimes import family_first_hits; "
            "P=np.arange(72,dtype=float).reshape(1,6,6,2); Q=P[:,::-1].copy(); "
            "S=np.random.default_rng(17).uniform(0,4,(2,6,2)); "
            "assert family_first_hits(Q,P,S,4,np.array([4,1,1]),False).tolist()==[1]"
        )
        for _ in range(2):
            result = subprocess.run([sys.executable, "-c", code], env=os.environ.copy(),
                                    capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_per_curve_parent_deduplication_and_repeated_query(self):
        X = np.zeros((4, 3, 4, 2))
        X[0, :, :, 0] = np.array([10, 20, 20])[:, None]
        X[1, :, :, 0] = np.array([10, 10, 20])[:, None]
        X[2] += 100
        X[3] += 200
        Q = X[1].copy()
        index = PerCurve(X, 2, 3, 5)
        for _ in range(3):
            complete = index.query(X, Q, small_config())
            self.assertEqual(set(index.candidates[:complete["candidates"]]), {0, 1})
            self.assertEqual(complete["id"], 1)
            self.assertEqual(complete["distance"], 0)
            rarest = index.query(X, Q, small_config(), rarest=True, active_L=1)
            self.assertEqual(rarest["id"], 1)
            self.assertEqual(rarest["distance"], 0)
            timed_config = dict(small_config(), query_deadline_seconds=0)
            self.assertTrue(index.query(X, Q, timed_config)["timeout"])
            self.assertEqual(index.query(X, Q, small_config())["id"], 1)
        self.assertEqual(core.dmatch(X[0], Q), 10)

    def test_ground_truth_matches_exhaustive_reference(self):
        rng = np.random.default_rng(99)
        X = rng.normal(size=(16, 3, 5, 2))
        Q = X[:3].copy()
        ids, distances, relevant = ground_truth(X, Q, 1.5)
        for q in range(len(Q)):
            values = np.array([core.dmatch(P, Q[q]) for P in X])
            self.assertAlmostEqual(distances[q], values.min())
            self.assertAlmostEqual(values[ids[q]], values.min())
            self.assertEqual(relevant[q], (values <= 1.5).sum())

    def test_nested_dataset_and_queries_are_preserved(self):
        path = Path(__file__).resolve().parents[1] / "configs" / "regimes-pilot-20261012.json"
        config = json.loads(path.read_text())
        case = dict(id="unit-small", kind="conjunctive", n=10000, k=3, frequency=.18)
        X, Q, info = dataset(config, case)
        larger, other, _ = dataset(config, dict(case, n=10001))
        self.assertTrue(np.array_equal(X, larger[:10000]))
        self.assertTrue(np.array_equal(Q, other))
        self.assertEqual(info["planted_sets"], 3*len(Q))

    def test_empty_answers_are_misses_not_successful_defaults(self):
        query = lambda Q: dict(id=-1, distance=np.inf, timeout=False, seconds=.001)
        rows, complete = timed_queries(query, np.zeros((2, 2, 4, 2)), np.ones(2), small_config())
        self.assertTrue(complete)
        self.assertTrue(all(not row["found"] and not row["exact_nn"] for row in rows))
        self.assertTrue(all(row["distance"] is None and row["ratio"] is None for row in rows))

    def test_zero_optimal_distance_and_unbounded_ratio(self):
        exact = lambda Q: dict(id=0,distance=0.,timeout=False,seconds=.001)
        wrong = lambda Q: dict(id=0,distance=1.,timeout=False,seconds=.001)
        Q = np.zeros((1,2,4,2))
        correct, _ = timed_queries(exact,Q,np.zeros(1),small_config())
        incorrect, _ = timed_queries(wrong,Q,np.zeros(1),small_config())
        self.assertEqual(correct[0]["ratio"],1)
        self.assertTrue(np.isinf(incorrect[0]["ratio"]))
        summary = approximation_summary([1.]*31+[np.inf])
        self.assertEqual(summary["mean_ratio"],"infinity")
        self.assertEqual(summary["p90_ratio"],1)
        self.assertEqual(summary["unbounded_ratio_count"],1)

    def test_end_to_end_smoke_selects_on_training_and_logs_heldout_queries(self):
        path = Path(__file__).resolve().parents[1] / "configs" / "regimes-pilot-20261012.json"
        config = json.loads(path.read_text())
        config.update(m=8, tuning_queries=8, test_queries=4, warmup_queries=1, Lmax=8, deltas=[128.0])
        case = dict(id="smoke", kind="conjunctive", n=10000, k=3, frequency=.18)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(root / "manifest.json", dict(commit="unit-fixture", config_sha256="unit-fixture"))
            write_json(root / "calibration.json", dict(entries=[1000,100000], seconds_per_probe=[2e-8,3e-8]))
            output = root / "smoke"
            output.mkdir()
            worker(config, case, output)
            results = json.loads((output / "results.json").read_text())
            selections = json.loads((output / "frozen-selections.json").read_text())
            self.assertIn("per_curve-intersection", selections)
            self.assertTrue(all(r["tuning_recall"] >= .9 for r in selections.values()))
            self.assertTrue(all(r["test_status"] == "complete" for r in results))
            self.assertTrue(all(r["test_queries"] == 4 for r in results))
            self.assertEqual(results[-1]["method"], "brute-force")


if __name__ == "__main__":
    unittest.main()
