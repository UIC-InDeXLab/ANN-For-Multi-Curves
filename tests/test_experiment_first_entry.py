import json
from pathlib import Path
import shutil
import unittest
import uuid

import numpy as np

from fresh.e1 import write_json
from fresh.experiment_first_entry import (aggregate_case_rows, build_first_entry_joint, choose_first_entry_configuration,
                                          first_entry_count, first_hit_query, keep_first_entries, worker)
from fresh.poc import Joint, dataset
from fresh.regimes import ground_truth
from test_experiment_matched_recall import selection_rows, small_config


PROJECT = Path(__file__).resolve().parents[1]


class FirstEntryTests(unittest.TestCase):
    def setUp(self):
        self.root = PROJECT / (".first-entry-test-" + uuid.uuid4().hex)
        self.root.mkdir()
        self.reference_root = self.root / "reference"
        self.reference_root.mkdir()
        self.output_root = self.root / "output"
        self.output_root.mkdir()
        write_json(self.output_root / "manifest.json", dict(commit="unit", config_sha256="unit"))

    def tearDown(self):
        shutil.rmtree(self.root)

    def fixture(self):
        config = small_config(self.reference_root, self.output_root)
        case = dict(id="unit-k2-seed1701", case_id="unit-k2", kind="conjunctive", n=24, k=2,
                    frequency=0.18, index_seed=1701)
        X, Q, info = dataset(config, case)
        ids, distances, _ = ground_truth(X, Q, config["relevance_radius"])
        folder = self.reference_root / case["id"]
        folder.mkdir()
        write_json(folder / "dataset.json", dict(info, **case))
        np.savez(folder / "ground-truth.npz", queries=Q, ids=ids, distances=distances)
        selections = selection_rows(X, case["index_seed"])
        selections["store_all"] = dict(selections["blocking"], config_id="store-u")
        write_json(folder / "frozen-main-selections.json", selections)
        return config, case, X, Q

    def test_one_entry_per_key_with_lowest_identifier(self):
        config, case, X, Q = self.fixture()
        full = Joint(X, 64.0, 2, 1701, (2,), False)
        for table in full.tables:
            reduced = keep_first_entries(table)
            keys, ids = table.keys[:table.N], table.ids[:table.N]
            self.assertEqual(reduced.N, first_entry_count(keys))
            self.assertEqual(len(set(reduced.keys[:reduced.N].tolist())), reduced.N)
            for key, kept in zip(reduced.keys[:reduced.N], reduced.ids[:reduced.N]):
                self.assertEqual(int(kept), int(ids[keys == key].min()))
            self.assertLessEqual(reduced.N, table.N)

    def test_streaming_build_matches_reduction_of_the_full_index(self):
        config, case, X, Q = self.fixture()
        full = Joint(X, 64.0, 2, 1701, (2,), False)
        streamed = build_first_entry_joint(X, 64.0, 2, 1701, (2,), False)
        for a, b in zip(full.tables, streamed.tables):
            reduced = keep_first_entries(a)
            self.assertTrue(np.array_equal(reduced.keys, b.keys))
            self.assertTrue(np.array_equal(reduced.ids, b.ids))

    def test_stored_curve_query_returns_a_collision_and_index_shrinks(self):
        config, case, X, Q = self.fixture()
        index = build_first_entry_joint(X, 64.0, 2, 1701, (2,), False)
        full = Joint(X, 64.0, 2, 1701, (2,), False)
        self.assertLessEqual(index.index_bytes, full.index_bytes)
        for p in range(len(X)):
            row = first_hit_query(index, X, X[p], config)
            self.assertGreaterEqual(row["id"], 0)
            self.assertEqual(row["distance"], row["distance"])
        one_table = first_hit_query(index, X, X[0], config, L=1)
        self.assertGreaterEqual(one_table["id"], 0)

    def test_configuration_rule_prefers_few_tables_then_best_recall(self):
        candidates = [dict(delta=8.0, L=3, tuning_recall=0.96, bytes=30),
                      dict(delta=16.0, L=2, tuning_recall=0.95, bytes=50),
                      dict(delta=32.0, L=2, tuning_recall=0.95, bytes=40)]
        chosen, reached = choose_first_entry_configuration(candidates, 0.95)
        self.assertTrue(reached)
        self.assertEqual((chosen["delta"], chosen["L"]), (32.0, 2))
        chosen, reached = choose_first_entry_configuration(candidates, 0.99)
        self.assertFalse(reached)
        self.assertEqual((chosen["delta"], chosen["L"]), (8.0, 3))

    def test_worker_runs_all_three_methods_with_tuning(self):
        config, case, X, Q = self.fixture()
        config.update(deltas=(32.0, 64.0), L_max=2, store_all_L_max=2, tuning_acceptance_recall=0.5)
        folder = self.output_root / case["id"]
        folder.mkdir()
        worker(config, case, folder)
        rows = json.loads((folder / "main-results.json").read_text())
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row["status"] == "complete" for row in rows))
        self.assertTrue(all(0 <= row["empty_rate"] <= 1 and row["delta"] in (32.0, 64.0) for row in rows))
        self.assertEqual(len(aggregate_case_rows(rows)), 3)


if __name__ == "__main__":
    unittest.main()
