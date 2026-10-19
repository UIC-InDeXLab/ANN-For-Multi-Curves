import json
from pathlib import Path
import shutil
import unittest
import uuid

import numpy as np

from fresh.e1 import write_json
from fresh.feasibility_first_hit import build_tables, first_hit_timed, index_bytes, study_jobs, summarize, worker
from fresh.poc import dataset
from fresh.regimes import ground_truth
from test_experiment_matched_recall import selection_rows, small_config

PROJECT = Path(__file__).resolve().parents[1]


class FirstHitFeasibilityTests(unittest.TestCase):
    def setUp(self):
        self.root = PROJECT / (".fh-feasibility-test-" + uuid.uuid4().hex)
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
        config.update(deltas=(32.0, 64.0), L_max=3, L_grid=(1, 2, 3), required_success=0.5, tuning_acceptance_recall=0.5,
                      compilation_deadline_seconds=60, blocking_storage_multipliers=(1, 2), include_balanced_in_ablation=True,
                      budgets=[dict(name="big", index_budget_bytes=10**9, query_deadline_seconds=1.0),
                               dict(name="tiny", index_budget_bytes=100, query_deadline_seconds=1.0)],
                      index_seeds=(1701,), cases=[], peak_ram_cap_bytes=10**10)
        base = dict(id="unit-k2", kind="conjunctive", n=24, k=2, frequency=0.18)
        config["cases"] = [base]
        case = study_jobs(config)[0]
        X, Q, info = dataset(config, case)
        ids, distances, _ = ground_truth(X, Q, config["relevance_radius"])
        folder = self.reference_root / case["id"]
        folder.mkdir()
        write_json(folder / "dataset.json", dict(info, **case))
        np.savez(folder / "ground-truth.npz", queries=Q, ids=ids, distances=distances)
        write_json(folder / "frozen-main-selections.json", selection_rows(X, 1701))
        return config, case, X

    def test_tables_stop_at_the_byte_cap_and_query_returns_in_time(self):
        config, case, X = self.fixture()
        full = build_tables(X, 64.0, 3, 1701, (2,), False, float("inf"))
        self.assertEqual(len(full.tables), 3)
        capped = build_tables(X, 64.0, 3, 1701, (2,), False, index_bytes(full, 2))
        self.assertEqual(len(capped.tables), 2)
        row = first_hit_timed(full, X, X[0], 3, 1.0, dict(config, query_deadline_seconds=60))
        self.assertGreaterEqual(row["id"], 0)
        self.assertFalse(row["timeout"])
        late = first_hit_timed(full, X, X[0], 3, -1.0, dict(config, query_deadline_seconds=60))
        self.assertTrue(late["timeout"] or late["id"] >= 0 and late["seconds"] > -1.0)

    def test_worker_and_summary_mark_budgets(self):
        config, case, X = self.fixture()
        folder = self.output_root / case["id"]
        folder.mkdir()
        worker(config, case, folder)
        rows = json.loads((folder / "main-results.json").read_text())
        self.assertEqual({r["budget"] for r in rows}, {"big", "tiny"})
        tiny = [r for r in rows if r["budget"] == "tiny"]
        self.assertTrue(all(r["status"] != "complete" for r in tiny))
        summarize(self.output_root, config)
        summary = json.loads((self.output_root / "summary.json").read_text())
        self.assertIn("big/blocking", summary["largest_feasible_tested_k"])


if __name__ == "__main__":
    unittest.main()
