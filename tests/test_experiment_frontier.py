import json
from pathlib import Path
import shutil
import unittest
import uuid

import numpy as np

from fresh.e1 import write_json
from fresh.experiment_frontier import partition_specs, selections, study_jobs, worker
from fresh.poc import dataset
from fresh.regimes import ground_truth
from test_experiment_matched_recall import small_config

PROJECT = Path(__file__).resolve().parents[1]


class FrontierTests(unittest.TestCase):
    def setUp(self):
        self.root = PROJECT / (".frontier-test-" + uuid.uuid4().hex)
        self.root.mkdir()
        self.reference_root = self.root / "reference"
        self.reference_root.mkdir()
        self.output_root = self.root / "output"
        self.output_root.mkdir()
        write_json(self.output_root / "manifest.json", dict(commit="unit", config_sha256="unit"))

    def tearDown(self):
        shutil.rmtree(self.root)

    def test_all_partitions_in_both_orientations(self):
        specs = partition_specs(8)
        self.assertEqual(len(specs), 43)
        self.assertEqual(len({(sizes, swapped) for _, _, sizes, swapped in specs}), 43)
        self.assertTrue(all(a * b == 40320 for a, b, _, _ in specs))
        self.assertEqual(sum(1 for a, _, _, _ in specs if a == 1), 1)
        self.assertEqual(sum(1 for a, _, _, _ in specs if a == 40320), 2)

    def test_worker_measures_every_buildable_partition(self):
        config = small_config(self.reference_root, self.output_root)
        config.update(deltas=(32.0, 64.0), Lmax=3, target_recall=0.5, tuning_acceptance_recall=0.5, index_budget_bytes=10**9,
                      ablation_latency_weights=[0.1, 10.0], cost_memory_scale_bytes=10**8,
                      cost_latency_scale_seconds=0.001, build_deadline_seconds=60, peak_ram_cap_bytes=10**10,
                      index_seeds=(1701,), case_ids=["unit"])
        base = dict(id="unit", kind="conjunctive", n=24, k=3, frequency=0.18)
        config["cases"] = [base]
        case = study_jobs(config)[0]
        X, Q, info = dataset(config, case)
        ids, distances, _ = ground_truth(X, Q, config["relevance_radius"])
        folder = self.reference_root / case["id"]
        folder.mkdir()
        write_json(folder / "dataset.json", dict(info, **case))
        np.savez(folder / "ground-truth.npz", queries=Q, ids=ids, distances=distances)
        write_json(folder / "frozen-main-selections.json", {})
        out = self.output_root / case["id"]
        out.mkdir()
        worker(config, case, out)
        rows = json.loads((out / "rows.json").read_text())
        self.assertEqual({r["config_id"].split("-", 1)[1] for r in rows}, {"3-swap0", "2+1-swap0", "2+1-swap1", "1+1+1-swap0", "1+1+1-swap1"})
        qualified = [r for r in rows if r["status"] == "qualified"]
        self.assertTrue(qualified)
        self.assertTrue(all(r["test_median_ms"] > 0 for r in qualified))
        chosen = selections(rows, config)
        self.assertEqual(set(chosen), {"0.1", "10.0"})


if __name__ == "__main__":
    unittest.main()
