import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from fresh.e1 import append_row, write_json
from fresh.experiment_first_hit_quality import (
    canonical_family, distance_ratio, evaluate, families, quality_summary, worker,
    construction_index_cap,
)
from fresh.feasibility_first_hit import build_tables
from fresh.poc import dataset
from fresh.regimes import ground_truth
from test_experiment_matched_recall import selection_rows, small_config


class FirstHitQualityFollowupTests(unittest.TestCase):
    def test_endpoint_duplicates_and_partition_count(self):
        self.assertEqual(canonical_family((8,), True), ((1,) * 8, False))
        self.assertEqual(canonical_family((1,) * 8, True), ((8,), False))
        with TemporaryDirectory() as temp:
            root = Path(temp)
            write_json(root / "frozen-main-selections.json", {})
            out = families(dict(case=dict(k=8), frontier=True), root)
            self.assertEqual(len(out), 42)
            self.assertEqual(len({(s, sw) for s, sw, labels in out}), 42)

    def test_quality_summary_counts_empty_answers_as_failures(self):
        base = dict(seconds=.001, timeout=False, exact_nn=False, first_table=1)
        rows = [
            dict(base, id=0, found=True, ratio=1.05, within_110=True, within_125=True),
            dict(base, id=1, found=True, ratio=1.2, within_110=False, within_125=True),
            dict(base, id=-1, found=False, ratio=None, within_110=False, within_125=False),
        ]
        out = quality_summary(rows)
        self.assertAlmostEqual(out["success_110"], 1 / 3)
        self.assertAlmostEqual(out["success_125"], 2 / 3)
        self.assertAlmostEqual(out["empty_rate"], 1 / 3)
        self.assertAlmostEqual(out["mean_ratio"], 1.125)
        self.assertEqual(out["median_ms"], 1)
        self.assertEqual(distance_ratio(0, 0), 1)
        self.assertEqual(distance_ratio(1, 0), "infinity")

    def test_blocking_only_does_not_add_unrequested_baseline_families(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            write_json(root / "frozen-main-selections.json", dict(
                blocking=dict(blocks=[3, 1, 1, 1, 1, 1], swapped=False),
                probe_all=dict(blocks=[1] * 8, swapped=False),
                store_all=dict(blocks=[8], swapped=False)))
            job = dict(case=dict(k=8), methods=["blocking"])
            found = families(job, root)
            self.assertEqual(found, [((3, 1, 1, 1, 1, 1), False, ["blocking"])])
            job["frontier"] = True
            self.assertEqual(len(families(job, root)), 42)

    def test_construction_headroom_reserves_raw_workspace(self):
        X = np.broadcast_to(np.zeros((1, 8, 32, 2)), (5_000_000, 8, 32, 2))
        config = dict(index_budget_bytes=24_000_000_000, peak_ram_cap_bytes=48_000_000_000)
        cap = construction_index_cap(X, 8, 24, 1680, config)
        self.assertEqual(cap, 13_359_986_368)
        self.assertEqual(construction_index_cap(X, 8, 96, 420, config), 0)

    def test_worker_uses_frozen_arrays_and_records_both_thresholds(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            ref = root / "reference"
            ref.mkdir()
            config = small_config(ref, root / "output")
            config.update(target_recall=.9, tuning_acceptance_recall=.5,
                          blocking_main_latency_weight=1, ablation_latency_weights=[1])
            case = dict(id="unit-k2-seed1701", case_id="unit-k2", kind="conjunctive",
                        n=24, k=2, frequency=.18, index_seed=1701)
            source = ref / case["id"]
            source.mkdir()
            write_json(ref / "config.json", config)
            X, Q, info = dataset(config, case)
            ids, optimum, _ = ground_truth(X, Q, config["relevance_radius"])
            write_json(source / "dataset.json", dict(info, **case))
            write_json(source / "frozen-main-selections.json", selection_rows(X, 1701))
            np.savez(source / "ground-truth.npz", queries=Q, ids=ids, distances=optimum)
            output = root / "output"
            output.mkdir()
            protocol = dict(quality_deltas=[32, 64], quality_Lmax=3, heavy_probe_threshold=500000,
                            heavy_deltas=[64], heavy_L_max=2, index_budget_bytes=10**9,
                            build_deadline_seconds=60, tuning_acceptance_recall=.5)
            job = dict(id="unit", case_id="unit-k2", case=case, metric="frechet",
                       reference_root=str(ref), reference_job=case["id"], table_counts=[1, 2])
            worker(dict(protocol=protocol), job, output)
            records = json.loads((output / "main-results.json").read_text())
            self.assertTrue(records)
            self.assertTrue(all(r["queries"] == config["test_queries"] for r in records))
            self.assertTrue(all(r["success_125"] >= r["success_110"] for r in records))
            self.assertTrue((output / "completion.json").is_file())
            with (output / "test-queries.csv").open() as handle:
                orphan = next(csv.DictReader(handle))
            orphan["config_id"] = "interrupted-config"
            append_row(output / "test-queries.csv", orphan)
            worker(dict(protocol=protocol), job, output)
            self.assertEqual(json.loads((output / "main-results.json").read_text()), records)
            with (output / "test-queries.csv").open() as handle:
                self.assertTrue(all(r["config_id"] != "interrupted-config" for r in csv.DictReader(handle)))
            self.assertTrue(list(output.glob("test-queries.partial-*.csv")))
            for row in records:
                row["labels"] = [label for label in row["labels"] if label != "probe_all"]
            write_json(output / "main-results.json", records)
            worker(dict(protocol=protocol), dict(job, methods=["probe_all"]), output)
            reused = json.loads((output / "main-results.json").read_text())
            endpoint = [r for r in reused if r["stored"] == 1]
            self.assertTrue(endpoint)
            self.assertTrue(all("probe_all" in r["labels"] for r in endpoint))
            self.assertEqual([r["mean_ratio"] for r in reused], [r["mean_ratio"] for r in records])
            for metric in ("frechet", "dtw"):
                index = build_tables(X, 64, 2, 1701, (2,), False, 10**9)
                rows = evaluate(index, X, Q[:1], np.full(1, np.inf), 1, config, metric)
                self.assertEqual(len(rows), 1)


if __name__ == "__main__":
    unittest.main()
