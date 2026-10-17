import csv
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
import uuid

import numba
import numpy as np

from fresh.e1 import write_json
from fresh.experiment_build_time import require_clean_tree, verify_index_bytes, worker
from fresh.poc import Joint, PerCurve, dataset


PROJECT = Path(__file__).resolve().parents[1]


def build_config(reference_root):
    return dict(dataset_seed=2026101217, m=8, d=2, rho=0.5, relevance_radius=1.5,
                tuning_queries=2, test_queries=4, warmup_queries=1, planted_per_query=1,
                plant_anchor_n=24, build_deadline_seconds=60, build_threads=1, pin_core=20,
                repetitions=2, rss_sampling_seconds=0.01,
                expanded_results_root=str(reference_root), scaleup_results_root=str(reference_root),
                methods=("blocking", "probe_all", "per_curve-intersection", "store_all"))


def frozen_rows(X, seed):
    blocking = Joint(X, 64.0, 2, seed, (2,), False)
    probe_all = Joint(X, 64.0, 2, seed, (1, 1), False)
    intersection = PerCurve(X, 64.0, 2, seed)
    store_all = Joint(X, 64.0, 2, seed, (2,), False)
    return {
        "blocking": dict(config_id="blocking-u", method="blocking-family", delta=64.0,
                           blocks=[2], swapped=False, stored=2, L=2, index_bytes=blocking.index_bytes),
        "probe_all": dict(config_id="probe-u", method="blocking-family", delta=64.0,
                            blocks=[1, 1], swapped=False, stored=1, L=2, index_bytes=probe_all.index_bytes),
        "per_curve-intersection": dict(config_id="pc-u", method="per_curve-intersection", delta=64.0,
                                        blocks=None, swapped=False, stored=None, L=2,
                                        index_bytes=intersection.index_bytes),
        "store_all": dict(config_id="store-u", method="blocking-family", delta=64.0,
                           blocks=[2], swapped=False, stored=2, L=2, index_bytes=store_all.index_bytes)
    }


class BuildTimeExperimentTests(unittest.TestCase):
    def setUp(self):
        self.root = PROJECT / (".build-time-test-" + uuid.uuid4().hex)
        self.root.mkdir()
        self.reference_root = self.root / "reference"
        self.reference_root.mkdir()
        self.output_root = self.root / "output"
        self.output_root.mkdir()
        write_json(self.output_root / "manifest.json", dict(commit="unit", config_sha256="unit"))

    def tearDown(self):
        shutil.rmtree(self.root)

    def make_fixture(self):
        config = build_config(self.reference_root)
        case = dict(id="unit-build-seed1701", case_id="unit-build", kind="conjunctive", n=24, k=2,
                    frequency=0.18, source="expanded", index_seed=1701)
        X, Q, info = dataset(config, case)
        folder = self.reference_root / case["id"]
        folder.mkdir()
        write_json(folder / "dataset.json", dict(info, **case))
        write_json(folder / "frozen-main-selections.json", frozen_rows(X, case["index_seed"]))
        return config, case

    def test_worker_repeats_builds_and_preserves_frozen_bytes(self):
        config, case = self.make_fixture()
        folder = self.output_root / case["id"]
        folder.mkdir()
        previous = numba.get_num_threads()
        try:
            numba.set_num_threads(1)
            worker(config, case, folder)
        finally:
            numba.set_num_threads(previous)
        rows = json.loads((folder / "main-results.json").read_text())
        complete = [row for row in rows if row["status"] == "complete"]
        self.assertEqual(len(complete), 4)
        self.assertTrue(all(row["repetitions"] == 2 for row in complete))
        self.assertTrue(all(row["index_bytes"] > 0 for row in complete))
        with (folder / "build-runs.csv").open() as handle:
            runs = list(csv.DictReader(handle))
        self.assertEqual(len(runs), 8)

    def test_verify_index_bytes_and_dirty_tree_guard(self):
        verify_index_bytes(7, 7, "unit")
        with self.assertRaisesRegex(AssertionError, "mismatch"):
            verify_index_bytes(7, 8, "unit")
        with patch("subprocess.check_output", return_value="?? untracked\n"):
            with self.assertRaisesRegex(RuntimeError, "clean frozen source tree"):
                require_clean_tree()


if __name__ == "__main__":
    unittest.main()
