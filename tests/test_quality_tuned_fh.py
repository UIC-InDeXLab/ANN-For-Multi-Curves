import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from fresh import experiment_quality_tuned_fh as runner
from fresh.e1 import write_json
from fresh.poc import dataset
from fresh.regimes import ground_truth
from fresh.nba_full import digest


class QualityTunedFirstHitTests(unittest.TestCase):
    def storage_trial(self, **overrides):
        return dict(dict(tuning_a125=.18, tuning_a110=.06, tuning_coverage=.94, tuning_mean_ratio=1.6,
                         index_bytes=100, L=64, delta=24, blocks=[2, 1, 1], swapped=False,
                         labels=["blocking"], family="2+1+1"), **overrides)

    def test_storage_selection_requires_both_quality_and_coverage(self):
        reference = dict(baseline=self.storage_trial(tuning_a125=.20, tuning_coverage=.95))
        config = dict(index_budget_bytes=1000,
                      storage_selection=dict(max_a125_loss=.02, max_coverage_loss=.02))
        acceptable = self.storage_trial(index_bytes=80, tuning_a125=.18, tuning_coverage=.93)
        low_quality = self.storage_trial(index_bytes=20, tuning_a125=.17)
        many_missing = self.storage_trial(index_bytes=10, tuning_coverage=.50)
        expensive = self.storage_trial(index_bytes=2000, tuning_a125=.9)
        higher_quality = self.storage_trial(index_bytes=100, tuning_a125=.3)
        self.assertIs(runner.choose_configuration(
            [low_quality, many_missing, expensive, higher_quality, acceptable], "blocking", config, reference),
            acceptable)
        with self.assertRaisesRegex(ValueError, "No tuning configuration"):
            runner.choose_configuration([low_quality, many_missing, expensive], "blocking", config, reference)

    def test_storage_policy_requires_a_reference_and_valid_tolerances(self):
        config = dict(index_budget_bytes=1000, storage_selection=dict(max_a125_loss=.02, max_coverage_loss=.02))
        with self.assertRaisesRegex(ValueError, "requires a frozen"):
            runner.choose_configuration([self.storage_trial()], "blocking", config)
        for loss in (-.01, 1.01, float("nan")):
            with self.subTest(loss=loss), self.assertRaisesRegex(ValueError, "finite fractions"):
                runner.choose_configuration([self.storage_trial()], "blocking",
                                            dict(config, storage_selection=dict(max_a125_loss=loss,
                                                                                max_coverage_loss=.02)),
                                            dict(baseline=self.storage_trial()))

    def test_interior_selection_does_not_relabel_either_endpoint(self):
        config = dict(index_budget_bytes=1000, interior_blocking=True)
        interior = self.storage_trial()
        for blocks, swapped in (([4], False), ([4], True), ([1, 1, 1, 1], False), ([1, 1, 1, 1], True)):
            endpoint = self.storage_trial(blocks=blocks, swapped=swapped, tuning_a125=1.)
            self.assertIs(runner.choose_configuration([interior, endpoint], "blocking", config), interior)
        self.assertIs(runner.choose_configuration(
            [interior, endpoint], "blocking", dict(index_budget_bytes=1000)), endpoint)

    def test_frozen_tuning_reference_checks_hashes_seed_splits_and_arrays(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            baseline = self.storage_trial(labels=["blocking", "store_all"], blocks=[4], family="4")
            config = dict(tuning_queries=500, test_queries=1000, warmup_queries=10)
            info = dict(dataset_sha256="database", queries_sha256="queries", nba_reference_artifacts={"truth": "hash"})
            write_json(root / "provenance.json",
                       dict(index_seed=2301, metric="frechet", protocol=config, dataset=info))
            write_json(root / "dataset.json", info)
            write_json(root / "tuning-configurations.json", [baseline])
            write_json(root / "frozen-selections.json", dict(store_all=baseline))
            names = ("provenance.json", "dataset.json", "tuning-configurations.json", "frozen-selections.json")
            write_json(root / "completion.json", dict(artifacts={name: digest(root / name) for name in names}))
            job = dict(tuning_reference=str(root), seed=2301, metric="frechet")
            trials, reference = runner.tuning_reference(job, config, info)
            self.assertEqual(trials, [baseline])
            self.assertEqual(reference["baseline"], baseline)
            for different_job, different_config, different_info in (
                    (dict(job, seed=2302), config, info),
                    (dict(job, metric="dtw"), config, info),
                    (job, dict(config, tuning_queries=400), info),
                    (job, config, dict(info, queries_sha256="changed")),
                    (job, config, dict(info, nba_reference_artifacts={"truth": "changed"}))):
                with self.assertRaisesRegex(ValueError, "differs"):
                    runner.tuning_reference(different_job, different_config, different_info)
            write_json(root / "tuning-configurations.json", [])
            with self.assertRaisesRegex(ValueError, "reference changed"):
                runner.tuning_reference(job, config, info)

    def test_approximation_success_controls_selection_instead_of_exact_hits(self):
        one = dict(tuning_a125=.4, tuning_a110=.4, tuning_coverage=1, tuning_mean_ratio=2,
                   index_bytes=100, L=1, delta=8, tuning_recall=.4)
        two = dict(one, tuning_a125=.95, tuning_a110=.8, tuning_mean_ratio=1.1, tuning_recall=.01)
        self.assertIs(min([one, two], key=runner.selection_key), two)

    def test_prefix_quality_keeps_first_wrong_answers_and_missing_queries(self):
        rows = [
            dict(found=True, timeout=False, first_table=1, within_125=False, within_110=False, ratio=1.5),
            dict(found=True, timeout=False, first_table=3, within_125=True, within_110=True, ratio=1.05),
            dict(found=False, timeout=False, first_table=4, within_125=False, within_110=False, ratio=None)]
        self.assertEqual(runner.prefix_quality(rows, 1)["tuning_a125"], 0)
        result = runner.prefix_quality(rows, 3)
        self.assertAlmostEqual(result["tuning_a125"], 1 / 3)
        self.assertAlmostEqual(result["tuning_coverage"], 2 / 3)
        self.assertAlmostEqual(result["tuning_mean_ratio"], 1.275)

    def test_nba_search_covers_all_twelve_distinct_families(self):
        specs = runner.family_specs(dict(study="nba"))
        self.assertEqual(len(specs), 12)
        self.assertEqual(sum("probe_all" in labels for _, _, labels in specs), 1)
        self.assertEqual(sum("store_all" in labels for _, _, labels in specs), 1)
        self.assertTrue(all(sum(sizes) == 5 for sizes, _, _ in specs))

    def test_selections_are_frozen_before_any_test_and_aliases_share_measurements(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            reference = root / "reference"
            reference.mkdir()
            config = dict(dataset_seed=55, m=3, d=2, rho=.1, tuning_queries=2, test_queries=3,
                          warmup_queries=1, planted_per_query=1, plant_anchor_n=8)
            case = dict(id="unit-seed1701", case_id="unit", kind="conjunctive",
                        n=16, k=2, frequency=.18, index_seed=1701)
            X, Q, info = dataset(config, case)
            ids, distances, _ = ground_truth(X, Q, 1.5)
            source = reference / case["id"]
            source.mkdir()
            write_json(reference / "config.json", config)
            write_json(source / "dataset.json", dict(info, **case))
            np.savez(source / "ground-truth.npz", queries=Q, ids=ids, distances=distances)
            output = root / "output"
            output.mkdir()
            protocol = dict(quality_deltas=[8, 16], quality_Lmax=3, index_budget_bytes=10**8,
                            build_deadline_seconds=60, query_deadline_seconds=10, probe_chunk=100)
            job = dict(id="unit", case_id="unit", seed=1701, metric="frechet", study="unit",
                       reference_root=str(reference), reference_job=case["id"], case=case,
                       methods=["blocking", "probe_all", "store_all"])
            original = runner.evaluate

            def guarded(*args, **kwargs):
                if len(args[2]) == 3:
                    self.assertTrue((output / "selection-frozen-before-test.json").exists())
                return original(*args, **kwargs)

            specs = [((1, 1), False, ["blocking", "probe_all"]), ((2,), False, ["blocking", "store_all"])]
            with patch.object(runner, "family_specs", return_value=specs), \
                    patch.object(runner, "evaluate", side_effect=guarded):
                runner.run(dict(protocol=protocol), job, output)
            measured = json.loads((output / "main-results.json").read_text())
            self.assertEqual({row["display_method"] for row in measured}, set(job["methods"]))
            self.assertTrue(all(row["quality_tuned"] for row in measured))
            with (output / "test-queries.csv").open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 9)
            self.assertTrue((output / "completion.json").exists())

            storage_output = root / "storage-output"
            storage_output.mkdir()
            storage_plan = dict(protocol=dict(
                protocol, storage_selection=dict(max_a125_loss=.02, max_coverage_loss=.02)))
            storage_job = dict(job, tuning_reference=str(output))

            def storage_guarded(*args, **kwargs):
                if len(args[2]) == 3:
                    self.assertTrue((storage_output / "selection-frozen-before-test.json").exists())
                return original(*args, **kwargs)

            with patch.object(runner, "family_specs", return_value=specs), \
                    patch.object(runner, "evaluate", side_effect=storage_guarded):
                runner.run(storage_plan, storage_job, storage_output)
            stored = json.loads((storage_output / "main-results.json").read_text())
            baseline = json.loads((output / "frozen-selections.json").read_text())["store_all"]
            for row in stored:
                self.assertGreaterEqual(row["tuning"]["tuning_a125"] + 1e-12, baseline["tuning_a125"] - .02)
                self.assertGreaterEqual(row["tuning"]["tuning_coverage"] + 1e-12, baseline["tuning_coverage"] - .02)
                self.assertLessEqual(row["index_bytes"], protocol["index_budget_bytes"])
            self.assertTrue((storage_output / "completion.json").exists())


if __name__ == "__main__":
    unittest.main()
