import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from fresh import experiment_nba_fh_accuracy as runner
from fresh.e1 import write_json
from fresh.experiment_first_hit_quality import evaluate, timed_hit
from fresh.feasibility_first_hit import build_tables
from fresh.poc import dataset
from fresh.regimes import ground_truth


class NBAFirstHitAccuracyTests(unittest.TestCase):
    def row(self, first_table, ratio, found=True, timeout=False):
        return dict(found=found, timeout=timeout, first_table=first_table, ratio=ratio,
                    within_125=found and ratio is not None and ratio <= 1.25,
                    within_110=found and ratio is not None and ratio <= 1.1)

    def test_schedule_scoring_keeps_the_first_wrong_answer_and_counts_misses(self):
        observations = {
            16: runner.observation_arrays([self.row(1, 1.6), self.row(5, 1.05), self.row(9, None, False)]),
            24: runner.observation_arrays([self.row(1, 1.), self.row(1, 1.2), self.row(9, None, False)])}
        result = runner.score_schedule([dict(delta=16, L=2), dict(delta=24, L=2)], observations)
        self.assertAlmostEqual(result["tuning_a125"], 1 / 3)
        self.assertAlmostEqual(result["tuning_coverage"], 2 / 3)
        self.assertAlmostEqual(result["tuning_mean_ratio"], 1.4)

    def test_only_fine_to_coarse_schedules_with_one_stage_per_delta_are_generated(self):
        options = {24: [dict(delta=24, L=1)], 16: [dict(delta=16, L=1), dict(delta=16, L=2)]}
        schedules = list(runner.schedules(options, 2))
        self.assertEqual(len(schedules), 5)
        for schedule in schedules:
            self.assertEqual([s["delta"] for s in schedule], sorted({s["delta"] for s in schedule}))

    def test_quality_choice_prioritizes_a125_then_mean_and_enforces_coverage(self):
        one = dict(kind="cascade", index_bytes=50, tuning_a125=.4, tuning_a110=.2,
                   tuning_coverage=.95, tuning_mean_ratio=1.4)
        two = dict(one, index_bytes=80, tuning_mean_ratio=1.3)
        missing = dict(one, tuning_a125=.9, tuning_coverage=.5)
        large = dict(one, tuning_a125=.9, index_bytes=101)
        config = dict(index_budget_bytes=100, minimum_tuning_coverage=.9)
        self.assertIs(runner.choose([one, two, missing, large], "cascade", config), two)
        with self.assertRaisesRegex(ValueError, "No uniform"):
            runner.choose([one, two], "uniform", config)

    def test_query_stops_at_first_hit_without_probing_later_stages(self):
        hit = dict(id=3, probes=5, timeout=False, first_table=2)
        with patch.object(runner, "timed_hit", return_value=hit) as query:
            result = runner.schedule_hit([("fine", 4), ("coarse", 8)], None, 2,
                                         dict(query_deadline_seconds=10))
        self.assertEqual(query.call_count, 1)
        self.assertEqual(result["id"], 3)
        self.assertEqual(result["first_table"], 2)
        self.assertEqual(result["hit_stage"], 0)

    def test_query_falls_back_only_on_misses_and_shares_one_deadline(self):
        clock = [0.]
        responses = [
            dict(id=-1, probes=4, timeout=False, first_table=5),
            dict(id=7, probes=2, timeout=False, first_table=2)]
        remaining = []

        def hit(index, query, count, config):
            remaining.append(config["query_deadline_seconds"])
            clock[0] += .6
            return responses.pop(0)

        with patch.object(runner.time, "perf_counter", side_effect=lambda: clock[0]), \
                patch.object(runner, "timed_hit", side_effect=hit):
            result = runner.schedule_hit([("fine", 4), ("coarse", 8)], None, 2,
                                         dict(query_deadline_seconds=1))
        self.assertAlmostEqual(remaining[1], .4)
        self.assertTrue(result["timeout"])
        self.assertEqual(result["id"], -1)
        self.assertEqual(result["probes"], 6)

    def test_real_schedule_answers_and_tuning_predictions_match_manual_first_hits(self):
        config = dict(dataset_seed=55, m=3, d=2, rho=.1, tuning_queries=3, test_queries=3,
                      warmup_queries=1, planted_per_query=1, plant_anchor_n=8,
                      query_deadline_seconds=10, probe_chunk=100)
        case = dict(id="unit", case_id="unit", kind="conjunctive", n=16, k=2, frequency=.18, index_seed=1701)
        X, Q, _ = dataset(config, case)
        ids, distances, _ = ground_truth(X, Q, 1.5)
        stages = [(build_tables(X, delta, 3, 1701, [1, 1], False, 10**8), 3) for delta in (1, 8)]
        rows = evaluate(stages, X, Q, distances, 2, config, "frechet", hit_function=runner.schedule_hit)
        observations = {delta: runner.observation_arrays(evaluate(
            stage, X, Q, distances, count, config, "frechet"))
            for delta, (stage, count) in zip((1, 8), stages)}
        predicted = runner.score_schedule([dict(delta=1, L=3), dict(delta=8, L=3)], observations)
        self.assertAlmostEqual(predicted["tuning_a125"], np.mean([r["within_125"] for r in rows]))
        self.assertAlmostEqual(predicted["tuning_coverage"], np.mean([r["found"] for r in rows]))
        for query, row in zip(Q, rows):
            expected = timed_hit(stages[0][0], query, 3, config)
            if expected["id"] < 0:
                expected = timed_hit(stages[1][0], query, 3, config)
            self.assertEqual(row["id"], expected["id"])

    def test_end_to_end_freezes_both_methods_before_test(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = dict(dataset_seed=55, m=3, d=2, rho=.1, tuning_queries=2, test_queries=3,
                          warmup_queries=1, planted_per_query=1, plant_anchor_n=8)
            case = dict(id="unit", case_id="unit", kind="conjunctive",
                        n=16, k=2, frequency=.18, index_seed=1701)
            X, Q, info = dataset(config, case)
            ids, distances, _ = ground_truth(X, Q, 1.5)
            protocol = dict(quality_deltas=[8, 16], table_grid=[1, 2], maximum_stages=2,
                            index_budget_bytes=10**8, minimum_tuning_coverage=.5,
                            query_deadline_seconds=10, probe_chunk=100, build_deadline_seconds=60,
                            **config)
            job = dict(id="unit", case_id="unit", metric="frechet", seed=1701,
                       frozen_family=dict(blocks=[1, 1], swapped=False))
            original = runner.evaluate

            def guarded(*args, **kwargs):
                if len(args[2]) == 3:
                    self.assertTrue((root / "selection-frozen-before-test.json").exists())
                    self.assertEqual(set(json.loads((root / "frozen-selections.json").read_text())),
                                     {"uniform", "cascade"})
                return original(*args, **kwargs)

            with patch.object(runner, "load_job", return_value=(protocol, case, X, Q, distances, info)), \
                    patch.object(runner, "evaluate", side_effect=guarded):
                runner.run(dict(protocol=protocol), job, root, exploratory=True)
            rows = json.loads((root / "main-results.json").read_text())
            self.assertEqual({r["display_method"] for r in rows}, {"uniform", "cascade"})
            self.assertTrue(all(r["queries"] == 3 for r in rows))
            self.assertTrue((root / "completion.json").exists())


if __name__ == "__main__":
    unittest.main()
