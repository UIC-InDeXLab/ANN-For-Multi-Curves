import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fresh.budget_quality_results import BUDGETS, METHODS, group_outcomes, test_metrics as summarize_test


class BudgetQualityResultsTests(unittest.TestCase):
    def test_late_answer_cannot_pass_even_if_saved_timeout_flag_is_false(self):
        rows = [
            dict(query=0, raw_id=1, raw_distance=1.2, id=1, distance=1.2,
                 nn_distance=1, seconds=.01, timeout=False),
            dict(query=1, raw_id=2, raw_distance=1, id=2, distance=1,
                 nn_distance=1, seconds=.011, timeout=False),
            dict(query=2, raw_id=-1, raw_distance=None, id=-1, distance=None,
                 nn_distance=1, seconds=.001, timeout=False)]
        stats = summarize_test(rows, .01, expected_queries=3)
        self.assertAlmostEqual(stats["a125"], 1 / 3)
        self.assertEqual(stats["ratios"], [1.2])
        self.assertAlmostEqual(stats["late"], 1 / 3)
        self.assertAlmostEqual(stats["empty_only"], 1 / 3)

    def test_zero_optimum_and_duplicate_query_ids_are_explicit(self):
        rows = [dict(query=0, raw_id=1, raw_distance=1e-12, nn_distance=0,
                     seconds=.001, timeout=False)]
        stats = summarize_test(rows, .01, expected_queries=1)
        self.assertEqual(stats["a125"], 0)
        self.assertEqual(stats["ratios"], ["infinity"])
        with self.assertRaises(ValueError):
            summarize_test(rows * 2, .01, expected_queries=2)

    def test_no_tuning_selection_is_not_reported_as_zero_measured_accuracy(self):
        rows = [dict(k=10, budget="1gb-10ms", method="fh_blocking",
                     seed=1701, tested=False, feasible=False, limits={})]
        group = group_outcomes(rows)[10, "1gb-10ms", "fh_blocking"]
        self.assertEqual(group["status"], "NQ")
        self.assertIsNone(group["metrics"])
        self.assertEqual(group["seeds"], 1)

    def test_compact_operating_point_uses_measured_shared_limits(self):
        from fresh.render_fh_blocking_paper import write_budget_results
        limits = {
            "1gb-10ms": (1_000_000_000, .01), "256mb-1ms": (256_000_000, .001),
            "16gb-10ms": (16_000_000_000, .01), "16gb-100ms": (16_000_000_000, .1)}
        records = []
        for budget in BUDGETS:
            memory, deadline = limits[budget]
            for method in METHODS:
                records.append(dict(k=10, budget=budget, method=method, seed=1701,
                                    tested=False, feasible=False, selection=None,
                                    limits=dict(index_budget_bytes=memory, query_deadline_seconds=deadline)))
        point = next(row for row in records
                     if row["method"] == "fh_blocking" and row["budget"] == "16gb-100ms")
        measurement = dict(summarize_test([
            dict(query=0, raw_id=1, raw_distance=1, nn_distance=1, seconds=.004, timeout=False)
        ], .1, expected_queries=1), seed=1701, index_mb=6900)
        point.update(tested=True, feasible=True, measurement=measurement,
                     selection=dict(blocks=[4, 2, 2, 1, 1], swapped=False, delta=128, L=6))
        report = dict(cases={10: [1701]}, records=records)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paper = root / "paper"
            (paper / "experiments/tables").mkdir(parents=True)
            (paper / "experiments/appendix").mkdir()
            with patch("fresh.render_fh_blocking_paper.budget_quality_results.load", return_value=report):
                write_budget_results(root, paper, root)
                table = (paper / "experiments/tables/feasibility-operating-point.tex").read_text()
                caption = (paper / "experiments/appendix/approx-budget-tables.tex").read_text()
                self.assertIn("Blocking & yes & 6.90 & 4.00", table)
                self.assertIn("Probe-all & NQ & ---", table)
                self.assertIn("16\\,GB", caption)
                self.assertIn("100\\,ms", caption)
                self.assertNotIn("256", caption)
                saved = json.loads((root / "paper-budget-summary.json").read_text())
                self.assertEqual(len(saved["outcomes"]), 16)
                point["limits"]["query_deadline_seconds"] = .2
                with self.assertRaisesRegex(ValueError, "measured 16 GB / 100 ms"):
                    write_budget_results(root, paper, root)


if __name__ == "__main__":
    unittest.main()
