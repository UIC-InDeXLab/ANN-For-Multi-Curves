import json
import tempfile
import unittest
from pathlib import Path

from fresh import render_experiment_tables as tables


def nba_report():
    rows = []
    for method, recall in (("blocking", .944), ("probe_all", .949), ("store_all", .963),
                           ("per_curve-intersection", .969), ("per_curve-rarest", .974)):
        rows.append(dict(method=method, status="complete", index_MB=9.5229, median_ms=1.3556, p95_ms=1.82,
                         exact_recall=recall, c_success={"1.1": .9887}, mean_excess_feet=.0487))
    return dict(main=rows)


class RenderTableTests(unittest.TestCase):
    def test_nba_table_lists_exactly_the_four_methods_without_the_removed_baseline(self):
        text = tables.nba_table(nba_report())
        self.assertNotIn("Rarest", text)
        self.assertEqual(len([line for line in text.splitlines() if line.endswith(r"\\")]), 5)
        self.assertIn("Blocking & 9.52 & 1.36 & 1.82 & 94.4 & 98.9 & 0.049", text)

    def test_nba_table_refuses_an_incomplete_method(self):
        report = nba_report()
        report["main"][1]["status"] = "timeout"
        with self.assertRaisesRegex(ValueError, "every method"):
            tables.nba_table(report)

    def test_ablation_requires_every_preference_and_reports_recall_range(self):
        def rows(kind, n):
            return [dict(kind=kind, n=n, k=8, latency_weight=weight, test_status="complete",
                         median_ms=10 / weight, index_MB=10 * weight, recall_at_1=.96 + weight / 10000)
                    for weight in (.1, 1.0, 10.0, 100.0)]
        report = dict(ablation=sum((rows(kind, n) for kind, n in (
            ("conjunctive", 100000), ("rare", 100000), ("repeated", 100000), ("conjunctive", 2000000),
            ("conjunctive", 5000000), ("rare", 5000000))), []))
        text, (low, high) = tables.ablation_table(report, report)
        self.assertEqual(text.count(r"\\"), 8)
        self.assertLess(low, high)
        report["ablation"] = [r for r in report["ablation"] if r["latency_weight"] != 10.0]
        with self.assertRaisesRegex(ValueError, "all four"):
            tables.ablation_table(report, report)

    def test_build_seconds_is_the_median_over_all_seeds_and_refuses_gaps(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for seed, seconds in zip(tables.SEEDS, (3., 1., 2.)):
                folder = root / f"case-seed{seed}"
                folder.mkdir()
                (folder / "main-results.json").write_text(json.dumps(
                    [dict(display_method="blocking", build_seconds=seconds)]))
            self.assertEqual(tables.build_seconds(root, "case", "blocking"), 2.)
            with self.assertRaisesRegex(ValueError, "Missing build time"):
                tables.build_seconds(root, "case", "probe_all")

    def test_selectivity_requires_all_six_settings(self):
        with self.assertRaisesRegex(ValueError, "six"):
            tables.selectivity_table(dict(comparisons=[]))

    def test_number_formatting_uses_latex_thousands_separators(self):
        self.assertEqual(tables.number(1440.58, 1), "1{,}440.6")
        self.assertEqual(tables.percent(.9447), "94.5")


if __name__ == "__main__":
    unittest.main()
