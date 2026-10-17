import shutil
import tempfile
import unittest
from pathlib import Path

from fresh.render_experiment_tables import matched_recall_table

CASES = ("conjunctive-n100000-k8", "rare-n100000-k8")
METHODS = ("blocking", "probe_all", "per_curve-intersection")


def summary():
    aggregates, matched = [], []
    for case in CASES:
        for method in METHODS:
            for L, recall, ms in ((1, .7, 1.), (4, .93, 2.), (16, .985, 5.)):
                aggregates.append(dict(case_id=case, display_method=method, target_L=L, mean_recall_at_1=recall,
                                       median_seed_median_ms=ms, median_index_mb=1.))
            for target in (.9, .95, .97, .99):
                reached = target <= .93 or target <= .985
                L, ms = (4, 2.) if target <= .93 else (16, 5.)
                if target == .99:
                    reached = False
                matched.append(dict(case_id=case, display_method=method, target_recall=target,
                                    status="reached" if reached else "not reached", target_L=L if reached else None,
                                    median_seed_median_ms=ms if reached else None))
    return dict(aggregates=aggregates, matched_recall=matched)


class MatchedRecallOutputTests(unittest.TestCase):
    def test_table_marks_unreached_targets_and_shows_the_selected_table_count(self):
        text = matched_recall_table(summary())
        self.assertIn("2.00 (4)", text)
        self.assertIn("5.00 (16)", text)
        self.assertIn("---", text)
        self.assertEqual(len([line for line in text.splitlines() if line.startswith(("Common", " &", "Rare"))]) >= 6, True)

    @unittest.skipUnless(shutil.which("kpsewhich"), "The figure uses the paper fonts")
    def test_figure_draws_both_panels_with_all_methods(self):
        from fresh.plot_experiment_figures import matched_recall
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "fig.pdf"
            matched_recall(summary(), destination)
            self.assertGreater(destination.stat().st_size, 1000)

    @unittest.skipUnless(shutil.which("kpsewhich"), "The figure uses the paper fonts")
    def test_figure_requires_every_method(self):
        from fresh.plot_experiment_figures import matched_recall
        data = summary()
        data["aggregates"] = [r for r in data["aggregates"] if r["display_method"] != "probe_all"]
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "every method"):
                matched_recall(data, Path(temporary) / "fig.pdf")


if __name__ == "__main__":
    unittest.main()
