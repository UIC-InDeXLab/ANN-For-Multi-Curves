import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

if importlib.util.find_spec("matplotlib") is None:
    raise unittest.SkipTest("Plot validation runs in the local graphics environment")

from fresh.e1 import write_json
from fresh.plot_selectivity import report_and_plot


class SelectivityTests(unittest.TestCase):
    def test_requires_complete_sweep_and_reports_crossovers_without_retuning(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            with self.assertRaises(ValueError):
                report_and_plot(root)
            write_json(root/"completion.json",{})
            cases=[dict(id="low",frequency=.005,k=8),dict(id="high",frequency=.65,k=8)]
            rows=[]
            for case in cases:
                for method,latency in (("blocking",1),("per_curve-intersection",.5 if case["id"]=="low" else 3),
                                        ("per_curve-rarest",.1)):
                    rows.append(dict(case_id=case["id"],display_method=method,test_status="complete",
                                     median_ms=latency,index_MB=2,recall_at_1=.95,test_target_met=True,
                                     mean_postings=10 if method=="blocking" else 1000,
                                     mean_candidates=2 if method=="blocking" else 1,
                                     mean_relevant_sets=1 if case["id"]=="low" else 100))
            write_json(root/"replicated-summary.json",dict(config=dict(cases=cases),main=rows))
            table=report_and_plot(root)
            self.assertEqual(table[0]["speedup_best_pc_over_blocking"],.5)
            self.assertEqual(table[1]["speedup_best_pc_over_blocking"],3)
            result=json.loads((root/"intersection-summary.json").read_text())
            self.assertEqual(table[0]["best_per_curve_variant"],"per_curve-intersection")
            self.assertFalse(table[0]["best_variant_is_posthoc_envelope"])
            self.assertEqual(len(result["observed_crossover_brackets"]),1)
            self.assertTrue((root/"selectivity-crossover.pdf").stat().st_size>1000)
            self.assertGreater(table[1]["mean_relevant_sets"],table[0]["mean_relevant_sets"])
            self.assertEqual(table[1]["blocking_mean_postings"],10)
            self.assertEqual(table[1]["best_pc_mean_postings"],1000)
            self.assertGreater(table[1]["blocking_mean_candidates"],table[1]["best_pc_mean_candidates"])


if __name__=="__main__":
    unittest.main()
