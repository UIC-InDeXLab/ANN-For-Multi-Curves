import json
import importlib.util
from pathlib import Path
import tempfile
import unittest

if importlib.util.find_spec("matplotlib") is None:
    raise unittest.SkipTest("Figure tests run locally in the environment with Matplotlib")

from fresh.e1 import write_json
from fresh.plot_poc import figures, metrics, summarize


class PocFigureTests(unittest.TestCase):
    def test_incomplete_results_cannot_be_summarized(self):
        with self.assertRaises(ValueError):
            metrics([],1000,[1,1.05])

    def test_quality_keeps_empty_answers_in_denominator(self):
        rows=[
            dict(query="0",timeout="False",seconds=".001",found="True",ratio="1",exact_nn="True",
                 distance="1",nn_distance="1",candidates="1",verified="1",postings="2",probes="4"),
            dict(query="1",timeout="False",seconds=".001",found="False",ratio="",exact_nn="False",
                 distance="",nn_distance="1",candidates="0",verified="0",postings="0",probes="4")]
        result=metrics(rows,2,[1,1.05])
        self.assertEqual(result["c_success"]["1.05"],.5)
        self.assertEqual(result["mean_ratio"],1)
        self.assertEqual(result["empty_rate"],.5)

    def test_replicated_job_seed_in_result_is_not_a_duplicate_keyword(self):
        import csv
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            write_json(root/"config.json",dict(test_queries=2,target_recall=.9,index_seeds=[17],
                                               c_success_factors=[1.,1.05],blocking_main_latency_weight=1))
            write_json(root/"manifest.json",dict(commit="fixture"))
            write_json(root/"cases.json",[dict(case="case-seed17",status="complete")])
            folder=root/"case-seed17"
            folder.mkdir()
            write_json(folder/"dataset.json",dict(kind="conjunctive",n=100000,k=8,case_id="case",index_seed=17,
                                                  raw_curve_bytes=100,planted_fraction=.01,
                                                  dataset_sha256="dataset",queries_sha256="queries",
                                                  mean_test_relevant_sets=1))
            item=dict(display_method="blocking",config_id="selected",index_seed=17,
                      test_status="complete",index_bytes=100,blocks=[4,2,2])
            write_json(folder/"main-results.json",[item])
            write_json(folder/"ablation-results.json",[])
            rows=[dict(query=str(q),config_id="selected",timeout="False",seconds=".001",found="True",
                       ratio="1",exact_nn="True",distance="1",nn_distance="1",
                       candidates="1",verified="1",postings="2",probes="4") for q in range(2)]
            with (folder/"test-queries.csv").open("w",newline="") as handle:
                writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
                writer.writeheader();writer.writerows(rows)
            result=summarize(root)
            self.assertEqual(result["main"][0]["index_seed"],17)
            self.assertEqual(result["main"][0]["case_id"],"case")
            item["index_seed"]=18
            write_json(folder/"main-results.json",[item])
            with self.assertRaises(ValueError):
                summarize(root)

    def test_real_summary_regenerates_all_figures_without_changed_measurements(self):
        root=Path(__file__).resolve().parents[1]/"results"/"fresh-poc-20261012"
        if not (root/"completion.json").exists():
            self.skipTest("Completed POC data is available only in the result workspace")
        # Reading the saved report avoids touching the measurement directory during the test.
        report=json.loads((root/"summary.json").read_text())
        self.assertEqual(len(report["cases"]),12)
        self.assertTrue(all(r["status"]=="complete" for r in report["cases"]))
        main=[r for r in report["main"] if r["display_method"]=="blocking"]
        self.assertEqual(len(main),12)
        self.assertTrue(all(r["test_queries"]==1000 for r in main))
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            figures(report,folder)
            self.assertEqual(len(list(folder.glob("*.png"))),5)
            self.assertEqual(len(list(folder.glob("*.pdf"))),5)
            self.assertTrue(all(p.stat().st_size>1000 for p in folder.iterdir()))


if __name__=="__main__":
    unittest.main()
