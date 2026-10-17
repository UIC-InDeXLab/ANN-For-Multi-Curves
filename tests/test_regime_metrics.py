import csv
import json
from pathlib import Path
import tempfile
import unittest

from fresh.analyze_regimes import summarize, wilson


class RegimeMetricTests(unittest.TestCase):
    def test_small_sample_interval_does_not_certify_target(self):
        low, high = wilson(29, 32)
        self.assertLess(low, .8)
        self.assertGreater(high, .96)

    def test_approximate_success_counts_empty_answers_as_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"config.json").write_text(json.dumps(dict(index_seed=1,target_recall=.9,test_queries=2,stage="unit")))
            (root/"cases.json").write_text(json.dumps([dict(case="case",status="complete")]))
            folder = root/"case"
            folder.mkdir()
            (folder/"dataset.json").write_text(json.dumps(
                dict(kind="unit",n=10,k=2,raw_curve_bytes=80,mean_test_relevant_sets=1)))
            (folder/"results.json").write_text(json.dumps(
                [dict(method="method",index_bytes=120,test_status="complete")]))
            (folder/"tuning-configurations.json").write_text("[]")
            rows = [
                dict(method="method",exact_nn="False",found="True",distance="1.02",nn_distance="1",
                     seconds=".001",ratio="1.02",candidates="1",verified="1",probes="1",postings="1"),
                dict(method="method",exact_nn="False",found="False",distance="",nn_distance="1",
                     seconds=".002",ratio="",candidates="0",verified="0",probes="1",postings="0")]
            with (folder/"queries.csv").open("w",newline="") as handle:
                writer = csv.DictWriter(handle,fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            result = summarize(root)[0]
            self.assertEqual(result["coverage"], .5)
            self.assertEqual(result["c_success"]["1.01"], 0)
            self.assertEqual(result["c_success"]["1.05"], .5)
            self.assertEqual(result["mean_ratio"], 1.02)
            self.assertEqual(result["evaluated_queries"], 2)


if __name__ == "__main__":
    unittest.main()
