import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from fresh.publish_feasibility import METHODS,audit,combine_seeds,detail_tables,figures,query_metrics,summary_table


def seed_row(seed,success=.95,measured=True):
    row=dict(k=8,method="blocking",index_seed=seed,measured=measured)
    if measured:
        row.update(success_rate=success,index_bytes=100000000,index_MB=100,
                   median_ms=1,p95_ms=2,timeout_rate=0,blocks=[4,4],delta=128.,L=3)
    else:
        row["reasons"]=["excluded: serving storage"]
    return row


class FeasibilityPublicationTests(unittest.TestCase):
    def test_all_seeds_must_meet_both_accuracy_and_storage_budgets(self):
        rows=[seed_row(i) for i in (1,2,3)]
        self.assertTrue(combine_seeds(rows,[1,2,3],.9,1000000000)["feasible"])
        rows[2]["success_rate"]=.89
        self.assertFalse(combine_seeds(rows,[1,2,3],.9,1000000000)["feasible"])
        rows[2]["success_rate"]=.95;rows[2]["index_bytes"]=1000000001
        self.assertFalse(combine_seeds(rows,[1,2,3],.9,1000000000)["feasible"])
        rows[2]=seed_row(3,measured=False)
        combined=combine_seeds(rows,[1,2,3],.9,1000000000)
        self.assertFalse(combined["feasible"])
        self.assertNotIn("median_ms",combined)
        with self.assertRaisesRegex(ValueError,"All required seeds"):
            combine_seeds(rows[:2],[1,2,3],.9,1000000000)

    def test_deadline_misses_are_failures_and_every_query_is_required(self):
        good=dict(query="0",seconds=".001",nn_distance="1",timeout="False",id="0",
                  found="True",distance="1",exact_nn="True")
        missed=dict(query="1",seconds=".012",nn_distance="1",timeout="True",id="-1",
                    found="False",distance="",exact_nn="False")
        r=query_metrics([good,missed],np.ones(2),.01,2)
        self.assertEqual(r["success_rate"],.5)
        self.assertEqual(r["timeout_rate"],.5)
        with self.assertRaisesRegex(ValueError,"Every held-out query"):
            query_metrics([good],np.ones(2),.01,2)
        missed.update(found="True",id="1",distance="1")
        with self.assertRaisesRegex(ValueError,"timed-out"):
            query_metrics([good,missed],np.ones(2),.01,2)

    def test_unfinished_benchmark_cannot_publish(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError,"incomplete"):
                audit(Path(temporary))

    def test_figures_and_tables_keep_infeasible_points_without_zero_timings(self):
        groups=[]
        for method in METHODS:
            for k in (4,6,8):
                rows=[dict(seed_row(seed),method=method,k=k) for seed in (1,2,3)]
                if method=="store_all" and k==8:
                    rows[2]=dict(seed_row(3,measured=False),method=method,k=k)
                groups.append(combine_seeds(rows,[1,2,3],.9,1000000000))
        report=dict(groups=groups,largest_feasible_tested_k={
                    method:6 if method=="store_all" else 8 for method in METHODS},
                    config=dict(cases=[dict(k=k) for k in (4,6,8)],index_budget_bytes=1000000000,
                                query_deadline_seconds=.01,target_recall=.9))
        before=copy.deepcopy(report)

        def inspect(fig,path):
            self.assertEqual(len(fig.axes),4)
            self.assertEqual(sum(line.get_marker()=="x" for line in fig.axes[0].lines),1)
            self.assertFalse(any(line.get_label()=="Store-all" and 8 in line.get_xdata()
                                 for line in fig.axes[2].lines))

        with patch("fresh.publish_feasibility.configure_paper_font"), \
                patch("fresh.publish_feasibility.save_row",side_effect=inspect):
            figures(report,Path("fixture"))
        self.assertEqual(report,before)
        self.assertIn("Store-all & 6",summary_table(report))
        self.assertIn("no qualifying configuration",detail_tables(report))


if __name__=="__main__":
    unittest.main()
