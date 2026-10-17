import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from fresh.e1 import write_json
from fresh.poc import jobs,resume_completed,tuning_target
from fresh.aggregate_replications import aggregate_group


class ReplicationTests(unittest.TestCase):
    def test_larger_sizes_preserve_expanded_protocol_and_three_seed_jobs(self):
        folder=Path(__file__).resolve().parents[1]/"configs"
        original=json.loads((folder/"synthetic-expanded-20261012.json").read_text())
        extension=json.loads((folder/"synthetic-scaleup-20261013.json").read_text())
        self.assertEqual({key for key in original if original[key]!=extension[key]},
                         {"experiment","stage","main_plot_n","cases"})
        grid=jobs(extension)
        self.assertEqual(len(grid),9)
        self.assertEqual(len({row["id"] for row in grid}),9)
        self.assertTrue(all(row["n"]>1000000 and row["k"]==8 for row in grid))
        self.assertEqual({row["index_seed"] for row in grid},{1701,1702,1703})
        self.assertEqual({(row["kind"],row["n"]) for row in grid},
                         {("conjunctive",2000000),("conjunctive",5000000),("rare",5000000)})

    def test_grid_contains_three_seeds_without_changing_dataset_ids(self):
        path=Path(__file__).resolve().parents[1]/"configs"/"synthetic-expanded-20261012.json"
        config=json.loads(path.read_text())
        grid=jobs(config)
        self.assertEqual(len(grid),48)
        self.assertEqual(len({r["id"] for r in grid}),48)
        self.assertEqual(grid[0]["case_id"],grid[1]["case_id"])
        self.assertNotEqual(grid[0]["index_seed"],grid[1]["index_seed"])
        self.assertEqual(tuning_target(config),.95)
        self.assertEqual(config["target_recall"],.9)
        self.assertEqual(config["blocking_main_latency_weight"],1)

    def test_resume_requires_identical_code_and_config_and_preserves_completed_jobs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            write_json(root/"manifest.json",dict(commit="source",config_sha256="configuration"))
            write_json(root/"cases.json",[dict(case="complete",status="complete"),
                                         dict(case="interrupted",status="incomplete")])
            folder=root/"complete"
            folder.mkdir()
            for name in ("main-results.json","ablation-results.json","test-queries.csv"):
                (folder/name).write_text("fixture")
            result=resume_completed(root,"source","configuration")
            self.assertEqual(list(result),["complete"])
            with self.assertRaises(ValueError):
                resume_completed(root,"different-source","configuration")
            with self.assertRaises(ValueError):
                resume_completed(root,"source","different-config")
            (root/"completion.json").write_text("{}")
            with self.assertRaises(ValueError):
                resume_completed(root,"source","configuration")

    def test_partial_seed_groups_have_no_performance_summary(self):
        row=dict(case="base-seed1",case_id="base",index_seed=1,kind="conjunctive",n=100000,k=8,
                 display_method="blocking",test_status="complete")
        result=aggregate_group([row],[1,2,3],Path("."),dict(test_queries=2))
        self.assertNotEqual(result["test_status"],"complete")
        self.assertNotIn("median_ms",result)
        self.assertNotIn("recall_at_1",result)

    def test_complete_seed_group_uses_paired_query_bootstrap_and_conditional_ratios(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            group=[]
            for seed in (1,2,3):
                folder=root/f"base-seed{seed}"
                folder.mkdir()
                queries=[
                    dict(query="0",config_id="config",exact_nn="True",ratio="1"),
                    dict(query="1",config_id="config",exact_nn="False",ratio="2" if seed==1 else "")]
                with (folder/"test-queries.csv").open("w",newline="") as handle:
                    writer=csv.DictWriter(handle,fieldnames=list(queries[0]))
                    writer.writeheader();writer.writerows(queries)
                row=dict(case=folder.name,case_id="base",index_seed=seed,kind="conjunctive",n=100000,k=8,
                         display_method="blocking",test_status="complete",config_id="config",
                         dataset_sha256="same-data",queries_sha256="same-queries",recall_at_1=.5,
                         test_target_met=False,c_success={"1.0":.5,"2.0":1 if seed==1 else .5},
                         blocks=[4,2,2],raw_curve_bytes=1000,mean_relevant_sets=1,index_MB=seed,index_bytes=seed*1e6,
                         median_ms=seed,p95_ms=2*seed,mean_ms=seed,coverage=1 if seed==1 else .5,
                         empty_rate=0 if seed==1 else .5,mean_candidates=1,mean_verified=1,mean_postings=2,
                         mean_probes=3,mean_nn_distance=1,mean_returned_distance=1.5 if seed==1 else 1)
                group.append(row)
            result=aggregate_group(group,[1,2,3],root,dict(test_queries=2))
            self.assertEqual(result["median_ms"],2)
            self.assertEqual(result["median_ms_seed_minmax"],[1,3])
            self.assertEqual(result["recall_at_1"],.5)
            self.assertEqual(result["mean_ratio"],1.25)
            self.assertEqual(result["returned_ratio_count"],4)
            self.assertAlmostEqual(result["coverage"],2/3)
            self.assertAlmostEqual(result["empty_rate"],1/3)
            self.assertAlmostEqual(result["coverage"]+result["empty_rate"],1)
            self.assertAlmostEqual(result["mean_returned_distance"],1.25)
            self.assertLessEqual(result["recall_at_1"],result["coverage"])
            self.assertFalse(result["test_target_met"])
            group[2]["queries_sha256"]="different"
            with self.assertRaises(ValueError):
                aggregate_group(group,[1,2,3],root,dict(test_queries=2))


if __name__=="__main__":
    unittest.main()
