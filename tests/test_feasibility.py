import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from fresh.e1 import write_json
from fresh.feasibility import METHODS,measure_queries,normalize_query,preflight_joint,select_fastest,tuning_summary,worker
from fresh.poc import jobs


def result(distance=1,id=0,timeout=False):
    return dict(id=id,distance=distance,timeout=timeout,candidates=1,verified=1,probes=1,postings=1,seconds=0)


class FeasibilityTests(unittest.TestCase):
    def test_deadline_applies_to_full_elapsed_time_and_late_answers_fail(self):
        row=normalize_query(result(),1,.02,.01)
        self.assertTrue(row["timeout"])
        self.assertFalse(row["found"])
        self.assertFalse(row["exact_nn"])
        self.assertEqual(row["id"],-1)
        self.assertIsNone(row["distance"])
        self.assertEqual(row["raw_id"],0)
        self.assertTrue(normalize_query(result(),1,.001,.01)["exact_nn"])
        self.assertFalse(normalize_query(result(2),1,.001,.01)["exact_nn"])
        self.assertFalse(normalize_query(result(timeout=True),1,.001,.01)["exact_nn"])

    def test_tuning_early_rejection_is_sound_but_tests_keep_every_timeout(self):
        config=dict(tuning_acceptance_recall=.95,target_recall=.9,query_deadline_seconds=.01)
        query=lambda curve:result(id=-1,timeout=True)
        with patch("fresh.feasibility.pinned"):
            rows,complete=measure_queries(query,np.zeros((100,1)),np.ones(100),config,tuning=True)
            self.assertEqual(len(rows),6)
            self.assertFalse(complete)
            summary=tuning_summary(rows,100,complete,config)
            self.assertEqual(summary["tuning_success_upper_bound"],.94)
            self.assertIsNone(summary["tuning_median_seconds"])
            self.assertNotEqual(summary["status"],"qualified")
            rows,complete=measure_queries(query,np.zeros((100,1)),np.ones(100),config)
            self.assertEqual(len(rows),100)
            self.assertTrue(complete)

    def test_preflight_only_rejects_sound_order_memory_or_construction_limits(self):
        config=dict(index_budget_bytes=100,peak_ram_cap_bytes=1000000)
        X=np.zeros((1,2,2,2))
        self.assertEqual(preflight_joint(X,10,2,2,100,config),
                         "excluded: mandatory order arrays exceed serving-index budget")
        self.assertEqual(preflight_joint(X,2**31,2,1,2,config),"excluded: integer capacity")
        self.assertIsNone(preflight_joint(X,10,2,2,2,config))
        config["peak_ram_cap_bytes"]=100
        self.assertEqual(preflight_joint(X,10,2,2,2,config),"excluded: construction RAM estimate")

    def test_selection_is_tuning_only_and_unqualified_rows_cannot_win(self):
        rows=[dict(config_id="a",status="qualified",index_bytes=100,tuning_median_seconds=.002,test_ms=0),
              dict(config_id="b",status="qualified",index_bytes=200,tuning_median_seconds=.001,test_ms=10),
              dict(config_id="c",status="rejected",index_bytes=0,tuning_median_seconds=0)]
        self.assertEqual(select_fastest(rows)["config_id"],"b")
        rows[0]["test_ms"],rows[1]["test_ms"]=100,0
        self.assertEqual(select_fastest(rows)["config_id"],"b")
        self.assertIsNone(select_fastest(rows[2:]))

    def test_frozen_protocol_tests_every_object_count_and_three_seeds(self):
        folder=Path(__file__).resolve().parents[1]/"configs"
        config=json.loads((folder/"feasibility-20261014.json").read_text())
        grid=jobs(config)
        self.assertEqual(len(grid),27)
        self.assertEqual({case["k"] for case in grid},set(range(4,13)))
        self.assertEqual({case["n"] for case in grid},{1000000})
        self.assertEqual(config["index_budget_bytes"],1000000000)
        self.assertEqual(config["query_deadline_seconds"],.01)
        self.assertEqual(config["tuning_queries"],500)
        self.assertEqual(config["test_queries"],1000)
        self.assertEqual(config["index_seeds"],[1701,1702,1703])

    def test_small_worker_keeps_full_test_logs_and_actual_serving_memory(self):
        config=json.loads((Path(__file__).resolve().parents[1]/"configs"/"feasibility-20261014.json").read_text())
        config.update(m=8,tuning_queries=8,test_queries=12,warmup_queries=1,
                      Lmax=8,deltas=[128.],plant_anchor_n=32,blocking_storage_multipliers=[1,6],
                      query_deadline_seconds=1.,compilation_deadline_seconds=60)
        case=dict(id="unit",case_id="unit",kind="conjunctive",n=40,k=3,frequency=.18,index_seed=1701)
        with tempfile.TemporaryDirectory() as temporary,patch("fresh.feasibility.pinned"):
            root=Path(temporary)
            write_json(root/"manifest.json",dict(commit="fixture",config_sha256="config"))
            folder=root/"unit";folder.mkdir()
            worker(config,case,folder)
            selections=json.loads((folder/"frozen-main-selections.json").read_text())
            self.assertTrue(selections)
            results=json.loads((folder/"main-results.json").read_text())
            self.assertEqual({row["display_method"] for row in results},set(METHODS))
            with (folder/"test-queries.csv").open() as handle:
                logs=list(csv.DictReader(handle))
            for row in selections.values():
                group=[log for log in logs if log["config_id"]==row["config_id"]]
                self.assertEqual(len(group),12)
                self.assertLessEqual(row["index_bytes"],config["index_budget_bytes"])
                self.assertTrue(all(log["commit"]=="fixture" for log in group))


if __name__=="__main__":
    unittest.main()
