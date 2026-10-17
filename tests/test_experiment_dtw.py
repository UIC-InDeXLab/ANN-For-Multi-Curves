import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from curveset.core import dmatch
from fresh.dtw import dtw_verify
from fresh.e1 import arrays_hash,write_json
from fresh.experiment_dtw import METHODS,exact_truth,load_dataset,study_jobs,worker
from fresh.poc import Joint,PerCurve,dataset


def protocol():
    path=Path(__file__).resolve().parents[1]/"configs"/"dtw-20261014.json"
    return json.loads(path.read_text())


class DtwExperimentTests(unittest.TestCase):
    def test_study_has_five_cases_and_three_seeds_each(self):
        config=protocol()
        jobs=study_jobs(config)
        self.assertEqual(len(jobs),15)
        self.assertEqual(len({job["id"] for job in jobs}),15)
        for case in config["cases"]:
            seeds={job["index_seed"] for job in jobs if job["case_id"]==case["id"]}
            expected=config["nba_index_seeds"] if case["kind"]=="nba" else config["synthetic_index_seeds"]
            self.assertEqual(seeds,set(expected))
        self.assertEqual(set(METHODS),{"blocking","probe_all","store_all","per_curve-intersection"})

    def test_metric_callback_changes_ranking_without_changing_default(self):
        X=np.zeros((2,1,3,2))
        X[0,0,:,0]=[0,0,2]
        X[1,0,:,0]=[1,1,1]
        Q=np.zeros((1,3,2))
        config=protocol()
        for index in (Joint(X,100.,3,1,(1,),False),PerCurve(X,100.,3,1)):
            with self.subTest(index=type(index).__name__):
                frechet=index.query(X,Q,config)
                dtw=index.query(X,Q,config,verifier=dtw_verify)
                self.assertEqual((frechet["id"],frechet["distance"]),(1,1.))
                self.assertEqual((dtw["id"],dtw["distance"]),(0,2.))
                self.assertEqual(index.query(X,Q,config)["id"],1)

    def test_cached_truth_preserves_metric_and_rejects_stale_arrays(self):
        X=np.zeros((2,1,3,2))
        X[0,0,:,0]=[0,0,2]
        X[1,0,:,0]=[1,1,1]
        Q=np.zeros((1,1,3,2))
        info=dict(dataset_sha256=arrays_hash(X),queries_sha256=arrays_hash(Q))
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/"job";output.mkdir()
            first=exact_truth(X,Q,info,np.array([1]),output,dict(case_id="unit"))
            second=exact_truth(X,Q,info,np.array([1]),output,dict(case_id="unit"))
            self.assertFalse(first[-1])
            self.assertTrue(second[-1])
            np.testing.assert_equal(first[0],[0])
            np.testing.assert_equal(first[1],[2.])
            np.testing.assert_equal(first[2],[3.])
            for left,right in zip(first[:3],second[:3]):
                np.testing.assert_equal(left,right)
            with self.assertRaisesRegex(ValueError,"stale provenance"):
                exact_truth(X,Q,dict(info,queries_sha256="changed"),np.array([1]),output,dict(case_id="unit"))
            with self.assertRaisesRegex(ValueError,"stale provenance"):
                exact_truth(X,Q,info,np.array([0]),output,dict(case_id="unit"))

    def test_synthetic_loader_requires_identical_control_queries(self):
        config=protocol()
        config.update(tuning_queries=4,test_queries=5,warmup_queries=2,plant_anchor_n=1000)
        case=dict(id="unit",kind="rare",n=1000,k=2,frequency=.005)
        X,Q,info=dataset(config,case)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            config["synthetic_reference"]=str(root)
            folder=root/"rare-n1000-k2-seed1701";folder.mkdir()
            write_json(folder/"dataset.json",info)
            np.savez(folder/"ground-truth.npz",queries=Q,ids=np.zeros(len(Q),np.int64),
                     distances=np.ones(len(Q)))
            loaded=load_dataset(config,case)
            np.testing.assert_equal(loaded[0],X)
            np.testing.assert_equal(loaded[1],Q)
            changed=Q.copy();changed[0,0,0,0]+=1
            np.savez(folder/"ground-truth.npz",queries=changed,ids=np.zeros(len(Q),np.int64),
                     distances=np.ones(len(Q)))
            with self.assertRaisesRegex(ValueError,"query order changed"):
                load_dataset(config,case)

    def test_worker_uses_exact_dtw_truth_for_every_retained_method(self):
        config=protocol()
        config.update(tuning_queries=4,test_queries=5,warmup_queries=2,deltas=[256.],Lmax=3,
                      verification_chunk=5,blocking_storage_multipliers=[1,2])
        rng=np.random.default_rng(829)
        X=rng.uniform(0,3,(30,2,4,2))
        Q=np.concatenate((X[:9]+rng.normal(0,.01,(9,2,4,2)),X[:2]))
        initial=np.arange(11)%30
        costs=np.asarray([dmatch(X[parent],query) for parent,query in zip(initial,Q)])
        info=dict(dataset_sha256=arrays_hash(X),queries_sha256=arrays_hash(Q))
        case=dict(id="unit-seed1701",case_id="unit",kind="rare",n=30,k=2,index_seed=1701)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            output=root/case["id"];output.mkdir()
            write_json(root/"manifest.json",dict(commit="unit",config_sha256="unit"))
            with patch("fresh.experiment_dtw.load_dataset",return_value=(X,Q,info,initial,costs)):
                worker(config,case,output)
            results=json.loads((output/"main-results.json").read_text())
            self.assertEqual({row["display_method"] for row in results},set(METHODS))
            self.assertTrue(all(row["test_status"]=="complete" for row in results))
            self.assertTrue(all(row["test_queries"]==5 for row in results))
            with (output/"test-queries.csv").open() as handle:
                rows=list(csv.DictReader(handle))
            self.assertTrue(rows)
            self.assertTrue(all(row["metric"]=="DTW" and row["exact_nn"]=="True" for row in rows))
            with np.load(output/"ground-truth.npz",allow_pickle=False) as truth:
                for row in rows:
                    self.assertAlmostEqual(float(row["nn_distance"]),truth["distances"][4+int(row["query"])])


if __name__=="__main__":
    unittest.main()
