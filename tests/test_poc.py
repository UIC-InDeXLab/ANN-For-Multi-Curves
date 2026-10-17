import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from curveset import core
from curveset.orders import blocking_families
from fresh.e1 import build_index, write_json
from curveset.index import make_index
from fresh.poc import (Joint, PerCurve, choose_blocking, compact_table, dataset,
                       family_specs, worker, weighted_cost)
from fresh.regimes import joint_query, PerCurve as UncompressedPerCurve


def config():
    path = Path(__file__).resolve().parents[1]/"configs"/"poc-20261012.json"
    return json.loads(path.read_text())


class PocTests(unittest.TestCase):
    def test_compaction_preserves_exact_parent_sets_and_storage(self):
        keys = np.array([7,7,7,8,7,9,7,9,7],np.uint64)
        table = compact_table(keys,3)
        observed = set(zip(table.keys.tolist(),table.ids.tolist()))
        expected = {(int(key),i//3) for i,key in enumerate(keys)}
        self.assertEqual(observed,expected)
        self.assertEqual(table.N,len(expected))
        self.assertTrue(np.all(table.keys[1:]>=table.keys[:-1]))
        self.assertEqual(int(table.dir[-1]),table.N)
        for j in range(table.B):
            low,high = int(table.dir[j]),int(table.dir[j+1])
            self.assertLessEqual(low,high)

    def test_joint_compression_preserves_candidates_for_every_family(self):
        rng = np.random.default_rng(419)
        X = rng.normal(0,.05,(15,4,5,2))
        X[0,:,0] = [[0,0],[10,0],[10,0],[0,0]]
        Q = X[0,[3,2,1,0]].copy()
        cfg = config()
        cfg.update(probe_chunk=3,verification_chunk=3)
        for sizes,swapped in [((1,1,1,1),False),((4,),False),((2,2),False),((2,1,1),True)]:
            old = build_index(make_index("blocking",4,2,4,3,sizes=sizes,swapped=swapped,seed=1),X)
            uncompressed = joint_query(old,X,Q,cfg)
            index = Joint(X,4,3,1,sizes,swapped)
            result = index.query(X,Q,cfg)
            self.assertEqual(set(index.base.cand[:result["candidates"]]),
                             set(old.cand[:uncompressed["candidates"]]))
            self.assertAlmostEqual(result["distance"],uncompressed["distance"])
            self.assertLessEqual(index.entries,index.raw_entries)
            self.assertLessEqual(index.index_bytes,old.index_bytes+old.stored.nbytes+old.probed.nbytes+
                                 old.shifts.nbytes+old.need_s.nbytes+old.need_q.nbytes)

    def test_compacted_per_curve_matches_uncompressed_and_does_not_reuse_objects(self):
        X = np.zeros((4,3,5,2))
        X[0,:,: ,0] = np.array([10,20,20])[:,None]
        X[1,:,: ,0] = np.array([10,10,20])[:,None]
        X[2] += 100
        X[3] += 200
        Q = X[1].copy()
        cfg = config()
        index = PerCurve(X,2,3,8)
        old = UncompressedPerCurve(X,2,3,8)
        self.assertLess(index.entries,index.raw_entries)
        for _ in range(3):
            result = index.query(X,Q,cfg)
            reference = old.query(X,Q,cfg)
            self.assertEqual(set(index.candidates[:result["candidates"]]),
                             set(old.candidates[:reference["candidates"]]))
            self.assertEqual(result["id"],1)
            self.assertEqual(result["distance"],0)
            self.assertEqual(index.query(X,Q,cfg,True)["distance"],0)
        self.assertEqual(core.dmatch(X[0],Q),10)

    def test_weights_use_tuning_values_only(self):
        cfg = config()
        rows = [
            dict(config_id="low-memory",method="blocking-family",status="qualified",index_bytes=1e6,
                 tuning_median_seconds=.001,test_median_seconds=100),
            dict(config_id="low-latency",method="blocking-family",status="qualified",index_bytes=50e6,
                 tuning_median_seconds=.0001,test_median_seconds=0),
            dict(config_id="bad-quality",method="blocking-family",status="tuning recall below target",
                 index_bytes=0,tuning_median_seconds=0)]
        self.assertEqual(choose_blocking(rows,.1,cfg)["config_id"],"low-memory")
        self.assertEqual(choose_blocking(rows,1.,cfg)["config_id"],"low-latency")
        rows[0]["test_median_seconds"],rows[1]["test_median_seconds"]=0,1000
        self.assertEqual(choose_blocking(rows,1.,cfg)["config_id"],"low-latency")
        self.assertAlmostEqual(weighted_cost(rows[0],1.,cfg),1.01)

    def test_small_dataset_planting_and_nested_sizes(self):
        cfg = config()
        case = dict(id="small",kind="conjunctive",n=2000,k=4,frequency=.18)
        X,Q,info = dataset(cfg,case)
        other,queries,_ = dataset(cfg,dict(case,n=2001))
        self.assertEqual(len(Q),1210)
        self.assertEqual(info["planted_sets"],1210)
        self.assertTrue(np.array_equal(X,other[:2000]))
        self.assertTrue(np.array_equal(Q,queries))
        for a,b,sizes,swapped in family_specs(4,cfg):
            S,T = blocking_families(sizes,swapped)
            self.assertEqual(len(S),a)
            self.assertEqual(len(T),b)

    def test_worker_selects_one_main_blocking_and_preserves_ablation(self):
        cfg = config()
        cfg.update(m=8,tuning_queries=8,test_queries=12,warmup_queries=1,Lmax=8,deltas=[128.],
                   plant_anchor_n=32,blocking_storage_multipliers=[1,6])
        case = dict(id="unit",kind="conjunctive",n=40,k=3,frequency=.18)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root/"manifest.json",dict(commit="test",config_sha256="test"))
            folder=root/"unit"
            folder.mkdir()
            worker(cfg,case,folder)
            rows=json.loads((folder/"main-results.json").read_text())
            self.assertEqual(sum(r["display_method"]=="blocking" for r in rows),1)
            self.assertNotIn("blocking-balanced",[r["display_method"] for r in rows])
            self.assertNotIn("blocking-cost",[r["display_method"] for r in rows])
            ablation=json.loads((folder/"ablation-results.json").read_text())
            self.assertEqual(len(ablation),4)
            with (folder/"test-queries.csv").open() as handle:
                queries=list(csv.DictReader(handle))
            for row in rows:
                if row["test_status"]=="complete":
                    self.assertEqual(sum(q["config_id"]==row["config_id"] for q in queries),12)

    def test_compaction_round_trips_adversarial_equal_keys_and_buckets(self):
        rng=np.random.default_rng(771)
        for divisor in (1,3,20):
            keys=np.concatenate([
                np.full(200,np.uint64(0x1234567800000003),np.uint64),
                np.full(200,np.uint64(0x1234567800000005),np.uint64),
                rng.integers(0,2**63,600,dtype=np.uint64)])
            rng.shuffle(keys)
            table=compact_table(keys,divisor)
            self.assertEqual(set(zip(table.keys.tolist(),table.ids.tolist())),
                             {(int(key),i//divisor) for i,key in enumerate(keys)})
            self.assertEqual(table.N,len({(int(key),i//divisor) for i,key in enumerate(keys)}))


if __name__=="__main__":
    unittest.main()
