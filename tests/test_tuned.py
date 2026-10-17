"""Tests of recall tuning and the cost model.  Run: pytest -q tests"""
import numpy as np

from curveset import data
from curveset.bench import ground_truth
from curveset.costmodel import CostModel
from curveset.index import index_bytes, make_index
from curveset.tune import first_hit_joint, first_hit_per_curve, tables_for_recall, tune_tables

rng = np.random.default_rng(99)


def setup(k=4, n=400, nq=12):
    X, Qs, _ = data.motif_collection(rng, n, k, 8, 2, extent=200.0, step=2.0, n_motifs=5 * k, rho=0.5,
                                     planted_per_query=2, nq=nq)
    gt = ground_truth(X, Qs)
    return X, Qs, gt


def test_first_hit_matches_indexes():
    k, delta, L = 4, 6.0, 8
    X, Qs, gt = setup(k)
    NN = X[gt[0]]
    Lj, Lp, fj, fp = tune_tables(Qs, NN, k, 2, delta, 0.5, L, 0, 0)
    for Lcheck in (1, 3, L):
        store = make_index("store_all", k, 2, delta, Lcheck, seed=0).build(X)
        pc = make_index("per_curve", k, 2, delta, Lcheck, seed=0).build(X)
        for q in range(len(Qs)):
            res = store.query(X, Qs[q], mode="verify")
            assert (gt[0][q] in set(store.cand[:res["candidates"]].tolist())) == (fj[q] <= Lcheck)
            res = pc.query(X, Qs[q], mode="verify")
            assert (gt[0][q] in set(pc.cand[:res["candidates"]].tolist())) == (fp[q] <= Lcheck)


def test_tables_for_recall():
    first = np.array([1, 2, 2, 3, 5, 9, 11])  # 11 = never within L_max = 10
    assert tables_for_recall(first, 0.5, 10) == 3
    assert tables_for_recall(first, 6 / 7, 10) == 9
    assert tables_for_recall(first, 1.0, 10) is None


def test_cost_model_choice_within_budgets():
    cm = CostModel([1e3, 1e9], [2e-8, 2e-7])
    for k, n, L in ((6, 10 ** 5, 10), (10, 10 ** 5, 80), (12, 10 ** 6, 20)):
        cands = cm.candidates(k, n, L)
        if not cands:
            continue
        cost, a, b, sizes, swapped, mem, t = cands[0]
        assert mem <= 24e9 and t <= 60 and mem == index_bytes(n * a, L)
        assert all(c[0] >= cost for c in cands)
    # a cheaper probe favors probing, a cheaper memory favors storing
    fast = CostModel([1e3, 1e9], [1e-9, 1e-9]).best_blocking(8, 10 ** 5, 20)
    slow = CostModel([1e3, 1e9], [1e-6, 1e-6]).best_blocking(8, 10 ** 5, 20)
    assert fast[1] <= slow[1]
