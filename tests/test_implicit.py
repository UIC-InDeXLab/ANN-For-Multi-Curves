"""Implicit collections: every kernel against the explicit implementation on materialized data.
Run: pytest -q tests"""
import numpy as np
import pytest

from curveset import core, data
from curveset.implicit import (Implicit, ImplicitPerCurve, fill_table, ground_truth, make_joint, scan, verify_sets)
from curveset.index import Tables, brute_force, make_index

rng = np.random.default_rng(2024)


def small(n=1500, k=4, P=8, rho=0.5):
    paths = data.random_walks(rng, P, 12, 2, extent=60.0, step=2.0)
    pid = rng.integers(0, P, size=(n, k)).astype(np.uint16)
    return Implicit(paths, pid, 77, rho)


def queries(D, nq=8, rho=0.5):
    src = rng.choice(D.n, size=nq, replace=False)
    X = D.materialize(src)
    return np.stack([data.jitter(rng, X[q][rng.permutation(D.k)], rho) for q in range(nq)])


def test_materialize_and_jitter():
    D = small()
    X = D.materialize(np.arange(D.n))
    base = D.paths[D.pid.astype(np.int64)]
    rad = np.linalg.norm(X - base, axis=-1)
    assert rad.max() <= D.rho + 1e-12 and abs(rad.mean() - 2 * D.rho / 3) < 0.01
    assert np.array_equal(X, D.materialize(np.arange(D.n)))  # deterministic


@pytest.mark.parametrize("method,sizes,swapped", [("probe_all", None, False), ("store_all", None, False),
                                                  ("blocking", (2, 1, 1), False), ("blocking", (2, 2), True)])
def test_joint_candidates_match_explicit(method, sizes, swapped):
    D = small()
    X = D.materialize(np.arange(D.n))
    Qs = queries(D)
    imp = make_joint(method, D.k, 6.0, 3, sizes=sizes, swapped=swapped, seed=5).build(D, chunk=256)
    exp = make_index(method, D.k, 2, 6.0, 3, seed=5, sizes=sizes, swapped=swapped).build(X)
    total = 0
    for Q in Qs:
        ri = imp.query(D, Q, r=1.0)
        re = exp.query(X, Q, mode="verify")
        ci = set(imp.cand[:ri["candidates"]].tolist())
        ce = set(exp.cand[:re["candidates"]].tolist())
        assert ci == ce
        total += len(ci)
        if ci:
            assert ri["dist"] == pytest.approx(re["dist"]) and ri["id"] == re["id"]
            assert ri["relevant_found"] == sum(core.dmatch(X[p], Q) < 1.0 for p in ci)
    assert total > 0


def test_per_curve_full_capped_topk():
    D = small()
    X = D.materialize(np.arange(D.n))
    Qs = queries(D)
    imp = ImplicitPerCurve(D.k, 6.0, 2, seed=3).build(D, chunk=300)
    exp = make_index("per_curve", D.k, 2, 6.0, 2, seed=3).build(X)
    for Q in Qs:
        full = imp.query(D, Q, r=1.0, cap=0)
        cf = set(imp.cand[:full["candidates"]].tolist())
        re = exp.query(X, Q)
        assert cf == set(exp.cand[:re["candidates"]].tolist()) and full["listed"] == re["listed"]
        big = imp.query(D, Q, r=1.0, cap=10 ** 9)
        assert set(imp.cand[:big["candidates"]].tolist()) == cf
        cap = imp.query(D, Q, r=1.0, cap=5, rng_seed=9)
        assert set(imp.cand[:cap["candidates"]].tolist()) <= cf and cap["listed"] <= full["listed"]
        top = imp.query_topk(D, Q, r=1.0, K=10 ** 9)
        assert set(imp.cand[:top["candidates"]].tolist()) == cf
        top = imp.query_topk(D, Q, r=1.0, K=3)
        assert set(imp.cand[:top["candidates"]].tolist()) <= cf


def test_ground_truth_and_brute_force():
    D = small(n=800)
    X = D.materialize(np.arange(D.n))
    Qs = queries(D, nq=6)
    ids, ds, nrel = ground_truth(D.paths, D.pid, D.seed, D.rho, Qs, 1.5)
    for q, Q in enumerate(Qs):
        exact = np.array([core.dmatch(X[p], Q) for p in range(D.n)])
        assert ds[q] == pytest.approx(exact.min()) and exact[ids[q]] == pytest.approx(exact.min())
        assert nrel[q] == int((exact < 1.5).sum())
        b_id, b_d, _ = scan(D.paths, D.pid, D.seed, D.rho, Q, 1.5, False)
        assert b_d == pytest.approx(exact.min())
        e_id, e_d = brute_force(X, Q)
        assert e_d == pytest.approx(b_d)
        cand = np.arange(D.n, dtype=np.int64)
        v_id, v_d, v_rel = verify_sets(D.paths, D.pid, D.seed, D.rho, Q, cand, 0, D.n, 1.5, np.inf, -1)
        assert v_d == pytest.approx(exact.min()) and v_rel == nrel[q]


def test_fill_table_adversarial_buckets():
    T = Tables(1, 6000)
    # two long runs of different keys in the same bucket, interleaved, plus random keys
    base = np.uint64(0x1234567800000000)
    keys = np.concatenate([np.full(2000, base + np.uint64(5)), np.full(2000, base + np.uint64(3)),
                           rng.integers(0, 2 ** 63, 2000, dtype=np.uint64)])
    perm = rng.permutation(keys.size)
    keys = keys[perm]
    fill_table(keys, 1, T.B, T.keys, T.ids, T.dir)
    assert sorted(T.ids.tolist()) == list(range(keys.size))
    assert np.array_equal(keys[T.ids], T.keys)
    for j in range(T.B):
        seg = T.keys[T.dir[j]:T.dir[j + 1]]
        assert np.all(seg[1:] >= seg[:-1])
