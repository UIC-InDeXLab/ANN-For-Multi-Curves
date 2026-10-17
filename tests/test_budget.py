"""Tests of the budgeted index implementation: order generators, fused query kernels, storage accounting, and
deadlines, each against an independent pure-Python reference.  Run: pytest -q tests"""
from itertools import combinations, permutations
from math import factorial

import numpy as np
import pytest

from curveset import core, data
from curveset.bench import budget_blocking
from curveset.index import index_bytes, make_index
from curveset.orders import (all_orders, allocations, blocking_families, blocking_points, partitions,
                             within_block_orders)

rng = np.random.default_rng(777)


def ref_within(sizes):
    k, rows, start = sum(sizes), [tuple(range(sum(sizes)))], 0
    for b in sizes:
        B = list(range(start, start + b))
        new = []
        for base in rows:
            for perm in permutations(B):
                row = list(base)
                for pos, v in zip(B, perm):
                    row[pos] = v
                new.append(tuple(row))
        rows, start = new, start + b
    return set(rows)


def ref_alloc(sizes):
    k, out = sum(sizes), set()
    starts = np.cumsum([0] + list(sizes[:-1]))

    def rec(j, remaining, row):
        if j == len(sizes):
            out.add(tuple(row))
            return
        for T in combinations(remaining, sizes[j]):
            for off, v in enumerate(T):
                row[starts[j] + off] = v
            rec(j + 1, [x for x in remaining if x not in T], row)

    rec(0, list(range(k)), [0] * k)
    return out


def is_lex_sorted(A):
    return all(tuple(A[i]) < tuple(A[i + 1]) for i in range(len(A) - 1))


@pytest.mark.parametrize("k", [1, 2, 3, 4, 5, 6])
def test_generators_match_reference(k):
    A = all_orders(k)
    assert A.dtype == np.int8 and A.shape == (factorial(k), k) and is_lex_sorted(A)
    assert set(map(tuple, A)) == set(permutations(range(k)))
    for sizes in partitions(k):
        U, T = within_block_orders(sizes), allocations(sizes)
        assert U.dtype == np.int8 and T.dtype == np.int8
        assert is_lex_sorted(U) and is_lex_sorted(T)
        assert set(map(tuple, U)) == ref_within(sizes) and len(U) == len(ref_within(sizes))
        assert set(map(tuple, T)) == ref_alloc(sizes) and len(T) == len(ref_alloc(sizes))


def ref_candidates(X, Q, idx):
    """Curve sets with a stored key equal to a probed key in some table (pure Python)."""
    k = X.shape[1]
    found = set()
    for u in range(idx.L):
        sh = idx.shifts[u]
        stored = {}
        for p in range(X.shape[0]):
            for s in idx.stored:
                fp = np.array([core.curve_fp(X[p, s[i]], sh[i], idx.delta) for i in range(k)], np.uint64)
                stored.setdefault(int(core.chain(fp)), set()).add(p)
        for t in idx.probed:
            fp = np.array([core.curve_fp(Q[t[i]], sh[i], idx.delta) for i in range(k)], np.uint64)
            found |= stored.get(int(core.chain(fp)), set())
    return found


@pytest.mark.parametrize("method,sizes,swapped", [("store_all", None, False), ("probe_all", None, False),
                                                  ("blocking", (2, 1, 1), False), ("blocking", (2, 2), True)])
def test_order_index_candidates_match_reference(method, sizes, swapped):
    k, m, d, n = 4, 6, 2, 120
    base = np.cumsum(rng.normal(0, 1.0, (20, k, m, d)), axis=2)
    X = np.concatenate([base[rng.integers(0, 20, n)][:, :, :, :] + rng.normal(0, 0.15, (n, k, m, d))])
    idx = make_index(method, k, d, 3.0, 3, seed=4, sizes=sizes, swapped=swapped).build(X)
    assert idx.index_bytes == index_bytes(n * idx.stored.shape[0], 3)
    total = 0
    for q in range(6):
        Q = base[q, rng.permutation(k)] + rng.normal(0, 0.15, (k, m, d))
        res = idx.query(X, Q, mode="verify")
        got = set(idx.cand[:res["candidates"]].tolist())
        ref = ref_candidates(X, Q, idx)
        assert got == ref
        total += len(ref)
        first = idx.query(X, Q, mode="first")
        assert (first["id"] >= 0) == bool(ref) and (first["id"] < 0 or first["id"] in ref)
        if ref:
            assert res["id"] >= 0 and res["dist"] == min(core.dmatch(X[p], Q) for p in ref)
    assert total > 0


def test_sorted_and_per_curve_match_reference():
    k, m, d, n, L, delta = 3, 6, 2, 150, 3, 3.0
    base = np.cumsum(rng.normal(0, 1.0, (15, k, m, d)), axis=2)
    X = base[rng.integers(0, 15, n)] + rng.normal(0, 0.15, (n, k, m, d))
    srt = make_index("sorted", k, d, delta, L, seed=5).build(X)
    pc = make_index("per_curve", k, d, delta, L, seed=6).build(X)
    nonempty = 0
    for q in range(8):
        Q = base[q, rng.permutation(k)] + rng.normal(0, 0.15, (k, m, d))
        ref_s = {p for u in range(L) for p in range(n)
                 if core.sorted_keys(X[p:p + 1], srt.shifts[u], delta)[0]
                 == core.sorted_keys(Q[None], srt.shifts[u], delta)[0]}
        res = srt.query(X, Q, mode="verify")
        assert set(srt.cand[:res["candidates"]].tolist()) == ref_s
        per = []
        for i in range(k):
            hit = set()
            for u in range(L):
                fq = core.curve_fp(Q[i], pc.shifts[u], delta)
                hit |= {p for p in range(n) for l in range(k) if core.curve_fp(X[p, l], pc.shifts[u], delta) == fq}
            per.append(hit)
        ref_c = set.intersection(*per)
        res = pc.query(X, Q)
        assert set(pc.cand[:res["candidates"]].tolist()) == ref_c
        assert res["listed"] == sum(len(h) for h in per)
        nonempty += bool(ref_s) + bool(ref_c)
    assert nonempty > 0


def test_deadline_stops_long_queries():
    k = 9
    X = data.collection(rng, 50, k, 8, 2, extent=100.0, step=1.0)
    idx = make_index("probe_all", k, 2, 4.0, 40, seed=1).build(X)
    res = idx.query(X, X[0], mode="verify", deadline=0.05)
    assert res["timeout"] and res["probes"] < 40 * factorial(k)


def test_budget_blocking():
    a, b, sizes, swapped = budget_blocking(4, 1000, 10, mem=1e12)
    assert (a, b) == (24, 1)  # storing all orders fits
    a, b, sizes, swapped = budget_blocking(8, 10 ** 5, 44, mem=24e9)
    assert index_bytes(10 ** 5 * a, 44) <= 24e9 and a * b == factorial(8)
    assert all(index_bytes(10 ** 5 * a2, 44) > 24e9 for a2, _, _, _ in blocking_points(8) if a2 > a)


def test_run_method_end_to_end(tmp_path):
    import pandas as pd
    from curveset.bench import Writer, ground_truth, run_method
    k, n = 3, 300
    X = data.collection(rng, n, k, 8, 2, extent=200.0, step=1.0)
    Qs, targets = data.planted_queries(rng, X, 8, rho=0.5)
    gt = ground_truth(X, Qs)
    w = Writer(tmp_path / "rows.csv")
    for method, sizes, swapped in (("store_all", None, False), ("probe_all", None, False),
                                   ("blocking", (2, 1), True), ("sorted", None, False), ("per_curve", None, False)):
        run_method(w, "t", "synthetic", X, Qs, targets, gt, method, k, 2, 40.0, 4, np.inf, sizes=sizes,
                   swapped=swapped, modes=("first", "verify"), max_queries=8)
    run_method(w, "t", "synthetic", X, Qs, targets, gt, "store_all", k, 2, 8.0, 4, np.inf, mem_budget=1e3)
    run_method(w, "t", "synthetic", X, Qs, targets, gt, "probe_all", k, 2, 8.0, 4, np.inf, time_budget=1e-12)
    w.f.close()
    df = pd.read_csv(tmp_path / "rows.csv")
    ok = df[df.status == "ok"]
    assert set(ok.method) == {"store_all", "probe_all", "blocking", "sorted", "per_curve"}
    found = ok[ok.found == 1]
    assert ok[ok["mode"] == "verify"].found.mean() > 0.5
    assert (found.d_ret >= found.d_nn - 1e-9).all() and (found[found["mode"] == "verify"].d_ret
                                                          <= found[found["mode"] == "verify"].d_nn * 100).all()
    assert set(df.status) == {"ok", "over memory budget", "over time budget (estimated)"}
