"""Correctness tests: every kernel against an independent brute-force reference.  Run: pytest -q tests"""
from itertools import permutations
from math import factorial

import numpy as np
import pytest

from curveset import core, data
from curveset.index import make_index, theory_params
from curveset.orders import (all_orders, allocations, blocking_families, blocking_points, compose_inverse,
                             partitions, within_block_orders)

rng = np.random.default_rng(12345)


def ref_hash(C, t, delta):
    pts = [tuple(np.floor((v - t) / delta + 0.5).astype(np.int64)) for v in C]
    out = [pts[0]]
    for p in pts[1:]:
        if p != out[-1]:
            out.append(p)
    return tuple(out)


def ref_frechet(A, B):
    D = np.linalg.norm(A[:, None] - B[None], axis=2)
    F = np.full(D.shape, np.inf)
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            if i == 0 and j == 0:
                prev = 0.0
            else:
                prev = min(F[i - 1, j] if i else np.inf, F[i, j - 1] if j else np.inf,
                           F[i - 1, j - 1] if i and j else np.inf)
            F[i, j] = max(prev, D[i, j])
    return F[-1, -1]


def test_hash_equality_matches_reference():
    agree = collide = 0
    for _ in range(3000):
        d = int(rng.integers(1, 4))
        m = int(rng.integers(1, 9))
        delta = rng.uniform(0.5, 3)
        A = np.cumsum(rng.normal(0, 1, (m, d)), axis=0)
        B = A + rng.normal(0, 0.3, A.shape) if rng.random() < 0.6 else np.cumsum(rng.normal(0, 1, (m, d)), 0)
        t = rng.uniform(0, delta, d)
        ref = ref_hash(A, t, delta) == ref_hash(B, t, delta)
        fast = core.curve_fp(A, t, delta) == core.curve_fp(B, t, delta)
        agree += ref == fast
        collide += ref
    assert agree == 3000 and collide > 300


def test_frechet_matches_reference():
    for _ in range(300):
        A = rng.normal(0, 1, (int(rng.integers(1, 8)), 2))
        B = rng.normal(0, 1, (int(rng.integers(1, 8)), 2))
        assert abs(core.frechet(A, B) - ref_frechet(A, B)) < 1e-12


def test_bottleneck_matches_bruteforce():
    for _ in range(500):
        k = int(rng.integers(1, 7))
        D = rng.integers(0, 5, size=(k, k)).astype(float)  # many ties
        ref = min(max(D[i, p[i]] for i in range(k)) for p in permutations(range(k)))
        assert core.bottleneck(D) == ref


def test_dmatch_and_lower_bound():
    for _ in range(200):
        k = int(rng.integers(1, 5))
        P = np.cumsum(rng.normal(0, 1, (k, 5, 2)), axis=1)
        Q = P[rng.permutation(k)] + rng.normal(0, 0.2, P.shape)
        D = np.array([[ref_frechet(Q[i], P[j]) for j in range(k)] for i in range(k)])
        ref = min(max(D[i, p[i]] for i in range(k)) for p in permutations(range(k)))
        assert abs(core.dmatch(P, Q) - ref) < 1e-12
        assert core.endpoint_lb(P, Q) <= ref + 1e-12


@pytest.mark.parametrize("k", [2, 3, 4, 5, 6])
def test_blocking_unique_factorization(k):
    """Lemma 5.6: every pi = sigma o tau^{-1} for exactly one (sigma, tau), for every blocking, both roles."""
    for sizes in partitions(k):
        for swapped in (False, True):
            S, T = blocking_families(sizes, swapped)
            assert S.shape[0] * T.shape[0] == factorial(k)
            seen = {tuple(compose_inverse(s, t)) for s in S for t in T}
            assert len(seen) == factorial(k)
    assert within_block_orders((k,)).shape[0] == factorial(k)
    assert allocations((1,) * k).shape[0] == factorial(k)
    assert len(blocking_points(k)) >= 1 and all_orders(k).shape[0] == factorial(k)


def test_sorted_key_lemma():
    """Lemma 5.1: equal sorted keys <=> some bijection pairs curves with equal hashes."""
    agree = equal = 0
    for _ in range(1500):
        k = int(rng.integers(2, 5))
        P = np.cumsum(rng.normal(0, 0.6, (k, 4, 2)), axis=1)
        P[:, 0] = rng.normal(0, 0.15, (k, 2))  # near-tied first points
        Q = P[rng.permutation(k)] + rng.normal(0, 0.05, P.shape)
        t = rng.uniform(0, 1.0, 2)
        lhs = core.sorted_keys(P[None], t, 1.0)[0] == core.sorted_keys(Q[None], t, 1.0)[0]
        hp = [ref_hash(c, t, 1.0) for c in P]
        hq = [ref_hash(c, t, 1.0) for c in Q]
        rhs = any(all(hq[i] == hp[p[i]] for i in range(k)) for p in permutations(range(k)))
        agree += lhs == rhs
        equal += lhs
    assert agree == 1500 and equal > 100


@pytest.mark.parametrize("method", ["store_all", "probe_all", "blocking", "sorted"])
def test_exact_copies_always_found_and_results_certified(method):
    """A shuffled exact copy collides in every table; every returned set is within sqrt(d)*delta."""
    k, m, d, n = 4, 8, 2, 300
    X = data.collection(rng, n, k, m, d, extent=400.0, step=3.0)
    delta = 20.0
    idx = make_index(method, k, d, delta, L=3, seed=1).build(X)
    for q in range(20):
        t = int(rng.integers(0, n))
        Q = X[t, rng.permutation(k)]
        res = idx.query(X, Q, mode="first")
        assert res["id"] >= 0
        assert core.dmatch(X[res["id"]], Q) <= np.sqrt(d) * delta + 1e-9


def test_recall_of_planted_neighbors_with_theory_parameters():
    """Theorem 5.7 (blocking): planted neighbors within r are found with probability >= 1 - eta."""
    k, m, d, r, p, eta = 3, 8, 2, 1.0, 0.75, 0.05
    tp = theory_params(d, m, r, p, k, eta)
    X = data.collection(rng, 2000, k, m, d, extent=50 * tp["delta"], step=0.2 * tp["delta"])
    Q, targets = data.planted_queries(rng, X, 200, rho=r)
    idx = make_index("blocking", k, d, tp["delta"], tp["L"], seed=2).build(X)
    hits = 0
    for q in range(len(Q)):
        res = idx.query(X, Q[q], mode="first")
        if res["id"] >= 0:
            assert core.dmatch(X[res["id"]], Q[q]) <= tp["c"] * r + 1e-9
            hits += 1
    assert hits / len(Q) >= 1 - eta - 0.05
