"""Parameter tuning for a target recall: the fewest tables L such that the exact nearest neighbor of at least a
target fraction of the queries becomes a candidate.

Joint keys (store all, probe all, blocking): in table u, the nearest neighbor P* of query Q is found iff some
matching pi makes every slot collide. For storing all orders, slot i holds query curve Q_i with shift t_{u,i}, so
this is a perfect matching in the bipartite graph {(i, l) : h_{t_{u,i}}(Q_i) = h_{t_{u,i}}(P*_l)}. Probing all
orders and blocking assign the shifts to curve pairs differently but have the same distribution.

Per-curve LSH: P* is a candidate after L tables iff every query curve collided with some curve of P* in one of the
first L tables (one shift per table, shared by all curves).
"""
import numpy as np
from numba import njit

from . import core


@njit(cache=True)
def first_hit_joint(Qs, NN, shifts, delta):
    """For every query, the first table (1-based) whose joint key finds its nearest neighbor; L+1 if none.
    Qs, NN: (nq, k, m, d); shifts: (L, k, d)."""
    nq, k = Qs.shape[0], Qs.shape[1]
    L = shifts.shape[0]
    out = np.full(nq, L + 1, np.int64)
    D = np.empty((k, k))
    row_of_col = np.empty(k, np.int64)
    col_of_row = np.empty(k, np.int64)
    parent = np.empty(k, np.int64)
    visited = np.empty(k, np.bool_)
    queue = np.empty(k, np.int64)
    for q in range(nq):
        for u in range(L):
            for i in range(k):
                fq = core.curve_fp(Qs[q, i], shifts[u, i], delta)
                for l in range(k):
                    D[i, l] = 0.0 if core.curve_fp(NN[q, l], shifts[u, i], delta) == fq else 1.0
            if core._has_perfect_matching(D, 0.5, row_of_col, col_of_row, parent, visited, queue):
                out[q] = u + 1
                break
    return out


@njit(cache=True)
def first_hit_per_curve(Qs, NN, shifts, delta):
    """For every query, the number of per-curve tables after which its nearest neighbor is a candidate; L+1 if
    never. shifts: (L, d)."""
    nq, k = Qs.shape[0], Qs.shape[1]
    L = shifts.shape[0]
    out = np.full(nq, L + 1, np.int64)
    first = np.empty(k, np.int64)
    for q in range(nq):
        first[:] = L + 1
        for u in range(L):
            done = True
            for i in range(k):
                if first[i] <= L:
                    continue
                fq = core.curve_fp(Qs[q, i], shifts[u], delta)
                for l in range(k):
                    if core.curve_fp(NN[q, l], shifts[u], delta) == fq:
                        first[i] = u + 1
                        break
                if first[i] > L:
                    done = False
            if done:
                break
        out[q] = first.max()
    return out


def tables_for_recall(first, target, L_max):
    """Smallest L with fraction(first <= L) >= target, or None if more than L_max tables would be needed."""
    f = np.sort(np.asarray(first))
    L = int(f[int(np.ceil(target * len(f))) - 1])
    return L if L <= L_max else None


def tune_tables(Qs, NN, k, d, delta, target, L_max, seed_joint, seed_pc):
    """(L for joint keys, L for per-curve LSH), each None if not reachable within L_max tables. The shift seeds are
    those of the indexes, so the first tables of an index are exactly the tables analyzed here."""
    sj = np.random.default_rng(seed_joint).uniform(0, delta, size=(L_max, k, d))
    sp = np.random.default_rng(seed_pc).uniform(0, delta, size=(L_max, d))
    fj = first_hit_joint(Qs, NN, sj, delta)
    fp = first_hit_per_curve(Qs, NN, sp, delta)
    return tables_for_recall(fj, target, L_max), tables_for_recall(fp, target, L_max), fj, fp
