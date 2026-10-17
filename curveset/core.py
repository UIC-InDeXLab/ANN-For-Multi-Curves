"""Numba kernels: Driemel-Silvestri grid hash, order-sensitive key chaining, discrete Frechet distance and the
matching distance d_match (bottleneck assignment over pairwise Frechet distances).

Curves are float64 arrays of shape (m, d); a curve set is (k, m, d); a collection is (n, k, m, d).
"""
import numpy as np
from numba import njit, prange

SEED = np.uint64(0x9E3779B97F4A7C15)
_OFF = np.int64(1 << 40)  # keeps cell indices positive before the unsigned cast


@njit(inline="always")
def mix64(x):
    """splitmix64 finalizer: a bijective 64-bit mixer."""
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


@njit
def curve_fp(C, t, delta):
    """64-bit fingerprint of h_delta^t(C): snap every vertex to the closest point of delta*Z^d + t, drop
    consecutive duplicates, and fold the resulting grid points in order (plus their number)."""
    m, d = C.shape
    prev = np.empty(d, np.int64)
    cell = np.empty(d, np.int64)
    h = SEED
    count = 0
    for j in range(m):
        same = j > 0
        for a in range(d):
            c = np.int64(np.floor((C[j, a] - t[a]) / delta + 0.5))
            cell[a] = c
            if j > 0 and c != prev[a]:
                same = False
        if not same:
            for a in range(d):
                h = mix64(h ^ np.uint64(cell[a] + _OFF))
                prev[a] = cell[a]
            count += 1
    return mix64(h ^ np.uint64(count))


@njit
def chain(fps):
    """Order-sensitive combination of per-slot fingerprints into one key."""
    h = SEED
    for i in range(fps.shape[0]):
        h = mix64(h + fps[i])
    return h


@njit(parallel=True)
def slot_fps(X, shifts, delta, need):
    """F[p, i, l] = fingerprint of curve l of set p under the shift of slot i, for the pairs with need[i, l].
    X: (n, k, m, d); shifts: (k, d)."""
    n, k = X.shape[0], X.shape[1]
    F = np.zeros((n, k, k), np.uint64)
    for p in prange(n):
        for i in range(k):
            for l in range(k):
                if need[i, l]:
                    F[p, i, l] = curve_fp(X[p, l], shifts[i], delta)
    return F


@njit(parallel=True)
def order_keys(F, orders):
    """keys[p, s] = chain(F[p, i, orders[s, i]] over slots i): the key of set p listed in order s."""
    n, k = F.shape[0], F.shape[1]
    a = orders.shape[0]
    keys = np.empty((n, a), np.uint64)
    for p in prange(n):
        buf = np.empty(k, np.uint64)
        for s in range(a):
            for i in range(k):
                buf[i] = F[p, i, orders[s, i]]
            keys[p, s] = chain(buf)
    return keys


@njit(parallel=True)
def sorted_keys(X, t, delta):
    """Sorted key K_t: fingerprints of all curves on one shared grid, sorted, then chained."""
    n, k = X.shape[0], X.shape[1]
    keys = np.empty(n, np.uint64)
    for p in prange(n):
        buf = np.empty(k, np.uint64)
        for l in range(k):
            buf[l] = curve_fp(X[p, l], t, delta)
        buf.sort()
        keys[p] = chain(buf)
    return keys


@njit(parallel=True)
def curve_fps_all(X, t, delta):
    """fps[p, l] = fingerprint of curve l of set p under one shift t (per-curve index)."""
    n, k = X.shape[0], X.shape[1]
    out = np.empty((n, k), np.uint64)
    for p in prange(n):
        for l in range(k):
            out[p, l] = curve_fp(X[p, l], t, delta)
    return out


# ------------------------------------------------------------------------------------------ distances
@njit
def frechet(A, B):
    """Discrete Frechet distance between polygonal curves A (m1, d) and B (m2, d)."""
    m1, m2, d = A.shape[0], B.shape[0], A.shape[1]
    prev = np.empty(m2)
    cur = np.empty(m2)
    for i in range(m1):
        for j in range(m2):
            s = 0.0
            for a in range(d):
                diff = A[i, a] - B[j, a]
                s += diff * diff
            e = np.sqrt(s)
            if i == 0 and j == 0:
                best = e
            elif i == 0:
                best = max(cur[j - 1], e)
            elif j == 0:
                best = max(prev[j], e)
            else:
                best = max(min(prev[j], prev[j - 1], cur[j - 1]), e)
            cur[j] = best
        prev, cur = cur, prev
    return prev[m2 - 1]


@njit
def _has_perfect_matching(D, theta, row_of_col, col_of_row, parent, visited, queue):
    """Is there a perfect matching in the bipartite graph {(i, j): D[i, j] <= theta}? (BFS augmenting paths)"""
    k = D.shape[0]
    for j in range(k):
        row_of_col[j] = -1
        col_of_row[j] = -1
    for root in range(k):
        for j in range(k):
            visited[j] = False
            parent[j] = -1
        qh, qt = 0, 1
        queue[0] = root
        end_col = -1
        while qh < qt and end_col < 0:
            i = queue[qh]
            qh += 1
            for j in range(k):
                if D[i, j] <= theta and not visited[j]:
                    visited[j] = True
                    parent[j] = i
                    if row_of_col[j] < 0:
                        end_col = j
                        break
                    queue[qt] = row_of_col[j]
                    qt += 1
        if end_col < 0:
            return False
        j = end_col
        while j >= 0:
            i = parent[j]
            nxt = col_of_row[i]
            row_of_col[j] = i
            col_of_row[i] = j
            j = nxt
    return True


@njit
def bottleneck(D):
    """min over permutations pi of max_i D[i, pi(i)]: binary search over the sorted entries."""
    k = D.shape[0]
    vals = np.sort(D.ravel())
    row_of_col = np.empty(k, np.int64)
    col_of_row = np.empty(k, np.int64)
    parent = np.empty(k, np.int64)
    visited = np.empty(k, np.bool_)
    queue = np.empty(k, np.int64)
    lo, hi = 0, vals.shape[0] - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if _has_perfect_matching(D, vals[mid], row_of_col, col_of_row, parent, visited, queue):
            hi = mid
        else:
            lo = mid + 1
    return vals[lo]


@njit
def endpoint_lb(P, Q):
    """Lower bound on d_match(P, Q): bottleneck over max(|first-first|, |last-last|), since the discrete
    Frechet distance is at least the distance of the first vertices and of the last vertices."""
    k, mq, d = Q.shape
    mp = P.shape[1]
    D = np.empty((k, k))
    for i in range(k):
        for j in range(k):
            s0 = 0.0
            s1 = 0.0
            for a in range(d):
                u = Q[i, 0, a] - P[j, 0, a]
                v = Q[i, mq - 1, a] - P[j, mp - 1, a]
                s0 += u * u
                s1 += v * v
            D[i, j] = np.sqrt(max(s0, s1))
    return bottleneck(D)


@njit
def dmatch(P, Q):
    """Matching distance: min over bijections of the max Frechet distance of matched curves."""
    k = Q.shape[0]
    D = np.empty((k, k))
    for i in range(k):
        for j in range(k):
            D[i, j] = frechet(Q[i], P[j])
    return bottleneck(D)


@njit
def dmatch_upto(P, Q, bound):
    """d_match(P, Q) if it is below `bound`, otherwise some value >= bound (cheap early exit)."""
    if endpoint_lb(P, Q) >= bound:
        return bound
    return dmatch(P, Q)


@njit(parallel=True)
def dmatch_many(X, Q):
    """d_match(X[p], Q) for every set p (brute force)."""
    n = X.shape[0]
    out = np.empty(n)
    for p in prange(n):
        out[p] = dmatch(X[p], Q)
    return out


@njit(parallel=True)
def dmatch_many_pruned(X, Q, bound):
    """d_match(X[p], Q) for every set p, or `bound` for sets whose endpoint lower bound reaches `bound`."""
    n = X.shape[0]
    out = np.empty(n)
    for p in prange(n):
        out[p] = dmatch_upto(X[p], Q, bound)
    return out
