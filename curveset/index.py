"""Hash indexes for curve sets (Sections 4 and 5 of the paper) and the baselines of the experiments.

Storage: the L tables of an index are kept in three flat arrays. Table u holds N entries (64-bit key fingerprint,
32-bit curve-set id) sorted by key, and a directory of B = N/4 buckets (uint32 offsets) indexed by the top bits
of the key, so a lookup reads one directory entry and scans a bucket of about four keys: 13 bytes per stored key.

Queries run in single-threaded Numba kernels that hash every probe (reusing the hash of the prefix shared with
the previous probe, since probe families are sorted lexicographically), look it up, and collect candidates.
Index construction is parallel.

Query modes:
  first  - the index of the theory: return the curve set stored with the first key found.
  verify - collect every curve set found in any table, compute d_match for each, and return the closest.
A query can be given a deadline in seconds; it is then checked between chunks of tables.
"""
import time
from math import ceil, log

import numpy as np
from numba import njit

from . import core
from .orders import all_orders, balanced_blocking, blocking_families, identity

KEY_BYTES, ID_BYTES, DIR_BYTES = 8, 4, 4
CHUNK_PROBES = 4_000_000  # probes per kernel call between deadline checks


def theory_params(d, m, r, p, k, eta, shared=False):
    """delta, c, alpha and L of Theorems 4.3/4.4 (and 5.2 for a shared shift)."""
    delta = 2 * d * m * r / (1 - p)
    c = np.sqrt(d) * delta / r
    alpha = 1 - k * (1 - p) if shared else p ** k
    L = None if alpha <= 0 else (1 if alpha >= 1 else ceil(log(eta) / log(1 - alpha)))
    return dict(delta=delta, c=c, alpha=alpha, L=L)


def n_buckets(N):
    return max(2, N // 4)


def index_bytes(N, L):
    """Bytes of L tables with N entries each."""
    return L * (N * (KEY_BYTES + ID_BYTES) + (n_buckets(N) + 1) * DIR_BYTES)


# ------------------------------------------------------------------------------------------ tables
@njit(inline="always")
def _bucket(key, B):
    return np.int64(((key >> np.uint64(32)) * np.uint64(B)) >> np.uint64(32))


@njit(cache=True)
def _fill_table(keys, id_div, B, out_keys, out_ids, out_dir):
    """Counting sort of `keys` by bucket, then insertion sort inside each bucket. Entry t of `keys` belongs to
    curve set t // id_div. Writes sorted keys, ids, and the directory (start offset of every bucket)."""
    N = keys.shape[0]
    cnt = np.zeros(B + 1, np.int64)
    for t in range(N):
        cnt[_bucket(keys[t], B) + 1] += 1
    for j in range(B):
        cnt[j + 1] += cnt[j]
    for j in range(B + 1):
        out_dir[j] = np.uint32(cnt[j])
    for t in range(N):
        bk = _bucket(keys[t], B)
        q = cnt[bk]
        out_keys[q] = keys[t]
        out_ids[q] = np.int32(t // id_div)
        cnt[bk] = q + 1
    for j in range(B):
        lo = np.int64(out_dir[j])
        hi = np.int64(out_dir[j + 1])
        for x in range(lo + 1, hi):
            kv = out_keys[x]
            iv = out_ids[x]
            y = x - 1
            while y >= lo and out_keys[y] > kv:
                out_keys[y + 1] = out_keys[y]
                out_ids[y + 1] = out_ids[y]
                y -= 1
            out_keys[y + 1] = kv
            out_ids[y + 1] = iv


class Tables:
    """L tables with N entries each, in flat arrays."""

    def __init__(self, L, N):
        self.L, self.N, self.B = L, N, n_buckets(N)
        self.keys = np.empty(L * N, np.uint64)
        self.ids = np.empty(L * N, np.int32)
        self.dir = np.empty(L * (self.B + 1), np.uint32)

    def fill(self, u, keys, id_div):
        N, B = self.N, self.B
        _fill_table(np.ascontiguousarray(keys).ravel(), id_div, B, self.keys[u * N:(u + 1) * N],
                    self.ids[u * N:(u + 1) * N], self.dir[u * (B + 1):(u + 1) * (B + 1)])

    @property
    def nbytes(self):
        return self.keys.nbytes + self.ids.nbytes + self.dir.nbytes

    @property
    def entries(self):
        return self.L * self.N


# ------------------------------------------------------------------------------------------ query kernels
@njit(cache=True)
def _probe_orders(Q, shifts, delta, probed, need_q, keys, ids, dirs, N, B, u0, u1, first, mark, stamp, cand,
                  ncand):
    """Probe tables u0..u1-1 with the keys of Q listed in every order of `probed` (sorted lexicographically).
    Returns (first hit or -1, probes done, number of candidates)."""
    k = Q.shape[0]
    b = probed.shape[0]
    F = np.zeros((k, k), np.uint64)
    H = np.empty(k + 1, np.uint64)
    H[0] = core.SEED
    probes = 0
    for u in range(u0, u1):
        for i in range(k):
            for l in range(k):
                if need_q[i, l]:
                    F[i, l] = core.curve_fp(Q[l], shifts[u, i], delta)
        kbase = u * N
        dbase = u * (B + 1)
        p = 0
        for s in range(b):
            if s > 0:
                p = 0
                while probed[s, p] == probed[s - 1, p]:
                    p += 1
            for i in range(p, k):
                H[i + 1] = core.mix64(H[i] + F[i, probed[s, i]])
            key = H[k]
            j = dbase + _bucket(key, B)
            lo = kbase + np.int64(dirs[j])
            hi = kbase + np.int64(dirs[j + 1])
            for t in range(lo, hi):
                kt = keys[t]
                if kt == key:
                    pid = ids[t]
                    if first:
                        return pid, probes + s + 1, ncand
                    if mark[pid] != stamp:
                        mark[pid] = stamp
                        cand[ncand] = pid
                        ncand += 1
                elif kt > key:
                    break
        probes += b
    return -1, probes, ncand


@njit(cache=True)
def _probe_sorted(Q, shifts, delta, keys, ids, dirs, N, B, first, mark, stamp, cand):
    k = Q.shape[0]
    L = shifts.shape[0]
    buf = np.empty(k, np.uint64)
    ncand = 0
    for u in range(L):
        for l in range(k):
            buf[l] = core.curve_fp(Q[l], shifts[u], delta)
        buf.sort()
        key = core.chain(buf)
        j = u * (B + 1) + _bucket(key, B)
        lo = u * N + np.int64(dirs[j])
        hi = u * N + np.int64(dirs[j + 1])
        for t in range(lo, hi):
            kt = keys[t]
            if kt == key:
                pid = ids[t]
                if first:
                    return pid, u + 1, 0
                if mark[pid] != stamp:
                    mark[pid] = stamp
                    cand[ncand] = pid
                    ncand += 1
            elif kt > key:
                break
    return -1, L, ncand


@njit(cache=True)
def _probe_per_curve(Q, shifts, delta, keys, ids, dirs, N, B, mark, cnt, stamp0, touched, cand):
    """Per-curve LSH: for every query curve, the curve sets with a colliding curve in some table; candidates are
    the sets reached by all k query curves. Returns (candidates, list entries)."""
    k = Q.shape[0]
    L = shifts.shape[0]
    ntouched = 0
    listed = 0
    for i in range(k):
        stamp = stamp0 + i
        for u in range(L):
            key = core.curve_fp(Q[i], shifts[u], delta)
            j = u * (B + 1) + _bucket(key, B)
            lo = u * N + np.int64(dirs[j])
            hi = u * N + np.int64(dirs[j + 1])
            for t in range(lo, hi):
                kt = keys[t]
                if kt == key:
                    pid = ids[t]
                    if mark[pid] != stamp:
                        mark[pid] = stamp
                        listed += 1
                        if cnt[pid] == 0:
                            touched[ntouched] = pid
                            ntouched += 1
                        cnt[pid] += 1
                elif kt > key:
                    break
    ncand = 0
    for x in range(ntouched):
        pid = touched[x]
        if cnt[pid] == k:
            cand[ncand] = pid
            ncand += 1
        cnt[pid] = 0
    return ncand, listed


@njit(cache=True)
def verify(X, Q, cand):
    """Exact d_match of every candidate with early exit; returns (best id, best distance)."""
    best, best_id = np.inf, -1
    for c in cand:
        dist = core.dmatch_upto(X[c], Q, best)
        if dist < best:
            best, best_id = dist, c
    return best_id, best


@njit(cache=True)
def brute_force(X, Q):
    """Exact nearest neighbor under d_match (single-threaded, endpoint lower bound pruning)."""
    best, best_id = np.inf, -1
    for p in range(X.shape[0]):
        dist = core.dmatch_upto(X[p], Q, best)
        if dist < best:
            best, best_id = dist, p
    return best_id, best


# ------------------------------------------------------------------------------------------ indexes
class _Base:
    def _buffers(self, n):
        self.mark = np.zeros(n, np.int32)
        self.cand = np.empty(n, np.int64)
        self.stamp = 0

    def _next_stamp(self, width=1):
        self.stamp += width
        if self.stamp > 2_000_000_000:
            self.mark[:] = 0
            self.stamp = width
        return self.stamp

    def _finish(self, X, Q, ncand, t0, **extra):
        cand = self.cand[:ncand]
        best_id, best = verify(X, Q, cand) if ncand else (-1, np.inf)
        return dict(id=int(best_id), dist=float(best), candidates=int(ncand), seconds=time.perf_counter() - t0,
                    timeout=False, **extra)

    @property
    def index_bytes(self):
        return self.tables.nbytes


class OrderIndex(_Base):
    """Slot-wise keys (Section 4.1) storing a family of orders of every data set and probing a family of orders
    of the query (Lemma 5.4): store all, probe all, blocking, and the warm-up (identity/identity)."""

    def __init__(self, k, d, delta, L, stored, probed, shared=False, seed=0, name="order"):
        rng = np.random.default_rng(seed)
        if shared:
            self.shifts = np.repeat(rng.uniform(0, delta, size=(L, 1, d)), k, axis=1)
        else:
            self.shifts = rng.uniform(0, delta, size=(L, k, d))
        self.k, self.d, self.delta, self.L = k, d, float(delta), L
        self.stored = np.ascontiguousarray(stored, dtype=np.int8)
        self.probed = np.ascontiguousarray(probed, dtype=np.int8)
        self.need_s = np.zeros((k, k), np.bool_)
        self.need_q = np.zeros((k, k), np.bool_)
        for i in range(k):
            self.need_s[i, np.unique(self.stored[:, i])] = True
            self.need_q[i, np.unique(self.probed[:, i])] = True
        self.name = name

    def build(self, X):
        t0 = time.perf_counter()
        n, a = X.shape[0], self.stored.shape[0]
        self.tables = Tables(self.L, n * a)
        for u in range(self.L):
            F = core.slot_fps(X, self.shifts[u], self.delta, self.need_s)
            self.tables.fill(u, core.order_keys(F, self.stored), a)
            del F
        self._buffers(n)
        self.build_seconds = time.perf_counter() - t0
        return self

    def query(self, X, Q, mode="first", deadline=None):
        t0 = time.perf_counter()
        T, b = self.tables, self.probed.shape[0]
        first = mode == "first"
        stamp = self._next_stamp()
        chunk = self.L if deadline is None else max(1, CHUNK_PROBES // max(b, 1))
        ncand, probes = 0, 0
        for u0 in range(0, self.L, chunk):
            u1 = min(self.L, u0 + chunk)
            hit, done, ncand = _probe_orders(Q, self.shifts, self.delta, self.probed, self.need_q, T.keys, T.ids,
                                             T.dir, T.N, T.B, u0, u1, first, self.mark, stamp, self.cand, ncand)
            probes += done
            if first and hit >= 0:
                return dict(id=int(hit), dist=np.nan, candidates=1, probes=probes,
                            seconds=time.perf_counter() - t0, timeout=False)
            if deadline is not None and time.perf_counter() - t0 > deadline:
                return dict(id=-1, dist=np.nan, candidates=ncand, probes=probes,
                            seconds=time.perf_counter() - t0, timeout=True)
        if first:
            return dict(id=-1, dist=np.nan, candidates=0, probes=probes, seconds=time.perf_counter() - t0,
                        timeout=False)
        return self._finish(X, Q, ncand, t0, probes=probes)


class SortedKeyIndex(_Base):
    """Sorted key with one shared shift per table (Section 5.1): one stored key and one probe per table."""

    def __init__(self, k, d, delta, L, seed=0, name="sorted key"):
        rng = np.random.default_rng(seed)
        self.shifts = rng.uniform(0, delta, size=(L, d))
        self.k, self.d, self.delta, self.L = k, d, float(delta), L
        self.name = name

    def build(self, X):
        t0 = time.perf_counter()
        n = X.shape[0]
        self.tables = Tables(self.L, n)
        for u in range(self.L):
            self.tables.fill(u, core.sorted_keys(X, self.shifts[u], self.delta), 1)
        self._buffers(n)
        self.build_seconds = time.perf_counter() - t0
        return self

    def query(self, X, Q, mode="first", deadline=None):
        t0 = time.perf_counter()
        T = self.tables
        hit, probes, ncand = _probe_sorted(Q, self.shifts, self.delta, T.keys, T.ids, T.dir, T.N, T.B,
                                           mode == "first", self.mark, self._next_stamp(), self.cand)
        if mode == "first":
            return dict(id=int(hit), dist=np.nan, candidates=int(hit >= 0), probes=probes,
                        seconds=time.perf_counter() - t0, timeout=False)
        return self._finish(X, Q, ncand, t0, probes=probes)


class PerCurveIndex(_Base):
    """Baseline: one single-curve LSH per table over all curves; a query collects, for every query curve, the
    curve sets containing a colliding curve in some table, intersects these k sets, and verifies."""

    def __init__(self, k, d, delta, L, seed=0, name="per-curve"):
        rng = np.random.default_rng(seed)
        self.shifts = rng.uniform(0, delta, size=(L, d))
        self.k, self.d, self.delta, self.L = k, d, float(delta), L
        self.name = name

    def build(self, X):
        t0 = time.perf_counter()
        n, k = X.shape[0], X.shape[1]
        self.tables = Tables(self.L, n * k)
        for u in range(self.L):
            self.tables.fill(u, core.curve_fps_all(X, self.shifts[u], self.delta), k)
        self._buffers(n)
        self.cnt = np.zeros(n, np.int32)
        self.touched = np.empty(n, np.int64)
        self.build_seconds = time.perf_counter() - t0
        return self

    def query(self, X, Q, mode="verify", deadline=None):
        t0 = time.perf_counter()
        T = self.tables
        ncand, listed = _probe_per_curve(Q, self.shifts, self.delta, T.keys, T.ids, T.dir, T.N, T.B, self.mark,
                                         self.cnt, self._next_stamp(self.k), self.touched, self.cand)
        return self._finish(X, Q, ncand, t0, probes=self.L * self.k, listed=int(listed))


def make_index(method, k, d, delta, L, seed=0, sizes=None, swapped=False):
    """Factory for the methods of the experiments."""
    if method == "store_all":
        return OrderIndex(k, d, delta, L, all_orders(k), identity(k), seed=seed, name="store all orders")
    if method == "probe_all":
        return OrderIndex(k, d, delta, L, identity(k), all_orders(k), seed=seed, name="probe all orders")
    if method == "blocking":
        if sizes is None:
            _, _, sizes, swapped = balanced_blocking(k)
        stored, probed = blocking_families(sizes, swapped)
        label = "+".join(map(str, sizes)) + (" (swapped)" if swapped else "")
        return OrderIndex(k, d, delta, L, stored, probed, seed=seed, name=f"blocking {label}")
    if method == "sorted":
        return SortedKeyIndex(k, d, delta, L, seed=seed)
    if method == "per_curve":
        return PerCurveIndex(k, d, delta, L, seed=seed)
    if method == "known":
        return OrderIndex(k, d, delta, L, identity(k), identity(k), seed=seed, name="known matching")
    raise ValueError(method)
