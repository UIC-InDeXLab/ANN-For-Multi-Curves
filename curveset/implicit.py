"""Implicit synthetic collections of curve sets, for experiments with up to 10^9 curves.

Curve l of curve set p follows path pid[p, l]: vertex j is paths[pid[p, l], j] plus a jitter vector drawn uniformly
from the disk of radius rho by a counter-based hash of (seed, p, l, j). Only the path ids are stored (one or two
bytes per curve), and curves are regenerated on the fly when they are hashed or compared. Kernels assume d = 2.

Indexes over implicit collections: ImplicitJoint (store all, probe all, blocking) and ImplicitPerCurve (per-curve LSH
with complete lists, capped lists, or exact top-K lists). Index construction is parallel; queries are single-threaded.
Every query returns, besides the answer, the number of candidates and of relevant candidates (d_match < r), from
which recall and precision with respect to the relevant curve sets are computed.
"""
import time

import numpy as np
from numba import njit, prange

from . import core
from .index import Tables, _bucket, _probe_orders, index_bytes
from .orders import all_orders, balanced_blocking, blocking_families, identity

INV53 = 1.0 / 9007199254740992.0
TWO_PI = 2.0 * np.pi
K1 = np.uint64(0x9E3779B97F4A7C15)
K2 = np.uint64(0xBF58476D1CE4E5B9)
K3 = np.uint64(0x94D049BB133111EB)


class Implicit:
    """paths: (P, m, 2) float64; pid: (n, k) uint8/uint16 path ids; seed: jitter seed; rho: jitter radius."""

    def __init__(self, paths, pid, seed, rho):
        self.paths = np.ascontiguousarray(paths, dtype=np.float64)
        self.pid = np.ascontiguousarray(pid)
        self.seed = np.uint64(seed)
        self.rho = float(rho)
        self.n, self.k = pid.shape
        self.m = paths.shape[1]

    def materialize(self, ids):
        ids = np.asarray(ids, dtype=np.int64)
        return materialize(self.paths, self.pid, self.seed, self.rho, ids)


# ------------------------------------------------------------------------------------------ curve generation
@njit(inline="always")
def _vertex(paths, pid, seed, rho, p, l, j, out):
    x = core.mix64(seed + np.uint64(p) * K1)
    x = core.mix64(x + np.uint64(l) * K2)
    x = core.mix64(x + np.uint64(j) * K3)
    u1 = (core.mix64(x + np.uint64(1)) >> np.uint64(11)) * INV53
    u2 = (core.mix64(x + np.uint64(2)) >> np.uint64(11)) * INV53
    ang = TWO_PI * u1
    rad = rho * np.sqrt(u2)
    c = pid[p, l]
    out[0] = paths[c, j, 0] + rad * np.cos(ang)
    out[1] = paths[c, j, 1] + rad * np.sin(ang)


@njit
def gen_curve(paths, pid, seed, rho, p, l, buf):
    for j in range(paths.shape[1]):
        _vertex(paths, pid, seed, rho, p, l, j, buf[j])


@njit
def gen_set(paths, pid, seed, rho, p, buf):
    for l in range(pid.shape[1]):
        gen_curve(paths, pid, seed, rho, p, l, buf[l])


@njit(parallel=True)
def materialize(paths, pid, seed, rho, ids):
    k, m = pid.shape[1], paths.shape[1]
    out = np.empty((ids.shape[0], k, m, 2))
    for x in prange(ids.shape[0]):
        gen_set(paths, pid, seed, rho, ids[x], out[x])
    return out


# ------------------------------------------------------------------------------------------ index construction
@njit(parallel=True)
def slot_fps_chunk(paths, pid, seed, rho, p0, p1, shifts, delta, need):
    """F[x, i, l] = fingerprint of curve l of set p0 + x under the shift of slot i, where need[i, l]."""
    k, m = pid.shape[1], paths.shape[1]
    use = np.zeros(k, np.bool_)
    for l in range(k):
        for i in range(k):
            if need[i, l]:
                use[l] = True
    F = np.zeros((p1 - p0, k, k), np.uint64)
    for x in prange(p1 - p0):
        buf = np.empty((m, 2))
        for l in range(k):
            if use[l]:
                gen_curve(paths, pid, seed, rho, p0 + x, l, buf)
                for i in range(k):
                    if need[i, l]:
                        F[x, i, l] = core.curve_fp(buf, shifts[i], delta)
    return F


@njit(parallel=True)
def curve_fps_chunk(paths, pid, seed, rho, p0, p1, t, delta):
    """Fingerprints of all curves of sets p0..p1-1 under one shift t, shape (p1 - p0, k)."""
    k, m = pid.shape[1], paths.shape[1]
    out = np.empty((p1 - p0, k), np.uint64)
    for x in prange(p1 - p0):
        buf = np.empty((m, 2))
        for l in range(k):
            gen_curve(paths, pid, seed, rho, p0 + x, l, buf)
            out[x, l] = core.curve_fp(buf, t, delta)
    return out


@njit
def fill_table(keys, id_div, B, out_keys, out_ids, out_dir):
    """Counting sort by bucket, then sort every bucket (already sorted runs are skipped; large unsorted buckets use
    merge sort, so the worst case is O(b log b)). Entry t belongs to id t // id_div."""
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
        if hi - lo < 2:
            continue
        ordered = True
        for x in range(lo + 1, hi):
            if out_keys[x] < out_keys[x - 1]:
                ordered = False
                break
        if ordered:
            continue
        if hi - lo <= 32:
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
        else:
            order = np.argsort(out_keys[lo:hi], kind="mergesort")
            ks = out_keys[lo:hi][order]
            vs = out_ids[lo:hi][order]
            out_keys[lo:hi] = ks
            out_ids[lo:hi] = vs


# ------------------------------------------------------------------------------------------ verification, ground truth
@njit
def verify_sets(paths, pid, seed, rho, Q, cand, c0, c1, r, best, best_id):
    """Exact d_match of candidates cand[c0:c1]: returns (best id, best distance, candidates with d_match < r)."""
    k, m = pid.shape[1], paths.shape[1]
    buf = np.empty((k, m, 2))
    nrel = 0
    for x in range(c0, c1):
        p = cand[x]
        gen_set(paths, pid, seed, rho, p, buf)
        bound = best if best > r else r
        dd = core.dmatch_upto(buf, Q, bound)
        if dd < r:
            nrel += 1
        if dd < best:
            best = dd
            best_id = p
    return best_id, best, nrel


@njit(inline="always")
def _dist2(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


@njit
def scan(paths, pid, seed, rho, Q, r, ground_truth):
    """Brute force over all curve sets. ground_truth: also count the relevant sets (d_match < r) and prune with
    max(r, best); otherwise prune with the best distance so far (the brute-force baseline). A set can be closer than
    a threshold only if every query curve has a data curve whose first and last vertices are closer than it."""
    n, k = pid.shape
    m = paths.shape[1]
    qe = np.empty((k, 2, 2))
    for i in range(k):
        qe[i, 0] = Q[i, 0]
        qe[i, 1] = Q[i, Q.shape[1] - 1]
    ends = np.empty((k, 2, 2))
    buf = np.empty((k, m, 2))
    best, best_id, nrel = np.inf, -1, 0
    for p in range(n):
        thr = (r if r > best else best) if ground_truth else best
        for l in range(k):
            _vertex(paths, pid, seed, rho, p, l, 0, ends[l, 0])
            _vertex(paths, pid, seed, rho, p, l, m - 1, ends[l, 1])
        ok = True
        for i in range(k):
            found = False
            for l in range(k):
                if _dist2(qe[i, 0], ends[l, 0]) < thr and _dist2(qe[i, 1], ends[l, 1]) < thr:
                    found = True
                    break
            if not found:
                ok = False
                break
        if not ok:
            continue
        gen_set(paths, pid, seed, rho, p, buf)
        dd = core.dmatch_upto(buf, Q, thr)
        if ground_truth and dd < r:
            nrel += 1
        if dd < best:
            best = dd
            best_id = p
    return best_id, best, nrel


@njit(parallel=True)
def ground_truth(paths, pid, seed, rho, Qs, r):
    """Exact nearest neighbor and number of relevant sets (d_match < r) of every query (parallel over queries)."""
    nq = Qs.shape[0]
    ids = np.empty(nq, np.int64)
    ds = np.empty(nq)
    nrel = np.empty(nq, np.int64)
    for q in prange(nq):
        a, b, c = scan(paths, pid, seed, rho, Qs[q], r, True)
        ids[q] = a
        ds[q] = b
        nrel[q] = c
    return ids, ds, nrel


# ------------------------------------------------------------------------------------------ per-curve query kernels
@njit(inline="always")
def _lower(keys, lo, hi, key):
    while lo < hi:
        mid = (lo + hi) >> 1
        if keys[mid] < key:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(inline="always")
def _upper(keys, lo, hi, key):
    while lo < hi:
        mid = (lo + hi) >> 1
        if keys[mid] <= key:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit
def pc_probe(Q, shifts, delta, keys, ids, dirs, N, B, k, cap, rng_state, mark, cnt, stamp0, touched, cand):
    """Per-curve LSH over curve ids (curve set = id // k). cap = 0: complete lists; cap > 0: every query curve takes at
    most cap colliding curves per table, a random window of the run of colliding curves. Candidates are the sets
    reached by all k query curves. Returns (candidates, list entries)."""
    L = shifts.shape[0]
    ntouched = 0
    listed = 0
    s = rng_state[0]
    for i in range(k):
        stamp = stamp0 + i
        for u in range(L):
            key = core.curve_fp(Q[i], shifts[u], delta)
            j = u * (B + 1) + _bucket(key, B)
            lo = u * N + np.int64(dirs[j])
            hi = u * N + np.int64(dirs[j + 1])
            a = _lower(keys, lo, hi, key)
            b = _upper(keys, a, hi, key)
            run = b - a
            take = run
            start = a
            if cap > 0 and run > cap:
                s ^= s << np.uint64(13)
                s ^= s >> np.uint64(7)
                s ^= s << np.uint64(17)
                start = a + np.int64(s % np.uint64(run))
                take = cap
            for x in range(take):
                t = start + x
                if t >= b:
                    t -= run
                p = np.int64(ids[t]) // k
                if mark[p] != stamp:
                    mark[p] = stamp
                    listed += 1
                    if cnt[p] == 0:
                        touched[ntouched] = p
                        ntouched += 1
                    cnt[p] += 1
    rng_state[0] = s
    ncand = 0
    for x in range(ntouched):
        p = touched[x]
        if cnt[p] == k:
            cand[ncand] = p
            ncand += 1
        cnt[p] = 0
    return ncand, listed


@njit
def pc_probe_topk(paths, pid, seed, rho, Q, shifts, delta, keys, ids, dirs, N, B, K, cmark, cstamp, cbuf, dist,
                  mark, cnt, stamp0, touched, cand):
    """Per-curve LSH returning the exact top-K curves per query curve: all colliding curves are collected, their
    Frechet distance to the query curve computed, and the K closest kept. Returns (candidates, curves compared)."""
    k = pid.shape[1]
    m = paths.shape[1]
    L = shifts.shape[0]
    buf = np.empty((m, 2))
    ntouched = 0
    compared = 0
    for i in range(k):
        stamp = stamp0 + i
        nc = 0
        for u in range(L):
            key = core.curve_fp(Q[i], shifts[u], delta)
            j = u * (B + 1) + _bucket(key, B)
            lo = u * N + np.int64(dirs[j])
            hi = u * N + np.int64(dirs[j + 1])
            a = _lower(keys, lo, hi, key)
            b = _upper(keys, a, hi, key)
            for t in range(a, b):
                c = np.int64(ids[t])
                if cmark[c] != cstamp + i:
                    cmark[c] = cstamp + i
                    cbuf[nc] = c
                    nc += 1
        for x in range(nc):
            c = cbuf[x]
            gen_curve(paths, pid, seed, rho, c // k, c % k, buf)
            dist[x] = core.frechet(Q[i], buf)
        compared += nc
        order = np.argsort(dist[:nc])
        for x in range(min(K, nc)):
            p = cbuf[order[x]] // k
            if mark[p] != stamp:
                mark[p] = stamp
                if cnt[p] == 0:
                    touched[ntouched] = p
                    ntouched += 1
                cnt[p] += 1
    ncand = 0
    for x in range(ntouched):
        p = touched[x]
        if cnt[p] == k:
            cand[ncand] = p
            ncand += 1
        cnt[p] = 0
    return ncand, compared


# ------------------------------------------------------------------------------------------ indexes
class _Base:
    VERIFY_CHUNK = 50_000

    def _scratch(self, n):
        self.mark = np.zeros(n, np.int32)
        self.cand = np.empty(n, np.int64)
        self.stamp = 0

    def _next_stamp(self, width=1):
        self.stamp += width
        if self.stamp > 2_000_000_000:
            self.mark[:] = 0
            self.stamp = width
        return self.stamp

    def _verify(self, data, Q, ncand, r, t0, deadline):
        best, best_id, nrel = np.inf, -1, 0
        for c0 in range(0, ncand, self.VERIFY_CHUNK):
            c1 = min(ncand, c0 + self.VERIFY_CHUNK)
            best_id, best, nr = verify_sets(data.paths, data.pid, data.seed, data.rho, Q, self.cand, c0, c1, r, best,
                                            best_id)
            nrel += nr
            if deadline is not None and time.perf_counter() - t0 > deadline:
                return dict(id=-1, dist=np.nan, candidates=ncand, relevant_found=nrel, timeout=True)
        return dict(id=int(best_id), dist=float(best), candidates=ncand, relevant_found=nrel, timeout=False)

    @property
    def index_bytes(self):
        return self.tables.nbytes


class ImplicitJoint(_Base):
    """Slot-wise keys with independent shifts that store a family of orders of every set and probe a family of
    orders of the query: store all, probe all, or blocking (same shifts as index.OrderIndex for a given seed)."""

    def __init__(self, k, delta, L, stored, probed, seed=0, name="joint"):
        self.shifts = np.random.default_rng(seed).uniform(0, delta, size=(L, k, 2))
        self.k, self.delta, self.L, self.name = k, float(delta), L, name
        self.stored = np.ascontiguousarray(stored, dtype=np.int8)
        self.probed = np.ascontiguousarray(probed, dtype=np.int8)
        self.need_s = np.zeros((k, k), np.bool_)
        self.need_q = np.zeros((k, k), np.bool_)
        for i in range(k):
            self.need_s[i, np.unique(self.stored[:, i])] = True
            self.need_q[i, np.unique(self.probed[:, i])] = True

    def build(self, data, chunk=1 << 20):
        t0 = time.perf_counter()
        n, a = data.n, self.stored.shape[0]
        self.tables = T = Tables(self.L, n * a)
        for u in range(self.L):
            keys = np.empty(n * a, np.uint64)
            for p0 in range(0, n, chunk):
                p1 = min(n, p0 + chunk)
                F = slot_fps_chunk(data.paths, data.pid, data.seed, data.rho, p0, p1, self.shifts[u], self.delta,
                                   self.need_s)
                keys[p0 * a:p1 * a] = core.order_keys(F, self.stored).ravel()
                del F
            N, B = T.N, T.B
            fill_table(keys, a, B, T.keys[u * N:(u + 1) * N], T.ids[u * N:(u + 1) * N],
                       T.dir[u * (B + 1):(u + 1) * (B + 1)])
            del keys
        self._scratch(n)
        self.build_seconds = time.perf_counter() - t0
        return self

    def query(self, data, Q, r, deadline=None):
        t0 = time.perf_counter()
        T, b = self.tables, self.probed.shape[0]
        stamp = self._next_stamp()
        chunk = self.L if deadline is None else max(1, 4_000_000 // max(b, 1))
        ncand, probes = 0, 0
        for u0 in range(0, self.L, chunk):
            u1 = min(self.L, u0 + chunk)
            _, done, ncand = _probe_orders(Q, self.shifts, self.delta, self.probed, self.need_q, T.keys, T.ids,
                                           T.dir, T.N, T.B, u0, u1, False, self.mark, stamp, self.cand, ncand)
            probes += done
            if deadline is not None and time.perf_counter() - t0 > deadline:
                return dict(id=-1, dist=np.nan, candidates=ncand, relevant_found=0, probes=probes, listed=0,
                            seconds=time.perf_counter() - t0, timeout=True)
        res = self._verify(data, Q, ncand, r, t0, deadline)
        res.update(probes=probes, listed=0, seconds=time.perf_counter() - t0)
        return res


class ImplicitPerCurve(_Base):
    """Per-curve LSH over curve ids: L tables with one shift each over all n k curves (same shifts as
    index.PerCurveIndex for a given seed)."""

    def __init__(self, k, delta, L, seed=0, name="per-curve"):
        self.shifts = np.random.default_rng(seed).uniform(0, delta, size=(L, 2))
        self.k, self.delta, self.L, self.name = k, float(delta), L, name

    def build(self, data, chunk=1 << 20):
        t0 = time.perf_counter()
        n, k = data.n, data.k
        self.tables = T = Tables(self.L, n * k)
        for u in range(self.L):
            keys = np.empty(n * k, np.uint64)
            for p0 in range(0, n, chunk):
                p1 = min(n, p0 + chunk)
                keys[p0 * k:p1 * k] = curve_fps_chunk(data.paths, data.pid, data.seed, data.rho, p0, p1,
                                                      self.shifts[u], self.delta).ravel()
            N, B = T.N, T.B
            fill_table(keys, 1, B, T.keys[u * N:(u + 1) * N], T.ids[u * N:(u + 1) * N],
                       T.dir[u * (B + 1):(u + 1) * (B + 1)])
            del keys
        self._scratch(n)
        self.cnt = np.zeros(n, np.int32)
        self.touched = np.empty(n, np.int64)
        self.build_seconds = time.perf_counter() - t0
        return self

    def query(self, data, Q, r, cap=0, rng_seed=1, deadline=None):
        t0 = time.perf_counter()
        T = self.tables
        state = np.array([(int(rng_seed) * 0x9E3779B97F4A7C15 + 1) & 0xFFFFFFFFFFFFFFFF or 1], np.uint64)
        ncand, listed = pc_probe(Q, self.shifts, self.delta, T.keys, T.ids, T.dir, T.N, T.B, self.k, cap, state,
                                 self.mark, self.cnt, self._next_stamp(self.k), self.touched, self.cand)
        res = self._verify(data, Q, ncand, r, t0, deadline)
        res.update(probes=self.L * self.k, listed=int(listed), seconds=time.perf_counter() - t0)
        return res

    def query_topk(self, data, Q, r, K, deadline=None):
        if not hasattr(self, "cmark"):
            self.cmark = np.zeros(data.n * data.k, np.int32)
            self.cbuf = np.empty(data.n * data.k, np.int64)
            self.cdist = np.empty(data.n * data.k)
            self.cstamp = 0
        self.cstamp += self.k
        if self.cstamp > 2_000_000_000:
            self.cmark[:] = 0
            self.cstamp = self.k
        t0 = time.perf_counter()
        T = self.tables
        ncand, compared = pc_probe_topk(data.paths, data.pid, data.seed, data.rho, Q, self.shifts, self.delta, T.keys,
                                        T.ids, T.dir, T.N, T.B, K, self.cmark, self.cstamp, self.cbuf, self.cdist,
                                        self.mark, self.cnt, self._next_stamp(self.k), self.touched, self.cand)
        res = self._verify(data, Q, ncand, r, t0, deadline)
        res.update(probes=self.L * self.k, listed=int(compared), seconds=time.perf_counter() - t0)
        return res


def make_joint(method, k, delta, L, sizes=None, swapped=False, seed=0):
    if method == "store_all":
        return ImplicitJoint(k, delta, L, all_orders(k), identity(k), seed, "store all orders")
    if method == "probe_all":
        return ImplicitJoint(k, delta, L, identity(k), all_orders(k), seed, "probe all orders")
    if sizes is None:
        _, _, sizes, swapped = balanced_blocking(k)
    stored, probed = blocking_families(sizes, swapped)
    return ImplicitJoint(k, delta, L, stored, probed, seed, "blocking")


def joint_bytes(n, a, L):
    return index_bytes(n * a, L)
