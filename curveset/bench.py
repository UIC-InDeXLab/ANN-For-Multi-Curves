"""Experiment harness: budgets, index construction, timed queries against exact ground truth, CSV rows.

Budgets (as in the paper): an index may use at most MEM_BUDGET bytes, and a query may take at most TIME_BUDGET
seconds. A configuration over the memory budget is recorded without being built. A configuration whose probes
could not finish within the time budget even at 5 ns per probe is recorded without being built; otherwise
every query runs with a deadline, and the first query over the deadline ends the configuration.

Timing: queries and brute force run single-threaded on one pinned core (PIN_CORE); index construction uses
all Numba threads. Every configuration answers at least MIN_QUERIES queries and stops after QUERY_SECONDS of
total query time or MAX_QUERIES queries.
"""
import csv
import os
import time
from math import factorial, prod
from pathlib import Path

import numpy as np

from . import core
from .index import brute_force, index_bytes, make_index
from .orders import blocking_points

MEM_BUDGET = 24e9
TIME_BUDGET = 60.0
MIN_PROBE_NS = 5.0
MIN_QUERIES, MAX_QUERIES, QUERY_SECONDS = 5, 100, 600.0
PIN_CORE = int(os.environ.get("PIN_CORE", str(min(os.sched_getaffinity(0)))))

FIELDS = ["exp", "dataset", "d", "k", "n", "method", "blocks", "stored", "probed", "L", "delta", "c_r", "mode",
          "query", "found", "planted_hit", "d_ret", "d_nn", "ratio", "within_cr", "exact", "probes", "candidates",
          "listed", "seconds", "bf_seconds", "build_seconds", "index_bytes", "entries", "status"]


class Writer:
    """Append rows to a CSV file (header written once)."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.path.exists()
        self.f = open(self.path, "a", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            self.w.writeheader()

    def row(self, **kw):
        self.w.writerow(kw)
        self.f.flush()


class pinned:
    """Pin the calling (main) thread to one core; Numba worker threads keep their affinity."""

    def __enter__(self):
        self.old = os.sched_getaffinity(0)
        os.sched_setaffinity(0, {PIN_CORE})

    def __exit__(self, *exc):
        os.sched_setaffinity(0, self.old)


def warmup():
    """Compile every kernel (parallel ones first, before any pinning) on a tiny problem."""
    rng = np.random.default_rng(0)
    X = np.cumsum(rng.normal(0, 1, size=(50, 3, 8, 2)), axis=2)
    for method, sizes in (("store_all", None), ("probe_all", None), ("blocking", (2, 1)), ("sorted", None),
                          ("per_curve", None)):
        idx = make_index(method, 3, 2, 4.0, 2, sizes=sizes).build(X)
        for mode in ("first", "verify"):
            idx.query(X, X[0], mode=mode, deadline=10.0)
    brute_force(X, X[0])
    core.dmatch(X[0], X[1])


def ground_truth(X, Qs):
    """Exact nearest neighbor of every query (single-threaded brute force on the pinned core, timed)."""
    ids, dists, secs = [], [], []
    with pinned():
        for Q in Qs:
            t0 = time.perf_counter()
            i, dist = brute_force(X, Q)
            secs.append(time.perf_counter() - t0)
            ids.append(i)
            dists.append(dist)
    return np.array(ids), np.array(dists), np.array(secs)


def family_sizes(method, k, sizes=None, swapped=False):
    """(stored orders a, probed orders b, blocks label) per data set and query per table."""
    if method == "store_all":
        return factorial(k), 1, ""
    if method == "probe_all":
        return 1, factorial(k), ""
    if method == "blocking":
        a = prod(factorial(s) for s in sizes)
        b = factorial(k) // a
        if swapped:
            a, b = b, a
        return a, b, "+".join(map(str, sizes)) + ("s" if swapped else "")
    return 1, 1, ""


def entries_per_table(method, n, k, a):
    return n * k if method == "per_curve" else (n if method == "sorted" else n * a)


def budget_blocking(k, n, L, mem=MEM_BUDGET):
    """Blocking with the fewest probes that fits the memory budget: the largest stored family a with
    index_bytes(n a, L) <= mem. While storing all orders fits, this is storing all orders."""
    best = None
    for a, b, sizes, swapped in blocking_points(k):
        if index_bytes(n * a, L) <= mem and (best is None or a > best[0]):
            best = (a, b, sizes, swapped)
    return best


def run_method(writer, exp, dataset, X, Qs, targets, gt, method, k, d, delta, L, c_r, modes=("verify",),
               sizes=None, swapped=False, seed=0, mem_budget=MEM_BUDGET, time_budget=TIME_BUDGET,
               min_queries=MIN_QUERIES, max_queries=MAX_QUERIES, query_seconds=QUERY_SECONDS):
    """Build one index within the budgets and answer queries in every mode, one CSV row per query."""
    n = X.shape[0]
    a, b, blocks = family_sizes(method, k, sizes, swapped)
    N = entries_per_table(method, n, k, a)
    est_bytes = index_bytes(N, L)
    probes = L * (k if method == "per_curve" else b)
    base = dict(exp=exp, dataset=dataset, d=d, k=k, n=n, method=method, blocks=blocks, stored=a, probed=b, L=L,
                delta=delta, c_r=c_r, index_bytes=est_bytes, entries=N * L, probes=probes)
    if est_bytes > mem_budget:
        writer.row(**base, status="over memory budget")
        return None
    if probes * MIN_PROBE_NS * 1e-9 > time_budget:
        writer.row(**base, status="over time budget (estimated)")
        return None
    idx = make_index(method, k, d, delta, L, seed=seed, sizes=sizes, swapped=swapped).build(X)
    base.update(index_bytes=idx.index_bytes, build_seconds=idx.build_seconds)
    gt_ids, gt_d, gt_s = gt

    def emit(**fields):
        row = dict(base)
        row.update(fields)
        writer.row(**row)

    for mode in modes:
        if method == "per_curve" and mode == "first":
            continue
        spent = 0.0
        with pinned():
            for q in range(min(len(Qs), max_queries)):
                if q >= min_queries and spent > query_seconds:
                    break
                res = idx.query(X, Qs[q], mode=mode, deadline=time_budget)
                spent += res["seconds"]
                if res["timeout"]:
                    emit(mode=mode, query=q, probes=res["probes"], seconds=res["seconds"], bf_seconds=gt_s[q],
                         status="over time budget")
                    break
                found = res["id"] >= 0
                d_ret = float(core.dmatch(X[res["id"]], Qs[q])) if found else np.nan
                emit(mode=mode, query=q, found=int(found),
                     planted_hit=int(found and targets is not None and res["id"] == targets[q]),
                     d_ret=d_ret, d_nn=gt_d[q], ratio=d_ret / gt_d[q] if found and gt_d[q] > 0 else np.nan,
                     within_cr=int(found and d_ret <= c_r * (1 + 1e-9)),
                     exact=int(found and d_ret <= gt_d[q] * (1 + 1e-9)), probes=res["probes"],
                     candidates=res["candidates"], listed=res.get("listed", ""), seconds=res["seconds"],
                     bf_seconds=gt_s[q], status="ok")
    return idx
