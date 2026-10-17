"""Large-scale experiments on implicit collections: up to 10^8 curve sets and 10^9 curves.

For one workload, number of curve sets n, number of objects k, and grid size delta:
  1. generate the collection (path ids only) and 2 nq queries: the first nq tune the tables, the last nq evaluate;
  2. ground truth (parallel, not timed): exact nearest neighbor and the relevant sets (d_match < r) of every query;
  3. tables for 90% recall on the tuning queries, for joint keys and for per-curve LSH (curveset.tune);
  4. every method within the budgets of 24 GB of index and 60 s per query, with single-threaded queries on one core
     (index construction uses all threads):
       blocking (block sizes from the cost model), probe all orders, store all orders,
       per-curve LSH with complete lists, with capped lists (K curves per query curve and table, chosen at random
       among the colliding ones), and with exact top-K lists (the K closest colliding curves; n <= --topk-nmax);
     and the single-threaded brute-force scan on a few queries;
  5. one CSV row per query: time, candidates, relevant candidates, recall, precision, exact nearest neighbor found.

Workloads (rho = 0.5, r = 1.5, m = 32, d = 2):
  popular - every curve follows one of 5k paths (each path occurs in about 18% of the sets); every query consists of k
            distinct paths and is planted in 3 sets, so only a few sets contain all of its objects;
  crowd   - every curve follows one of k paths; a query is a data set, shuffled and jittered.
"""
import argparse
import csv
import os
import time
from math import factorial
from pathlib import Path

import numpy as np

from curveset import data as gen
from curveset.bench import MEM_BUDGET, TIME_BUDGET, pinned
from curveset.costmodel import CostModel
from curveset.implicit import Implicit, ImplicitPerCurve, ground_truth, make_joint, scan
from curveset.index import index_bytes
from curveset.tune import tune_tables

M, RHO, R, TARGET = 32, 0.5, 1.5, 0.9
FIELDS = ["exp", "workload", "n", "k", "delta", "method", "variant", "blocks", "stored", "probed", "L", "L_tuned",
          "query", "found_nn", "d_ret", "d_nn", "relevant", "relevant_found", "candidates", "listed", "recall",
          "precision", "seconds", "build_seconds", "index_bytes", "status"]


class Writer:
    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        new = not path.exists() or path.stat().st_size == 0
        self.f = open(path, "a", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            self.w.writeheader()

    def row(self, **kw):
        self.w.writerow(kw)
        self.f.flush()


def workload(name, n, k, nq_total, seed):
    rng = np.random.default_rng(seed)
    if name == "popular":
        P = 5 * k
        paths = gen.random_walks(rng, P, M, 2, extent=1000.0, step=3.0)
        pid = rng.integers(0, P, size=(n, k), dtype=np.uint16)
        Qs = np.empty((nq_total, k, M, 2))
        planted = rng.choice(n, size=3 * nq_total, replace=False)
        for q in range(nq_total):
            mot = rng.choice(P, size=k, replace=False)
            Qs[q] = gen.jitter(rng, paths[mot], RHO)
            for s in planted[3 * q:3 * q + 3]:
                pid[s] = mot[rng.permutation(k)]
        return Implicit(paths, pid, seed + 1, RHO), Qs
    if name == "crowd":
        paths = gen.random_walks(rng, k, M, 2, extent=1000.0, step=3.0)
        pid = rng.integers(0, k, size=(n, k), dtype=np.uint16)
        D = Implicit(paths, pid, seed + 1, RHO)
        src = rng.choice(n, size=nq_total, replace=False)
        X = D.materialize(src)
        Qs = np.stack([gen.jitter(rng, X[q][rng.permutation(k)], RHO) for q in range(nq_total)])
        return D, Qs
    raise ValueError(name)


def run_queries(w, base, index, D, Qs, gt, call, max_queries, query_seconds):
    ids, ds, nrel = gt
    spent = 0.0
    with pinned():
        call(index, D, Qs[0])  # compile outside the timed loop
        for q in range(min(len(Qs), max_queries)):
            if q >= 5 and spent > query_seconds:
                break
            res = call(index, D, Qs[q])
            spent += res["seconds"]
            row = dict(base, query=q, d_nn=ds[q], relevant=nrel[q], candidates=res["candidates"],
                       listed=res.get("listed", ""), seconds=res["seconds"], build_seconds=index.build_seconds,
                       index_bytes=index.index_bytes)
            if res["timeout"] or res["seconds"] > TIME_BUDGET:
                w.row(**row, status="over time budget")
                break
            found = res["id"] >= 0
            rf = res["relevant_found"]
            w.row(**row, found_nn=int(found and res["dist"] <= ds[q] * (1 + 1e-9)), d_ret=res["dist"] if found else "",
                  relevant_found=rf, recall=rf / nrel[q] if nrel[q] else "",
                  precision=rf / res["candidates"] if res["candidates"] else "", status="ok")


def warmup(k):
    """Compile the single-threaded kernels on a tiny collection, outside every timed region."""
    rng = np.random.default_rng(0)
    paths = gen.random_walks(rng, 4, M, 2, extent=50.0, step=2.0)
    D = Implicit(paths, rng.integers(0, 4, size=(64, k), dtype=np.uint16), 3, RHO)
    Q = D.materialize(np.array([0]))[0]
    scan(D.paths, D.pid, D.seed, D.rho, Q, R, False)
    idx = ImplicitPerCurve(k, 8.0, 1).build(D)
    idx.query(D, Q, R, cap=0)
    idx.query(D, Q, R, cap=2)
    idx.query_topk(D, Q, R, 2)
    make_joint("probe_all", k, 8.0, 1).build(D).query(D, Q, R) if k <= 8 else None


def main(a):
    seed = {"popular": 11, "crowd": 13}[a.workload] * 1000 + a.k
    t0 = time.time()
    D, Qall = workload(a.workload, a.n, a.k, 2 * a.nq, seed)
    print(f"{a.workload} n={a.n} k={a.k}: generated in {time.time() - t0:.0f}s", flush=True)
    warmup(a.k)
    t0 = time.time()
    gt_all = ground_truth(D.paths, D.pid, D.seed, D.rho, Qall, R)
    print(f"  ground truth: {time.time() - t0:.0f}s; median NN distance {np.median(gt_all[1]):.3f}, "
          f"median relevant sets {np.median(gt_all[2]):.0f}", flush=True)
    h = a.nq
    Qt, Qe = Qall[:h], Qall[h:]
    gt = (gt_all[0][h:], gt_all[1][h:], gt_all[2][h:])
    NNt = D.materialize(gt_all[0][:h])
    Lj, Lp, fj, fp = tune_tables(Qt, NNt, a.k, 2, a.delta, TARGET, a.Lmax, 0, 0)
    print(f"  tables for {TARGET:.0%} recall: joint {Lj}, per-curve {Lp}", flush=True)
    w = Writer(f"{a.results}/large_{a.workload}.csv")
    cm = CostModel.load()
    n, k = a.n, a.k
    common = dict(exp="large", workload=a.workload, n=n, k=k, delta=a.delta)

    # brute force (single-threaded, a few queries)
    with pinned():
        for q in range(a.bf_queries):
            t1 = time.perf_counter()
            b_id, b_d, _ = scan(D.paths, D.pid, D.seed, D.rho, Qe[q], R, False)
            sec = time.perf_counter() - t1
            w.row(**common, method="brute_force", query=q, found_nn=int(b_d <= gt[1][q] * (1 + 1e-9)), d_ret=b_d,
                  d_nn=gt[1][q], relevant=gt[2][q], seconds=sec,
                  status="ok" if sec <= TIME_BUDGET else "over time budget")
            print(f"  brute force query {q}: {sec:.1f}s", flush=True)

    # joint keys
    runs = []
    if Lj:
        runs.append(("probe_all", None, False, 1, factorial(k)))
        runs.append(("store_all", None, False, factorial(k), 1))
        best = cm.best_blocking(k, n, Lj)
        if best:
            runs.append(("blocking", best[3], best[4], best[1], best[2]))
        else:
            w.row(**common, method="blocking", L=Lj, status="no blocking within both budgets")
    else:
        for m_ in ("probe_all", "store_all", "blocking"):
            w.row(**common, method=m_, status=f"recall target not reached with {a.Lmax} tables")
    for method, sizes, swapped, st, pr in runs:
        blocks = "+".join(map(str, sizes)) + ("s" if swapped else "") if sizes else ""
        base = dict(common, method=method, blocks=blocks, stored=st, probed=pr, L=Lj, L_tuned=Lj)
        mem = index_bytes(n * st, Lj)
        if mem > MEM_BUDGET:
            w.row(**base, index_bytes=mem, status="over memory budget")
            continue
        if Lj * pr * cm.tau(n * st) > TIME_BUDGET:
            w.row(**base, index_bytes=mem, status="over time budget (estimated)")
            continue
        t1 = time.time()
        idx = make_joint(method, k, a.delta, Lj, sizes=sizes, swapped=swapped).build(D)
        print(f"  {method} {blocks}: built in {time.time() - t1:.0f}s ({idx.index_bytes / 1e9:.2f} GB)", flush=True)
        run_queries(w, base, idx, D, Qe, gt, lambda ix, DD, Q: ix.query(DD, Q, R, deadline=TIME_BUDGET),
                    a.nq, a.query_seconds)
        print(f"  {method}: done", flush=True)
        del idx

    # per-curve LSH: complete, capped, and exact top-K lists on one index
    if Lp:
        L = Lp
        while L >= 1 and index_bytes(n * k, L) > MEM_BUDGET:
            L -= 1
        base = dict(common, method="per_curve", L=L, L_tuned=Lp)
        if L == 0:
            w.row(**base, index_bytes=index_bytes(n * k, 1), status="over memory budget")
        else:
            t1 = time.time()
            idx = ImplicitPerCurve(k, a.delta, L).build(D)
            print(f"  per-curve L={L}: built in {time.time() - t1:.0f}s ({idx.index_bytes / 1e9:.2f} GB)", flush=True)
            note = "" if L == Lp else f" (L reduced from {Lp} to fit 24 GB)"
            run_queries(w, dict(base, variant="complete" + note), idx, D, Qe, gt,
                        lambda ix, DD, Q: ix.query(DD, Q, R, cap=0, deadline=TIME_BUDGET), a.nq, a.query_seconds)
            print("  per-curve complete: done", flush=True)
            for K in a.Ks:
                run_queries(w, dict(base, method="per_curve_capped", variant=f"K={K}"), idx, D, Qe, gt,
                            lambda ix, DD, Q, K=K: ix.query(DD, Q, R, cap=K, rng_seed=K, deadline=TIME_BUDGET),
                            a.nq, a.query_seconds)
                print(f"  per-curve capped K={K}: done", flush=True)
            if n <= a.topk_nmax:
                for K in a.Ks_topk:
                    run_queries(w, dict(base, method="per_curve_topk", variant=f"K={K}"), idx, D, Qe, gt,
                                lambda ix, DD, Q, K=K: ix.query_topk(DD, Q, R, K, deadline=TIME_BUDGET),
                                a.nq, a.query_seconds)
                    print(f"  per-curve exact top-K K={K}: done", flush=True)
            del idx
    else:
        w.row(**common, method="per_curve", status=f"recall target not reached with {a.Lmax} tables")
    print(f"done {a.workload} n={n} k={k}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", default="popular", choices=["popular", "crowd"])
    ap.add_argument("--n", type=int, default=10 ** 8)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--delta", type=float, default=128.0)
    ap.add_argument("--nq", type=int, default=100)
    ap.add_argument("--Lmax", type=int, default=200)
    ap.add_argument("--Ks", type=int, nargs="+", default=[100, 1000, 10000, 100000])
    ap.add_argument("--Ks-topk", type=int, nargs="+", default=[100, 1000, 10000])
    ap.add_argument("--topk-nmax", type=int, default=10 ** 7)
    ap.add_argument("--bf-queries", type=int, default=3)
    ap.add_argument("--query-seconds", type=float, default=300.0)
    ap.add_argument("--results", default="results")
    main(ap.parse_args())
