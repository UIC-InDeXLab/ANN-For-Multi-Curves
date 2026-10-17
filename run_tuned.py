"""Tuned experiments: every method runs with the fewest tables that reach the target recall (the exact nearest
neighbor returned for at least 90% of the queries), within 24 GB of index and 60 s per query. Run everything
single-threaded (NUMBA_NUM_THREADS=1), one experiment at a time.

  popular : synthetic curve sets whose single trajectories are popular (each in about 20% of the sets) while a
            given combination of k trajectories is rare
  mot     : pedestrian groups from MOT17 and MOT20, natural queries, 2-D and 3-D
  pareto  : every blocking within the budgets for one workload, k, and grid size

For every grid size delta, the tables needed by joint keys (store all, probe all, blocking) and by per-curve LSH
are computed from the queries' exact nearest neighbors (curveset.tune), and every method then runs with its own L.
Blocking uses the block sizes that minimize the cost model (curveset.costmodel).
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np

from curveset import data
from curveset.bench import Writer, ground_truth, run_method, warmup
from curveset.costmodel import CostModel
from curveset.tune import tune_tables

M_POINTS = 32


class TuningLog:
    """One row per (workload, d, k, delta): the tables needed by joint keys and by per-curve LSH."""
    FIELDS = ["dataset", "d", "k", "n", "delta", "target", "L_joint", "L_per_curve", "recall_joint_at_L",
              "recall_per_curve_at_L"]

    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        new = not path.exists()
        self.f = open(path, "a", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=self.FIELDS)
        if new:
            self.w.writeheader()

    def row(self, **kw):
        self.w.writerow(kw)
        self.f.flush()


def popular(rng, n, k, nq):
    """Every curve is a jittered copy (radius 0.5 per vertex) of one of 5k random-walk motifs, so each motif occurs
    in about 1 - exp(-1/5) = 18% of the sets. Every query is k distinct motifs, planted in 3 sets."""
    return data.motif_collection(rng, n, k, M_POINTS, 2, extent=1000.0, step=3.0, n_motifs=5 * k, rho=0.5,
                                 planted_per_query=3, nq=nq)[:2]


def crowd(rng, n, k, nq):
    """Every object follows one of k popular paths (uniform), so each path occurs in about 1 - (1 - 1/k)^k = 65% of
    the sets and several objects of a set share a path. Queries are data sets, shuffled and jittered by 0.5."""
    X = data.crowd_collection(rng, n, k, M_POINTS, 2, extent=1000.0, step=3.0, n_paths=k, rho=0.5)
    Qs, _ = data.planted_queries(rng, X, nq, rho=0.5)
    return X, Qs


def lanes(rng, n, k, nq):
    """A fixed scene with 4 lanes: every object follows one of 4 paths (uniform), so nearly every curve set has
    objects on every lane and clips differ in how many objects use each lane. Queries are data sets, shuffled
    and jittered by 0.5."""
    X = data.crowd_collection(rng, n, k, M_POINTS, 2, extent=1000.0, step=3.0, n_paths=4, rho=0.5)
    Qs, _ = data.planted_queries(rng, X, nq, rho=0.5)
    return X, Qs


WORKLOADS = {"popular": popular, "crowd": crowd, "lanes": lanes}


def run_config(w, tw, exp, dataset, X, Qs, gt, k, d, deltas, args, cm):
    """Tune on the first half of the queries, evaluate every method on the second half."""
    h = len(Qs) // 2
    Qt, NNt = Qs[:h], X[gt[0][:h]]
    Qe, gte = Qs[h:], (gt[0][h:], gt[1][h:], gt[2][h:])
    for delta in deltas:
        t0 = time.time()
        Lj, Lp, fj, fp = tune_tables(Qt, NNt, k, d, delta, args.target, args.Lmax, 0, 0)
        tw.row(dataset=dataset, d=d, k=k, n=X.shape[0], delta=delta, target=args.target, L_joint=Lj or "",
               L_per_curve=Lp or "", recall_joint_at_L=(fj <= Lj).mean() if Lj else "",
               recall_per_curve_at_L=(fp <= Lp).mean() if Lp else "")
        print(f"  {dataset} d={d} k={k} delta={delta:g}: joint L={Lj}, per-curve L={Lp} ({time.time() - t0:.0f}s)",
              flush=True)
        runs = []
        base = dict(exp=exp, dataset=dataset, d=d, k=k, n=X.shape[0], delta=delta)
        if Lj:
            runs += [("store_all", Lj, None, False), ("probe_all", Lj, None, False)]
            best = cm.best_blocking(k, X.shape[0], Lj)
            if best:
                runs.append(("blocking", Lj, best[3], best[4]))
            else:
                w.row(**base, method="blocking", L=Lj, status="no blocking within both budgets")
        else:
            for m in ("store_all", "probe_all", "blocking"):
                w.row(**base, method=m, status=f"recall target not reached with {args.Lmax} tables")
        if Lp:
            runs.append(("per_curve", Lp, None, False))
        else:
            w.row(**base, method="per_curve", status=f"recall target not reached with {args.Lmax} tables")
        for method, L, sizes, swapped in runs:
            t1 = time.time()
            run_method(w, exp, dataset, X, Qe, None, gte, method, k, d, delta, L, np.inf, sizes=sizes,
                       swapped=swapped, max_queries=len(Qe), query_seconds=args.query_seconds)
            print(f"    {method} L={L} {sizes or ''}{'s' if swapped else ''}: {time.time() - t1:.0f}s", flush=True)


def exp_synthetic(args):
    name = args.exp
    w = Writer(f"{args.results}/tuned_{name}.csv")
    tw = TuningLog(f"{args.results}/tuned_{name}_tuning.csv")
    cm = CostModel.load()
    for n in args.ns:
        for k in args.ks:
            rng = np.random.default_rng(1000 * k + n % 997)
            X, Qs = WORKLOADS[name](rng, n, k, 2 * args.nq)
            gt = ground_truth(X, Qs)
            print(f"{name} n={n} k={k}: brute force {gt[2].mean():.3f} s per query", flush=True)
            run_config(w, tw, f"tuned_{name}", name, X, Qs, gt, k, 2, args.deltas, args, cm)


def exp_mot(args):
    import run_mot
    w, tw = Writer(f"{args.results}/tuned_mot.csv"), TuningLog(f"{args.results}/tuned_mot_tuning.csv")
    cm = CostModel.load()
    for dims in (2, 3):
        deltas = args.deltas2 if dims == 2 else args.deltas3
        for k in args.ks:
            rng = np.random.default_rng(100 + k)
            X, meta = run_mot.collection(k, dims)
            Xd, Qn = run_mot.natural_split(rng, X, meta, 2 * args.nq)
            gt = ground_truth(Xd, Qn)
            print(f"MOT d={dims} k={k} n={len(Xd)}: brute force {gt[2].mean():.3f} s per query", flush=True)
            run_config(w, tw, "tuned_mot", "MOT17+20", Xd, Qn, gt, k, dims, deltas, args, cm)


def exp_pareto(args):
    """Every blocking that fits the memory budget, plus the other methods, on the same collection, test queries,
    and tuned tables as the popular experiment (read from its tuning log)."""
    import pandas as pd
    from curveset.orders import blocking_points
    from curveset.index import index_bytes
    w = Writer(f"{args.results}/tuned_pareto.csv")
    n, k, delta = args.ns[0], args.k, args.delta
    t = pd.read_csv(f"{args.results}/tuned_{args.workload}_tuning.csv")
    row = t[(t.n == n) & (t.k == k) & (t.delta == delta)].iloc[-1]
    L, Lp = int(row.L_joint), int(row.L_per_curve)
    rng = np.random.default_rng(1000 * k + n % 997)
    X, Qs = WORKLOADS[args.workload](rng, n, k, 2 * args.nq)
    h = len(Qs) // 2
    gt = ground_truth(X, Qs[h:])
    Qe = Qs[h:]
    print(f"pareto n={n} k={k} delta={delta:g} L={L} per-curve L={Lp}", flush=True)
    for a, b, sizes, swapped in blocking_points(k):
        if index_bytes(n * a, L) > 24e9:
            continue
        t0 = time.time()
        run_method(w, "tuned_pareto", args.workload, X, Qe, None, gt, "blocking", k, 2, delta, L, np.inf, sizes=sizes,
                   swapped=swapped, max_queries=len(Qe), query_seconds=args.query_seconds)
        print(f"  {sizes}{'s' if swapped else ''} a={a} b={b}: {time.time() - t0:.0f}s", flush=True)
    run_method(w, "tuned_pareto", args.workload, X, Qe, None, gt, "per_curve", k, 2, delta, Lp, np.inf,
               max_queries=len(Qe), query_seconds=args.query_seconds)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", choices=["popular", "crowd", "lanes", "mot", "pareto"])
    ap.add_argument("--ns", type=int, nargs="+", default=[1000000])
    ap.add_argument("--ks", type=int, nargs="+", default=[4, 6, 8, 10, 12])
    ap.add_argument("--deltas", type=float, nargs="+", default=[16.0, 32.0, 64.0, 128.0])
    ap.add_argument("--deltas2", type=float, nargs="+", default=[100.0, 200.0, 400.0, 800.0])
    ap.add_argument("--deltas3", type=float, nargs="+", default=[1.0, 2.0, 4.0, 8.0])
    ap.add_argument("--target", type=float, default=0.9)
    ap.add_argument("--nq", type=int, default=100)
    ap.add_argument("--Lmax", type=int, default=400)
    ap.add_argument("--query-seconds", type=float, default=300.0)
    ap.add_argument("--workload", default="crowd", choices=["popular", "crowd", "lanes"])
    ap.add_argument("--dims", type=int, default=2)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--delta", type=float, default=64.0)
    ap.add_argument("--L", type=int, default=10)
    ap.add_argument("--Lp", type=int, default=0)
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    warmup()
    {"popular": exp_synthetic, "crowd": exp_synthetic, "lanes": exp_synthetic, "mot": exp_mot,
     "pareto": exp_pareto}[a.exp](a)
