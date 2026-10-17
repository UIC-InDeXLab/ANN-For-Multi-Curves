"""Experiments under the budgets of the paper: at most 24 GB of index and at most 60 s per query.

  k      : memory and query time of every method as k grows (synthetic, n = 1e5, theory parameters)
  pareto : every blocking for given k (synthetic, n = 1e5), measured memory and query time
  n      : memory and query time as n grows (k = 6)
  motif  : per-curve LSH on curve sets whose curves match individually but rarely jointly

Blocking uses the split with the fewest probes that fits in the memory budget (bench.budget_blocking).
Run on the server:  python run_budget.py <experiment> [options]
"""
import argparse
import time

import numpy as np

from curveset import data
from curveset.bench import Writer, budget_blocking, ground_truth, run_method, warmup
from curveset.index import theory_params

M, R, P_SLOT, ETA, D = 32, 1.0, 0.75, 0.01, 2


def per_curve_tables(k):
    """Tables so that every matched curve collides in some table with probability >= 1 - eta/k."""
    return int(np.ceil(np.log(ETA / k) / np.log(1 - P_SLOT)))


def synth(rng, n, k, delta, nq):
    X = data.collection(rng, n, k, M, D, extent=100 * delta, step=0.05 * delta)
    Qs, targets = data.planted_queries(rng, X, nq, rho=R)
    return X, Qs, targets


def all_methods(w, exp, dataset, X, Qs, targets, gt, k, tp, args):
    n = X.shape[0]
    blk = budget_blocking(k, n, tp["L"])
    runs = [("sorted", tp["L"], None, False), ("per_curve", per_curve_tables(k), None, False),
            ("probe_all", tp["L"], None, False), ("store_all", tp["L"], None, False)]
    if blk is not None:
        runs.append(("blocking", tp["L"], blk[2], blk[3]))
    for method, L, sizes, swapped in runs:
        t0 = time.time()
        run_method(w, exp, dataset, X, Qs, targets, gt, method, k, D, tp["delta"], L, tp["c"] * R,
                   sizes=sizes, swapped=swapped, max_queries=args.nq)
        print(f"  {exp} n={n} k={k} {method} {sizes or ''}{'s' if swapped else ''}: {time.time() - t0:.0f}s",
              flush=True)


def exp_k(args):
    w = Writer(f"{args.results}/budget_k.csv")
    rng = np.random.default_rng(21)
    for k in range(args.kmin, args.kmax + 1):
        tp = theory_params(D, M, R, P_SLOT, k, ETA)
        X, Qs, targets = synth(rng, args.n, k, tp["delta"], args.nq)
        gt = ground_truth(X, Qs)
        print(f"k={k}: brute force {1000 * gt[2].mean():.1f} ms per query", flush=True)
        all_methods(w, "budget_k", "synthetic", X, Qs, targets, gt, k, tp, args)


def exp_n(args):
    w = Writer(f"{args.results}/budget_n.csv")
    rng = np.random.default_rng(23)
    k = args.k
    tp = theory_params(D, M, R, P_SLOT, k, ETA)
    for n in (1000, 3000, 10000, 30000, 100000, 300000, 1000000):
        if n > args.nmax:
            break
        X, Qs, targets = synth(rng, n, k, tp["delta"], args.nq)
        gt = ground_truth(X, Qs)
        print(f"n={n}: brute force {1000 * gt[2].mean():.1f} ms per query", flush=True)
        all_methods(w, "budget_n", "synthetic", X, Qs, targets, gt, k, tp, args)


def exp_pareto(args):
    from curveset.orders import blocking_points
    w = Writer(f"{args.results}/budget_pareto.csv")
    rng = np.random.default_rng(22)
    for k in args.ks:
        tp = theory_params(D, M, R, P_SLOT, k, ETA)
        X, Qs, targets = synth(rng, args.n, k, tp["delta"], args.nq)
        gt = ground_truth(X, Qs)
        for a, b, sizes, swapped in blocking_points(k):
            t0 = time.time()
            run_method(w, "budget_pareto", "synthetic", X, Qs, targets, gt, "blocking", k, D, tp["delta"],
                       tp["L"], tp["c"] * R, sizes=sizes, swapped=swapped, max_queries=args.nq,
                       query_seconds=args.query_seconds)
            print(f"  pareto k={k} {sizes}{'s' if swapped else ''} a={a} b={b}: {time.time() - t0:.0f}s", flush=True)


def exp_motif(args):
    w = Writer(f"{args.results}/budget_motif.csv")
    rng = np.random.default_rng(25)
    rho = R / 2
    for k in (4, 6):
        tp = theory_params(D, M, R, P_SLOT, k, ETA)
        for n in args.ns:
            X, Qs, _ = data.motif_collection(rng, n, k, M, D, extent=100 * tp["delta"], step=0.05 * tp["delta"],
                                             n_motifs=5 * k, rho=rho, planted_per_query=3, nq=args.nq)
            gt = ground_truth(X, Qs)
            all_methods(w, "budget_motif", "synthetic-motif", X, Qs, None, gt, k, tp, args)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", choices=["k", "n", "pareto", "motif"])
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--nq", type=int, default=100)
    ap.add_argument("--kmin", type=int, default=2)
    ap.add_argument("--kmax", type=int, default=13)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--ks", type=int, nargs="+", default=[10, 11])
    ap.add_argument("--nmax", type=int, default=1000000)
    ap.add_argument("--ns", type=int, nargs="+", default=[1000, 3000, 10000, 30000, 100000])
    ap.add_argument("--query-seconds", type=float, default=300.0)
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    warmup()
    {"k": exp_k, "n": exp_n, "pareto": exp_pareto, "motif": exp_motif}[a.exp](a)
