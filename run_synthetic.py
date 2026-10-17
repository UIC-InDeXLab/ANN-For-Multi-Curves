"""Synthetic correctness experiments (plan questions Q1, Q2, Q6). The cost experiments are in run_budget.py.

  guarantees : failure rate vs eta, nothing beyond cr, ratio to the exact nearest neighbor (near misses added)
  sorted     : per-table collision rate of the planted neighbor, sorted key vs blocking, for random, spread,
               and formation placements

Run on the server:  python run_synthetic.py <experiment>
"""
import argparse
import time

import numpy as np

from curveset import data
from curveset.bench import Writer, ground_truth, run_method, warmup
from curveset.index import theory_params
from curveset.orders import balanced_blocking

M, R, P_SLOT, ETA = 32, 1.0, 0.75, 0.01


def synth(rng, n, k, d, delta, nq, near_miss=True):
    X = data.collection(rng, n, k, M, d, extent=100 * delta, step=0.05 * delta)
    Qs, targets = data.planted_queries(rng, X, nq, rho=R)
    if near_miss:
        X = data.add_near_misses(rng, X, targets, radii=[2, 4, 8, 16, 32, 64, 128, 256, 512])
    return X, Qs, targets


def exp_guarantees(args):
    w = Writer(f"{args.results}/guarantees.csv")
    rng = np.random.default_rng(1)
    for d in (2, 3):
        for k in range(2, 9):
            tp = theory_params(d, M, R, P_SLOT, k, ETA)
            X, Qs, targets = synth(rng, args.n, k, d, tp["delta"], args.nq)
            gt = ground_truth(X, Qs)
            _, _, sizes, swapped = balanced_blocking(k)
            for method in ("blocking", "probe_all", "store_all", "sorted", "per_curve"):
                L = tp["L"]
                if method == "per_curve":  # each query curve must find its match in some table
                    L = int(np.ceil(np.log(ETA / k) / np.log(1 - P_SLOT)))
                t0 = time.time()
                run_method(w, "guarantees", "synthetic", X, Qs, targets, gt, method, k, d, tp["delta"], L,
                           tp["c"] * R, modes=("first", "verify"), sizes=sizes, swapped=swapped,
                           max_queries=args.nq, query_seconds=np.inf)
                print(f"d={d} k={k} {method}: {time.time() - t0:.1f}s", flush=True)


def exp_sorted(args):
    """Collision rate per table (L = 1) of planted neighbors, for random, spread, and formation placements."""
    w = Writer(f"{args.results}/sorted.csv")
    rng = np.random.default_rng(6 if args.placement == "all" else 60)
    d = 2
    placements = ("random", "spread", "formation") if args.placement == "all" else (args.placement,)
    for placement in placements:
        delta = (160.0 if placement == "formation" else 40.0) * R
        for k in range(2, 11):
            if placement == "random":
                X = data.collection(rng, args.n, k, M, d, extent=100 * delta, step=0.05 * delta)
            elif placement == "spread":
                X = data.spread_collection(rng, args.n, k, M, d, delta)
            else:
                X = data.formation_collection(rng, args.n, k, M, d, delta)
            Qs, targets = data.planted_queries(rng, X, args.nq, rho=R)
            gt = ground_truth(X, Qs)
            _, _, sizes, swapped = balanced_blocking(k)
            for method in ("sorted", "blocking"):
                for seed in range(args.tables):
                    run_method(w, "sorted", placement, X, Qs, targets, gt, method, k, d, delta, 1, np.inf,
                               modes=("first",), sizes=sizes, swapped=swapped, seed=seed, max_queries=args.nq,
                               query_seconds=np.inf)
            print(f"{placement} k={k} done", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", choices=["guarantees", "sorted"])
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--nq", type=int, default=200)
    ap.add_argument("--tables", type=int, default=20)
    ap.add_argument("--placement", default="all", choices=["all", "random", "spread", "formation"])
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    warmup()
    {"guarantees": exp_guarantees, "sorted": exp_sorted}[a.exp](a)
