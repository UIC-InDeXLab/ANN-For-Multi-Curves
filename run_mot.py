"""Real multi-object tracking data (plan question Q7): MOT17 + MOT20 ground truth, 2-D and 3-D.

  sweep   : grid size delta vs recall, candidates and time for blocking (verify mode), k in {2, 4, 6}
  methods : every method at a fixed practical delta and L, k = 2..8
  stats   : sizes and nearest-neighbor distance distributions of the curve-set collections

Queries:
  planted - a data set, shuffled and jittered by at most rho per vertex (a known neighbor within rho)
  natural - a data set removed from the collection, together with every set from the same sequence whose
            window overlaps it; its nearest neighbor is exact (brute force)
"""
import argparse
import time

import numpy as np

from curveset import data, mot
from curveset.bench import Writer, budget_blocking, ground_truth, run_method, warmup
from curveset.orders import balanced_blocking

ROOT = "data/mot"
RHO = {2: 5.0, 3: 0.1}          # planted jitter: 5 px at 1080p, 10 cm
DELTAS = {2: [100, 200, 400, 800, 1600], 3: [1, 2, 4, 8, 16]}


M = 32


def collection(k, dims):
    X17, m17 = mot.load(ROOT, "MOT17", k, m=M, dims=dims)
    X20, m20 = mot.load(ROOT, "MOT20", k, m=M, dims=dims)
    return np.concatenate([X17, X20]), m17 + m20


def natural_split(rng, X, meta, nq, overlap_frames=150):
    """Pick nq query sets and remove every set that shares a tracked person with a query in an overlapping
    window of the same sequence (the same people seen again); other groups remain searchable."""
    q_idx = rng.choice(len(meta), size=nq, replace=False)
    seqs = np.array([s for s, _, _ in meta])
    starts = np.array([t for _, t, _ in meta])
    keep = np.ones(len(meta), bool)
    for q in q_idx:
        near = np.nonzero((seqs == seqs[q]) & (np.abs(starts - starts[q]) < overlap_frames))[0]
        ids_q = set(meta[q][2])
        for p in near:
            if ids_q.intersection(meta[p][2]):
                keep[p] = False
    return X[keep], X[q_idx]


def exp_stats(args):
    w = Writer(f"{args.results}/mot_stats.csv")
    rng = np.random.default_rng(10)
    for dims in (2, 3):
        for k in (2, 4, 6, 8):
            X, meta = collection(k, dims)
            Xd, Qs = natural_split(rng, X, meta, args.nq)
            gt = ground_truth(Xd, Qs)
            for q in range(len(Qs)):
                w.row(exp="stats", dataset=f"MOT17+20 m={M}", d=dims, k=k, n=len(Xd), method="brute force", query=q,
                      d_nn=gt[1][q], bf_seconds=gt[2][q], status="ok")
            print(f"d={dims} k={k}: n={len(Xd)} median NN distance {np.median(gt[1]):.3g}", flush=True)


def exp_sweep(args):
    w = Writer(f"{args.results}/mot_sweep.csv")
    rng = np.random.default_rng(11)
    for dims in (2, 3):
        for k in (2, 4, 6):
            X, meta = collection(k, dims)
            Xd, Qn = natural_split(rng, X, meta, args.nq)
            Qp, tp_ = data.planted_queries(rng, Xd, args.nq, rho=RHO[dims])
            gt_n, gt_p = ground_truth(Xd, Qn), ground_truth(Xd, Qp)
            _, _, sizes, swapped = balanced_blocking(k)
            for delta in DELTAS[dims]:
                for name, Qs, targets, gt in (("natural", Qn, None, gt_n), ("planted", Qp, tp_, gt_p)):
                    t0 = time.time()
                    run_method(w, f"sweep-{name}", f"MOT17+20 m={M}", Xd, Qs, targets, gt, "blocking", k, dims, delta,
                               args.L, np.inf, sizes=sizes, swapped=swapped, max_queries=args.nq,
                               query_seconds=np.inf)
                    print(f"d={dims} k={k} delta={delta} {name}: {time.time() - t0:.1f}s", flush=True)


def exp_methods(args):
    """Every method under the budgets (24 GB of index, 60 s per query), natural queries."""
    w = Writer(f"{args.results}/mot_budget.csv")
    rng = np.random.default_rng(12)
    for dims in (2, 3):
        delta = args.delta2 if dims == 2 else args.delta3
        for k in args.ks:
            X, meta = collection(k, dims)
            Xd, Qn = natural_split(rng, X, meta, args.nq)
            gt = ground_truth(Xd, Qn)
            blk = budget_blocking(k, len(Xd), args.L)
            runs = [("sorted", None, False), ("per_curve", None, False), ("probe_all", None, False),
                    ("store_all", None, False)] + ([("blocking", blk[2], blk[3])] if blk else [])
            for method, sizes, swapped in runs:
                t0 = time.time()
                run_method(w, "methods-natural", f"MOT17+20 m={M}", Xd, Qn, None, gt, method, k, dims, delta,
                           args.L, np.inf, sizes=sizes, swapped=swapped, max_queries=args.nq)
                print(f"d={dims} k={k} n={len(Xd)} {method} {sizes or ''}{'s' if swapped else ''}: "
                      f"{time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", choices=["stats", "sweep", "methods"])
    ap.add_argument("--nq", type=int, default=200)
    ap.add_argument("--L", type=int, default=20)
    ap.add_argument("--delta2", type=float, default=400.0)
    ap.add_argument("--delta3", type=float, default=8.0)
    ap.add_argument("--ks", type=int, nargs="+", default=[2, 4, 6, 8, 10, 11, 12, 13, 14])
    ap.add_argument("--results", default="results")
    ap.add_argument("--m", type=int, default=32)
    a = ap.parse_args()
    M = a.m
    warmup()
    {"stats": exp_stats, "sweep": exp_sweep, "methods": exp_methods}[a.exp](a)
