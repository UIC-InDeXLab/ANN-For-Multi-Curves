"""Small-scale check of the lanes workload: blocking vs per-curve LSH (not part of the experiments)."""
import sys

import numpy as np

import run_tuned
from curveset.bench import ground_truth, warmup
from curveset.costmodel import CostModel
from curveset.index import make_index
from curveset.tune import tune_tables

warmup()
cm = CostModel.load(sys.argv[1])
n = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
for k in (8, 12):
    rng = np.random.default_rng(5)
    X, Qs = run_tuned.lanes(rng, n, k, 40)
    gt = ground_truth(X, Qs)
    Lj, Lp, fj, fp = tune_tables(Qs, X[gt[0]], k, 2, 256.0, 0.9, 200, 0, 0)
    print(f"k={k}: brute {1000 * gt[2].mean():.1f} ms, L_joint={Lj}, L_per_curve={Lp}", flush=True)
    best = cm.best_blocking(k, n, Lj)
    for name, L, sizes, sw in (("blocking", Lj, best[3], best[4]), ("per_curve", Lp, None, False)):
        idx = make_index(name, k, 2, 256.0, L, sizes=sizes, swapped=sw).build(X)
        res = [idx.query(X, Q, mode="verify") for Q in Qs]
        t = 1000 * np.median([r["seconds"] for r in res])
        cand = np.mean([r["candidates"] for r in res])
        exact = np.mean([r["id"] == g for r, g in zip(res, gt[0])])
        print(f"   {name:10s} {sizes} t={t:.1f} ms cand={cand:.0f} exact={exact:.2f}", flush=True)
