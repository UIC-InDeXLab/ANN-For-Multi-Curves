"""Explore the popular-trajectories workload: tables needed for 90% recall per grid size, and curve popularity."""
import numpy as np

from curveset.bench import ground_truth, warmup
from curveset.index import make_index
from curveset.tune import tune_tables
from run_tuned import popular

warmup()
n, nq = 20000, 100
for k in (4, 6, 8, 10, 12):
    rng = np.random.default_rng(7 + k)
    X, Qs = popular(rng, n, k, nq)
    gt = ground_truth(X, Qs)
    NN = X[gt[0]]
    print(f"k={k}: NN distance median {np.median(gt[1]):.2f}, brute force {1000 * gt[2].mean():.1f} ms", flush=True)
    for delta in (8.0, 16.0, 32.0, 64.0, 128.0):
        Lj, Lp, fj, fp = tune_tables(Qs, NN, k, 2, delta, 0.9, 2000, 0, 0)
        pc = make_index("per_curve", k, 2, delta, 1, seed=0).build(X)
        lists = [pc.query(X, Q)["listed"] for Q in Qs[:20]]
        print(f"   delta={delta:5g}: L_joint={Lj}  L_per_curve={Lp}  per-curve list per query curve and table = "
              f"{np.mean(lists) / k / n:.1%} of the sets", flush=True)
