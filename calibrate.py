"""Calibrate the budgeted implementation near the budgets (not part of the experiments)."""
import sys
import time

import numpy as np

from curveset import data
from curveset.bench import budget_blocking, pinned, warmup
from curveset.index import make_index, theory_params

warmup()
rng = np.random.default_rng(0)
n = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
cases = [(9, "probe_all"), (10, "probe_all"), (10, "blocking"), (11, "blocking"), (12, "blocking")]
for k, method in cases:
    tp = theory_params(2, 32, 1.0, 0.75, k, 0.01)
    X = data.collection(rng, n, k, 32, 2, extent=100 * tp["delta"], step=0.05 * tp["delta"])
    Qs, targets = data.planted_queries(rng, X, 2, rho=1.0)
    sizes, swapped = None, False
    if method == "blocking":
        _, _, sizes, swapped = budget_blocking(k, n, tp["L"])
    t0 = time.time()
    idx = make_index(method, k, 2, tp["delta"], tp["L"], sizes=sizes, swapped=swapped).build(X)
    build = time.time() - t0
    with pinned():
        res = [idx.query(X, Q, mode="verify", deadline=90) for Q in Qs]
    probes = idx.probed.shape[0] * tp["L"]
    sec = float(np.mean([r["seconds"] for r in res]))
    label = f"{sizes}{'s' if swapped else ''}"
    print(f"k={k} {method} {label} L={tp['L']} a={idx.stored.shape[0]} b={idx.probed.shape[0]} "
          f"GB={idx.index_bytes / 1e9:.2f} build={build:.0f}s query={sec:.2f}s ({1e9 * sec / probes:.1f} ns/probe) "
          f"timeout={[r['timeout'] for r in res]} planted found={[r['id'] == t for r, t in zip(res, targets)]}",
          flush=True)
    del idx
