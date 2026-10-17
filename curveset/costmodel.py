"""Cost model for choosing the block sizes, and the calibration of its per-probe cost.

Cost of a blocking that stores a orders and probes b orders per table, with L tables and n curve sets:
    memory M(a) = index_bytes(n a, L)                       (13 bytes per stored key)
    time   T(b) = L * b * tau(n a)                           (tau: measured seconds per probe for a table of n a keys)
    cost        = M(a) / MEM_BUDGET + T(b) / TIME_BUDGET     (the fraction of each budget that is used)
The balanced blocking minimizes the cost among all block sizes (both roles) that fit in both budgets. Storing all
orders (a = k!) and probing all orders (b = k!) are two of the candidates.

Calibration (python -m curveset.costmodel): single-threaded probes with the real kernel against tables of random
keys; the probed family is all orders of 9 curves, so every probe pays the prefix-shared hashing of the queries.
"""
import json
import time
from pathlib import Path

import numpy as np

from .bench import MEM_BUDGET, TIME_BUDGET
from .index import Tables, _probe_orders, index_bytes
from .orders import all_orders, blocking_points

DEFAULT_FILE = Path(__file__).resolve().parent.parent / "results" / "cost_model.json"


class CostModel:
    def __init__(self, sizes, seconds, mem_budget=MEM_BUDGET, time_budget=TIME_BUDGET):
        self.logN = np.log(np.asarray(sizes, float))
        self.logt = np.log(np.asarray(seconds, float))
        self.mem_budget, self.time_budget = mem_budget, time_budget

    @classmethod
    def load(cls, path=DEFAULT_FILE, **kw):
        d = json.loads(Path(path).read_text())
        return cls(d["entries"], d["seconds_per_probe"], **kw)

    def tau(self, N):
        """Seconds per probe for a table of N keys (log-log interpolation, constant beyond the calibrated range)."""
        return float(np.exp(np.interp(np.log(max(N, 1)), self.logN, self.logt)))

    def memory(self, n, L, a):
        return index_bytes(n * a, L)

    def time(self, n, L, a, b):
        return L * b * self.tau(n * a)

    def cost(self, n, L, a, b):
        return self.memory(n, L, a) / self.mem_budget + self.time(n, L, a, b) / self.time_budget

    def candidates(self, k, n, L):
        """(cost, a, b, sizes, swapped, memory, time) for every blocking within both budgets, cheapest first."""
        out = []
        for a, b, sizes, swapped in blocking_points(k):
            mem, t = self.memory(n, L, a), self.time(n, L, a, b)
            if mem <= self.mem_budget and t <= self.time_budget:
                out.append((mem / self.mem_budget + t / self.time_budget, a, b, sizes, swapped, mem, t))
        return sorted(out)

    def best_blocking(self, k, n, L):
        c = self.candidates(k, n, L)
        return c[0] if c else None


def calibrate(sizes=(1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9), k=9, repeats=3, path=DEFAULT_FILE):
    rng = np.random.default_rng(0)
    probed = all_orders(k)
    Q = np.cumsum(rng.normal(0, 1, size=(k, 32, 2)), axis=1)
    shifts = rng.uniform(0, 8.0, size=(1, k, 2))
    need = np.ones((k, k), np.bool_)
    secs = []
    for N in sizes:
        N = int(N)
        T = Tables(1, N)
        T.fill(0, rng.integers(0, np.iinfo(np.uint64).max, size=N, dtype=np.uint64), 1)
        mark = np.zeros(N, np.int32)
        cand = np.empty(N, np.int64)
        _probe_orders(Q, shifts, 8.0, probed[:1000], need, T.keys, T.ids, T.dir, T.N, T.B, 0, 1, False, mark, 1,
                      cand, 0)  # compile and warm the caches
        best = np.inf
        for r in range(repeats):
            t0 = time.perf_counter()
            _probe_orders(Q, shifts, 8.0, probed, need, T.keys, T.ids, T.dir, T.N, T.B, 0, 1, False, mark, r + 2,
                          cand, 0)
            best = min(best, time.perf_counter() - t0)
        secs.append(best / probed.shape[0])
        print(f"table of {N:.0e} keys ({T.nbytes / 1e9:.2f} GB): {1e9 * secs[-1]:.1f} ns per probe", flush=True)
        del T, mark, cand
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"entries": [int(N) for N in sizes], "seconds_per_probe": secs,
                                      "k": k, "note": "single thread, random keys, all orders of k curves"}, indent=1))
    return secs


if __name__ == "__main__":
    calibrate()
