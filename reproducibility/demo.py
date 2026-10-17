"""Small synthetic retrieval example with independently computed exact ground truth."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from curveset import core
from fresh.poc import Joint, PerCurve, dataset


def run(output, n=500):
    if n < 16:
        raise ValueError("The demonstration needs at least 16 database multi-curves")
    config = dict(
        dataset_seed=2026101217, m=8, d=2, rho=0.5, relevance_radius=1.5,
        tuning_queries=2, test_queries=5, warmup_queries=1,
        planted_per_query=1, plant_anchor_n=n, probe_chunk=1000,
        verification_chunk=1000, query_deadline_seconds=60,
    )
    case = dict(kind="conjunctive", n=n, k=3, frequency=0.18)
    X, Q, _ = dataset(config, case)
    query = Q[config["tuning_queries"]]
    distances = np.empty(n)
    for p, curves in enumerate(X):
        pairwise = np.array([[core.frechet(a, b) for b in query] for a in curves])
        distances[p] = min(max(pairwise[order[i], i] for i in range(3))
                           for order in itertools.permutations(range(3)))
    optimum = float(distances.min())
    records = []
    methods = (
        ("Blocking", Joint(X, 64.0, 4, 1701, (2, 1), False)),
        ("Probe-all", Joint(X, 64.0, 4, 1701, (1, 1, 1), False)),
        ("Store-all", Joint(X, 64.0, 4, 1701, (3,), False)),
        ("Intersection", PerCurve(X, 64.0, 4, 1701)),
    )
    for method, index in methods:
        answer = index.query(X, query, config)
        if answer["timeout"] or answer["id"] < 0:
            raise RuntimeError(f"{method} did not return a result")
        distance = float(core.dmatch(X[answer["id"]], query))
        if not np.isclose(distance, answer["distance"], rtol=0, atol=1e-9):
            raise AssertionError("Returned distance does not match the returned multi-curve")
        records.append(dict(method=method, id=answer["id"], distance=distance, optimum=optimum,
                            exact=bool(np.isclose(distance, optimum, rtol=0, atol=1e-9)),
                            candidates=answer["candidates"], index_bytes=index.index_bytes))
    result = dict(n=n, k=3, m=8, note="Functional example, not a paper timing measurement", results=records)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    for row in records:
        print(f"{row['method']:14} distance={row['distance']:.6f} optimum={optimum:.6f}"
              f" exact={row['exact']} candidates={row['candidates']}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/demo.json"))
    parser.add_argument("--n", type=int, default=500)
    args = parser.parse_args()
    run(args.output, args.n)
