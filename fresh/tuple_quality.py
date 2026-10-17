"""Exact all-tuple distances for a uniform-random answer, separate from timed queries."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
from numba import njit, prange

from curveset import core


@njit(cache=True)
def frechet_squared(A, B):
    lower = 0.0
    for a, b in ((0, 0), (len(A) - 1, len(B) - 1)):
        value = 0.0
        for d in range(A.shape[1]):
            diff = A[a, d] - B[b, d]
            value += diff * diff
        lower = max(lower, value)
    diagonal = lower
    if len(A) == len(B):
        for j in range(len(A)):
            value = 0.0
            for d in range(A.shape[1]):
                diff = A[j, d] - B[j, d]
                value += diff * diff
            diagonal = max(diagonal, value)
        if diagonal == lower:
            return lower
    previous = np.empty(len(B))
    current = np.empty(len(B))
    for i in range(len(A)):
        for j in range(len(B)):
            value = 0.0
            for d in range(A.shape[1]):
                diff = A[i, d] - B[j, d]
                value += diff * diff
            if i == j == 0:
                prior = 0.0
            elif i == 0:
                prior = current[j - 1]
            elif j == 0:
                prior = previous[j]
            else:
                prior = min(previous[j], previous[j - 1], current[j - 1])
            current[j] = max(prior, value)
        previous, current = current, previous
    return previous[-1]


@njit(cache=True)
def exact_distance(P, Q):
    k = len(Q)
    endpoints = np.empty((k, k))
    for i in range(k):
        for j in range(k):
            start, end = 0.0, 0.0
            for d in range(Q.shape[2]):
                a = Q[i, 0, d] - P[j, 0, d]
                b = Q[i, -1, d] - P[j, -1, d]
                start += a * a
                end += b * b
            endpoints[i, j] = max(start, end)
    lower = core.bottleneck(endpoints)
    row_of_col = np.empty(k, np.int64)
    col_of_row = np.empty(k, np.int64)
    parent = np.empty(k, np.int64)
    visited = np.empty(k, np.bool_)
    queue = np.empty(k, np.int64)
    if not core._has_perfect_matching(endpoints, lower, row_of_col, col_of_row, parent, visited, queue):
        raise ValueError("Endpoint bottleneck failed to produce a matching")
    distances = np.full((k, k), np.inf)
    computed = np.zeros((k, k), np.bool_)
    upper = 0.0
    for i in range(k):
        j = col_of_row[i]
        value = frechet_squared(Q[i], P[j])
        distances[i, j] = value
        computed[i, j] = True
        upper = max(upper, value)
    if upper == lower:
        return np.sqrt(upper)
    # An edge above a feasible matching's cost cannot be needed by an optimal matching.
    for i in range(k):
        for j in range(k):
            if not computed[i, j] and endpoints[i, j] <= upper:
                distances[i, j] = frechet_squared(Q[i], P[j])
    return np.sqrt(core.bottleneck(distances))


@njit(cache=True, parallel=True)
def all_tuple_statistics(X, Qs, r):
    output = np.empty((len(Qs), 4))
    for q in prange(len(Qs)):
        total, nearest, farthest, relevant = 0.0, np.inf, 0.0, 0
        for p in range(len(X)):
            value = exact_distance(X[p], Qs[q])
            total += value
            nearest = min(nearest, value)
            farthest = max(farthest, value)
            relevant += int(value <= r)
        output[q, 0] = total / len(X)
        output[q, 1] = nearest
        output[q, 2] = farthest
        output[q, 3] = relevant
    return output


def save_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def first_hit_rows(folder, method, config):
    statuses = json.loads((folder / "configurations.json").read_text())
    matching = [r for r in statuses if r["method"] == method and r["mode"] == "first"]
    if len(matching) != len(config["index_seeds"]) or any(r["status"] != "complete" for r in matching):
        raise ValueError(f"First-hit configuration not complete for all seeds: {folder.name}, {method}")
    with (folder / "queries.csv").open() as handle:
        rows = [r for r in csv.DictReader(handle) if r["method"] == method and r["mode"] == "first"]
    expected = {(seed, q) for seed in config["index_seeds"] for q in range(config["test_queries"])}
    observed = {(int(r["seed"]), int(r["query"])) for r in rows}
    if len(rows) != len(expected) or observed != expected or any(r["status"] != "ok" for r in rows):
        raise ValueError(f"Incomplete query rows: {folder.name}, {method}")
    return rows, matching[0]


def analyze_case(root, output, name, method):
    folder = root / name
    config = json.loads((root / "config.json").read_text())
    rows, metadata = first_hit_rows(folder, method, config)
    with np.load(folder / "dataset-ground-truth.npz") as dataset:
        X, Q = dataset["X"], dataset["Q"]
        optimum, relevant = dataset["nn_distance"], dataset["relevant"]
    if len(X) != metadata["n"] or len(Q) != config["test_queries"]:
        raise ValueError("Dataset shape does not match the logged configuration")
    started = time.perf_counter()
    statistics = all_tuple_statistics(X, Q, config["r"])
    elapsed = time.perf_counter() - started
    if not np.allclose(statistics[:, 1], optimum, atol=1e-8, rtol=1e-10):
        raise AssertionError("Exact all-tuple scan disagrees with cached nearest-neighbor distances")
    if not np.array_equal(statistics[:, 3], relevant):
        raise AssertionError("Exact all-tuple scan disagrees with cached relevance counts")
    np.savez(output / f"{name}-all-tuples.npz", query_mean_distance=statistics[:, 0],
             query_min_distance=statistics[:, 1], query_max_distance=statistics[:, 2],
             query_relevance_count=statistics[:, 3])
    first = np.array([float(r["returned_distance"]) if r["found"] == "True" else np.nan for r in rows])
    query_ids = np.array([int(r["query"]) for r in rows])
    positive = optimum > 0
    found_positive = np.isfinite(first) & positive[query_ids]
    result = dict(
        case=name, method=method, workload=metadata["workload"], d=metadata["d"], k=metadata["k"],
        experiment_commit=metadata["commit"], dataset_sha256=metadata["dataset_sha256"],
        query_count=len(Q), tuple_count=len(X), first_hit_evaluations=len(first),
        exact_tuple_query_distances=len(X) * len(Q), scan_seconds=elapsed,
        mean_optimal_distance=float(optimum.mean()),
        mean_first_hit_distance=float(first[np.isfinite(first)].mean()) if np.isfinite(first).any() else None,
        first_hit_empty_answers=int((~np.isfinite(first)).sum()),
        mean_uniform_random_tuple_distance=float(statistics[:, 0].mean()),
        mean_first_hit_over_optimum=float((first[found_positive] / optimum[query_ids[found_positive]]).mean())
        if found_positive.any() else None,
        mean_uniform_random_over_optimum=float((statistics[positive, 0] / optimum[positive]).mean())
        if positive.any() else None,
        zero_optimum_queries=int((~positive).sum()))
    save_json(output / f"{name}-quality.json", result)
    print(json.dumps(result), flush=True)
    return result


def main(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "analysis-manifest.json", dict(
        purpose="Exact uniform-random tuple distance; offline derived statistics, not query timing",
        experiment_root=str(args.results.resolve()), analysis_host=platform.node(),
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        numpy=np.__version__, affinity=sorted(os.sched_getaffinity(0)),
        started=time.time(), cases=args.cases))
    results = []
    for name in args.cases:
        results.append(analyze_case(args.results, output, name, args.method))
        save_json(output / "table.json", results)
    save_json(output / "completion.json", dict(completed=time.time(), cases=len(results)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", required=True)
    parser.add_argument("--method", default="cost-model")
    main(parser.parse_args())
