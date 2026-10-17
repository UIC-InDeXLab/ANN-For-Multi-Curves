"""Small matched-accuracy comparisons across synthetic retrieval regimes."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
from numba import njit, prange

from curveset import core, data
from curveset.bench import pinned
from curveset.costmodel import CostModel
from curveset.implicit import Implicit, _lower, _upper, fill_table
from curveset.index import Tables, _bucket, _probe_orders, index_bytes, make_index
from curveset.orders import balanced_blocking
from curveset.tune import first_hit_per_curve, tables_for_recall
from fresh.e1 import (append_row, arrays_hash, build_index, monitor_worker, resident_bytes,
                      selected_blocking, terminate_worker, write_json)
from fresh.tuple_quality import exact_distance


@njit(cache=True)
def _block_match(within, allocated, starts, ends):
    k = within.shape[0]
    next_pair = np.zeros(k, np.int64)
    chosen_query, chosen_data = np.empty(k, np.int64), np.empty(k, np.int64)
    slot, used_within, used_allocated = 0, 0, 0
    while slot >= 0:
        if slot == k:
            return True
        previous = -1 if slot == starts[slot] else chosen_query[slot-1]
        found = False
        for pair in range(next_pair[slot], k*k):
            j, l = pair//k, pair%k
            if j <= previous or l < starts[slot] or l >= ends[slot]:
                continue
            if used_allocated & (1 << j) or used_within & (1 << l):
                continue
            if within[slot, l] == allocated[slot, j]:
                next_pair[slot] = pair+1
                chosen_query[slot], chosen_data[slot] = j, l
                used_allocated |= 1 << j
                used_within |= 1 << l
                slot += 1
                if slot < k:
                    next_pair[slot] = 0
                found = True
                break
        if not found:
            next_pair[slot] = 0
            slot -= 1
            if slot >= 0:
                used_allocated &= ~(1 << chosen_query[slot])
                used_within &= ~(1 << chosen_data[slot])
    return False


@njit(cache=True)
def family_first_hits(Qs, NN, shifts, delta, sizes, swapped):
    nq, k = Qs.shape[:2]
    starts, ends = np.empty(k, np.int64), np.empty(k, np.int64)
    start = 0
    for size in sizes:
        for i in range(start, start + size):
            starts[i], ends[i] = start, start + size
        start += size
    result = np.full(nq, len(shifts) + 1, np.int64)
    within, allocated = np.empty((k, k), np.uint64), np.empty((k, k), np.uint64)
    edges = np.empty((k, k))
    row_of_col, col_of_row = np.empty(k, np.int64), np.empty(k, np.int64)
    parent, queue = np.empty(k, np.int64), np.empty(k, np.int64)
    visited = np.empty(k, np.bool_)
    for q in range(nq):
        for u in range(len(shifts)):
            for i in range(k):
                for l in range(k):
                    A = Qs[q, l] if swapped else NN[q, l]
                    B = NN[q, l] if swapped else Qs[q, l]
                    within[i, l] = core.curve_fp(A, shifts[u, i], delta)
                    allocated[i, l] = core.curve_fp(B, shifts[u, i], delta)
            if len(sizes) == 1 or len(sizes) == k:
                for i in range(k):
                    for j in range(k):
                        equal = within[i, j] == allocated[i, i] if len(sizes) == 1 else within[i, i] == allocated[i, j]
                        edges[i, j] = 0.0 if equal else 1.0
                matched = core._has_perfect_matching(edges, .5, row_of_col, col_of_row, parent, visited, queue)
            else:
                matched = _block_match(within, allocated, starts, ends)
            if matched:
                result[q] = u + 1
                break
    return result


@njit(cache=True)
def endpoint_rows_covered(P, Q, threshold):
    squared = threshold * threshold
    for i in range(len(Q)):
        matched = False
        for j in range(len(P)):
            start, end = 0.0, 0.0
            for d in range(Q.shape[2]):
                a = P[j, 0, d] - Q[i, 0, d]
                b = P[j, -1, d] - Q[i, -1, d]
                start += a * a
                end += b * b
            if max(start, end) <= squared:
                matched = True
                break
        if not matched:
            return False
    return True


@njit(cache=True)
def verify_chunk(X, Q, parents, best, best_id):
    checked = 0
    for p in parents:
        if endpoint_rows_covered(X[p], Q, best) and core.endpoint_lb(X[p], Q) <= best:
            value = exact_distance(X[p], Q)
            checked += 1
            if value < best:
                best, best_id = value, p
    return best, best_id, checked


@njit(cache=True, parallel=True)
def ground_truth(X, Qs, radius):
    ids, distances = np.empty(len(Qs), np.int64), np.empty(len(Qs))
    relevant = np.zeros(len(Qs), np.int64)
    for q in prange(len(Qs)):
        best, best_id = np.inf, -1
        for p in range(len(X)):
            bound = max(best, radius)
            if endpoint_rows_covered(X[p], Qs[q], bound) and core.endpoint_lb(X[p], Qs[q]) <= bound:
                value = exact_distance(X[p], Qs[q])
                relevant[q] += int(value <= radius)
                if value < best:
                    best, best_id = value, p
        ids[q], distances[q] = best_id, best
    return ids, distances, relevant


@njit(cache=True)
def intersect_postings(ids, lo, hi, k, marks, stamp, counts, touched, nt):
    for t in range(lo, hi):
        p = np.int64(ids[t]) // k
        if marks[p] != stamp:
            marks[p] = stamp
            if counts[p] == 0:
                touched[nt] = p
                nt += 1
            counts[p] += 1
    return nt


@njit(cache=True)
def finish_intersection(counts, touched, nt, k, candidates):
    nc = 0
    for i in range(nt):
        p = touched[i]
        if counts[p] == k:
            candidates[nc] = p
            nc += 1
        counts[p] = 0
    return nc


@njit(cache=True)
def union_postings(ids, lo, hi, k, marks, stamp, candidates, nc):
    for t in range(lo, hi):
        p = np.int64(ids[t]) // k
        if marks[p] != stamp:
            marks[p] = stamp
            candidates[nc] = p
            nc += 1
    return nc


def verify(X, Q, parents, config, started):
    best, best_id, checked = np.inf, -1, 0
    for start in range(0, len(parents), config["verification_chunk"]):
        if time.perf_counter() - started > config["query_deadline_seconds"]:
            return -1, np.inf, checked, True
        best, best_id, count = verify_chunk(
            X, Q, parents[start:start + config["verification_chunk"]], best, best_id)
        checked += count
    timeout = time.perf_counter() - started > config["query_deadline_seconds"]
    return int(best_id) if not timeout else -1, float(best), checked, timeout


def joint_query(index, X, Q, config):
    started = time.perf_counter()
    T = index.tables
    nc, probes = 0, 0
    stamp = index._next_stamp()
    for u in range(index.L):
        for start in range(0, len(index.probed), config["probe_chunk"]):
            _, count, nc = _probe_orders(
                Q, index.shifts, index.delta, index.probed[start:start + config["probe_chunk"]],
                index.need_q, T.keys, T.ids, T.dir, T.N, T.B, u, u + 1,
                False, index.mark, stamp, index.cand, nc)
            probes += count
            if time.perf_counter() - started > config["query_deadline_seconds"]:
                return dict(id=-1, distance=np.inf, timeout=True, candidates=nc, verified=0,
                            probes=probes, postings=None, seconds=time.perf_counter() - started)
    p, value, checked, timeout = verify(X, Q, index.cand[:nc], config, started)
    return dict(id=p, distance=value, timeout=timeout, candidates=nc, verified=checked,
                probes=probes, postings=None, seconds=time.perf_counter() - started)


class PerCurve:
    def __init__(self, X, delta, L, seed):
        started = time.perf_counter()
        self.k, self.n, self.delta, self.L = X.shape[1], len(X), delta, L
        self.shifts = np.random.default_rng(seed).uniform(0, delta, (L, X.shape[3]))
        self.tables = T = Tables(L, self.n * self.k)
        for u in range(L):
            keys = core.curve_fps_all(X, self.shifts[u], delta).ravel()
            fill_table(keys, 1, T.B, T.keys[u*T.N:(u+1)*T.N], T.ids[u*T.N:(u+1)*T.N],
                       T.dir[u*(T.B+1):(u+1)*(T.B+1)])
        self.marks = np.zeros(self.n, np.int32)
        self.counts = np.zeros(self.n, np.int16)
        self.touched = np.empty(self.n, np.int64)
        self.candidates = np.empty(self.n, np.int64)
        self.stamp = 0
        self.build_seconds = time.perf_counter() - started

    @property
    def index_bytes(self):
        return self.tables.nbytes + self.shifts.nbytes

    def query(self, X, Q, config, rarest=False, active_L=None):
        started = time.perf_counter()
        L = self.L if active_L is None else active_L
        self.stamp += self.k + 1
        if self.stamp > 2_000_000_000:
            self.marks.fill(0)
            self.stamp = self.k + 1
        T = self.tables
        ranges = posting_ranges(Q, self.shifts, self.delta, T.keys, T.dir, T.N, T.B, L)
        chosen = int(np.argmin((ranges[:, :, 1] - ranges[:, :, 0]).sum(axis=1)))
        nt, nc, postings = 0, 0, 0
        for i in ([chosen] if rarest else range(self.k)):
            for u in range(L):
                lo, hi = ranges[i, u]
                for start in range(lo, hi, config["probe_chunk"]):
                    end = min(hi, start + config["probe_chunk"])
                    if rarest:
                        nc = union_postings(T.ids, start, end, self.k, self.marks, self.stamp,
                                            self.candidates, nc)
                    else:
                        nt = intersect_postings(T.ids, start, end, self.k, self.marks, self.stamp+i,
                                                self.counts, self.touched, nt)
                    postings += end - start
                    if time.perf_counter() - started > config["query_deadline_seconds"]:
                        if not rarest:
                            finish_intersection(self.counts, self.touched, nt, self.k, self.candidates)
                        return dict(id=-1, distance=np.inf, timeout=True, candidates=0, verified=0,
                                    probes=self.k*L, postings=postings, seconds=time.perf_counter()-started)
        if not rarest:
            nc = finish_intersection(self.counts, self.touched, nt, self.k, self.candidates)
        p, value, checked, timeout = verify(X, Q, self.candidates[:nc], config, started)
        return dict(id=p, distance=value, timeout=timeout, candidates=nc, verified=checked,
                    probes=self.k*L, postings=postings, seconds=time.perf_counter()-started)


@njit(cache=True)
def posting_ranges(Q, shifts, delta, keys, directory, N, B, L):
    ranges = np.empty((len(Q), L, 2), np.int64)
    for i in range(len(Q)):
        for u in range(L):
            fp = core.curve_fp(Q[i], shifts[u], delta)
            j = u * (B + 1) + _bucket(fp, B)
            lo, hi = u * N + np.int64(directory[j]), u * N + np.int64(directory[j+1])
            a = _lower(keys, lo, hi, fp)
            ranges[i, u, 0] = a
            ranges[i, u, 1] = _upper(keys, a, hi, fp)
    return ranges


def dataset(config, case):
    k, n = case["k"], case["n"]
    kind_id = {"conjunctive": 1, "rare": 2, "repeated": 3}[case["kind"]]
    seed = config["dataset_seed"] + 100*k + kind_id
    count = config["tuning_queries"] + config["test_queries"] + config["warmup_queries"]
    P = case.get("paths")
    if P is None:
        P = int(round(1 / (1 - (1-case["frequency"])**(1/k))))
    rng = np.random.default_rng(seed)
    paths = data.random_walks(rng, P, config["m"], config["d"], 1000, 3)
    pid = np.random.default_rng(seed+1).integers(0, P, (n, k), dtype=np.uint16)
    q_rng = np.random.default_rng(seed+2)
    motifs = np.empty((count, k), np.uint16)
    for q in range(count):
        motifs[q] = q_rng.choice(P, k, replace=case["kind"] == "repeated")
    Q = data.jitter(q_rng, paths[motifs], config["rho"])
    positions = np.random.default_rng(seed+3).choice(10000, 3*count, replace=False)
    for q in range(count):
        for p in positions[3*q:3*q+3]:
            pid[p] = motifs[q, q_rng.permutation(k)]
        Q[q] = Q[q, q_rng.permutation(k)]
    implicit = Implicit(paths, pid, seed+4, config["rho"])
    X = implicit.materialize(np.arange(n))
    return X, Q, dict(paths=P, nominal_path_frequency=1-(1-1/P)**k,
                      background_n=n, planted_sets=len(positions), dataset_seed=seed,
                      dataset_sha256=arrays_hash(X), queries_sha256=arrays_hash(Q))


def timed_queries(query, Qs, optimum, config, progress=None):
    results = []
    with pinned():
        for q, Q in enumerate(Qs):
            row = query(Q)
            found = row["id"] >= 0
            row["found"] = found
            row["exact_nn"] = bool(found and row["distance"] <= optimum[q] + 1e-8)
            if not found:
                row["ratio"] = None
            elif optimum[q] > 0:
                row["ratio"] = float(row["distance"]/optimum[q])
            else:
                row["ratio"] = 1.0 if row["distance"] == 0 else float("inf")
            if not found:
                row["distance"] = None
            results.append(row)
            if progress is not None and (q+1) % 100 == 0:
                progress(q+1)
            if row["timeout"]:
                return results, False
    return results, True


def warm(query, Qs):
    with pinned():
        for Q in Qs:
            result = query(Q)
            if result["timeout"]:
                raise RuntimeError("Warm-up exceeded query deadline")


def approximation_summary(ratios):
    if not ratios:
        return dict(mean_ratio=None, p90_ratio=None, returned_ratio_count=0, unbounded_ratio_count=0)
    values = np.asarray(ratios)
    unbounded = int(np.isinf(values).sum())
    upper_rank = int(np.ceil(.9*(len(values)-1)))
    tail_unbounded = bool(np.isinf(np.sort(values)[upper_rank]))
    return dict(mean_ratio="infinity" if unbounded else float(values.mean()),
                p90_ratio="infinity" if tail_unbounded else float(np.quantile(values,.9)),
                returned_ratio_count=len(ratios), unbounded_ratio_count=unbounded)


def worker(config, case, output):
    started = time.perf_counter()
    X, Qs, info = dataset(config, case)
    write_json(output / "active_configuration.json", dict(method="ground-truth", phase="preparation"))
    ids, distances, relevant = ground_truth(X, Qs, config["relevance_radius"])
    h, s = config["tuning_queries"], config["test_queries"]
    Qt, Qtest, Qwarm = Qs[:h], Qs[h:h+s], Qs[h+s:]
    NN = X[ids[:h]]
    np.savez(output / "ground-truth.npz", ids=ids, distances=distances, relevant=relevant, queries=Qs)
    write_json(output / "dataset.json", dict(info, **case, raw_curve_bytes=X.nbytes,
                                            preparation_seconds=time.perf_counter()-started,
                                            mean_test_nn_distance=float(distances[h:h+s].mean()),
                                            mean_test_relevant_sets=float(relevant[h:h+s].mean())))
    manifest = json.loads((output.parent / "manifest.json").read_text())
    model = CostModel.load(output.parent / "calibration.json")
    k, d, seed = case["k"], config["d"], config["index_seed"]
    configs, choices = [], {}

    def record(method, delta, L, index_memory, elapsed, rows, status, sizes=None, swapped=False, build=None):
        result = dict(method=method, delta=delta, L=L, index_bytes=index_memory,
                      blocks=list(sizes) if sizes else None, swapped=swapped, build_seconds=build,
                      status=status, tuning_recall=sum(r["exact_nn"] for r in rows)/len(rows) if rows else None,
                      tuning_median_seconds=float(np.median([r["seconds"] for r in rows])) if rows else None,
                      tuning_seconds=elapsed)
        configs.append(result)
        write_json(output / "tuning-configurations.json", configs)
        if status == "qualified":
            old = choices.get(method)
            if old is None or result["tuning_median_seconds"] < old["tuning_median_seconds"]:
                choices[method] = result
        return result

    for delta in config["deltas"]:
        shifts = np.random.default_rng(seed).uniform(0, delta, (config["Lmax"], k, d))
        cache = {}

        def hits(sizes, swapped):
            key = (tuple(sizes), swapped)
            if key not in cache:
                cache[key] = family_first_hits(Qt, NN, shifts, delta, np.array(sizes), swapped)
            return cache[key]

        specs = [("probe_all", (1,)*k, False), ("store_all", (k,), False)]
        _, _, sizes, swapped = balanced_blocking(k)
        specs.append(("blocking-balanced", sizes, swapped))
        auto = None
        for L in range(1, config["Lmax"]+1):
            selected = selected_blocking(model, config, len(X), k, d, L)
            if selected is not None:
                _, a, b, sizes, swapped = selected
                if (hits(sizes, swapped) <= L).mean() >= config["target_recall"]:
                    auto = (L, sizes, swapped)
                    break
        if auto is not None:
            specs.append(("blocking-cost", auto[1], auto[2]))
        else:
            record("blocking-cost", delta, None, None, 0, [], "no tuned configuration within pilot Lmax")
        for method, sizes, swapped in specs:
            L = auto[0] if method == "blocking-cost" else tables_for_recall(
                hits(sizes, swapped), config["target_recall"], config["Lmax"])
            if L is None:
                record(method, delta, None, None, 0, [], "no tuned configuration within pilot Lmax", sizes, swapped)
                continue
            from math import factorial, prod
            a, b = prod(factorial(x) for x in sizes), factorial(k)//prod(factorial(x) for x in sizes)
            if swapped:
                a, b = b, a
            memory = resident_bytes(len(X), a, b, k, d, L)
            if memory > config["index_budget_bytes"]:
                record(method, delta, L, memory, 0, [], "over index memory (computed)", sizes, swapped)
                continue
            write_json(output / "active_configuration.json", dict(method=method, phase="building", delta=delta, L=L))
            signal.alarm(config["build_deadline_seconds"])
            try:
                build_started = time.perf_counter()
                index = build_index(make_index("blocking", k, d, delta, L, sizes=sizes, swapped=swapped, seed=seed), X)
                index.build_seconds = time.perf_counter()-build_started
            finally:
                signal.alarm(0)
            query = lambda Q: joint_query(index, X, Q, config)
            warm(query, Qwarm)
            write_json(output / "active_configuration.json", dict(method=method, phase="tuning", delta=delta, L=L))
            rows, complete = timed_queries(query, Qt, distances[:h], config)
            recall = sum(r["exact_nn"] for r in rows)/len(rows)
            status = "qualified" if complete and recall >= config["target_recall"] else "tuning target not met" if complete else "over query time"
            record(method, delta, L, memory, sum(r["seconds"] for r in rows), rows, status, sizes, swapped, index.build_seconds)
            del index

        sp = np.random.default_rng(seed).uniform(0, delta, (config["Lmax"], d))
        L = tables_for_recall(first_hit_per_curve(Qt, NN, sp, delta), config["target_recall"], config["Lmax"])
        if L is None:
            for method in ("per_curve-intersection", "per_curve-rarest"):
                record(method, delta, None, None, 0, [], "no tuned configuration within pilot Lmax")
            continue
        memory = index_bytes(len(X)*k, L)+L*d*8
        if memory > config["index_budget_bytes"]:
            for method in ("per_curve-intersection", "per_curve-rarest"):
                record(method, delta, L, memory, 0, [], "over index memory (computed)")
            continue
        write_json(output / "active_configuration.json", dict(method="per_curve", phase="building", delta=delta, L=L))
        signal.alarm(config["build_deadline_seconds"])
        try:
            build_started = time.perf_counter()
            index = PerCurve(X, delta, L, seed)
        finally:
            signal.alarm(0)
        for method, rarest in (("per_curve-intersection", False), ("per_curve-rarest", True)):
            limits = range(1, L+1) if rarest else (L,)
            for active in limits:
                query = lambda Q: index.query(X, Q, config, rarest, active)
                warm(query, Qwarm)
                write_json(output / "active_configuration.json", dict(method=method, phase="tuning", delta=delta, L=active))
                rows, complete = timed_queries(query, Qt, distances[:h], config)
                recall = sum(r["exact_nn"] for r in rows)/len(rows)
                status = "qualified" if complete and recall >= config["target_recall"] else "tuning target not met" if complete else "over query time"
                record(method, delta, active, index_bytes(len(X)*k, active)+active*d*8,
                       sum(r["seconds"] for r in rows), rows, status, build=index.build_seconds)
                if status == "qualified" or not complete:
                    break
        del index

    write_json(output / "frozen-selections.json", choices)
    final = []
    for method, chosen in choices.items():
        L, delta = chosen["L"], chosen["delta"]
        write_json(output / "active_configuration.json", dict(method=method, phase="building-final", delta=delta, L=L))
        signal.alarm(config["build_deadline_seconds"])
        try:
            build_started = time.perf_counter()
            if method.startswith("per_curve"):
                index = PerCurve(X, delta, L, seed)
                query = lambda Q: index.query(X, Q, config, method == "per_curve-rarest")
                actual = index.index_bytes
            else:
                index = build_index(make_index("blocking", k, d, delta, L, sizes=chosen["blocks"],
                                               swapped=chosen["swapped"], seed=seed), X)
                query = lambda Q: joint_query(index, X, Q, config)
                actual = index.index_bytes+index.stored.nbytes+index.probed.nbytes+index.shifts.nbytes+index.need_s.nbytes+index.need_q.nbytes
                index.build_seconds = time.perf_counter()-build_started
        finally:
            signal.alarm(0)
        if actual != chosen["index_bytes"]:
            raise AssertionError("Final resident memory disagrees with selected configuration")
        warm(query, Qwarm)
        write_json(output / "active_configuration.json", dict(method=method, phase="test", delta=delta, L=L))
        rows, complete = timed_queries(query, Qtest, distances[h:h+s], config)
        for q, row in enumerate(rows):
            append_row(output / "queries.csv", dict(
                commit=manifest["commit"], config_sha256=manifest["config_sha256"],
                case=case["id"], dataset_sha256=info["dataset_sha256"], phase="test", method=method,
                delta=delta, L=L, index_bytes=actual, query=q, nn_distance=float(distances[h+q]), **row))
        if complete:
            times = [r["seconds"] for r in rows]
            ratios = [r["ratio"] for r in rows if r["ratio"] is not None]
            final.append(dict(chosen, build_seconds=index.build_seconds, test_status="complete", test_queries=len(rows),
                              test_recall=sum(r["exact_nn"] for r in rows)/len(rows),
                              median_seconds=float(np.median(times)), p95_seconds=float(np.quantile(times,.95)),
                              **approximation_summary(ratios),
                              empty_rate=sum(not r["found"] for r in rows)/len(rows)))
        else:
            final.append(dict(chosen, test_status="over query time; no partial summary"))
        write_json(output / "results.json", final)
        del index
    all_parents = np.arange(len(X), dtype=np.int64)

    def brute(Q):
        started = time.perf_counter()
        p, distance, checked, timeout = verify(X, Q, all_parents, config, started)
        return dict(id=p, distance=distance, timeout=timeout, candidates=len(X), verified=checked,
                    probes=0, postings=0, seconds=time.perf_counter()-started)

    warm(brute, Qwarm)
    rows, complete = timed_queries(brute, Qtest, distances[h:h+s], config)
    for q, row in enumerate(rows):
        append_row(output / "queries.csv", dict(
            commit=manifest["commit"], config_sha256=manifest["config_sha256"], case=case["id"],
            dataset_sha256=info["dataset_sha256"], phase="test", method="brute-force",
            delta=None, L=None, index_bytes=0, query=q, nn_distance=float(distances[h+q]), **row))
    if not complete or any(not r["exact_nn"] for r in rows):
        raise AssertionError("Brute-force reference failed")
    final.append(dict(method="brute-force", index_bytes=0, test_status="complete", test_queries=len(rows),
                      test_recall=1, median_seconds=float(np.median([r["seconds"] for r in rows])),
                      p95_seconds=float(np.quantile([r["seconds"] for r in rows],.95)),
                      mean_ratio=1, p90_ratio=1, empty_rate=0))
    write_json(output / "results.json", final)
    print(f"CASE {case['id']} complete", flush=True)


def main(args):
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    output = Path(args.output).resolve()
    if args.worker is not None:
        worker(config, config["cases"][args.worker], output)
        return
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise RuntimeError("Run from a clean frozen code commit")
    output.mkdir(parents=True, exist_ok=False)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    import numba
    write_json(output / "config.json", config)
    write_json(output / "manifest.json", dict(commit=commit, config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
                                             started=time.time(), stage=config["stage"], python=sys.version,
                                             numpy=np.__version__, numba=numba.__version__,
                                             affinity=sorted(os.sched_getaffinity(0)), pin_core=config["pin_core"],
                                             numba_threads=numba.get_num_threads(),
                                             cpu_model=next(line.split(":",1)[1].strip() for line in
                                                            Path("/proc/cpuinfo").read_text().splitlines()
                                                            if line.startswith("model name"))))
    (output / "calibration.json").write_bytes(Path(config["calibration_file"]).read_bytes())
    started, results = time.monotonic(), []
    for i, case in enumerate(config["cases"]):
        folder = output / case["id"]
        folder.mkdir()
        remaining = config["run_cap_seconds"]-(time.monotonic()-started)
        if remaining <= 0:
            results.append(dict(case=case["id"], status="not run: pilot run cap"))
            continue
        with (folder / "worker.log").open("w") as log:
            process = subprocess.Popen([sys.executable, "-m", "fresh.regimes", "--config", str(config_path),
                                        "--output", str(folder), "--worker", str(i)], stdout=log, stderr=subprocess.STDOUT)
            try:
                code, interruption = monitor_worker(process, folder, config, remaining)
            finally:
                terminate_worker(process)
        status = interruption or ("complete" if code == 0 else "build deadline" if code == -signal.SIGALRM else "implementation failure")
        results.append(dict(case=case["id"], status=status, returncode=code))
        write_json(output / "cases.json", results)
        print(f"{case['id']}: {status}", flush=True)
        if code or interruption:
            raise RuntimeError(f"Pilot stopped: {status}; inspect {folder/'worker.log'}")
    write_json(output / "completion.json", dict(completed=time.time(), elapsed_seconds=time.monotonic()-started))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--worker", type=int)
    main(parser.parse_args())
