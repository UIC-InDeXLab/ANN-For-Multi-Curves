"""Object-count feasibility of the first-hit indexes (one entry per key, first collision returned)."""
import argparse
from collections import defaultdict
from math import factorial
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numba
import numpy as np

from curveset import core
from curveset.bench import pinned
from curveset.index import make_index
from fresh.e1 import append_row, monitor_worker, terminate_worker, write_json
from fresh.experiment_first_entry import (first_hit_in_table, keep_first_entries, load_reference_dataset,
                                          percentile_ms, require_clean_tree, resolve_path, worker_state)
from fresh.nba_full import digest
from fresh.poc import Joint, compact_table, family_specs, observed_peak_gate
from fresh.regimes import verify

METHODS = ("blocking", "probe_all", "store_all")


def load_protocol(path):
    config_path = Path(path).resolve()
    config = json.loads(config_path.read_text())
    config["config_path"] = str(config_path)
    config["reference_results_root"] = str(resolve_path(config_path.parent, config["reference_results_root"]))
    return config


def study_jobs(config):
    return [dict(case, case_id=case["id"], index_seed=seed, id=f"{case['id']}-seed{seed}")
            for case in config["cases"] for seed in config["index_seeds"]]


def index_bytes(index, L):
    base = index.base
    return (sum(t.nbytes for t in index.tables[:L]) + base.stored.nbytes + base.probed.nbytes + base.shifts.nbytes +
            base.need_s.nbytes + base.need_q.nbytes)


def build_tables(X, delta, L_max, seed, sizes, swapped, byte_cap):
    """Build up to L_max tables with one entry per key, stopping before the index exceeds byte_cap."""
    index = Joint.__new__(Joint)
    index.base = make_index("blocking", X.shape[1], X.shape[3], delta, L_max, sizes=sizes, swapped=swapped, seed=seed)
    index.tables = []
    started = time.perf_counter()
    for u in range(L_max):
        fingerprints = core.slot_fps(X, index.base.shifts[u], delta, index.base.need_s)
        keys = core.order_keys(fingerprints, index.base.stored).ravel()
        index.tables.append(keep_first_entries(compact_table(keys, len(index.base.stored))))
        del fingerprints, keys
        if index_bytes(index, u + 1) > byte_cap:
            index.tables.pop()
            break
    index.base._buffers(len(X))
    index.build_seconds = time.perf_counter() - started
    return index


def first_hit_timed(index, X, Q, L, deadline, config):
    """First hit within the deadline; the clock stops before the distance of the answer is computed."""
    started = time.perf_counter()
    base = index.base
    answer, timeout = -1, False
    for u in range(L):
        table = index.tables[u]
        for start in range(0, len(base.probed), config["probe_chunk"]):
            family = base.probed[start:start + config["probe_chunk"]]
            found, _ = first_hit_in_table(Q, base.shifts[u], base.delta, family, base.need_q,
                                          table.keys, table.ids, table.dir, table.B)
            if found >= 0:
                answer = int(found)
                break
            if time.perf_counter() - started > deadline:
                timeout = True
                break
        if answer >= 0 or timeout:
            break
    seconds = time.perf_counter() - started
    timeout = timeout or seconds > deadline
    distance = np.inf
    if answer >= 0 and not timeout:
        distance = float(verify(X, Q, np.array([answer], np.int64), config, time.perf_counter())[1])
    return dict(id=answer if not timeout else -1, distance=distance, timeout=timeout, seconds=seconds)


def run_queries(index, X, L, queries, optimum, deadline, config, need=None):
    rows, successes = [], 0
    with pinned():
        for q, curve in enumerate(queries):
            result = first_hit_timed(index, X, curve, L, deadline, config)
            found = result["id"] >= 0 and not result["timeout"]
            ok = found and result["distance"] <= optimum[q] + 1e-8
            successes += int(ok)
            distance = float(result["distance"]) if found else None
            ratio = (distance / optimum[q] if optimum[q] > 0 else 1.0 if distance == 0 else "infinity") if found else None
            rows.append(dict(seconds=result["seconds"], ok=bool(ok), timeout=bool(result["timeout"]),
                             id=int(result["id"]), found=found, distance=distance,
                             nn_distance=float(optimum[q]), ratio=ratio,
                             within_110=bool(found and distance <= 1.1 * optimum[q] + 1e-8),
                             within_125=bool(found and distance <= 1.25 * optimum[q] + 1e-8)))
            if need is not None and successes + len(queries) - len(rows) < need:
                return rows, False
    return rows, True


def preflight(X, case, a, b, budget, config):
    n, k = len(X), case["k"]
    if n >= 2**31 or n * a >= 2**32:
        return "excluded: integer capacity"
    if (a + b) * k > budget["index_budget_bytes"]:
        return "excluded: mandatory order arrays exceed serving-index budget"
    reason = observed_peak_gate(X, n, k, a, 1, config)
    return "excluded: construction RAM estimate" if reason else None


def method_of(a, k):
    return [m for m, hit in (("probe_all", a == 1), ("store_all", a == factorial(k)), ("blocking", True)) if hit]


def worker(config, case, output):
    manifest = json.loads((output.parent / "manifest.json").read_text())
    worker_state(output, method="ground-truth", seed=case["index_seed"], phase="preparation")
    X, Q, info, ids, distances, _ = load_reference_dataset(config, case)
    h, s = config["tuning_queries"], config["test_queries"]
    Qt, Qtest, Qwarm = Q[:h], Q[h:h + s], Q[h + s:]
    opt_t, opt_s = distances[:h], distances[h:h + s]
    seed, k = case["index_seed"], case["k"]
    budgets = config["budgets"]
    max_cap = max(b["index_budget_bytes"] for b in budgets)
    need = int(np.ceil(config["tuning_acceptance_recall"] * h))
    relaxed = dict(config, query_deadline_seconds=config["compilation_deadline_seconds"])
    candidates, log = [], []
    for a, b, sizes, swapped in family_specs(k, config):
        fits = [bud for bud in budgets if preflight(X, case, a, b, bud, config) is None]
        for bud in budgets:
            reason = preflight(X, case, a, b, bud, config)
            if reason:
                log.append(dict(budget=bud["name"], a=a, b=b, blocks=list(sizes), status=reason))
        if not fits:
            continue
        cap = max(bud["index_budget_bytes"] for bud in fits)
        for delta in config["deltas"]:
            worker_state(output, method=f"a{a}", seed=seed, phase="building", delta=delta)
            signal.alarm(int(config["build_deadline_seconds"]))
            try:
                index = build_tables(X, delta, config["L_max"], seed, sizes, swapped, cap)
            finally:
                signal.alarm(0)
            for q in Qwarm:
                first_hit_timed(index, X, q, 1, config["compilation_deadline_seconds"], relaxed)
            for bud in fits:
                limit = max((L for L in range(len(index.tables) + 1) if L == 0 or index_bytes(index, L) <= bud["index_budget_bytes"]), default=0)
                chosen = None
                for L in [v for v in config["L_grid"] if 1 <= v <= limit]:
                    worker_state(output, method=f"a{a}", seed=seed, phase="tuning", delta=delta, L=L, budget=bud["name"])
                    rows, complete = run_queries(index, X, L, Qt, opt_t, bud["query_deadline_seconds"], relaxed, need)
                    if complete and sum(r["ok"] for r in rows) >= need:
                        chosen = dict(budget=bud["name"], a=a, b=b, blocks=list(sizes), swapped=bool(swapped),
                                      delta=float(delta), L=L, bytes=int(index_bytes(index, L)),
                                      tuning_success=sum(r["ok"] for r in rows) / h,
                                      tuning_median=float(np.median([r["seconds"] for r in rows])))
                        break
                if chosen is None:
                    log.append(dict(budget=bud["name"], a=a, b=b, blocks=list(sizes), delta=float(delta),
                                    status="rejected: on-time tuning recall" if limit else "excluded: measured serving-index storage"))
                else:
                    candidates.append(chosen)
            del index
    write_json(output / "tuning-configurations.json", dict(candidates=candidates, excluded=log))
    results = []
    for bud in budgets:
        for method in METHODS:
            options = [c for c in candidates if c["budget"] == bud["name"] and method in method_of(c["a"], k)]
            if not options:
                row = dict(budget=bud["name"], display_method=method, k=k, index_seed=seed, case_id=case["case_id"],
                           status="no qualified configuration", feasible=False)
                results.append(row)
                continue
            pick = min(options, key=lambda c: (c["tuning_median"], c["bytes"], c["delta"]))
            worker_state(output, method=method, seed=seed, phase="test", budget=bud["name"])
            signal.alarm(int(config["build_deadline_seconds"]))
            try:
                index = build_tables(X, pick["delta"], pick["L"], seed, tuple(pick["blocks"]), pick["swapped"], float("inf"))
            finally:
                signal.alarm(0)
            for q in Qwarm:
                first_hit_timed(index, X, q, 1, config["compilation_deadline_seconds"], relaxed)
            rows, _ = run_queries(index, X, pick["L"], Qtest, opt_s, bud["query_deadline_seconds"], relaxed)
            for q, r in enumerate(rows):
                append_row(output / "test-queries.csv", dict(budget=bud["name"], method=method, query=q, **r))
            seconds = [r["seconds"] for r in rows]
            memory = index_bytes(index, len(index.tables))
            success = float(np.mean([r["ok"] for r in rows]))
            ratios = [r["ratio"] for r in rows if r["found"]]
            finite = all(r != "infinity" for r in ratios)
            results.append(dict(budget=bud["name"], display_method=method, k=k, index_seed=seed, case_id=case["case_id"],
                                status="complete", blocks=pick["blocks"], swapped=pick["swapped"], delta=pick["delta"],
                                L=pick["L"], index_bytes=int(memory), index_mb=float(memory / 1e6),
                                tuning_success=pick["tuning_success"], success_rate=success,
                                timeout_rate=float(np.mean([r["timeout"] for r in rows])),
                                median_ms=percentile_ms(seconds, 0.5), p95_ms=percentile_ms(seconds, 0.95),
                                success_110=float(np.mean([r["within_110"] for r in rows])),
                                success_125=float(np.mean([r["within_125"] for r in rows])),
                                empty_rate=float(np.mean([not r["found"] for r in rows])),
                                mean_ratio=float(np.mean(ratios)) if ratios and finite else "infinity" if ratios else None,
                                feasible=bool(memory <= bud["index_budget_bytes"] and success >= config["required_success"]),
                                commit=manifest["commit"]))
            del index
    write_json(output / "main-results.json", results)
    write_json(output / "completion.json", dict(completed=time.time(), case=case["id"], rows=len(results)))


def summarize(output, config):
    rows = []
    for job in study_jobs(config):
        rows.extend(json.loads((output / job["id"] / "main-results.json").read_text()))
    groups = defaultdict(list)
    for r in rows:
        groups[(r["budget"], r["display_method"], r["k"])].append(r)
    seeds = len(config["index_seeds"])
    summary = []
    for (budget, method, k), group in sorted(groups.items()):
        ok = len(group) == seeds and all(g["feasible"] for g in group)
        entry = dict(budget=budget, method=method, k=k, feasible=ok, seeds=len(group))
        done = [g for g in group if g["status"] == "complete"]
        if done:
            entry.update(median_ms=float(np.median([g["median_ms"] for g in done])),
                         index_mb=float(np.median([g["index_mb"] for g in done])),
                         success_min=min(g["success_rate"] for g in done), success_mean=float(np.mean([g["success_rate"] for g in done])),
                         blocks=[g["blocks"] for g in done], tables=[g["L"] for g in done], deltas=[g["delta"] for g in done])
        summary.append(entry)
    largest = {}
    for budget in {s["budget"] for s in summary}:
        for method in METHODS:
            ks = [s["k"] for s in summary if s["budget"] == budget and s["method"] == method and s["feasible"]]
            largest[f"{budget}/{method}"] = max(ks) if ks else None
    write_json(output / "summary.json", dict(groups=summary, largest_feasible_tested_k=largest,
                                              ks=sorted({c["k"] for c in config["cases"]}),
                                              note="First-hit indexes: one entry per key, first collision returned; a count is feasible when every seed fits the budget and answers at least 90% of queries with an exact neighbor on time."))


def main(args):
    config = load_protocol(args.config)
    output = Path(args.output).resolve()
    config["output_root"] = str(output if args.worker is None else output.parent)
    if args.worker is not None:
        worker(config, study_jobs(config)[args.worker], output)
        return
    require_clean_tree()
    if numba.get_num_threads() != config["build_threads"] or int(os.environ.get("PIN_CORE", "-1")) != config["pin_core"]:
        raise ValueError("CPU settings differ from the frozen first-hit feasibility protocol")
    output.mkdir(parents=True, exist_ok=False)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    write_json(output / "config.json", config)
    write_json(output / "manifest.json", dict(commit=commit, config_sha256=digest(Path(args.config).resolve()),
                                              started=time.time(), python=sys.version, numpy=np.__version__,
                                              numba=numba.__version__, pin_core=config["pin_core"],
                                              build_threads=numba.get_num_threads()))
    started = time.monotonic()
    cases = []
    for i, case in enumerate(study_jobs(config)):
        remaining = config["run_cap_seconds"] - (time.monotonic() - started)
        if remaining <= 0:
            raise RuntimeError("First-hit feasibility study exceeded its run cap")
        folder = output / case["id"]
        folder.mkdir()
        with (folder / "worker.log").open("w") as log:
            process = subprocess.Popen([sys.executable, "-u", "-m", "fresh.feasibility_first_hit", "--config",
                                        str(Path(args.config).resolve()), "--output", str(folder), "--worker", str(i)],
                                       stdout=log, stderr=subprocess.STDOUT)
            try:
                code, interruption = monitor_worker(process, folder, config, remaining)
            finally:
                terminate_worker(process)
        status = interruption or ("complete" if code == 0 else "implementation or build failure")
        cases.append(dict(case=case["id"], index_seed=case["index_seed"], status=status, returncode=code))
        write_json(output / "cases.json", cases)
        print(f"{case['id']}: {status}", flush=True)
        if code or interruption:
            raise RuntimeError(f"Stopped at {folder}: {status}")
    summarize(output, config)
    write_json(output / "completion.json", dict(completed=time.time(), elapsed_seconds=time.monotonic() - started))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--worker", type=int)
    main(parser.parse_args())
