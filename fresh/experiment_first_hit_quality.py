"""Measure missing first-hit studies against frozen datasets and exact references.

The historical tuning rule is retained: smallest table count reaching the exact
neighbor target, otherwise best tuning hit rate. Test reporting uses distance
ratios, both approximation-success thresholds, tails, and empty-answer rates.
"""
import argparse
import csv
import gc
import json
from math import factorial, prod
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
from curveset.orders import partitions
from fresh.dtw import dtw_distance
from fresh.e1 import append_row, arrays_hash, monitor_worker, terminate_worker, write_json
from fresh.experiment_first_entry import (
    choose_first_entry_configuration, first_hit_in_table, require_clean_tree, worker_state,
)
from fresh.feasibility_first_hit import build_tables, index_bytes
from fresh.nba_full import digest
from fresh.poc import dataset, observed_peak_gate
from fresh.tuple_quality import exact_distance


def canonical_family(sizes, swapped=False):
    sizes = tuple(int(s) for s in sizes)
    if not sizes or min(sizes) < 1:
        raise ValueError("Block sizes must be positive")
    if swapped and len(sizes) == 1:
        return (1,) * sum(sizes), False
    if swapped and len(sizes) == sum(sizes):
        return (sum(sizes),), False
    return sizes, bool(swapped)


def families(job, reference):
    k = job["case"]["k"]
    requested = tuple(job.get("methods", ("blocking", "probe_all", "store_all")))
    if not requested or any(method not in ("blocking", "probe_all", "store_all") for method in requested):
        raise ValueError("Requested first-hit methods must be explicit joint-key methods")
    selected = json.loads((reference / "frozen-main-selections.json").read_text())
    weighted_path = reference / "frozen-weight-selections.json"
    weighted = json.loads(weighted_path.read_text()) if job.get("include_weights") and weighted_path.is_file() else {}
    out = {}

    def add(sizes, swapped=False, label=None):
        key = canonical_family(sizes, swapped)
        if sum(key[0]) != k:
            raise ValueError("The stored blocks do not partition the curve slots")
        out.setdefault(key, [])
        if label is not None and label not in out[key]:
            out[key].append(label)

    for method in requested:
        if method in selected:
            row = selected[method]
            add(row["blocks"], row["swapped"], method)
    for weight, row in weighted.items():
        add(row["blocks"], row["swapped"], f"lambda={weight}")
    if "store_all" in requested:
        add((k,), label="store_all")
    if "probe_all" in requested:
        add((1,) * k, label="probe_all")
    if job.get("frontier"):
        for sizes in partitions(k):
            for swapped in (False, True):
                add(sizes, swapped)
    return [(sizes, swapped, labels) for (sizes, swapped), labels in out.items()]


def load_reference(job):
    root = Path(job["reference_root"])
    config = json.loads((root / "config.json").read_text())
    case = job["case"]
    reference = root / job["reference_job"]
    info = json.loads((reference / "dataset.json").read_text())
    if case["kind"] == "nba":
        data = Path(job["nba_data"])
        with np.load(data / "database.npz", allow_pickle=False) as arrays:
            X = arrays["curves"]
        with np.load(data / "tune.npz", allow_pickle=False) as arrays:
            Qt = arrays["curves"]
        with np.load(data / "test.npz", allow_pickle=False) as arrays:
            Qs = arrays["curves"]
        Q = np.concatenate((Qt, Qs, X[:config["warmup_queries"]]))
        if arrays_hash(X) != info["dataset_sha256"] or arrays_hash(Q) != info["queries_sha256"]:
            raise ValueError("NBA arrays differ from the frozen metric-specific evaluation")
    else:
        X, Q, generated = dataset(config, case)
        if any(generated[k] != info[k] for k in ("dataset_sha256", "queries_sha256")):
            raise ValueError("Synthetic arrays differ from the frozen evaluation")
    truth = reference / "ground-truth.npz"
    with np.load(truth, allow_pickle=False) as arrays:
        if not np.array_equal(arrays["queries"], Q):
            raise ValueError("Reference query order changed")
        optimum = arrays["distances"].copy()
    if len(optimum) != len(Q) or not np.all(np.isfinite(optimum)) or np.any(optimum < 0):
        raise ValueError("Every query requires a finite nonnegative exact optimum")
    return config, reference, X, Q, optimum, info, digest(truth)


def timed_hit(index, Q, L, config):
    started = time.perf_counter()
    answer, probes, table_number = -1, 0, L + 1
    for u in range(L):
        table, base = index.tables[u], index.base
        for start in range(0, len(base.probed), config["probe_chunk"]):
            found, read = first_hit_in_table(
                Q, base.shifts[u], base.delta, base.probed[start:start + config["probe_chunk"]],
                base.need_q, table.keys, table.ids, table.dir, table.B)
            probes += int(read)
            if found >= 0:
                answer, table_number = int(found), u + 1
                break
            if time.perf_counter() - started > config["query_deadline_seconds"]:
                return dict(id=-1, timeout=True, probes=probes,
                            seconds=time.perf_counter() - started, first_table=L + 1)
        if answer >= 0:
            break
    seconds = time.perf_counter() - started
    timeout = seconds > config["query_deadline_seconds"]
    return dict(id=-1 if timeout else answer, timeout=timeout, probes=probes,
                seconds=seconds, first_table=L + 1 if timeout else table_number)


def distance_ratio(distance, optimum):
    if optimum > 0:
        return float(distance / optimum)
    return 1.0 if distance == 0 else "infinity"


def evaluate(index, X, queries, optimum, L, config, metric, progress=None, hit_function=None):
    distance = dtw_distance if metric == "dtw" else exact_distance
    query_hit = timed_hit if hit_function is None else hit_function
    rows = []
    with pinned():
        for q, curve in enumerate(queries):
            row = query_hit(index, curve, L, config)
            found = row["id"] >= 0 and not row["timeout"]
            value = float(distance(X[row["id"]], curve)) if found else None
            row.update(query=q, found=found, distance=value, nn_distance=float(optimum[q]),
                       ratio=distance_ratio(value, optimum[q]) if found else None,
                       exact_nn=bool(found and value <= optimum[q] + 1e-8),
                       within_110=bool(found and value <= 1.1 * optimum[q] + 1e-8),
                       within_125=bool(found and value <= 1.25 * optimum[q] + 1e-8))
            if found and np.isfinite(optimum[q]) and value < optimum[q] - 1e-7:
                raise AssertionError("Returned distance is smaller than the exact reference")
            rows.append(row)
            if progress is not None and (q + 1) % 100 == 0:
                progress(q + 1)
    return rows


def quality_summary(rows):
    if not rows:
        raise ValueError("Cannot summarize an empty evaluation")
    returned = [r for r in rows if r["found"] and not r["timeout"]]
    ratios = [float("inf") if r["ratio"] == "infinity" else r["ratio"] for r in returned]
    finite = all(np.isfinite(r) for r in ratios)
    result = dict(
        queries=len(rows), returned=len(returned), empty_rate=1 - len(returned) / len(rows),
        timeout_rate=float(np.mean([r["timeout"] for r in rows])),
        success_110=float(np.mean([r["within_110"] for r in rows])),
        success_125=float(np.mean([r["within_125"] for r in rows])),
        exact_rate=float(np.mean([r["exact_nn"] for r in rows])),
        median_ms=float(np.median([r["seconds"] for r in rows]) * 1000),
        p95_ms=float(np.quantile([r["seconds"] for r in rows], .95) * 1000),
    )
    for key, quantile in (("median_ratio", .5), ("p95_ratio", .95), ("p99_ratio", .99)):
        value = np.quantile(ratios, quantile, method="linear" if finite else "inverted_cdf") if ratios else None
        result[key] = "infinity" if value is not None and not np.isfinite(value) else float(value) if value is not None else None
    result["mean_ratio"] = (float(np.mean(ratios)) if finite else "infinity") if ratios else None
    result["max_ratio"] = (float(max(ratios)) if finite else "infinity") if ratios else None
    return result


def tune_family(X, Qt, optimum, warm, seed, sizes, swapped, config, metric, state):
    options = []
    a = prod(factorial(s) for s in sizes)
    if swapped:
        a = factorial(sum(sizes)) // a
    b = factorial(sum(sizes)) // a
    deltas = config["quality_deltas"]
    limit = config["quality_Lmax"]
    if b > config["heavy_probe_threshold"]:
        deltas = config["heavy_deltas"]
        limit = min(limit, config["heavy_L_max"])
    for delta in deltas:
        state(phase="building-tuning", delta=delta)
        signal.alarm(config["build_deadline_seconds"])
        try:
            index = build_tables(X, delta, limit, seed, sizes, swapped, config["effective_index_cap_bytes"])
        finally:
            signal.alarm(0)
        if not index.tables:
            continue
        evaluate(index, X, warm, np.full(len(warm), np.inf), 1, config, metric)
        state(phase="tuning", delta=delta)
        rows = evaluate(index, X, Qt, optimum, len(index.tables), config, metric,
                        lambda count: state(phase="tuning", delta=delta, completed_queries=count))
        for L in range(1, len(index.tables) + 1):
            hit = [r["exact_nn"] and r["first_table"] <= L for r in rows]
            options.append(dict(delta=float(delta), L=L, tuning_recall=float(np.mean(hit)),
                                bytes=int(index_bytes(index, L))))
        del index
        gc.collect()
    if not options:
        return None, False, []
    choice, reached = choose_first_entry_configuration(options, config["tuning_acceptance_recall"])
    return choice, reached, options


def construction_index_cap(X, k, a, b, config):
    reserve = min(2_000_000_000, int(config["peak_ram_cap_bytes"] * .05))
    raw_workspace = len(X) * a * 80
    fingerprints = len(X) * k * k * 8
    metadata = (a + b) * k
    headroom = config["peak_ram_cap_bytes"] - X.nbytes - raw_workspace - fingerprints - metadata - reserve
    return min(config["index_budget_bytes"], max(0, int(headroom)))


def worker(plan, job, output):
    if job.get("study_type") == "budget":
        from fresh.feasibility_first_hit import worker as budget_worker
        config = dict(plan["budget_protocol"])
        config.update(reference_results_root=job["reference_root"], output_root=str(output.parent))
        budget_worker(config, job["case"], output)
        complete = json.loads((output / "completion.json").read_text())
        complete["artifacts"] = {p.name: digest(p) for p in output.iterdir() if p.name in
                                 ("main-results.json", "test-queries.csv", "tuning-configurations.json")}
        write_json(output / "completion.json", complete)
        return
    config, reference, X, Q, optimum, info, truth_hash = load_reference(job)
    config.update(plan["protocol"])
    config.update(job.get("protocol_overrides", {}))
    metric = job.get("metric", "frechet")
    h, s = config["tuning_queries"], config["test_queries"]
    Qt, Qs, Qwarm = Q[:h], Q[h:h + s], Q[h + s:]
    opt_t, opt_s = optimum[:h], optimum[h:h + s]
    write_json(output / "dataset.json", dict(info, reference=str(reference),
                                            exact_ground_truth_sha256=truth_hash, metric=metric))
    def saved(name):
        path = output / name
        return json.loads(path.read_text()) if path.exists() else []

    selected = saved("frozen-selections.json")
    profiles = saved("tuning-configurations.json")
    measured = saved("main-results.json")
    excluded = saved("excluded.json")
    requested_counts = set(job.get("table_counts", []))
    query_path = output / "test-queries.csv"
    if query_path.exists():
        with query_path.open() as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames
            existing = list(reader)
        complete_ids = {r["config_id"] for r in measured}
        retained = [r for r in existing if r["config_id"] in complete_ids]
        if len(retained) != len(existing):
            query_path.rename(query_path.with_name(f"test-queries.partial-{time.time_ns()}.csv"))
            with query_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(retained)
    for sizes, swapped, labels in families(job, reference):
        family = "+".join(map(str, sizes)) + ("T" if swapped else "")
        a = prod(factorial(v) for v in sizes)
        if swapped:
            a = factorial(sum(sizes)) // a
        b = factorial(sum(sizes)) // a
        base = dict(family=family, blocks=list(sizes), swapped=swapped, labels=labels,
                    stored=a, probed=b, metric=metric, case_id=job["case_id"], index_seed=job["case"]["index_seed"])
        finished = [r for r in measured if r["family"] == family and r["status"] == "complete"]
        for row in finished:
            row["labels"] = list(dict.fromkeys([*row["labels"], *labels]))
        accounted_counts = {r["L"] for r in finished} | {
            r["L"] for r in excluded if r.get("family") == family and "L" in r}
        if any(r["selected_table_count"] for r in finished) and requested_counts <= accounted_counts:
            continue
        reason = observed_peak_gate(X, len(X), sum(sizes), a, 1, config)
        cap = construction_index_cap(X, sum(sizes), a, b, config)
        if not reason and cap <= 0:
            reason = "not run: no retained-index headroom under construction RAM limit"
        if reason:
            excluded.append(dict(base, status=reason))
            write_json(output / "excluded.json", excluded)
            continue

        def state(**details):
            worker_state(output, method=family, seed=job["case"]["index_seed"], **details)

        family_config = dict(config, effective_index_cap_bytes=cap)
        choice, reached, options = tune_family(
            X, Qt, opt_t, Qwarm, job["case"]["index_seed"], sizes, swapped, family_config, metric, state)
        profiles.append(dict(base, candidates=options, construction_index_cap_bytes=cap))
        write_json(output / "tuning-configurations.json", profiles)
        if choice is None:
            excluded.append(dict(base, status="not built: no table fits index/construction-memory headroom"))
            write_json(output / "excluded.json", excluded)
            continue
        selected.append(dict(base, **choice, tuning_target_reached=reached))
        write_json(output / "frozen-selections.json", selected)
        counts = sorted(set([choice["L"], *job.get("table_counts", [])]))
        state(phase="building-final", delta=choice["delta"])
        signal.alarm(config["build_deadline_seconds"])
        try:
            index = build_tables(X, choice["delta"], max(counts), job["case"]["index_seed"],
                                 sizes, swapped, cap)
        finally:
            signal.alarm(0)
        for L in counts:
            if any(r["family"] == family and r["L"] == L for r in measured):
                continue
            if L > len(index.tables):
                excluded.append(dict(base, L=L, delta=choice["delta"],
                                     status="not built: requested table prefix exceeds index budget"))
                write_json(output / "excluded.json", excluded)
                continue
            evaluate(index, X, Qwarm, np.full(len(Qwarm), np.inf), L, config, metric)
            state(phase="test", delta=choice["delta"], L=L)
            rows = evaluate(index, X, Qs, opt_s, L, config, metric,
                            lambda count: state(phase="test", delta=choice["delta"], L=L, completed_queries=count))
            key = f"{family}-d{choice['delta']:g}-L{L}"
            for row in rows:
                append_row(output / "test-queries.csv", dict(config_id=key, **base, L=L,
                                                            delta=choice["delta"], **row))
            measured.append(dict(base, config_id=key, delta=choice["delta"], L=L,
                                 selected_table_count=L == choice["L"], tuning_target_reached=reached,
                                 index_bytes=int(index_bytes(index, L)), build_seconds=index.build_seconds,
                                 status="complete", **quality_summary(rows)))
            write_json(output / "main-results.json", measured)
        del index
        gc.collect()
    for filename, data in (("main-results.json", measured), ("excluded.json", excluded)):
        write_json(output / filename, data)
    write_json(output / "completion.json", dict(
        completed=time.time(), job=job["id"],
        artifacts={p.name: digest(p) for p in output.iterdir() if p.name in
                   ("main-results.json", "excluded.json", "test-queries.csv", "frozen-selections.json")}))


def main(args):
    plan_path = Path(args.plan).resolve()
    plan = json.loads(plan_path.read_text())
    output = Path(args.output).resolve()
    if args.worker is not None:
        output.mkdir(parents=True, exist_ok=args.resume)
        worker(plan, plan["jobs"][args.worker], output)
        complete = json.loads((output / "completion.json").read_text())
        complete.update(source_revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(), plan_sha256=digest(plan_path))
        write_json(output / "completion.json", complete)
        return
    require_clean_tree()
    protocol = plan["protocol"]
    if numba.get_num_threads() != protocol["build_threads"] or int(os.environ.get("PIN_CORE", -1)) != protocol["pin_core"]:
        raise ValueError("CPU settings differ from the follow-up protocol")
    identity = dict(commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                    plan_sha256=digest(plan_path), protocol=protocol)
    if output.exists():
        if not args.resume:
            raise ValueError("Output exists; use explicit --resume with the same plan and source revision")
        if json.loads((output / "manifest.json").read_text()) != identity:
            raise ValueError("Resume requires the original source revision, plan, and protocol")
    else:
        output.mkdir(parents=True)
        write_json(output / "manifest.json", identity)
        write_json(output / "plan.json", plan)
    outcomes = []
    for i, job in enumerate(plan["jobs"]):
        folder = output / job["id"]
        complete = folder / "completion.json"
        if complete.exists():
            for name, expected in json.loads(complete.read_text())["artifacts"].items():
                if digest(folder / name) != expected:
                    raise ValueError("Completed result artifact changed")
            outcomes.append(dict(job=job["id"], status="complete"))
            continue
        if folder.exists():
            folder.rename(folder.with_name(folder.name + f".interrupted-{time.time_ns()}"))
        with (output / f"{job['id']}.log").open("w") as log:
            process = subprocess.Popen(
                [sys.executable, "-u", "-m", "fresh.experiment_first_hit_quality", "--plan", str(plan_path),
                 "--output", str(folder), "--worker", str(i)], stdout=log, stderr=subprocess.STDOUT)
            try:
                code, interruption = monitor_worker(process, folder, protocol, protocol["job_timeout_seconds"])
            finally:
                terminate_worker(process)
        state = interruption or ("complete" if code == 0 else "implementation failure")
        outcomes.append(dict(job=job["id"], status=state, returncode=code))
        write_json(output / "cases.json", outcomes)
        print(f"{job['id']}: {state}", flush=True)
        if code or interruption:
            raise RuntimeError(f"Stopped at {job['id']}: {state}")
    write_json(output / "completion.json", dict(completed=time.time(), jobs=len(outcomes)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--worker", type=int)
    parser.add_argument("--resume", action="store_true")
    main(parser.parse_args())
