"""First-entry index variant: one stored entry per key and the first collision as the answer."""
import argparse
from collections import defaultdict
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

from fresh.e1 import append_row, monitor_worker, terminate_worker, write_json
from fresh.nba_full import digest
from numba import njit

from curveset import core
from curveset.index import Tables, _bucket
from fresh.poc import Joint, dataset, observed_peak_gate
from fresh.regimes import timed_queries, verify, warm


METHODS = ("blocking", "probe_all", "store_all")
EXCLUSION_STATUSES = {"excluded", "missing frozen selection"}


def require_clean_tree():
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise RuntimeError("Run only from a clean frozen source tree")


def percentile_ms(seconds, quantile):
    if not seconds:
        return None
    return float(np.quantile(np.asarray(seconds, float), quantile) * 1000.0)


def worker_state(output, **payload):
    write_json(output / "active_configuration.json", payload)


def study_jobs(config):
    return [dict(case, case_id=case["id"], index_seed=seed, id=f"{case['id']}-seed{seed}")
            for case in config["cases"] for seed in config["index_seeds"]]


def resolve_path(base, value):
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def load_protocol(path):
    config_path = Path(path).resolve()
    raw = json.loads(config_path.read_text())
    source_path = resolve_path(config_path.parent, raw["synthetic_config"])
    source = json.loads(source_path.read_text())
    cases = {case["id"]: case for case in source["cases"]}
    chosen = [dict(cases[case_id]) for case_id in raw["case_ids"]]
    missing = [case_id for case_id in raw["case_ids"] if case_id not in cases]
    if missing:
        raise ValueError(f"Unknown source cases: {missing}")
    config = dict(source)
    config.update(raw)
    config["cases"] = chosen
    config["config_path"] = str(config_path)
    config["source_config"] = str(source_path)
    config["source_config_sha256"] = digest(source_path)
    config["reference_results_root"] = str(resolve_path(config_path.parent, raw["reference_results_root"]))
    config["methods"] = tuple(raw.get("methods", METHODS))
    if tuple(config["methods"]) != METHODS:
        raise ValueError("The first-entry experiment expects Blocking, Probe-all, and Store-all only")
    config["deltas"] = tuple(float(v) for v in raw["deltas"])
    config["L_max"] = int(raw["L_max"])
    config["store_all_L_max"] = int(raw.get("store_all_L_max", raw["L_max"]))
    config["index_seeds"] = tuple(int(v) for v in raw.get("index_seeds", source["index_seeds"]))
    return config


def load_reference_dataset(config, case):
    reference = Path(config["reference_results_root"]) / case["id"]
    original = json.loads((reference / "dataset.json").read_text())
    X, Q, info = dataset(config, case)
    if info["dataset_sha256"] != original["dataset_sha256"] or info["queries_sha256"] != original["queries_sha256"]:
        raise ValueError(f"Synthetic arrays changed for {case['id']}")
    truth_path = reference / "ground-truth.npz"
    with np.load(truth_path, allow_pickle=False) as truth:
        if "queries" in truth and not np.array_equal(truth["queries"], Q):
            raise ValueError(f"Frozen ground-truth queries changed for {case['id']}")
        ids = truth["ids"].copy()
        distances = truth["distances"].copy()
    selections = json.loads((reference / "frozen-main-selections.json").read_text())
    info.update(reference_dataset_json=str(reference / "dataset.json"),
                reference_ground_truth=str(truth_path),
                reference_ground_truth_sha256=digest(truth_path),
                reference_frozen_selections=str(reference / "frozen-main-selections.json"),
                reference_frozen_selections_sha256=digest(reference / "frozen-main-selections.json"),
                dataset_json_sha256=digest(reference / "dataset.json"))
    return X, Q, info, ids, distances, selections


@njit(cache=True)
def first_entry_count(keys):
    count = 0
    for i in range(len(keys)):
        if i == 0 or keys[i] != keys[i - 1]:
            count += 1
    return count


@njit(cache=True)
def first_entry_into(keys, ids, out_keys, out_ids, directory, B):
    out, bucket = 0, 0
    directory[0] = 0
    for i in range(len(keys)):
        if i and keys[i] == keys[i - 1]:
            continue
        target = _bucket(keys[i], B)
        while bucket < target:
            bucket += 1
            directory[bucket] = out
        out_keys[out], out_ids[out] = keys[i], ids[i]
        out += 1
    while bucket < B:
        bucket += 1
        directory[bucket] = out


def keep_first_entries(table):
    """Keep the first stored multi-curve (lowest identifier) for every key, as in the dictionary of the design."""
    keys, ids = table.keys[:table.N], table.ids[:table.N]
    result = Tables(1, first_entry_count(keys))
    first_entry_into(keys, ids, result.keys, result.ids, result.dir, result.B)
    return result


@njit(cache=True)
def first_hit_in_table(Q, shift, delta, family, needed, keys, ids, directory, B):
    k = len(Q)
    fingerprints = np.zeros((k, k), np.uint64)
    prefix = np.empty(k + 1, np.uint64)
    prefix[0] = core.SEED
    for i in range(k):
        for j in range(k):
            if needed[i, j]:
                fingerprints[i, j] = core.curve_fp(Q[j], shift[i], delta)
    for s in range(len(family)):
        common = 0
        if s:
            while common < k and family[s, common] == family[s - 1, common]:
                common += 1
        for i in range(common, k):
            prefix[i + 1] = core.mix64(prefix[i] + fingerprints[i, family[s, i]])
        fp = prefix[k]
        bucket = _bucket(fp, B)
        lo, hi = np.int64(directory[bucket]), np.int64(directory[bucket + 1])
        for t in range(lo, hi):
            if keys[t] > fp:
                break
            if keys[t] == fp:
                return np.int64(ids[t]), s + 1
    return np.int64(-1), len(family)


def table_first_hit(index, u, Q, config):
    """First entry matched in table u by the query keys in probing order, or -1, and the keys probed."""
    base = index.base
    table = index.tables[u]
    probes = 0
    for start in range(0, len(base.probed), config["probe_chunk"]):
        family = base.probed[start:start + config["probe_chunk"]]
        found, read = first_hit_in_table(Q, base.shifts[u], base.delta, family, base.need_q,
                                         table.keys, table.ids, table.dir, table.B)
        probes += read
        if found >= 0:
            return int(found), probes
    return -1, probes


def first_hit_query(index, X, Q, config, L=None):
    """Return the first multi-curve whose key a probed key matches; the timed part ends before the distance check."""
    started = time.perf_counter()
    answer, probes = -1, 0
    for u in range(len(index.tables) if L is None else L):
        answer, read = table_first_hit(index, u, Q, config)
        probes += read
        if answer >= 0:
            break
    seconds = time.perf_counter() - started
    distance = verify(X, Q, np.array([answer], np.int64), config, time.perf_counter())[1] if answer >= 0 else np.inf
    return dict(id=answer, distance=float(distance), timeout=False, candidates=int(answer >= 0), verified=0,
                probes=probes, postings=int(answer >= 0), seconds=seconds)


def build_first_entry_joint(X, delta, L, seed, sizes, swapped):
    """Joint index that reduces every table to one entry per key as soon as the table is built."""
    from curveset.index import make_index
    from fresh.poc import compact_table
    started = time.perf_counter()
    index = Joint.__new__(Joint)
    index.base = make_index("blocking", X.shape[1], X.shape[3], delta, L, sizes=sizes, swapped=swapped, seed=seed)
    index.tables = []
    for u in range(L):
        fingerprints = core.slot_fps(X, index.base.shifts[u], delta, index.base.need_s)
        keys = core.order_keys(fingerprints, index.base.stored).ravel()
        index.tables.append(keep_first_entries(compact_table(keys, len(index.base.stored))))
    index.base._buffers(len(X))
    index.build_seconds = time.perf_counter() - started
    index.raw_entries = len(X) * len(index.base.stored) * L
    return index


def prefix_bytes(index, L):
    return sum(t.nbytes for t in index.tables[:L]) + index.base.shifts.nbytes


def tuning_recalls(index, X, Qt, optimum, config):
    """Tuning recall of every table prefix, from the first hit of each table."""
    L = len(index.tables)
    first = np.full((len(Qt), L), -1, np.int64)
    for q, Q in enumerate(Qt):
        for u in range(L):
            first[q, u], _ = table_first_hit(index, u, Q, config)
            if first[q, u] >= 0:
                break
    cache, recalls = {}, np.zeros(L)
    for L_active in range(1, L + 1):
        hits = 0
        for q, Q in enumerate(Qt):
            ids = first[q, :L_active]
            found = ids[ids >= 0]
            if len(found):
                p = int(found[0])
                if (q, p) not in cache:
                    cache[(q, p)] = verify(X, Q, np.array([p], np.int64), config, time.perf_counter())[1]
                hits += cache[(q, p)] <= optimum[q] + 1e-8
        recalls[L_active - 1] = hits / len(Qt)
    return recalls


def choose_first_entry_configuration(candidates, target):
    """Smallest table count that reaches the tuning target, else the best tuning recall; ties go to the smaller index."""
    reached = [c for c in candidates if c["tuning_recall"] >= target]
    if reached:
        return min(reached, key=lambda c: (c["L"], c["bytes"], c["delta"])), True
    return min(candidates, key=lambda c: (-c["tuning_recall"], c["L"], c["bytes"], c["delta"])), False


def tune_and_test(X, Qt, optimum_tune, Qtest, optimum_test, Qwarm, selection, seed, L_max, config, state=None):
    """Tune the grid size and table count on the tuning queries only, then test the chosen configuration."""
    sizes, swapped = tuple(selection["blocks"]), bool(selection["swapped"])
    stored = prod(factorial(b) for b in sizes)
    probed = factorial(sum(sizes)) // stored
    if swapped:
        stored, probed = probed, stored
    deltas = config["deltas"]
    if probed > config.get("heavy_probe_threshold", float("inf")):
        # Misses scan every probed key; restrict the search where the full grid is too slow to tune.
        deltas = config["heavy_deltas"]
        L_max = min(L_max, config["heavy_L_max"])
    candidates = []
    for delta in deltas:
        if state:
            state(phase="tuning", delta=delta)
        signal.alarm(int(config["build_deadline_seconds"]))
        try:
            index = build_first_entry_joint(X, delta, L_max, seed, sizes, swapped)
        finally:
            signal.alarm(0)
        recalls = tuning_recalls(index, X, Qt, optimum_tune, config)
        for L in range(1, L_max + 1):
            candidates.append(dict(delta=float(delta), L=L, tuning_recall=float(recalls[L - 1]),
                                   bytes=int(prefix_bytes(index, L))))
        del index
    chosen, reached = choose_first_entry_configuration(candidates, config["tuning_acceptance_recall"])
    if state:
        state(phase="building", delta=chosen["delta"], target_L=chosen["L"])
    index = build_first_entry_joint(X, chosen["delta"], chosen["L"], seed, sizes, swapped)
    call = lambda query: first_hit_query(index, X, query, config)
    warm(call, Qwarm)
    rows, complete = timed_queries(call, Qtest, optimum_test, config)
    return index, rows, complete, chosen, reached, candidates


def peak_gate_reason(X, case, selection, L, config):
    stored = selection.get("stored") or case["k"]
    return observed_peak_gate(X, len(X), case["k"], int(stored), int(L), config)


def exclusion_row(config, case, info, selection, method, L, reason):
    return dict(case=case["id"], case_id=case["case_id"], index_seed=case["index_seed"], kind=case["kind"],
                n=case["n"], k=case["k"], display_method=method, target_L=int(L),
                source_config_id=None if selection is None else selection["config_id"],
                source_L=None if selection is None else selection["L"],
                delta=None if selection is None else float(selection["delta"]),
                build_seconds=None, index_bytes=None, index_mb=None, entries=None, raw_entries=None,
                scratch_bytes=None, recall_at_1=None, median_ms=None, p95_ms=None, queries_evaluated=0,
                test_status="not run", status="missing frozen selection" if selection is None else "excluded",
                exclusion_reason=reason, commit=json.loads((Path(config["output_root"]) / "manifest.json").read_text())["commit"],
                config_sha256=json.loads((Path(config["output_root"]) / "manifest.json").read_text())["config_sha256"],
                dataset_sha256=info["dataset_sha256"], queries_sha256=info["queries_sha256"])


def summarize_query_rows(rows):
    recalls = [bool(row["exact_nn"]) for row in rows]
    seconds = [float(row["seconds"]) for row in rows]
    return dict(recall_at_1=float(np.mean(recalls)) if recalls else None,
                median_ms=percentile_ms(seconds, 0.5),
                p95_ms=percentile_ms(seconds, 0.95),
                min_ms=percentile_ms(seconds, 0.0),
                max_ms=percentile_ms(seconds, 1.0),
                mean_ms=float(np.mean(np.asarray(seconds, float)) * 1000.0) if seconds else None,
                empty_rate=float(np.mean([not row["found"] for row in rows])) if rows else None,
                within_1_1=float(np.mean([bool(row["found"] and row["ratio"] is not None and row["ratio"] <= 1.1 + 1e-8)
                                          for row in rows])) if rows else None,
                mean_ratio=(float(np.mean([row["ratio"] for row in rows if row["ratio"] is not None]))
                            if any(row["ratio"] is not None for row in rows) else None))


def append_query_rows(path, manifest, info, case, method, target_L, source_config_id, optimum, rows):
    for q, row in enumerate(rows):
        append_row(path, dict(config_id=source_config_id, commit=manifest["commit"],
                              config_sha256=manifest["config_sha256"], dataset_sha256=info["dataset_sha256"],
                              queries_sha256=info["queries_sha256"], case=case["id"], case_id=case["case_id"],
                              index_seed=case["index_seed"], display_method=method, target_L=target_L,
                              phase="test", query=q, nn_distance=float(optimum[q]), **row))


def worker(config, case, output):
    if any(path.name != "worker.log" for path in output.iterdir()):
        raise ValueError(f"Worker output must start empty apart from worker.log: {output}")
    manifest = json.loads((output.parent / "manifest.json").read_text())
    worker_state(output, method="ground-truth", seed=case["index_seed"], phase="preparation")
    X, Q, info, ids, distances, selections = load_reference_dataset(config, case)
    h, s = config["tuning_queries"], config["test_queries"]
    Qt, Qtest, Qwarm = Q[:h], Q[h:h + s], Q[h + s:]
    optimum_tune, optimum = distances[:h], distances[h:h + s]
    write_json(output / "dataset.json", dict(info, **case, reference_exact_nn_ids_sha256=digest(Path(info["reference_ground_truth"])),
                                             mean_test_nn_distance=float(optimum.mean())))
    results, tuning = [], {}
    for method in METHODS:
        selection = selections.get(method)
        L_max = config["store_all_L_max"] if method == "store_all" else config["L_max"]
        if selection is None:
            row = exclusion_row(config, case, info, None, method, 0, "not run: no frozen qualified selection")
            append_row(output / "results.csv", row)
            results.append(row)
            continue
        reason = peak_gate_reason(X, case, selection, L_max, config)
        if reason is not None:
            row = exclusion_row(config, case, info, selection, method, L_max, reason)
            append_row(output / "results.csv", row)
            results.append(row)
            continue
        state = lambda m=method, **extra: worker_state(output, method=m, seed=case["index_seed"], **extra)
        index, rows, complete, chosen, reached, candidates = tune_and_test(
            X, Qt, optimum_tune, Qtest, optimum, Qwarm, selection, case["index_seed"], L_max, config, state)
        append_query_rows(output / "test-queries.csv", manifest, info, case, method, chosen["L"],
                          selection["config_id"], optimum, rows)
        tuning[method] = candidates
        result = dict(case=case["id"], case_id=case["case_id"], index_seed=case["index_seed"],
                      kind=case["kind"], n=case["n"], k=case["k"], display_method=method,
                      target_L=int(chosen["L"]), source_config_id=selection["config_id"],
                      source_L=int(selection["L"]), source_delta=float(selection["delta"]),
                      delta=float(chosen["delta"]), tuning_recall=float(chosen["tuning_recall"]),
                      tuning_target_reached=bool(reached), blocks=list(selection["blocks"]),
                      swapped=bool(selection["swapped"]), build_seconds=float(index.build_seconds),
                      index_bytes=int(index.index_bytes), index_mb=float(index.index_bytes / 1e6),
                      entries=int(index.entries), raw_entries=int(index.raw_entries),
                      queries_evaluated=len(rows), test_status="complete" if complete else "test timeout",
                      status="complete" if complete else "test timeout", exclusion_reason=None,
                      commit=manifest["commit"], config_sha256=manifest["config_sha256"],
                      dataset_sha256=info["dataset_sha256"], queries_sha256=info["queries_sha256"],
                      **summarize_query_rows(rows))
        append_row(output / "results.csv", result)
        results.append(result)
        del index
    write_json(output / "tuning-grid.json", tuning)
    write_json(output / "main-results.json", results)
    write_json(output / "completion.json", dict(completed=time.time(), case=case["id"], rows=len(results)))


def aggregate_case_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        if row["status"] == "complete":
            grouped[(row["case_id"], row["display_method"])].append(row)
    aggregates = []
    for (case_id, method), group in sorted(grouped.items()):
        recall = np.asarray([row["recall_at_1"] for row in group], float)
        aggregates.append(dict(
            case_id=case_id, display_method=method, seed_count=len(group),
            mean_recall_at_1=float(recall.mean()), min_recall_at_1=float(recall.min()),
            mean_within_1_1=float(np.mean([row["within_1_1"] for row in group])),
            mean_empty_rate=float(np.mean([row["empty_rate"] for row in group])),
            mean_ratio=float(np.mean([row["mean_ratio"] for row in group if row["mean_ratio"] is not None]))
            if any(row["mean_ratio"] is not None for row in group) else None,
            median_ms=float(np.median([row["median_ms"] for row in group])),
            median_p95_ms=float(np.median([row["p95_ms"] for row in group])),
            max_median_ms=float(max(row["median_ms"] for row in group)),
            median_index_mb=float(np.median([row["index_mb"] for row in group])),
            median_L=float(np.median([row["target_L"] for row in group])),
            deltas=sorted({float(row["delta"]) for row in group}),
            tables=sorted({int(row["target_L"]) for row in group}),
            reached_count=sum(bool(row["tuning_target_reached"]) for row in group),
            mean_tuning_recall=float(np.mean([row["tuning_recall"] for row in group]))))
    return aggregates


def collect_rows(output, config):
    rows = []
    for job in study_jobs(config):
        path = output / job["id"] / "main-results.json"
        rows.extend(json.loads(path.read_text()))
    return rows


def write_top_level_summaries(output, config):
    rows = collect_rows(output, config)
    for row in rows:
        append_row(output / "all-results.csv", row)
    write_json(output / "all-results.json", rows)
    aggregates = aggregate_case_rows(rows)
    for row in aggregates:
        append_row(output / "aggregate-results.csv", row)
    exclusions = [row for row in rows if row["status"] in EXCLUSION_STATUSES]
    write_json(output / "summary.json", dict(
        aggregates=aggregates, exclusions=exclusions,
        note="Grid size and table count are tuned on the tuning queries for one stored entry per key and the first "
             "collision as the answer; block structures come from the frozen candidate-reranking selections."))


def main(args):
    config = load_protocol(args.config)
    output = Path(args.output).resolve()
    config["output_root"] = str(output if args.worker is None else output.parent)
    if args.worker is not None:
        worker(config, study_jobs(config)[args.worker], output)
        return
    require_clean_tree()
    if numba.get_num_threads() != config["build_threads"] or int(os.environ.get("PIN_CORE", "-1")) != config["pin_core"]:
        raise ValueError("CPU settings differ from the frozen first-entry protocol")
    output.mkdir(parents=True, exist_ok=False)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    write_json(output / "config.json", config)
    write_json(output / "manifest.json",
               dict(commit=commit, config_sha256=digest(Path(args.config).resolve()), started=time.time(),
                    python=sys.version, numpy=np.__version__, numba=numba.__version__,
                    cpu_model=next(line.split(":", 1)[1].strip()
                                   for line in Path("/proc/cpuinfo").read_text().splitlines()
                                   if line.startswith("model name")),
                    build_threads=numba.get_num_threads(), pin_core=config["pin_core"],
                    source_config_sha256=config["source_config_sha256"], implementation_sha256=digest(__file__)))
    started = time.monotonic()
    cases = []
    for i, case in enumerate(study_jobs(config)):
        remaining = config["run_cap_seconds"] - (time.monotonic() - started)
        if remaining <= 0:
            raise RuntimeError("First-entry study exceeded its run cap")
        folder = output / case["id"]
        folder.mkdir()
        with (folder / "worker.log").open("w") as log:
            process = subprocess.Popen([sys.executable, "-u", "-m", "fresh.experiment_first_entry",
                                        "--config", str(Path(args.config).resolve()),
                                        "--output", str(folder), "--worker", str(i)],
                                       stdout=log, stderr=subprocess.STDOUT)
            try:
                code, interruption = monitor_worker(process, folder, config, remaining)
            finally:
                terminate_worker(process)
        status = interruption or ("complete" if code == 0 else "implementation or build failure")
        cases.append(dict(case=case["id"], case_id=case["case_id"], index_seed=case["index_seed"],
                          status=status, returncode=code))
        write_json(output / "cases.json", cases)
        print(f"{case['id']}: {status}", flush=True)
        if code or interruption:
            raise RuntimeError(f"Stopped at {folder}: {status}")
    write_top_level_summaries(output, config)
    write_json(output / "completion.json", dict(completed=time.time(), elapsed_seconds=time.monotonic() - started))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--worker", type=int)
    main(parser.parse_args())
