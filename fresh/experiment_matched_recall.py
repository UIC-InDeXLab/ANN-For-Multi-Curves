"""Matched-recall recall/latency curves from frozen synthetic selections."""
import argparse
from collections import defaultdict
import json
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
from fresh.poc import Joint, PerCurve, dataset, observed_peak_gate
from fresh.regimes import timed_queries, warm


METHODS = ("blocking", "probe_all", "per_curve-intersection")
EXCLUSION_STATUSES = {"excluded", "missing frozen selection"}


def validate_l_values(values):
    L_values = tuple(int(v) for v in values)
    if not L_values or any(v <= 0 for v in L_values):
        raise ValueError("L_values must be positive integers")
    if any(a >= b for a, b in zip(L_values, L_values[1:])):
        raise ValueError("L_values must be strictly increasing")
    return L_values


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
        raise ValueError("Matched-recall experiment expects Blocking, Probe-all, and Intersection only")
    config["L_values"] = validate_l_values(raw["L_values"])
    config["matched_recall_targets"] = tuple(float(v) for v in raw["matched_recall_targets"])
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


def build_call(X, selection, seed, L, config):
    signal.alarm(int(config["build_deadline_seconds"]))
    try:
        if selection["method"] == "blocking-family":
            index = Joint(X, selection["delta"], L, seed, tuple(selection["blocks"]), bool(selection["swapped"]))
            call = lambda query: index.query(X, query, config)
        else:
            index = PerCurve(X, selection["delta"], L, seed)
            call = lambda query: index.query(X, query, config, False)
    finally:
        signal.alarm(0)
    return index, call


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
                mean_ms=float(np.mean(np.asarray(seconds, float)) * 1000.0) if seconds else None)


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
    Qtest, Qwarm = Q[h:h+s], Q[h+s:]
    optimum = distances[h:h+s]
    write_json(output / "dataset.json", dict(info, **case, reference_exact_nn_ids_sha256=digest(Path(info["reference_ground_truth"])),
                                             mean_test_nn_distance=float(optimum.mean())))
    write_json(output / "reference-ground-truth.json",
               dict(reference_ground_truth=info["reference_ground_truth"], ids_sha256=digest(Path(info["reference_ground_truth"]))))
    results = []
    for method in METHODS:
        selection = selections.get(method)
        for L in config["L_values"]:
            if selection is None:
                row = exclusion_row(config, case, info, None, method, L, "not run: no frozen qualified selection")
                append_row(output / "results.csv", row)
                results.append(row)
                continue
            reason = peak_gate_reason(X, case, selection, L, config)
            if reason is not None:
                row = exclusion_row(config, case, info, selection, method, L, reason)
                append_row(output / "results.csv", row)
                results.append(row)
                continue
            worker_state(output, method=method, seed=case["index_seed"], phase="building", target_L=L,
                         source_config_id=selection["config_id"])
            index, call = build_call(X, selection, case["index_seed"], L, config)
            if index.index_bytes > config["index_budget_bytes"]:
                row = exclusion_row(config, case, info, selection, method, L,
                                    "not run: actual index storage budget")
                row.update(build_seconds=float(index.build_seconds), index_bytes=int(index.index_bytes),
                           index_mb=float(index.index_bytes / 1e6), entries=int(index.entries),
                           raw_entries=int(index.raw_entries), scratch_bytes=int(index.scratch_bytes))
                append_row(output / "results.csv", row)
                results.append(row)
                del call, index
                continue
            worker_state(output, method=method, seed=case["index_seed"], phase="warmup", target_L=L,
                         source_config_id=selection["config_id"])
            warm(call, Qwarm)
            worker_state(output, method=method, seed=case["index_seed"], phase="test", target_L=L,
                         source_config_id=selection["config_id"], completed_queries=0)
            progress = lambda count, m=method, ll=L: worker_state(
                output, method=m, seed=case["index_seed"], phase="test", target_L=ll,
                source_config_id=selection["config_id"], completed_queries=count)
            rows, complete = timed_queries(call, Qtest, optimum, config, progress)
            append_query_rows(output / "test-queries.csv", manifest, info, case, method, L,
                              selection["config_id"], optimum, rows)
            summary = summarize_query_rows(rows)
            result = dict(case=case["id"], case_id=case["case_id"], index_seed=case["index_seed"],
                          kind=case["kind"], n=case["n"], k=case["k"], display_method=method,
                          target_L=int(L), source_config_id=selection["config_id"], source_L=int(selection["L"]),
                          delta=float(selection["delta"]), build_seconds=float(index.build_seconds),
                          index_bytes=int(index.index_bytes), index_mb=float(index.index_bytes / 1e6),
                          entries=int(index.entries), raw_entries=int(index.raw_entries),
                          scratch_bytes=int(index.scratch_bytes), queries_evaluated=len(rows),
                          test_status="complete" if complete else "test timeout",
                          status="complete" if complete else "test timeout", exclusion_reason=None,
                          commit=manifest["commit"], config_sha256=manifest["config_sha256"],
                          dataset_sha256=info["dataset_sha256"], queries_sha256=info["queries_sha256"],
                          **summary)
            append_row(output / "results.csv", result)
            results.append(result)
            del call, index
    write_json(output / "main-results.json", results)
    write_json(output / "completion.json", dict(completed=time.time(), case=case["id"], rows=len(results)))


def aggregate_case_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        if row["status"] == "complete":
            grouped[(row["case_id"], row["display_method"], int(row["target_L"]))].append(row)
    aggregates = []
    for (case_id, method, L), group in sorted(grouped.items()):
        recall = np.asarray([row["recall_at_1"] for row in group], float)
        median_ms = np.asarray([row["median_ms"] for row in group], float)
        p95_ms = np.asarray([row["p95_ms"] for row in group], float)
        index_mb = np.asarray([row["index_mb"] for row in group], float)
        build_seconds = np.asarray([row["build_seconds"] for row in group], float)
        aggregates.append(dict(case_id=case_id, display_method=method, target_L=L, seed_count=len(group),
                               mean_recall_at_1=float(recall.mean()),
                               median_seed_median_ms=float(np.median(median_ms)),
                               median_seed_p95_ms=float(np.median(p95_ms)),
                               median_index_mb=float(np.median(index_mb)),
                               median_build_seconds=float(np.median(build_seconds)),
                               min_recall_at_1=float(recall.min()), max_recall_at_1=float(recall.max()),
                               min_median_ms=float(median_ms.min()), max_median_ms=float(median_ms.max()),
                               min_p95_ms=float(p95_ms.min()), max_p95_ms=float(p95_ms.max()),
                               min_index_mb=float(index_mb.min()), max_index_mb=float(index_mb.max())))
    return aggregates


def matched_recall_rows(aggregates, targets):
    grouped = defaultdict(list)
    for row in aggregates:
        grouped[(row["case_id"], row["display_method"])].append(row)
    matches = []
    for (case_id, method), group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda row: row["target_L"])
        for target in targets:
            chosen = next((row for row in ordered if row["mean_recall_at_1"] >= target), None)
            if chosen is None:
                matches.append(dict(case_id=case_id, display_method=method, target_recall=target,
                                    status="not reached", target_L=None, mean_recall_at_1=None,
                                    median_seed_median_ms=None, median_seed_p95_ms=None, median_index_mb=None))
            else:
                matches.append(dict(case_id=case_id, display_method=method, target_recall=target,
                                    status="reached", target_L=chosen["target_L"],
                                    mean_recall_at_1=chosen["mean_recall_at_1"],
                                    median_seed_median_ms=chosen["median_seed_median_ms"],
                                    median_seed_p95_ms=chosen["median_seed_p95_ms"],
                                    median_index_mb=chosen["median_index_mb"]))
    return matches


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
    matched = matched_recall_rows(aggregates, config["matched_recall_targets"])
    for row in matched:
        append_row(output / "matched-recall.csv", row)
    exclusions = [row for row in rows if row["status"] in EXCLUSION_STATUSES]
    write_json(output / "summary.json", dict(aggregates=aggregates, matched_recall=matched,
                                              exclusions=exclusions,
                                              note="Frozen test-set configurations vary only L; ground truth is reused from the original runs."))


def main(args):
    config = load_protocol(args.config)
    output = Path(args.output).resolve()
    config["output_root"] = str(output if args.worker is None else output.parent)
    if args.worker is not None:
        worker(config, study_jobs(config)[args.worker], output)
        return
    require_clean_tree()
    if numba.get_num_threads() != config["build_threads"] or int(os.environ.get("PIN_CORE", "-1")) != config["pin_core"]:
        raise ValueError("CPU settings differ from the frozen matched-recall protocol")
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
            raise RuntimeError("Matched-recall study exceeded its run cap")
        folder = output / case["id"]
        folder.mkdir()
        with (folder / "worker.log").open("w") as log:
            process = subprocess.Popen([sys.executable, "-u", "-m", "fresh.experiment_matched_recall",
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
