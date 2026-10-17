"""Cost of every block partition: index size and latency of blocking in both orientations (k = 8)."""
import argparse
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

from curveset.orders import partitions
from curveset.tune import tables_for_recall
from fresh.e1 import append_row, monitor_worker, terminate_worker, write_json
from fresh.experiment_first_entry import load_reference_dataset, require_clean_tree, resolve_path, worker_state
from fresh.nba_full import digest
from fresh.poc import Joint, observed_peak_gate, tuning_target, weighted_cost
from fresh.regimes import family_first_hits, timed_queries, warm


def load_protocol(path):
    config_path = Path(path).resolve()
    raw = json.loads(config_path.read_text())
    source_path = resolve_path(config_path.parent, raw["synthetic_config"])
    source = json.loads(source_path.read_text())
    cases = {case["id"]: case for case in source["cases"]}
    config = dict(source)
    config.update(raw)
    config["cases"] = [dict(cases[i]) for i in raw["case_ids"]]
    config["config_path"] = str(config_path)
    config["source_config_sha256"] = digest(source_path)
    config["reference_results_root"] = str(resolve_path(config_path.parent, raw["reference_results_root"]))
    return config


def study_jobs(config):
    return [dict(case, case_id=case["id"], index_seed=seed, id=f"{case['id']}-seed{seed}")
            for case in config["cases"] for seed in config["index_seeds"]]


def partition_specs(k):
    """Every partition of k slots in both orientations; Probe-all is listed once."""
    K, specs = factorial(k), []
    for sizes in partitions(k):
        stored = prod(factorial(s) for s in sizes)
        for swapped in (False, True):
            if swapped and len(sizes) == 1:
                continue
            a, b = (K // stored, stored) if swapped else (stored, K // stored)
            specs.append((a, b, tuple(sizes), swapped))
    return specs


def evaluate(index, X, Qt, opt_t, Qtest, opt_s, Qwarm, config, progress=None):
    call = lambda query: index.query(X, query, config)
    warm(call, Qwarm)
    rows, complete = timed_queries(call, Qt, opt_t, config)
    success = sum(r["exact_nn"] for r in rows) / len(rows)
    out = dict(tuning_complete=bool(complete), tuning_recall=float(success),
               tuning_median_seconds=float(np.median([r["seconds"] for r in rows])) if complete else None)
    out["status"] = "qualified" if complete and success >= tuning_target(config) else (
        "tuning recall below target" if complete else "tuning timeout")
    if out["status"] == "qualified":
        test_rows, done = timed_queries(call, Qtest, opt_s, config, progress)
        seconds = [r["seconds"] for r in test_rows]
        out.update(test_complete=bool(done), test_recall=float(np.mean([r["exact_nn"] for r in test_rows])),
                   test_median_ms=float(np.median(seconds) * 1000.0), test_p95_ms=float(np.quantile(seconds, 0.95) * 1000.0),
                   test_mean_candidates=float(np.mean([r["candidates"] for r in test_rows])),
                   test_mean_probes=float(np.mean([r["probes"] for r in test_rows])))
    return out


def worker(config, case, output):
    worker_state(output, method="ground-truth", seed=case["index_seed"], phase="preparation")
    X, Q, info, ids, distances, _ = load_reference_dataset(config, case)
    h, s = config["tuning_queries"], config["test_queries"]
    Qt, Qtest, Qwarm = Q[:h], Q[h:h + s], Q[h + s:]
    opt_t, opt_s = distances[:h], distances[h:h + s]
    NN = X[ids[:h]]
    seed, k, d = case["index_seed"], case["k"], config["d"]
    rows = []
    for delta in config["deltas"]:
        shifts = np.random.default_rng(seed).uniform(0, delta, (config["Lmax"], k, d))
        for a, b, sizes, swapped in partition_specs(k):
            cfg_id = f"d{delta:g}-{'+'.join(map(str, sizes))}-swap{int(swapped)}"
            base = dict(config_id=cfg_id, case_id=case["case_id"], index_seed=seed, delta=float(delta),
                        blocks=list(sizes), swapped=swapped, stored=a, probed=b)
            worker_state(output, method=cfg_id, seed=seed, phase="collision-tuning")
            hit = family_first_hits(Qt, NN, shifts, delta, np.asarray(sizes), swapped)
            L = tables_for_recall(hit, tuning_target(config), config["Lmax"])
            base["L"] = L
            reason = "no tuned L in grid" if L is None else observed_peak_gate(X, len(X), k, a, L, config)
            if reason:
                rows.append(dict(base, status=reason))
                write_json(output / "rows.json", rows)
                continue
            worker_state(output, method=cfg_id, seed=seed, phase="building")
            signal.alarm(int(config["build_deadline_seconds"]))
            try:
                index = Joint(X, delta, L, seed, sizes, swapped)
            finally:
                signal.alarm(0)
            base.update(index_bytes=int(index.index_bytes), entries=int(index.entries), raw_entries=int(index.raw_entries),
                        build_seconds=float(index.build_seconds))
            if index.index_bytes > config["index_budget_bytes"]:
                rows.append(dict(base, status="over index storage"))
            else:
                worker_state(output, method=cfg_id, seed=seed, phase="tuning")
                rows.append(dict(base, **evaluate(index, X, Qt, opt_t, Qtest, opt_s, Qwarm, config)))
            del index
            write_json(output / "rows.json", rows)
    write_json(output / "dataset.json", dict(info, **case))
    write_json(output / "completion.json", dict(completed=time.time(), case=case["id"], rows=len(rows)))


def selections(rows, config):
    """For every latency weight the configuration that minimizes the tuning cost J_lambda."""
    eligible = [r for r in rows if r["status"] == "qualified"]
    chosen = {}
    for weight in config["ablation_latency_weights"]:
        chosen[str(weight)] = min(eligible, key=lambda r: (weighted_cost(r, weight, config), r["index_bytes"], r["config_id"]))
    return chosen


def summarize(output, config):
    summary = []
    for job in study_jobs(config):
        rows = json.loads((output / job["id"] / "rows.json").read_text())
        summary.append(dict(case=job["id"], case_id=job["case_id"], index_seed=job["index_seed"], rows=rows,
                            selections=selections(rows, config)))
    write_json(output / "summary.json", dict(jobs=summary, note="Every partition of k slots in both orientations; "
                                             "grid size and number of tables are tuned for each, and every qualified "
                                             "(partition, grid size) pair is measured on the test queries."))


def main(args):
    config = load_protocol(args.config)
    output = Path(args.output).resolve()
    config["output_root"] = str(output if args.worker is None else output.parent)
    if args.worker is not None:
        worker(config, study_jobs(config)[args.worker], output)
        return
    require_clean_tree()
    if numba.get_num_threads() != config["build_threads"] or int(os.environ.get("PIN_CORE", "-1")) != config["pin_core"]:
        raise ValueError("CPU settings differ from the frozen frontier protocol")
    output.mkdir(parents=True, exist_ok=False)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    write_json(output / "config.json", config)
    write_json(output / "manifest.json", dict(commit=commit, config_sha256=digest(Path(args.config).resolve()),
                                              started=time.time(), python=sys.version, numpy=np.__version__,
                                              numba=numba.__version__, pin_core=config["pin_core"],
                                              build_threads=numba.get_num_threads(),
                                              source_config_sha256=config["source_config_sha256"]))
    started = time.monotonic()
    cases = []
    for i, case in enumerate(study_jobs(config)):
        remaining = config["run_cap_seconds"] - (time.monotonic() - started)
        if remaining <= 0:
            raise RuntimeError("Frontier study exceeded its run cap")
        folder = output / case["id"]
        folder.mkdir()
        with (folder / "worker.log").open("w") as log:
            process = subprocess.Popen([sys.executable, "-u", "-m", "fresh.experiment_frontier", "--config",
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
