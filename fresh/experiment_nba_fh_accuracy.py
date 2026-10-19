"""Accuracy-focused uniform and fine-to-coarse NBA FH, without candidate verification."""
import argparse
import gc
from itertools import combinations, product
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import numba
import numpy as np

from fresh.e1 import append_row, arrays_hash, write_json
from fresh.experiment_first_entry import require_clean_tree
from fresh.experiment_first_hit_quality import evaluate, quality_summary, timed_hit
from fresh.experiment_quality_tuned_fh import load_job, serving_bytes
from fresh.feasibility_first_hit import build_tables, index_bytes
from fresh.nba_full import digest


def schedule_hit(stages, query, L, config):
    if L != len(stages):
        raise ValueError("A schedule query must include every configured stage")
    started = time.perf_counter()
    probes, preceding = 0, 0
    answer, first_table, hit_stage, timeout = -1, sum(n for _, n in stages) + 1, -1, False
    for stage_number, (index, count) in enumerate(stages):
        remaining = config["query_deadline_seconds"] - (time.perf_counter() - started)
        if remaining <= 0:
            timeout = True
            break
        row = timed_hit(index, query, count, dict(config, query_deadline_seconds=remaining))
        probes += row["probes"]
        if row["timeout"]:
            timeout = True
            break
        if row["id"] >= 0:
            answer, first_table, hit_stage = row["id"], preceding + row["first_table"], stage_number
            break
        preceding += count
    seconds = time.perf_counter() - started
    timeout = timeout or seconds > config["query_deadline_seconds"]
    return dict(id=-1 if timeout else answer, timeout=timeout, probes=probes, seconds=seconds,
                first_table=sum(n for _, n in stages) + 1 if timeout else first_table,
                hit_stage=-1 if timeout else hit_stage)


def observation_arrays(rows):
    return dict(
        available=np.array([r["found"] and not r["timeout"] for r in rows]),
        first_table=np.array([r["first_table"] for r in rows]),
        a125=np.array([r["within_125"] for r in rows]),
        a110=np.array([r["within_110"] for r in rows]),
        ratios=np.array([np.inf if r["ratio"] == "infinity" else
                         np.nan if r["ratio"] is None else r["ratio"] for r in rows]))


def score_schedule(schedule, observations):
    h = len(observations[schedule[0]["delta"]]["available"])
    returned, a125, a110 = (np.zeros(h, bool) for _ in range(3))
    ratios = np.full(h, np.nan)
    for stage in schedule:
        source = observations[stage["delta"]]
        take = ~returned & source["available"] & (source["first_table"] <= stage["L"])
        a125[take], a110[take], ratios[take] = source["a125"][take], source["a110"][take], source["ratios"][take]
        returned |= take
    mean = float(np.mean(ratios[returned])) if returned.any() else None
    return dict(tuning_a125=float(a125.mean()), tuning_a110=float(a110.mean()),
                tuning_coverage=float(returned.mean()),
                tuning_mean_ratio=mean if mean is None or np.isfinite(mean) else "infinity")


def quality_key(row):
    mean = row["tuning_mean_ratio"]
    mean = np.inf if mean in (None, "infinity") else mean
    return (-row["tuning_a125"], mean, -row["tuning_a110"], -row["tuning_coverage"], row["index_bytes"])


def choose(trials, kind, config):
    eligible = [r for r in trials if r["kind"] == kind
                and r["index_bytes"] <= config["index_budget_bytes"]
                and r["tuning_coverage"] + 1e-12 >= config["minimum_tuning_coverage"]]
    if not eligible:
        raise ValueError(f"No {kind} schedule meets the storage and coverage requirements")
    return min(eligible, key=quality_key)


def schedules(stage_options, maximum_stages):
    deltas = sorted(stage_options)
    for count in range(1, maximum_stages + 1):
        for selected in combinations(deltas, count):
            yield from product(*(stage_options[delta] for delta in selected))


def run(plan, job, output, exploratory=False):
    config, case, X, Q, optimum, info = load_job(plan, job)
    h, s = config["tuning_queries"], config["test_queries"]
    if len(Q) <= h + s or len(optimum) != len(Q):
        raise ValueError("Frozen tuning/test/warm-up splits are incomplete")
    deltas = config["quality_deltas"]
    grid = config["table_grid"]
    if (not deltas or len(set(deltas)) != len(deltas) or any(not np.isfinite(d) or d <= 0 for d in deltas)
            or not grid or grid != sorted(set(grid)) or any(not isinstance(n, int) or n <= 0 for n in grid)
            or not 2 <= config["maximum_stages"] <= len(deltas)
            or not 0 < config["minimum_tuning_coverage"] <= 1):
        raise ValueError("Invalid delta, table, stage-count, or coverage search grid")
    sizes, swapped = job["frozen_family"]["blocks"], job["frozen_family"]["swapped"]
    Qt, Qtest, Qwarm = Q[:h], Q[h:h + s], Q[h + s:]
    identity = dict(job=job, protocol=config, dataset=info, metric=job["metric"], index_seed=job["seed"],
                    source_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                    source_sha256=digest(Path(__file__)), exploratory=exploratory,
                    actual_numba_threads=numba.get_num_threads(), pin_core=os.environ.get("PIN_CORE"),
                    selection="maximize tuning A1.25, then minimize mean ratio, with common coverage and storage guards",
                    semantics="fine-to-coarse first collision; answer distances computed only after the query timer")
    sources = subprocess.check_output(["git", "ls-files", "curveset/*.py", "fresh/*.py"], text=True).splitlines()
    identity["source_files_sha256"] = {name: digest(Path(name)) for name in sources}
    identity["source_tree_status"] = subprocess.check_output(["git", "status", "--porcelain"], text=True)
    write_json(output / "provenance.json", identity)
    write_json(output / "dataset.json", dict(info, **case))

    def state(**details):
        write_json(output / "active_configuration.json", dict(seed=job["seed"], **details))

    observations, options = {}, {}
    for delta in deltas:
        state(phase="building-tuning", delta=delta)
        signal.alarm(config["build_deadline_seconds"])
        try:
            index = build_tables(X, delta, max(grid), job["seed"], sizes, swapped, config["index_budget_bytes"])
        finally:
            signal.alarm(0)
        if not index.tables:
            raise ValueError(f"No table fits the serving-storage cap for delta={delta}")
        evaluate(index, X, Qwarm, optimum[h + s:], 1, config, job["metric"])
        state(phase="tuning", delta=delta)
        rows = evaluate(index, X, Qt, optimum[:h], len(index.tables), config, job["metric"])
        if any(row["timeout"] for row in rows):
            raise ValueError("Tuning timeouts prevent exact prefix/schedule rescoring")
        for row in rows:
            append_row(output / "tuning-queries.csv", dict(delta=delta, source_L=len(index.tables), **row))
        observations[delta] = observation_arrays(rows)
        options[delta] = [dict(delta=delta, L=L, index_bytes=serving_bytes(index, L))
                          for L in grid if L <= len(index.tables)]
        print(f"TUNED {job['id']} delta={delta} L={len(index.tables)} "
              f"A125={quality_summary(rows)['success_125']:.3f}", flush=True)
        del index
        gc.collect()

    trials = []
    for schedule in schedules(options, config["maximum_stages"]):
        memory = sum(stage["index_bytes"] for stage in schedule)
        if memory > config["index_budget_bytes"]:
            continue
        trials.append(dict(kind="uniform" if len(schedule) == 1 else "cascade", schedule=list(schedule),
                           index_bytes=memory, **score_schedule(schedule, observations)))
    write_json(output / "tuning-configurations.json", trials)
    chosen = {kind: choose(trials, kind, config) for kind in ("uniform", "cascade")}
    write_json(output / "frozen-selections.json", chosen)
    write_json(output / "selection-frozen-before-test.json", dict(time=time.time(), methods=list(chosen)))
    measured = []
    for kind, winner in chosen.items():
        state(phase="building-test", method=kind, schedule=winner["schedule"])
        stages = []
        allocated = 0
        for stage in winner["schedule"]:
            signal.alarm(config["build_deadline_seconds"])
            try:
                index = build_tables(X, stage["delta"], stage["L"], job["seed"], sizes, swapped,
                                     config["index_budget_bytes"] - allocated)
            finally:
                signal.alarm(0)
            if len(index.tables) != stage["L"] or index_bytes(index, stage["L"]) != stage["index_bytes"]:
                raise ValueError("Rebuilt stage differs from its frozen serving-storage projection")
            stages.append((index, stage["L"]))
            allocated += stage["index_bytes"]
        evaluate(stages, X, Qwarm, optimum[h + s:], len(stages), config, job["metric"], hit_function=schedule_hit)
        state(phase="test", method=kind)
        rows = evaluate(stages, X, Qtest, optimum[h:h + s], len(stages), config, job["metric"],
                        hit_function=schedule_hit)
        for row in rows:
            append_row(output / "test-queries.csv", dict(display_method=kind, **row))
        measured.append(dict(case_id=job["case_id"], metric=job["metric"], index_seed=job["seed"],
                             display_method=kind, status="complete", blocks=sizes, swapped=swapped,
                             schedule=winner["schedule"], index_bytes=allocated, tuning=winner,
                             **quality_summary(rows)))
        write_json(output / "main-results.json", measured)
        print(f"TESTED {job['id']} {kind}: {json.dumps(measured[-1])}", flush=True)
        del stages, index
        gc.collect()
    if arrays_hash(X) != info["dataset_sha256"] or arrays_hash(Q) != info["queries_sha256"]:
        raise ValueError("Evaluation arrays changed")
    state(phase="complete")
    names = ("provenance.json", "dataset.json", "tuning-configurations.json", "tuning-queries.csv",
             "frozen-selections.json", "selection-frozen-before-test.json", "main-results.json", "test-queries.csv")
    write_json(output / "completion.json", dict(completed=time.time(),
                                              artifacts={name: digest(output / name) for name in names}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--worker", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--exploratory", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if not args.exploratory:
        require_clean_tree()
    if (numba.get_num_threads() != plan["protocol"]["build_threads"]
            or int(os.environ.get("PIN_CORE", -1)) != plan["protocol"]["pin_core"]):
        raise ValueError("CPU settings differ from the frozen accuracy protocol")
    args.output.mkdir(parents=True, exist_ok=False)
    run(plan, plan["jobs"][args.worker], args.output, args.exploratory)
