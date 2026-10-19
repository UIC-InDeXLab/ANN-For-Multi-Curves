"""FH tuning on approximation success, with selections frozen before held-out tests."""
import argparse
import gc
import json
from math import factorial, prod
import os
from pathlib import Path
import signal
import subprocess
import time

import numba
import numpy as np

from curveset.orders import partitions
from fresh.e1 import append_row, arrays_hash, write_json
from fresh.experiment_baselines import load_dataset
from fresh.experiment_first_hit_quality import canonical_family, evaluate, load_reference, quality_summary
from fresh.experiment_first_entry import require_clean_tree
from fresh.feasibility_first_hit import build_tables, index_bytes
from fresh.nba_full import digest


def selection_key(row):
    mean = row["tuning_mean_ratio"]
    mean = float("inf") if mean in (None, "infinity") else mean
    return (-row["tuning_a125"], -row["tuning_a110"], -row["tuning_coverage"],
            mean, row["index_bytes"], row["L"], row["delta"])


def prefix_quality(rows, L):
    eligible = [r for r in rows if r["found"] and not r["timeout"] and r["first_table"] <= L]
    ratios = [float("inf") if r["ratio"] == "infinity" else r["ratio"] for r in eligible]
    mean = float(np.mean(ratios)) if ratios else None
    return dict(
        tuning_a125=sum(r["within_125"] for r in eligible) / len(rows),
        tuning_a110=sum(r["within_110"] for r in eligible) / len(rows),
        tuning_coverage=len(eligible) / len(rows),
        tuning_mean_ratio=mean if mean is None or np.isfinite(mean) else "infinity")


def serving_bytes(index, L):
    return int(index_bytes(index, L) - index.base.shifts.nbytes + index.base.shifts[:L].nbytes)


def tuning_reference(job, config, info):
    path = Path(job["tuning_reference"])
    completion = json.loads((path / "completion.json").read_text())
    names = ("provenance.json", "dataset.json", "tuning-configurations.json", "frozen-selections.json")
    for name in names:
        if digest(path / name) != completion["artifacts"][name]:
            raise ValueError(f"Frozen tuning reference changed: {path / name}")
    provenance = json.loads((path / "provenance.json").read_text())
    if provenance["index_seed"] != job["seed"] or provenance["metric"] != job["metric"]:
        raise ValueError("Tuning reference seed or metric differs")
    for key in ("tuning_queries", "test_queries", "warmup_queries"):
        if provenance["protocol"][key] != config[key]:
            raise ValueError(f"Tuning reference split differs: {key}")
    for key in ("dataset_sha256", "queries_sha256", "nba_reference_artifacts"):
        if provenance["dataset"].get(key) != info.get(key):
            raise ValueError(f"Tuning reference dataset differs: {key}")
    trials = json.loads((path / "tuning-configurations.json").read_text())
    baseline = json.loads((path / "frozen-selections.json").read_text())["store_all"]
    if baseline not in trials or "store_all" not in baseline["labels"]:
        raise ValueError("Store-all reference is not a recorded tuning configuration")
    return trials, dict(path=str(path.resolve()), completion_sha256=digest(path / "completion.json"),
                        baseline=baseline)


def choose_configuration(trials, method, config, reference=None):
    eligible = [row for row in trials
                if method in row["labels"] and row["index_bytes"] <= config["index_budget_bytes"]]
    if method == "blocking" and config.get("interior_blocking", False):
        eligible = [row for row in eligible
                    if 1 < (factorial(sum(row["blocks"])) // prod(factorial(b) for b in row["blocks"])
                            if row["swapped"] else prod(factorial(b) for b in row["blocks"]))
                    < factorial(sum(row["blocks"]))]
    if reference is not None:
        policy = config["storage_selection"]
        losses = (policy["max_a125_loss"], policy["max_coverage_loss"])
        if not all(np.isfinite(v) and 0 <= v <= 1 for v in losses):
            raise ValueError("Storage-selection losses must be finite fractions between zero and one")
        baseline = reference["baseline"]
        eligible = [row for row in eligible
                    if row["tuning_a125"] + 1e-12 >= baseline["tuning_a125"] - losses[0]
                    and row["tuning_coverage"] + 1e-12 >= baseline["tuning_coverage"] - losses[1]]
    elif "storage_selection" in config:
        raise ValueError("Storage-aware selection requires a frozen Store-all tuning reference")
    if not eligible:
        raise ValueError(f"No tuning configuration meets the storage/quality requirements for {method}")
    if reference is not None:
        return min(eligible, key=lambda row: (row["index_bytes"], selection_key(row)))
    return min(eligible, key=selection_key)


def family_specs(job):
    if job["study"] == "nba":
        families = {}
        for sizes in partitions(5):
            for swapped in (False, True):
                key = canonical_family(sizes, swapped)
                labels = ["blocking"]
                if key == ((1, 1, 1, 1, 1), False):
                    labels.append("probe_all")
                if key == ((5,), False):
                    labels.append("store_all")
                families[key] = labels
        return [(sizes, swapped, labels) for (sizes, swapped), labels in families.items()]
    family = job["frozen_family"]
    return [(tuple(family["blocks"]), bool(family["swapped"]), ["blocking"])]


def load_job(plan, job):
    if job["study"] == "nba":
        config = json.loads(Path(job["nba_config"]).read_text())
        case = dict(config["case"], kind="nba", case_id=config["case"]["id"], index_seed=job["seed"])
        X, Q, info, _, optimum = load_dataset(config, case)
    else:
        config, _, X, Q, optimum, info, truth_hash = load_reference(job)
        info = dict(info, exact_ground_truth_sha256=truth_hash)
        case = job["case"]
    config.update(plan["protocol"])
    config.update(job.get("protocol_overrides", {}))
    return config, case, X, Q, optimum, info


def run(plan, job, output, exploratory=False):
    config, case, X, Q, optimum, info = load_job(plan, job)
    seed, metric = job["seed"], job["metric"]
    h, s = config["tuning_queries"], config["test_queries"]
    if len(Q) <= h + s or len(optimum) != len(Q):
        raise ValueError("Frozen tuning/test/warm-up splits are incomplete")
    Qt, Qtest, Qwarm = Q[:h], Q[h:h + s], Q[h + s:]
    trials, reference = tuning_reference(job, config, info) if "tuning_reference" in job else ([], None)
    if reference is not None and "storage_selection" not in config:
        raise ValueError("A tuning reference requires an explicit storage-selection policy")
    identity = dict(job=job, protocol=config, dataset=info, metric=metric, index_seed=seed,
                    source_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                    source_sha256=digest(Path(__file__)), actual_numba_threads=numba.get_num_threads(),
                    pin_core=os.environ.get("PIN_CORE"),
                    exploratory=exploratory, tuning_reference=reference,
                    selection=("minimize serving storage within explicit A1.25 and coverage losses against frozen Store-all"
                               if reference is not None else
                               "maximize tuning A1.25, then A1.1, coverage; minimize mean ratio and serving storage"))
    if exploratory:
        identity["source_tree_status"] = subprocess.check_output(["git", "status", "--porcelain"], text=True)
        sources = subprocess.check_output(["git", "ls-files", "curveset/*.py", "fresh/*.py"], text=True).splitlines()
        identity["source_files_sha256"] = {name: digest(Path(name)) for name in sources}
    write_json(output / "provenance.json", identity)
    dataset_info = dict(info, **case)
    dataset_info.update(case_id=job["case_id"], metric=metric)
    write_json(output / "dataset.json", dataset_info)

    def state(**details):
        write_json(output / "active_configuration.json", dict(seed=seed, **details))

    for sizes, swapped, labels in family_specs(job):
        family = "+".join(map(str, sizes)) + ("T" if swapped else "")
        for delta in config["quality_deltas"]:
            state(phase="building-tuning", method=family, delta=delta)
            signal.alarm(config["build_deadline_seconds"])
            try:
                index = build_tables(X, delta, config["quality_Lmax"], seed, sizes, swapped,
                                     config["index_budget_bytes"])
            finally:
                signal.alarm(0)
            if not index.tables:
                append_row(output / "excluded.csv", dict(family=family, delta=delta,
                                                         reason="no table fits serving storage"))
                continue
            evaluate(index, X, Qwarm, np.full(len(Qwarm), np.inf), 1, config, metric)
            state(phase="tuning", method=family, delta=delta)
            rows = evaluate(index, X, Qt, optimum[:h], len(index.tables), config, metric)
            for row in rows:
                append_row(output / "tuning-queries.csv",
                           dict(family=family, delta=delta, source_L=len(index.tables), **row))
            for L in range(1, len(index.tables) + 1):
                trials.append(dict(family=family, blocks=list(sizes), swapped=swapped, labels=labels,
                                   delta=float(delta), L=L, index_bytes=serving_bytes(index, L),
                                   **prefix_quality(rows, L)))
            write_json(output / "tuning-configurations.json", trials)
            print(f"TUNED {job['id']} {family} delta={delta} "
                  f"A125={max(t['tuning_a125'] for t in trials if t['family'] == family):.3f}", flush=True)
            del index
            gc.collect()
    selected = {}
    for method in job["methods"]:
        selected[method] = choose_configuration(trials, method, config, reference)
    write_json(output / "frozen-selections.json", selected)
    write_json(output / "selection-frozen-before-test.json", dict(time=time.time(), methods=list(selected)))

    measured, cache = [], {}
    for method, winner in selected.items():
        key = f"{winner['family']}-d{winner['delta']:g}-L{winner['L']}"
        if key not in cache:
            state(phase="building-test", method=method, delta=winner["delta"], L=winner["L"])
            index = build_tables(X, winner["delta"], winner["L"], seed, winner["blocks"], winner["swapped"],
                                 config["index_budget_bytes"])
            if len(index.tables) != winner["L"] or index_bytes(index, winner["L"]) != winner["index_bytes"]:
                raise ValueError("Selected serving index does not match its frozen tuning projection")
            evaluate(index, X, Qwarm, np.full(len(Qwarm), np.inf), winner["L"], config, metric)
            state(phase="test", method=method, delta=winner["delta"], L=winner["L"])
            rows = evaluate(index, X, Qtest, optimum[h:h + s], winner["L"], config, metric)
            cache[key] = rows
            del index
            gc.collect()
        rows = cache[key]
        for row in rows:
            append_row(output / "test-queries.csv", dict(display_method=method, config_id=key, **row))
        measured.append(dict(
            case_id=job["case_id"], study=job["study"], metric=metric, index_seed=seed,
            display_method=method, status="complete", quality_tuned=True,
            blocks=winner["blocks"], swapped=winner["swapped"], delta=winner["delta"], L=winner["L"],
            index_bytes=winner["index_bytes"], tuning=winner, **quality_summary(rows)))
        write_json(output / "main-results.json", measured)
    if arrays_hash(X) != info["dataset_sha256"] or arrays_hash(Q) != info["queries_sha256"]:
        raise ValueError("Evaluation arrays changed")
    state(phase="complete", method="all")
    names = ("provenance.json", "dataset.json", "tuning-configurations.json", "tuning-queries.csv", "frozen-selections.json",
             "selection-frozen-before-test.json", "main-results.json", "test-queries.csv")
    write_json(output / "completion.json", dict(
        job=job["id"], source_revision=identity["source_revision"], completed=time.time(),
        artifacts={name: digest(output / name) for name in names}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--worker", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--exploratory", action="store_true",
                        help="Allow a working tree with changes; record source hashes and label results exploratory")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if not args.exploratory:
        require_clean_tree()
    if (numba.get_num_threads() != plan["protocol"]["build_threads"] or
            int(os.environ.get("PIN_CORE", -1)) != plan["protocol"]["pin_core"]):
        raise ValueError("CPU settings differ from the frozen quality-tuning protocol")
    args.output.mkdir(parents=True, exist_ok=False)
    run(plan, plan["jobs"][args.worker], args.output, exploratory=args.exploratory)
