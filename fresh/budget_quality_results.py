"""Validate approximate hard-budget outcomes from actual on-time query returns."""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from fresh import quality_paper_results as quality
from fresh.nba_full import digest

METHODS = ("fh_blocking", "fh_probe_all", "fh_store_all", "cl_blocking")
LABELS = {"fh_blocking": "Blocking FH", "fh_probe_all": "Probe-all FH",
          "fh_store_all": "Store-all FH", "cl_blocking": "Blocking CL"}
BUDGETS = ("1gb-10ms", "256mb-1ms", "16gb-10ms", "16gb-100ms")


def test_metrics(rows, deadline, expected_queries=1000):
    if not np.isfinite(deadline) or deadline < 0:
        raise ValueError("Budget deadline must be finite and nonnegative")
    if len(rows) != expected_queries or sorted(int(r["query"]) for r in rows) != list(range(expected_queries)):
        raise ValueError("Budget test requires exactly one return record per held-out query")
    normalized, late, empty, successes = [], [], [], [[], []]
    for row in rows:
        elapsed, optimum = float(row["seconds"]), float(row["nn_distance"])
        timeout = quality.flag(row["timeout"]) or elapsed > deadline
        found = int(row["raw_id"]) >= 0 and not timeout
        distance = float(row["raw_distance"]) if found else None
        normalized.append(dict(row, id=int(row["raw_id"]) if found else -1,
                               distance=distance, timeout=timeout))
        late.append(timeout)
        empty.append(not found and not timeout)
        for factor, values in zip((1.1, 1.25), successes):
            values.append(bool(found and (distance == 0 if optimum == 0
                                          else distance <= factor * optimum + 1e-8)))
    stats = quality.metrics(normalized)
    stats.update(a110=float(np.mean(successes[0])), a125=float(np.mean(successes[1])),
                 late=float(np.mean(late)), empty_only=float(np.mean(empty)))
    return stats


def load(root):
    records, cases = [], {}
    for folder in sorted(Path(root).glob("budget-*seed*")):
        if ".interrupted" in folder.name or not (folder / "completion.json").exists():
            continue
        completion = json.loads((folder / "completion.json").read_text())
        for name, expected in completion["artifacts"].items():
            if digest(folder / name) != expected:
                raise ValueError(f"Budget artifact changed: {folder / name}")
        main = json.loads((folder / "main-results.json").read_text())
        protocol, case = main["protocol"], main["case"]
        if (protocol["approximation_factor"] != 1.25 or protocol["required_success"] != .9 or
                protocol["tuning_acceptance_recall"] != .95 or protocol["test_queries"] != 1000 or
                protocol["tuning_queries"] != 500 or protocol["L_max"] != 8 or
                set(protocol["methods"]) != set(METHODS)):
            raise ValueError("Unexpected approximation-budget protocol; do not mix historical exact targets")
        if case["n"] != 1000000 or case["k"] not in range(4, 13):
            raise ValueError("Unexpected collection size or curve count in the hard-budget study")
        budgets = {b["name"]: b for b in protocol["budgets"]}
        if set(budgets) != set(BUDGETS):
            raise ValueError("Four declared serving budgets are required")
        outcomes = main["results"]
        keys = [(row["budget"], row["method"]) for row in outcomes]
        if len(keys) != 16 or len(set(keys)) != 16 or set(keys) != {
                (budget, method) for budget in BUDGETS for method in METHODS}:
            raise ValueError("Each completed case must account for all sixteen method/budget outcomes")
        groups = defaultdict(list)
        with (folder / "raw-test-queries.jsonl").open() as handle:
            for line in handle:
                row = json.loads(line)
                groups[row["budget"], row["method"]].append(row)
        trials = json.loads((folder / "tuning-configurations.json").read_text())
        selected_keys = {(row["budget"], row["config_id"]) for row in outcomes if row["test_evaluated"]}
        tuning_groups = defaultdict(list)
        with (folder / "raw-tuning-queries.jsonl").open() as handle:
            for line in handle:
                row = json.loads(line)
                key = (row["budget"], row["config_id"])
                if key in selected_keys:
                    tuning_groups[key].append(row)
        for outcome in outcomes:
            budget, method = outcome["budget"], outcome["method"]
            limits = budgets[budget]
            base = dict(k=case["k"], seed=case["index_seed"], method=method, budget=budget,
                        limits=limits, source=str(folder), status=outcome["status"],
                        tested=bool(outcome["test_evaluated"]), feasible=bool(outcome["feasible"]))
            rows = groups.pop((budget, method), [])
            candidates = [t for t in trials if t["budget"] == budget and method in t["method_labels"]
                          and t["status"] == "qualified"]
            if not outcome["test_evaluated"]:
                if (rows or candidates or outcome["feasible"] or
                        outcome["status"] != "no_qualifying_configuration_in_bounded_search"):
                    raise ValueError("Untested budget outcomes must be explicit search exclusions")
                records.append(dict(base, metrics=None, selection=None))
                continue
            if outcome["index_bytes"] > limits["index_budget_bytes"]:
                raise ValueError("A tested serving index exceeds its declared budget")
            if not candidates:
                raise ValueError("Tested method lacks a qualified tuning selection")
            best = min(candidates, key=lambda t: (t["tuning_summary"]["median_seconds"],
                                                  t["index_bytes"], t["config_id"]))
            if best["config_id"] != outcome["config_id"]:
                raise ValueError("Budget winner was not the fastest qualified tuning configuration")
            if best["tuning_summary"]["queries"] != 500 or best["tuning_summary"]["success_125"] < .95:
                raise ValueError("Selected configuration fails the tuning guard")
            tuning = test_metrics(tuning_groups[budget, outcome["config_id"]],
                                  limits["query_deadline_seconds"], expected_queries=500)
            if tuning["a125"] < .95 or abs(tuning["a125"] - best["tuning_summary"]["success_125"]) > 1e-12:
                raise ValueError("Raw tuning returns disagree with the selection guard")
            if abs(tuning["median_ms"] - 1000 * best["tuning_summary"]["median_seconds"]) > 1e-7:
                raise ValueError("Raw tuning times disagree with the selection objective")
            if best["index_bytes"] != outcome["index_bytes"]:
                raise ValueError("Serving memory differs from the frozen selection")
            if any(row["config_id"] != outcome["config_id"] for row in rows):
                raise ValueError("Held-out records do not belong to the frozen serving selection")
            stats = test_metrics(rows, limits["query_deadline_seconds"])
            for name, field in (("a110", "success_110"), ("a125", "success_125"),
                                ("empty", "no_answer_rate"), ("late", "timeout_rate"),
                                ("empty_only", "empty_ontime_rate")):
                if abs(stats[name] - outcome["test_summary"][field]) > 1e-12:
                    raise ValueError(f"Budget quality disagrees with actual returns: {name}")
            if (stats["a125"] >= .9) != outcome["feasible"]:
                raise ValueError("Budget feasibility disagrees with held-out on-time success")
            measurement = dict(stats, seed=case["index_seed"], index_mb=outcome["index_bytes"] / 1e6)
            records.append(dict(base, metrics=quality.summarize([measurement]),
                                measurement=measurement, selection=best))
        if groups:
            raise ValueError("Unmatched held-out records in a budget experiment")
        cases.setdefault(case["k"], set()).add(case["index_seed"])
    return dict(records=records, cases={k: sorted(seeds) for k, seeds in cases.items()})


def group_outcomes(records):
    grouped = defaultdict(list)
    for row in records:
        grouped[row["k"], row["budget"], row["method"]].append(row)
    result = {}
    for key, rows in grouped.items():
        if len({r["seed"] for r in rows}) != len(rows):
            raise ValueError("Duplicate index seed in a budget outcome")
        tested = all(r["tested"] for r in rows)
        stats = quality.summarize([r["measurement"] for r in rows]) if tested else None
        if tested:
            stats["p95_ms"] = float(np.median([r["measurement"]["p95_ms"] for r in rows]))
        result[key] = dict(
            tested=tested, feasible=all(r["feasible"] for r in rows),
            seeds=len(rows), metrics=stats, limits=rows[0]["limits"],
            status="yes" if all(r["feasible"] for r in rows) else "no" if tested else "NQ")
    return result
