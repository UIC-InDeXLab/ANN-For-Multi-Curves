"""First-entry index variant on the NBA collection with the frozen per-seed selections."""
import argparse
import json
import os
from pathlib import Path
import time

import numba
import numpy as np

from fresh.e1 import write_json
from fresh.experiment_baselines import load_dataset
from fresh.e1 import append_row
from fresh.experiment_first_entry import aggregate_case_rows, append_query_rows, summarize_query_rows, tune_and_test

METHODS = ("blocking", "probe_all", "store_all")


def selections_by_seed(summary_path):
    summary = json.loads(Path(summary_path).read_text())
    chosen = {}
    for row in summary["main"]:
        if row["method"] in METHODS:
            for configuration in row["configurations"]:
                chosen[(row["method"], int(configuration["seed"]))] = dict(
                    config_id=configuration["config_id"], method="blocking-family", delta=float(configuration["delta"]),
                    blocks=list(configuration["blocks"]), swapped=bool(configuration["swapped"]), L=int(configuration["L"]))
    return chosen


def main(args):
    config = json.loads(Path(args.config).read_text())
    case = dict(config["case"], kind="nba")
    X, Q, info, ids, distances = load_dataset(config, case)
    h, s = config["tuning_queries"], config["test_queries"]
    Qt, Qtest, Qwarm = Q[:h], Q[h:h + s], Q[h + s:]
    optimum_tune, optimum = distances[:h], distances[h:h + s]
    selections = selections_by_seed(config["selection_summary"])
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    rows, tuning = [], {}
    for seed in config["index_seeds"]:
        for method in METHODS:
            selection = selections[(method, seed)]
            L_max = config["store_all_L_max"] if method == "store_all" else config["L_max"]
            index, test_rows, complete, chosen, reached, candidates = tune_and_test(
                X, Qt, optimum_tune, Qtest, optimum, Qwarm, selection, seed, L_max, config)
            summary = summarize_query_rows(test_rows)
            folder = output / f"{case['id']}-seed{seed}"
            folder.mkdir(exist_ok=True)
            append_query_rows(folder / "test-queries.csv", dict(commit="nba", config_sha256="nba"), info,
                              dict(case, case_id=case["id"], index_seed=seed), method, chosen["L"],
                              selection["config_id"], optimum, test_rows)
            rows.append(dict(case=case["id"], case_id=case["id"], index_seed=seed, kind="nba", n=len(X), k=X.shape[1],
                             display_method=method, target_L=int(chosen["L"]), source_L=int(selection["L"]),
                             source_delta=selection["delta"], delta=float(chosen["delta"]),
                             tuning_recall=float(chosen["tuning_recall"]), tuning_target_reached=bool(reached),
                             blocks=selection["blocks"], swapped=selection["swapped"],
                             build_seconds=float(index.build_seconds), index_bytes=int(index.index_bytes),
                             index_mb=float(index.index_bytes / 1e6), entries=int(index.entries),
                             raw_entries=int(index.raw_entries), queries_evaluated=len(test_rows),
                             status="complete" if complete else "test timeout",
                             test_status="complete" if complete else "test timeout", **summary))
            tuning[f"{method}-{seed}"] = candidates
            append_row(folder / "results.csv", rows[-1])
            print(method, seed, chosen, rows[-1]["recall_at_1"], rows[-1]["median_ms"], flush=True)
            del index
    for seed in config["index_seeds"]:
        write_json(output / f"{case['id']}-seed{seed}" / "main-results.json", [r for r in rows if r["index_seed"] == seed])
    write_json(output / "all-results.json", rows)
    write_json(output / "tuning-grid.json", tuning)
    write_json(output / "summary.json", dict(aggregates=aggregate_case_rows(rows), exclusions=[],
                                              note="Frozen per-seed NBA block structures; grid size and table count tuned on the tuning queries."))
    write_json(output / "completion.json", dict(completed=time.time()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    main(parser.parse_args())
