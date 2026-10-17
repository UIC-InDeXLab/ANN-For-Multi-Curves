"""Primary ANN metrics; counts are diagnostics, not separate quality metrics."""
import argparse
import csv
import json
from math import sqrt
from pathlib import Path

import numpy as np

from fresh.e1 import write_json
from fresh.regimes import approximation_summary


def wilson(successes, total):
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z*z/total
    center = (p + z*z/(2*total)) / denominator
    margin = z*sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    return [max(0, center-margin), min(1, center+margin)]


def summarize(root):
    config = json.loads((root / "config.json").read_text())
    cases = json.loads((root / "cases.json").read_text())
    table = []
    for case in cases:
        if case["status"] != "complete":
            continue
        folder = root / case["case"]
        info = json.loads((folder / "dataset.json").read_text())
        results = json.loads((folder / "results.json").read_text())
        tuning = json.loads((folder / "tuning-configurations.json").read_text())
        methods = {r["method"] for r in results}
        with (folder / "queries.csv").open() as handle:
            queries = list(csv.DictReader(handle))
        for row in results:
            out = dict(case=case["case"], kind=info["kind"], n=info["n"], k=info["k"],
                       method=row["method"], index_bytes=row["index_bytes"],
                       raw_curve_bytes=info["raw_curve_bytes"],
                       test_status=row["test_status"], index_seed=config["index_seed"],
                       stage=config["stage"], delta=row.get("delta"), L=row.get("L"),
                       mean_relevant_sets=info["mean_test_relevant_sets"])
            if row["test_status"] == "complete":
                q = [r for r in queries if r["method"] == row["method"]]
                if len(q) != config["test_queries"]:
                    raise ValueError(f"Incomplete query log: {case['case']}, {row['method']}")
                successes = sum(r["exact_nn"] == "True" for r in q)
                times = [float(r["seconds"]) for r in q]
                ratios = [float(r["ratio"]) for r in q if r["ratio"]]
                factors = [1.0, 1.01, 1.05, 1.1, 1.5, 2.0, 5.0]
                success = {
                    str(c): sum(
                        r["found"] == "True" and float(r["distance"]) <= c*float(r["nn_distance"])+1e-8
                        for r in q)/len(q)
                    for c in factors
                }
                out.update(recall_at_1=successes/len(q), recall_95_interval=wilson(successes,len(q)),
                           exact_nn_successes=successes, evaluated_queries=len(q),
                           c_success=success, coverage=sum(r["found"] == "True" for r in q)/len(q),
                           observed_target_met=successes/len(q) >= config["target_recall"],
                           median_ms=1000*float(np.median(times)), p95_ms=1000*float(np.quantile(times,.95)),
                           serial_qps=len(q)/sum(times), **approximation_summary(ratios),
                           returned_queries=len(ratios), empty_rate=sum(r["found"] != "True" for r in q)/len(q),
                           diagnostics=dict(mean_candidates=float(np.mean([int(r["candidates"]) for r in q])),
                                            mean_verified=float(np.mean([int(r["verified"]) for r in q])),
                                            mean_probes=float(np.mean([int(r["probes"]) for r in q])),
                                            mean_postings_read=float(np.mean([int(r["postings"]) for r in q if r["postings"]]))
                                            if any(r["postings"] for r in q) else None))
            table.append(out)
        for method in {r["method"] for r in tuning} - methods:
            reasons = sorted({r["status"] for r in tuning if r["method"] == method})
            table.append(dict(case=case["case"], kind=info["kind"], n=info["n"], k=info["k"],
                              method=method, test_status="no feasible tuned configuration in pilot grid",
                              reasons=reasons))
    write_json(root / "summary.json", dict(
        stage=config["stage"], cases=cases, configurations=table,
        warning="One index seed and 32 test queries per case: exploratory trends, not final accuracy certification.",
        primary_metrics=["tie-aware recall@1","c-success including empty-answer failures","median/p95 latency",
                         "index array storage bytes","conditional mean/p90 distance ratio","empty-answer rate"],
        memory_definition="index_bytes sums allocated index-array payloads, not an RSS difference. Shared raw curves are separate; case resources report total process RSS high-water.",
        c_success_note="Derived post hoc from the completed pilot with fixed generic factors 1/1.01/1.05/1.1/1.5/2/5; no retuning or change to recorded outputs. Preregister this grid for subsequent replication.",
        diagnostics=["probes","postings read","candidates","exact distances verified"],
        random_return="Context only, not a competitive ANN baseline or headline metric"))
    print("case method index_MB latency_ms p95_ms recall% ratio empty%")
    for r in table:
        if r["test_status"] == "complete":
            ratio = ("--" if r["mean_ratio"] is None else r["mean_ratio"]
                     if isinstance(r["mean_ratio"],str) else f"{r['mean_ratio']:.4f}")
            print(f"{r['case']} {r['method']} {r['index_bytes']/1e6:.2f} {r['median_ms']:.3f} "
                  f"{r['p95_ms']:.3f} {100*r['recall_at_1']:.1f} {ratio} {100*r['empty_rate']:.1f}")
        else:
            print(f"{r['case']} {r['method']} {r['test_status']}")
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    summarize(parser.parse_args().results)
