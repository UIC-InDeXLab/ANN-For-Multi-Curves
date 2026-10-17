"""Summaries only for complete fresh E1 configurations."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def cluster_interval(values, seed=701):
    means = values.mean(axis=1)
    rng = np.random.default_rng(seed)
    bootstrap = means[rng.integers(0, len(means), (2000, len(means)))].mean(axis=1)
    return [float(x) for x in np.quantile(bootstrap, [0.025, 0.975])]


def summarize(root):
    config = json.loads((root / "config.json").read_text())
    collisions = pd.read_csv(root / "collisions.csv")
    result = dict(stage=config["stage"], test_queries_per_seed=config["test_queries"],
                  evaluation_finished=(root / "completion.json").exists(),
                  manifest=json.loads((root / "manifest.json").read_text()), collisions=[], configurations=[])
    for (workload, d, k), group in collisions.groupby(["workload", "d", "k"]):
        samples = [
            np.load(root / "collision-samples" / f"{workload}-d{d}-k{k}-seed{seed}.npz")["independent"]
            for seed in config["index_seeds"]
        ]
        # Same pair identifiers across seeds: keep their results together.
        paired = np.concatenate(samples, axis=1)
        expected = group.construction_joint_probability.dropna()
        result["collisions"].append(dict(
            workload=workload, d=int(d), k=int(k), mean_rate=float(paired.mean()),
            query_cluster_interval=cluster_interval(paired), bound=float(group.bound.iloc[0]),
            ratio=float(paired.mean() / group.bound.iloc[0]),
            analytical_probability=float(expected.iloc[0]) if len(expected) else None,
            mean_shared_rate=float(group.shared_rate.mean()),
            fingerprint_mismatches=int(group.fingerprint_mismatches.sum())))
    for folder in sorted(root.iterdir()):
        statuses = folder / "configurations.json"
        if not folder.is_dir() or not statuses.exists():
            continue
        groups = {}
        for row in json.loads(statuses.read_text()):
            groups.setdefault((row["method"], row["mode"]), []).append(row)
        queries = pd.read_csv(folder / "queries.csv") if (folder / "queries.csv").exists() else None
        for (method, mode), rows in groups.items():
            record = dict(case=folder.name, method=method, mode=mode,
                          seed_statuses=[dict(seed=r["seed"], status=r["status"]) for r in rows])
            if len(rows) == len(config["index_seeds"]) and all(r["status"] == "complete" for r in rows):
                group = queries[(queries.method == method) & (queries["mode"] == mode)]
                if len(group) != len(config["index_seeds"]) * config["test_queries"] or (group.status != "ok").any():
                    raise AssertionError(f"Incomplete query log: {folder.name}, {method}, {mode}")
                per_seed = group.groupby("seed").seconds.median()
                exact_pairs = group.pivot(index="query", columns="seed", values="exact_nn").to_numpy()
                success_pairs = group.pivot(index="query", columns="seed", values="within_cr").to_numpy()
                ratio_rows = group[group.found & (group.nn_distance > 0)]
                ratios = ratio_rows.returned_distance / ratio_rows.nn_distance
                bound = np.sqrt(float(group.d.iloc[0])) * float(group.delta.iloc[0])
                record.update(status="complete", queries_per_seed=config["test_queries"],
                              within_cr=float(group.within_cr.mean()), exact_nn=float(group.exact_nn.mean()),
                              within_cr_query_interval=cluster_interval(success_pairs),
                              exact_nn_query_interval=cluster_interval(exact_pairs),
                              certificate_violations=int((group.returned_distance > bound + 1e-7).sum()),
                              empty_rate=float((~group.found).mean()),
                              median_seconds=float(per_seed.median()),
                              per_seed_median_seconds=[float(x) for x in per_seed],
                              per_seed_p95_seconds=[float(g.seconds.quantile(0.95)) for _, g in group.groupby("seed")],
                              mean_candidate_recall=float(group.recall.mean()) if group.recall.notna().any() else None,
                              mean_candidate_precision=float(group.precision.mean()) if group.precision.notna().any() else None,
                              defined_candidate_recall_queries=int(group.recall.notna().sum()),
                              defined_candidate_precision_queries=int(group.precision.notna().sum()),
                              positive_nn_return_ratio_mean=float(ratios.mean()) if len(ratios) else None,
                              positive_nn_return_ratio_max=float(ratios.max()) if len(ratios) else None,
                              positive_distance_with_zero_nn=int(((group.nn_distance == 0) &
                                                                 (group.returned_distance > 0)).sum()),
                              resident_index_bytes=int(group.index_bytes.iloc[0]))
            else:
                record["status"] = "not completely evaluated"
            result["configurations"].append(record)
    result["case_resources"] = {folder.name: json.loads((folder / "resources.json").read_text())
                                for folder in root.iterdir() if folder.is_dir() and (folder / "resources.json").exists()}
    result["factorization_checks"] = len(pd.read_csv(root / "factorizations.csv"))
    (root / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    completed = [r for r in result["configurations"] if r["status"] == "complete"]
    print(f"Factorization checks: {result['factorization_checks']}")
    print(f"Collision cells: {len(result['collisions'])}; fingerprint mismatches: "
          f"{sum(r['fingerprint_mismatches'] for r in result['collisions'])}")
    print(f"Complete method/mode/case groups: {len(completed)}")
    print(f"Returned answers outside cr: {sum(r['certificate_violations'] for r in completed)}")
    for r in result["collisions"]:
        if r["workload"] == "separated-star" and r["k"] == 10:
            print(f"Star d={r['d']} k=10: observed={r['mean_rate']:.5f}, "
                  f"analytical={r['analytical_probability']:.5f}, bound={r['bound']:.5f}, "
                  f"interval={r['query_cluster_interval']}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    summarize(parser.parse_args().results)
