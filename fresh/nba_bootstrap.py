"""Re-derive NBA held-out quality intervals from the audited per-seed query logs."""
import argparse
import json
from pathlib import Path

import numpy as np

from fresh.e1 import write_json
from fresh.nba_full import digest


METHODS = {
    "blocking": "Blocking",
    "probe_all": "Probe-all",
    "store_all": "Store-all",
    "per_curve-intersection": "Per-curve intersection",
}
EXPECTED_RECALL = {
    "blocking": 0.944,
    "probe_all": 0.9493333333333334,
    "store_all": 0.9626666666666667,
    "per_curve-intersection": 0.969,
}
PAIRED_DIFFERENCES = (
    ("per_curve-intersection", "blocking", "Intersection minus Blocking"),
    ("store_all", "blocking", "Store-all minus Blocking"),
    ("probe_all", "blocking", "Probe-all minus Blocking"),
)
EPSILON = 1e-8


def _method_rows(seed_root):
    rows = json.loads((seed_root / "main-results.json").read_text())
    return rows if isinstance(rows, dict) else {row["display_method"]: row for row in rows}


def load_audited_matrices(results_root, metadata_path):
    results_root = Path(results_root).resolve()
    metadata_path = Path(metadata_path).resolve()
    metadata = json.loads(metadata_path.read_text())
    summary = json.loads((results_root / "replicated-summary.json").read_text())
    summary_rows = {row["method"]: row for row in summary["main"]}
    seed_dirs = sorted(path for path in results_root.iterdir() if path.is_dir() and path.name.startswith("seed-"))
    if not seed_dirs:
        raise ValueError("No audited seed directories were found")
    seeds = [int(path.name.split("-", 1)[1]) for path in seed_dirs]
    method_rows = {method: [] for method in METHODS}
    reference = None
    game_ids = np.asarray([row["game_id"] for row in metadata])
    if len(set(game_ids)) != 56:
        raise ValueError("Expected exactly 56 held-out games in the metadata")

    for seed, seed_root in zip(seeds, seed_dirs):
        rows = _method_rows(seed_root)
        for method in METHODS:
            row = rows.get(method)
            if row is None:
                raise ValueError(f"Missing {method} results for seed {seed}")
            status = row.get("status", row.get("test_status"))
            if status != "complete":
                raise ValueError(f"Seed {seed} method {method} is not complete")
            queries = row["queries"]
            if len(queries) != len(metadata):
                raise ValueError(f"Seed {seed} method {method} has {len(queries)} queries, expected {len(metadata)}")
            identity = [(q["query"], q["optimal_feet"]) for q in queries]
            if reference is None:
                reference = identity
            elif identity != reference:
                raise ValueError("Query identities or exact optima changed across seed logs")
            method_rows[method].append(dict(seed=seed, queries=queries))

    outputs = {}
    for method, per_seed in method_rows.items():
        recall = np.empty((len(metadata), len(seeds)), dtype=float)
        success = np.empty((len(metadata), len(seeds)), dtype=float)
        for column, entry in enumerate(sorted(per_seed, key=lambda item: item["seed"])):
            for query, row in enumerate(entry["queries"]):
                found = row["id"] >= 0 and row["distance"] is not None
                optimum = float(row["optimal_feet"])
                distance = float(row["distance"]) if found else np.inf
                recall[query, column] = float(found and distance <= optimum + EPSILON)
                success[query, column] = float(found and distance <= 1.1 * optimum + EPSILON)
        outputs[method] = dict(recall=recall, success_1_1=success)
        point = float(recall.mean())
        if not np.isclose(point, EXPECTED_RECALL[method], atol=1e-12, rtol=0.0):
            raise AssertionError(f"Point estimate mismatch for {method}: got {point}, expected {EXPECTED_RECALL[method]}")
        audited = float(summary_rows[method]["exact_recall"])
        if not np.isclose(point, audited, atol=1e-12, rtol=0.0):
            raise AssertionError(f"Replicated summary mismatch for {method}: got {point}, summary has {audited}")
    return outputs, game_ids, seeds, summary


def game_bootstrap(block_values, game_ids, *, samples):
    labels = sorted(set(game_ids))
    totals = np.asarray([block_values[game_ids == label].sum() for label in labels], dtype=float)
    counts = np.asarray([block_values[game_ids == label].size for label in labels], dtype=float)
    estimates = totals[samples].sum(axis=1) / counts[samples].sum(axis=1)
    return float(block_values.mean()), [float(x) for x in np.quantile(estimates, [0.025, 0.975])]


def build_summary(results_root, metadata_path, *, bootstrap_repetitions=10000, bootstrap_seed=20261015):
    matrices, game_ids, seeds, replicated = load_audited_matrices(results_root, metadata_path)
    rng = np.random.default_rng(bootstrap_seed)
    unique_games = sorted(set(game_ids))
    samples = rng.integers(0, len(unique_games), size=(bootstrap_repetitions, len(unique_games)))

    methods = []
    for method, label in METHODS.items():
        recall_mean, recall_interval = game_bootstrap(matrices[method]["recall"], game_ids, samples=samples)
        success_mean, success_interval = game_bootstrap(matrices[method]["success_1_1"], game_ids, samples=samples)
        methods.append(dict(
            method=method,
            label=label,
            recall_mean=recall_mean,
            recall_percent=100.0 * recall_mean,
            recall_interval=recall_interval,
            recall_interval_percent=[100.0 * x for x in recall_interval],
            success_1_1_mean=success_mean,
            success_1_1_percent=100.0 * success_mean,
            success_1_1_interval=success_interval,
            success_1_1_interval_percent=[100.0 * x for x in success_interval],
        ))

    paired = []
    for left, right, label in PAIRED_DIFFERENCES:
        mean, interval = game_bootstrap(matrices[left]["recall"] - matrices[right]["recall"], game_ids, samples=samples)
        paired.append(dict(
            comparison=label,
            minuend=left,
            subtrahend=right,
            mean=mean,
            mean_percent_points=100.0 * mean,
            interval=interval,
            interval_percent_points=[100.0 * x for x in interval],
        ))

    return dict(
        bootstrap_repetitions=int(bootstrap_repetitions),
        bootstrap_seed=int(bootstrap_seed),
        results_root=str(Path(results_root).resolve()),
        metadata_path=str(Path(metadata_path).resolve()),
        replicated_summary_sha256=digest(Path(results_root) / "replicated-summary.json"),
        metadata_sha256=digest(metadata_path),
        game_count=len(unique_games),
        query_count=len(game_ids),
        index_seeds=seeds,
        methods=methods,
        paired_recall_differences=paired,
        replicated_summary_exact_recall={row["method"]: float(row["exact_recall"]) for row in replicated["main"]},
    )


def main(args):
    results_root = Path(args.results).resolve()
    metadata_path = Path(args.metadata).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    summary = build_summary(
        results_root,
        metadata_path,
        bootstrap_repetitions=args.bootstrap_repetitions,
        bootstrap_seed=args.bootstrap_seed,
    )
    write_json(output / "summary.json", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results/nba-full-20261012")
    parser.add_argument("--metadata", default="data/nba-full-20261012/test-metadata.json")
    parser.add_argument("--output", default="results/nba-bootstrap-20261015")
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261015)
    main(parser.parse_args())
