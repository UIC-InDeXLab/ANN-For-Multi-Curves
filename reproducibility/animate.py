"""Render a trajectories-only animation of a completed NBA blocking retrieval."""
import argparse
import json
from pathlib import Path

import numpy as np

from fresh.nba import render


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/nba-full-20261012"))
    parser.add_argument("--results", type=Path, default=Path("results/nba-full-20261012"))
    parser.add_argument("--seed", type=int, default=2301)
    parser.add_argument("--query", type=int, default=327)
    parser.add_argument("--output", type=Path, default=Path("outputs/basketball"))
    args = parser.parse_args()
    seed = args.results / f"seed-{args.seed}"
    if not (seed / "complete.json").is_file():
        raise ValueError("The chosen seed has not completed")
    selection = json.loads((seed / "frozen-selections.json").read_text())["main"]["blocking"]
    record = json.loads((seed / f"{selection['config_id']}-test.json").read_text())
    row = record["queries"][args.query]
    if row["id"] < 0 or row["timeout"]:
        raise ValueError("This query has no completed answer; choose another query explicitly")
    with np.load(args.data / "database.npz", allow_pickle=False) as arrays:
        X, balls = arrays["curves"], arrays["balls"]
    with np.load(args.data / "test.npz", allow_pickle=False) as arrays:
        Q, query_balls = arrays["curves"], arrays["balls"]
    query_metadata = json.loads((args.data / "test-metadata.json").read_text())
    database_metadata = json.loads((args.data / "database-metadata.json").read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    distance, optimum = row["distance"], row["optimal_feet"]
    match = dict(database_index=row["id"], distance_feet=distance, rank=1,
                 approximation_ratio=distance / optimum if optimum else 1 if distance == 0 else "infinity")
    render(args.output, (Q[args.query], query_balls[args.query], query_metadata[args.query]),
           [match], X, balls, database_metadata)


if __name__ == "__main__":
    main()
