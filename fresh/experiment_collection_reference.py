"""Exact full-collection distance reference, outside all index query timings."""
import argparse
import json
import time
from pathlib import Path

import numba
import numpy as np

from fresh.e1 import append_row, write_json
from fresh.experiment_first_hit_quality import load_reference
from fresh.nba_full import digest
from fresh.poc import jobs
from fresh.tuple_quality import all_tuple_statistics_radii


def main(args):
    source = json.loads((args.reference_root / "config.json").read_text())
    choices = [case for case in jobs(source) if case["case_id"] == args.case]
    if not choices:
        raise ValueError(f"Case absent from frozen reference: {args.case}")
    case = min(choices, key=lambda item: item["index_seed"])
    job = dict(case=case, case_id=case["case_id"], reference_root=str(args.reference_root),
               reference_job=case["id"])
    config, reference, X, Q, optimum, info, truth_hash = load_reference(job)
    h, total = config["tuning_queries"], min(config["test_queries"], args.limit)
    Qs, optimum = Q[h:h + total], optimum[h:h + total]
    if np.any(optimum <= 0):
        raise ValueError("This mean-ratio reference requires positive exact optimum distances")
    identity = dict(case=args.case, n=len(X), queries=total, metric="discrete-Frechet-bottleneck",
                    source=digest(Path(__file__)), distance_source=digest(Path(__file__).with_name("tuple_quality.py")),
                    reference=str(reference), ground_truth_sha256=truth_hash,
                    dataset_sha256=info["dataset_sha256"], queries_sha256=info["queries_sha256"],
                    factor=1.25, threads=numba.get_num_threads(), evaluated_database_entries="all")
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "manifest.json", identity)
    all_stats = []
    started = time.perf_counter()
    for start in range(0, total, args.batch):
        stop = min(total, start + args.batch)
        stats = all_tuple_statistics_radii(X, Qs[start:stop], 1.25 * optimum[start:stop], 1e-8)
        if not np.allclose(stats[:, 1], optimum[start:stop], rtol=0, atol=1e-8):
            raise AssertionError("Full distance scan disagrees with frozen exact nearest-neighbor distances")
        all_stats.extend(stats.tolist())
        for q, values in enumerate(stats, start):
            append_row(args.output / "queries.csv", dict(
                query=q, database_entries=len(X), optimal_distance=float(optimum[q]),
                mean_database_distance=float(values[0]), mean_database_ratio=float(values[0] / optimum[q]),
                nearest_distance=float(values[1]), farthest_distance=float(values[2]),
                database_within_125=int(values[3]), database_within_125_fraction=float(values[3] / len(X))))
        data = np.asarray(all_stats)
        write_json(args.output / "summary.json", dict(
            **identity, evaluated_queries=stop, complete=stop == total,
            mean_database_distance=float(np.mean(data[:, 0])),
            mean_database_ratio=float(np.mean(data[:, 0] / optimum[:stop])),
            mean_database_within_125=float(np.mean(data[:, 3])),
            mean_database_within_125_fraction=float(np.mean(data[:, 3] / len(X))),
            min_database_within_125=int(np.min(data[:, 3])), max_database_within_125=int(np.max(data[:, 3])),
            elapsed_seconds=time.perf_counter() - started))
        print(f"{args.case}: {stop}/{total} full-collection queries, "
              f"{time.perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument("--case", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()
    if args.limit <= 0 or args.batch <= 0:
        parser.error("Query limit and batch size must be positive")
    main(args)
