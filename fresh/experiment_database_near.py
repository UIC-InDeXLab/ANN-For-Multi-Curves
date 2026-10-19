"""Exact database-wide near fractions with bounded-memory, hash-verified streaming."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numba
from numba import njit, prange
import numpy as np

from curveset import core
from fresh.e1 import arrays_hash, write_rows
from fresh.nba_full import atomic_json, digest
from fresh.poc import implicit_dataset, jobs
from fresh.regimes import endpoint_rows_covered
from fresh.tuple_quality import exact_distance


@njit(cache=True, parallel=True)
def count_near(X, queries, radii):
    if len(queries) != len(radii):
        raise ValueError("Each query requires its own radius")
    if X.shape[1] != queries.shape[1] or X.shape[3] != queries.shape[3]:
        raise ValueError("Database and query curve counts and dimensions must match")
    for radius in radii:
        if not np.isfinite(radius) or radius < 0:
            raise ValueError("Query radii must be finite and nonnegative")
    counts = np.zeros(len(queries), np.int64)
    for q in prange(len(queries)):
        radius = radii[q]
        safe_radius = np.nextafter(radius, np.inf)
        count = 0
        for p in range(len(X)):
            if (endpoint_rows_covered(X[p], queries[q], safe_radius) and
                    core.endpoint_lb(X[p], queries[q]) <= radius and
                    exact_distance(X[p], queries[q]) <= radius):
                count += 1
        counts[q] = count
    return counts


def dataset_digest_header(implicit):
    result = hashlib.sha256()
    result.update(str(np.dtype(np.float64)).encode())
    result.update(str((implicit.n, implicit.k, implicit.m, 2)).encode())
    return result


def run(args):
    full = args.results / "collection-reference-20261017" / args.case / "summary.json"
    if full.exists():
        summary = json.loads(full.read_text())
        if summary["complete"] and summary["evaluated_queries"] == 1000:
            print(f"REUSED exact full-collection result: {args.case}", flush=True)
            return summary
    output = args.results / "database-near-20261017" / args.case
    finished = output / "completion.json"
    if finished.exists():
        for name, expected in json.loads(finished.read_text())["artifacts"].items():
            if digest(output / name) != expected:
                raise ValueError(f"Completed database fraction changed: {output / name}")
        return json.loads((output / "summary.json").read_text())
    reference_root = args.results / ("fresh-selectivity-expanded-20261013"
                                    if args.case.startswith("frequency-") else "fresh-synthetic-expanded-20261012")
    source = json.loads((reference_root / "config.json").read_text())
    if source["d"] != 2:
        raise ValueError("The frozen implicit generator supports two-dimensional data")
    choices = [row for row in jobs(source) if row["case_id"] == args.case]
    if not choices:
        raise ValueError(f"Unknown frozen case: {args.case}")
    case = min(choices, key=lambda row: row["index_seed"])
    reference = reference_root / case["id"]
    info = json.loads((reference / "dataset.json").read_text())
    implicit, Q, generated = implicit_dataset(source, case)
    if arrays_hash(Q) != info["queries_sha256"] or generated["dataset_seed"] != info["dataset_seed"]:
        raise ValueError("Reconstructed queries differ from the frozen experiment")
    truth_path = reference / "ground-truth.npz"
    with np.load(truth_path, allow_pickle=False) as truth:
        if not np.array_equal(Q, truth["queries"]):
            raise ValueError("Frozen ground-truth query order changed")
        h, s = source["tuning_queries"], source["test_queries"]
        optimum = truth["distances"][h:h + s].copy()
    if s != 1000 or len(optimum) != 1000 or np.any(~np.isfinite(optimum)) or np.any(optimum < 0):
        raise ValueError("Exactly 1000 finite reference optimum distances are required")
    output.mkdir(parents=True, exist_ok=False)
    identity = dict(
        case=args.case, n=implicit.n, queries=s, factor=1.25, tolerance=1e-8,
        source=digest(Path(__file__)), generator_source=digest(Path(__file__).with_name("poc.py")),
        reference=str(reference), ground_truth_sha256=digest(truth_path),
        dataset_sha256=info["dataset_sha256"], queries_sha256=info["queries_sha256"],
        threads=numba.get_num_threads(), chunk_entries=args.chunk,
        method="exact distance with sound endpoint rejection; every database entry considered")
    atomic_json(output / "manifest.json", identity)
    queries, radii = Q[h:h + s], 1.25 * optimum + 1e-8
    counts = np.zeros(s, np.int64)
    hasher = dataset_digest_header(implicit)
    started = time.perf_counter()
    for start in range(0, implicit.n, args.chunk):
        stop = min(implicit.n, start + args.chunk)
        X = implicit.materialize(np.arange(start, stop))
        hasher.update(X.tobytes())
        counts += count_near(X, queries, radii)
        del X
        atomic_json(output / "progress.json", dict(
            case=args.case, phase="counting", database_entries_processed=stop,
            database_entries_total=implicit.n, queries=s, elapsed_seconds=time.perf_counter() - started))
        print(f"{args.case}: all {s} queries against {stop}/{implicit.n} entries", flush=True)
    if hasher.hexdigest() != info["dataset_sha256"]:
        raise ValueError("Streamed database differs from the frozen dataset")
    if np.any(counts < 1) or np.any(counts > implicit.n):
        raise ValueError("A query must have at least its exact nearest neighbor within 1.25d*")
    rows = [dict(query=q, nn_distance=float(optimum[q]), radius=float(radii[q]),
                 database_entries=implicit.n, database_within_125=int(counts[q]),
                 database_within_125_fraction=float(counts[q] / implicit.n)) for q in range(s)]
    write_rows(output / "queries.csv", rows)
    summary = dict(
        **identity, complete=True, evaluated_queries=s, evaluated_database_entries="all",
        mean_database_within_125=float(np.mean(counts)),
        mean_database_within_125_fraction=float(np.mean(counts / implicit.n)),
        min_database_within_125=int(np.min(counts)), max_database_within_125=int(np.max(counts)),
        elapsed_seconds=time.perf_counter() - started)
    atomic_json(output / "summary.json", summary)
    atomic_json(finished, dict(artifacts={
        name: digest(output / name) for name in ("manifest.json", "summary.json", "queries.csv")}))
    return summary


def refresh(args, summary):
    from fresh.render_fh_blocking_paper import first_seed_joint_view, paired_comparison_table, write_collection_reference
    records = first_seed_joint_view(json.loads(args.records.read_text()))
    references = args.results / "collection-reference-20261017"
    near = args.results / "database-near-20261017"
    target = args.paper / "experiments/tables/first-hit-comparison.tex"
    target.write_text("% Generated from frozen measurements and exact database-wide near counts.\n" +
                      paired_comparison_table(records, references, near))
    write_collection_reference(records, references, args.paper)
    log_path = args.results / "database-near-20261017" / f"{args.case}-paper-build.log"
    log_path.parent.mkdir(exist_ok=True)
    with log_path.open("w") as log:
        subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
                       cwd=args.paper, stdout=log, stderr=subprocess.STDOUT, check=True)
    atomic_json(log_path.with_name(f"{args.case}-refreshed.json"), dict(
        case=args.case, complete=True, database_near_percent=100 * summary["mean_database_within_125_fraction"],
        table_sha256=digest(target), pdf_sha256=digest(args.paper / "main.pdf")))
    print(f"REFRESHED {args.case}: DB Near = "
          f"{100 * summary['mean_database_within_125_fraction']:.8f}% ; paper compiled", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--chunk", type=int, default=16384)
    parser.add_argument("--paper", type=Path)
    parser.add_argument("--records", type=Path)
    args = parser.parse_args()
    if args.chunk <= 0 or bool(args.paper) != bool(args.records):
        parser.error("Use a positive chunk size and supply both --paper and --records for refreshes")
    result = run(args)
    if args.paper:
        refresh(args, result)
