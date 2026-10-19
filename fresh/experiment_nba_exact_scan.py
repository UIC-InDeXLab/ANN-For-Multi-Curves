"""Timed exact-scan baseline for the NBA multi-curve benchmark.

This runner reuses the synthetic exact-scan verifier: a full database pass with a
self-contained running best, sound endpoint lower-bound pruning, and the exact
bottleneck/Frechet distance routine used elsewhere in the paper code.

Warm-up matches ``fresh.nba_full``: the first untimed tuning queries, not held-out
 test queries and not planted ground-truth parents.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numba
import numpy as np

from fresh.e1 import write_json
from fresh.nba_full import digest
from fresh.regimes import timed_queries, verify, warm


REPO_ROOT = Path(__file__).resolve().parents[1]
EPSILON = 1e-8


def clean_commit(repo_root=REPO_ROOT):
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=repo_root, text=True).strip():
        raise RuntimeError("Run only from a clean frozen source tree")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()


def cpu_model():
    return next(
        line.split(":", 1)[1].strip()
        for line in Path("/proc/cpuinfo").read_text().splitlines()
        if line.startswith("model name")
    )


def _resolve_root(root):
    return Path(root).expanduser().resolve()


def load_exact_scan_inputs(config, *, data_root=None, warmup_root=None):
    data_root = _resolve_root(data_root or config["data_directory"])
    warmup_root = _resolve_root(warmup_root or config.get("warmup_data_directory", data_root))
    database_path = data_root / config["database_file"]
    test_path = data_root / config["test_file"]
    truth_path = data_root / config["ground_truth_file"]
    warmup_path = warmup_root / config["warmup_file"]

    with np.load(database_path, allow_pickle=False) as source:
        X = source["curves"].copy()
    with np.load(test_path, allow_pickle=False) as source:
        Qtest = source["curves"].copy()
    with np.load(truth_path, allow_pickle=False) as source:
        expected_ids = source["ids"].copy()
        expected_distances = source["distances"].copy()
    with np.load(warmup_path, allow_pickle=False) as source:
        Qtune = source["curves"].copy()

    qwarm = Qtune[:config["warmup_queries"]].copy()
    if len(qwarm) != config["warmup_queries"]:
        raise ValueError("Not enough tuning queries for the requested warm-up count")
    if X.shape[1:] != (config["k"], config["m"], config["d"]):
        raise ValueError("Database multi-curve shape differs from the declared protocol")
    if Qtest.shape[1:] != X.shape[1:] or qwarm.shape[1:] != X.shape[1:]:
        raise ValueError("Query multi-curve shape differs from the database shape")
    if len(Qtest) != config["expected_test_queries"]:
        raise ValueError("Held-out test query count differs from the declared protocol")
    if len(X) != config["expected_database_size"]:
        raise ValueError("Database size differs from the declared protocol")
    if len(expected_ids) != len(Qtest) or len(expected_distances) != len(Qtest):
        raise ValueError("Ground-truth arrays do not align with the held-out test queries")
    if np.any(expected_ids < 0) or np.any(expected_ids >= len(X)):
        raise ValueError("Ground-truth parent ids are invalid for the loaded database")

    info = dict(
        data_root=str(data_root),
        warmup_data_root=str(warmup_root),
        database_sha256=digest(database_path),
        test_sha256=digest(test_path),
        ground_truth_sha256=digest(truth_path),
        warmup_sha256=digest(warmup_path),
        warmup_source=config["warmup_source"],
        database_count=len(X),
        test_query_count=len(Qtest),
        warmup_query_count=len(qwarm),
    )
    return X, Qtest, qwarm, expected_ids, expected_distances, info


def brute_query_factory(X, config):
    all_parents = np.arange(len(X), dtype=np.int64)

    def brute(Q):
        started = time.perf_counter()
        parent, distance, checked, timeout = verify(X, Q, all_parents, config, started)
        return dict(
            id=parent,
            distance=distance,
            timeout=timeout,
            candidates=len(X),
            verified=checked,
            probes=0,
            postings=0,
            seconds=time.perf_counter() - started,
        )

    return brute


def validate_exact_rows(rows, expected_ids, expected_distances, *, tolerance=EPSILON):
    validated = []
    mismatches = []
    for query, (row, expected_id, expected_distance) in enumerate(zip(rows, expected_ids, expected_distances)):
        distance = row.get("distance")
        found = row["id"] >= 0 and distance is not None
        tie_aware = bool(found and np.isclose(distance, expected_distance, atol=tolerance, rtol=0.0))
        strict = bool(tie_aware and row["id"] == int(expected_id))
        enriched = dict(
            query=query,
            expected_id=int(expected_id),
            expected_distance=float(expected_distance),
            tie_aware_exact=tie_aware,
            strict_ground_truth_id=strict,
            **row,
        )
        validated.append(enriched)
        if not tie_aware:
            mismatches.append(
                dict(
                    query=query,
                    returned_id=int(row["id"]),
                    returned_distance=None if distance is None else float(distance),
                    expected_id=int(expected_id),
                    expected_distance=float(expected_distance),
                    timeout=bool(row["timeout"]),
                )
            )
    if mismatches:
        sample = mismatches[:5]
        raise AssertionError(
            f"Exact scan disagreed with the audited ground truth on {len(mismatches)} query/repetition rows; "
            f"first mismatches: {json.dumps(sample, allow_nan=False)}"
        )
    return validated


def summarize_rows(repetitions, *, tolerance=EPSILON):
    flat = [row for repetition in repetitions for row in repetition["rows"]]
    seconds = np.asarray([row["seconds"] for row in flat], dtype=float)
    query_total = len(flat)
    query_seconds = float(seconds.sum())
    wall_seconds = float(sum(rep["wall_seconds"] for rep in repetitions))
    return dict(
        repetitions=len(repetitions),
        queries_per_repetition=len(repetitions[0]["rows"]) if repetitions else 0,
        total_queries=query_total,
        agreement_tolerance=tolerance,
        exact_agreement_count=int(sum(row["tie_aware_exact"] for row in flat)),
        strict_ground_truth_id_count=int(sum(row["strict_ground_truth_id"] for row in flat)),
        timeout_count=int(sum(bool(row["timeout"]) for row in flat)),
        mean_ms=float(seconds.mean() * 1000.0),
        median_ms=float(np.median(seconds) * 1000.0),
        p95_ms=float(np.quantile(seconds, 0.95) * 1000.0),
        throughput_qps=float(query_total / query_seconds),
        wall_seconds=wall_seconds,
        wall_throughput_qps=float(query_total / wall_seconds),
        per_repetition=[
            dict(
                repetition=rep["repetition"],
                wall_seconds=float(rep["wall_seconds"]),
                query_seconds=float(sum(row["seconds"] for row in rep["rows"])),
                exact_agreement_count=int(sum(row["tie_aware_exact"] for row in rep["rows"])),
                strict_ground_truth_id_count=int(sum(row["strict_ground_truth_id"] for row in rep["rows"])),
            )
            for rep in repetitions
        ],
    )


def run_exact_scan(config, output, *, data_root=None, warmup_root=None, repetitions=None):
    output = Path(output).resolve()
    X, Qtest, qwarm, expected_ids, expected_distances, data_info = load_exact_scan_inputs(
        config, data_root=data_root, warmup_root=warmup_root
    )
    repetition_count = int(config["repetitions"] if repetitions is None else repetitions)
    if repetition_count <= 0:
        raise ValueError("Repetitions must be positive")

    query = brute_query_factory(X, config)
    write_json(output / "active.json", dict(phase="warmup", warmup_queries=len(qwarm), **data_info))
    warm(query, qwarm)

    all_repetitions = []
    for repetition in range(1, repetition_count + 1):
        write_json(
            output / "active.json",
            dict(phase="timing", repetition=repetition, requested_queries=len(Qtest), **data_info),
        )
        started = time.perf_counter()
        rows, complete = timed_queries(
            query,
            Qtest,
            expected_distances,
            config,
            progress=lambda count, rep=repetition: write_json(
                output / "active.json",
                dict(phase="timing", repetition=rep, completed_queries=count,
                     requested_queries=len(Qtest), **data_info),
            ),
        )
        if not complete:
            raise RuntimeError(f"Exact scan timed out during repetition {repetition}")
        validated = validate_exact_rows(
            rows,
            expected_ids,
            expected_distances,
            tolerance=float(config.get("agreement_tolerance", EPSILON)),
        )
        payload = dict(repetition=repetition, rows=validated, wall_seconds=time.perf_counter() - started)
        all_repetitions.append(payload)
        write_json(output / f"repetition-{repetition:03d}.json", payload)

    summary = summarize_rows(all_repetitions, tolerance=float(config.get("agreement_tolerance", EPSILON)))
    summary.update(data_info)
    return summary


def main(args):
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    if args.repetitions is not None:
        config["repetitions"] = args.repetitions
    output = Path(args.output).resolve()
    if numba.get_num_threads() != config["build_threads"] or int(os.environ.get("PIN_CORE", "-1")) != config["pin_core"]:
        raise ValueError("CPU settings differ from the frozen exact-scan protocol")
    commit = clean_commit()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", config)
    manifest = dict(
        commit=commit,
        config_sha256=digest(config_path),
        started=time.time(),
        python=sys.version,
        numpy=np.__version__,
        numba=numba.__version__,
        cpu_model=cpu_model(),
        pin_core=config["pin_core"],
        build_threads=numba.get_num_threads(),
        implementation_sha256=digest(__file__),
        curveset_core_sha256=digest(REPO_ROOT / "curveset" / "core.py"),
        regimes_sha256=digest(REPO_ROOT / "fresh" / "regimes.py"),
        tuple_quality_sha256=digest(REPO_ROOT / "fresh" / "tuple_quality.py"),
    )
    write_json(output / "manifest.json", manifest)
    summary = run_exact_scan(
        config,
        output,
        data_root=args.data,
        warmup_root=args.warmup_data,
        repetitions=args.repetitions,
    )
    summary.update(experiment=config["experiment"], commit=commit, config_sha256=manifest["config_sha256"])
    write_json(output / "summary.json", summary)
    write_json(output / "complete.json", dict(completed=time.time(), summary_sha256=digest(output / "summary.json")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data", help="Override the configured benchmark data root")
    parser.add_argument("--warmup-data", help="Override the configured warm-up data root (must contain tune.npz)")
    parser.add_argument("--repetitions", type=int, help="Number of full held-out 1,000-query passes to run")
    main(parser.parse_args())
