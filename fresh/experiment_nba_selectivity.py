"""Exact discrete-Frechet selectivity measurements on the frozen NBA multi-curve corpus."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numba
import numpy as np
from numba import njit, prange

from curveset import core
from fresh.e1 import write_json


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def repo_commit(root):
    override = os.environ.get("EXPERIMENT_GIT_COMMIT")
    if override:
        return override, os.environ.get("EXPERIMENT_GIT_DIRTY", "").lower() in {"1", "true", "yes"}
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True).strip())
    return commit, dirty


def load_arrays(config):
    with np.load(config["database_path"], allow_pickle=False) as source:
        database = source["curves"].copy()
    with np.load(config["test_path"], allow_pickle=False) as source:
        queries = source["curves"].copy()
    with np.load(config["ground_truth_path"], allow_pickle=False) as source:
        ids = source["ids"].copy()
        distances = source["distances"].copy()
    count = config["query_count"]
    if count > len(queries) or count > len(distances):
        raise ValueError("Configured query_count exceeds available test queries")
    if database.ndim != 4 or queries.ndim != 4 or queries.shape[1:] != database.shape[1:]:
        raise ValueError("Database and test arrays must have compatible (k, m, d) shapes")
    if np.any(distances[:count] < 0) or np.any(ids[:count] < 0) or np.any(ids[:count] >= len(database)):
        raise ValueError("Ground-truth distances and ids must be valid")
    return database, queries[:count], ids[:count], distances[:count]


@njit(cache=True, parallel=True)
def query_curve_distances(query, database):
    qk = query.shape[0]
    plays = database.shape[0]
    dbk = database.shape[1]
    out = np.empty((qk, plays, dbk), np.float64)
    for flat in prange(plays * dbk):
        play = flat // dbk
        slot = flat % dbk
        for query_slot in range(qk):
            out[query_slot, play, slot] = core.frechet(query[query_slot], database[play, slot])
    return out


@njit(cache=True, parallel=True)
def query_curve_statistics(distances, radius, radius_15, radius_2, fixed_radius):
    qk, plays, dbk = distances.shape
    occurrence_r = np.empty(qk, np.float64)
    occurrence_15 = np.empty(qk, np.float64)
    occurrence_2 = np.empty(qk, np.float64)
    occurrence_fixed = np.empty(qk, np.float64)
    curve_fraction_r = np.empty(qk, np.float64)
    total_curves = plays * dbk
    for query_slot in prange(qk):
        plays_r = 0
        plays_15 = 0
        plays_2 = 0
        plays_fixed = 0
        curves_r = 0
        for play in range(plays):
            hit_r = False
            hit_15 = False
            hit_2 = False
            hit_fixed = False
            for slot in range(dbk):
                value = distances[query_slot, play, slot]
                if value <= radius:
                    hit_r = True
                    curves_r += 1
                if value <= radius_15:
                    hit_15 = True
                if value <= radius_2:
                    hit_2 = True
                if value <= fixed_radius:
                    hit_fixed = True
            plays_r += int(hit_r)
            plays_15 += int(hit_15)
            plays_2 += int(hit_2)
            plays_fixed += int(hit_fixed)
        occurrence_r[query_slot] = plays_r / plays
        occurrence_15[query_slot] = plays_15 / plays
        occurrence_2[query_slot] = plays_2 / plays
        occurrence_fixed[query_slot] = plays_fixed / plays
        curve_fraction_r[query_slot] = curves_r / total_curves
    return occurrence_r, occurrence_15, occurrence_2, occurrence_fixed, curve_fraction_r


@njit(cache=True)
def matching_exists_threshold(D, threshold):
    k = D.shape[0]
    row_of_col = np.empty(k, np.int64)
    col_of_row = np.empty(k, np.int64)
    parent = np.empty(k, np.int64)
    visited = np.empty(k, np.bool_)
    queue = np.empty(k, np.int64)
    for j in range(k):
        row_of_col[j] = -1
        col_of_row[j] = -1
    for root in range(k):
        for j in range(k):
            visited[j] = False
            parent[j] = -1
        qh = 0
        qt = 1
        queue[0] = root
        end_col = -1
        while qh < qt and end_col < 0:
            row = queue[qh]
            qh += 1
            for col in range(k):
                if D[row, col] <= threshold and not visited[col]:
                    visited[col] = True
                    parent[col] = row
                    if row_of_col[col] < 0:
                        end_col = col
                        break
                    queue[qt] = row_of_col[col]
                    qt += 1
        if end_col < 0:
            return False
        col = end_col
        while col >= 0:
            row = parent[col]
            nxt = col_of_row[row]
            row_of_col[col] = row
            col_of_row[row] = col
            col = nxt
    return True


@njit(cache=True, parallel=True)
def query_play_counts(distances, radius, radius_15):
    plays = distances.shape[1]
    count_r = 0
    count_15 = 0
    for play in prange(plays):
        if matching_exists_threshold(distances[:, play, :], radius):
            count_r += 1
        if matching_exists_threshold(distances[:, play, :], radius_15):
            count_15 += 1
    return count_r, count_15


def stats(values, scale=100.0):
    values = np.asarray(values, dtype=np.float64)
    scaled = scale * values
    return dict(
        mean=float(np.mean(scaled)),
        median=float(np.median(scaled)),
        p10=float(np.quantile(scaled, 0.10)),
        p25=float(np.quantile(scaled, 0.25)),
        p75=float(np.quantile(scaled, 0.75)),
        p90=float(np.quantile(scaled, 0.90)),
        max=float(np.max(scaled)),
    )


def count_stats(values):
    values = np.asarray(values, dtype=np.int64)
    return dict(
        mean=float(np.mean(values)),
        median=float(np.median(values)),
        p10=float(np.quantile(values, 0.10)),
        p25=float(np.quantile(values, 0.25)),
        p75=float(np.quantile(values, 0.75)),
        p90=float(np.quantile(values, 0.90)),
        max=int(np.max(values)),
        min=int(np.min(values)),
    )


def run(config_path, output_path):
    started = time.time()
    workspace = Path(__file__).resolve().parents[1]
    config = json.loads(Path(config_path).read_text())
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=False)
    numba.set_num_threads(min(config["max_threads"], numba.get_num_threads()))
    database, queries, ids, radii = load_arrays(config)
    commit, dirty = repo_commit(workspace)
    write_json(output / "config.json", config)
    write_json(
        output / "manifest.json",
        dict(
            commit=commit,
            dirty=dirty,
            config_sha256=digest(config_path),
            started=started,
            python=sys.version,
            numpy=np.__version__,
            numba=numba.__version__,
            numba_threads=numba.get_num_threads(),
            cpu_model=next(
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
        ),
    )
    rows = []
    play_hits_r = []
    play_hits_15 = []
    for query_index, query in enumerate(queries):
        if query_index % 10 == 0:
            print(f"nba-selectivity query={query_index}/{len(queries)}", flush=True)
        radius = float(radii[query_index])
        distances = query_curve_distances(query, database)
        occurrence_r, occurrence_15, occurrence_2, occurrence_fixed, curve_fraction_r = query_curve_statistics(
            distances, radius, 1.5 * radius, 2.0 * radius, config["fixed_radius_feet"]
        )
        hits_r, hits_15 = query_play_counts(distances, radius, 1.5 * radius)
        play_hits_r.append(hits_r)
        play_hits_15.append(hits_15)
        for curve_index in range(query.shape[0]):
            rows.append(
                dict(
                    query_index=query_index,
                    curve_index=curve_index,
                    nn_id=int(ids[query_index]),
                    nn_distance_feet=radius,
                    occurrence_at_r=occurrence_r[curve_index],
                    occurrence_at_1_5r=occurrence_15[curve_index],
                    occurrence_at_2r=occurrence_2[curve_index],
                    occurrence_at_5ft=occurrence_fixed[curve_index],
                    curve_fraction_at_r=curve_fraction_r[curve_index],
                    plays_within_bottleneck_r=hits_r,
                    plays_within_bottleneck_1_5r=hits_15,
                )
            )
    print(f"nba-selectivity query={len(queries)}/{len(queries)}", flush=True)
    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(
        experiment=config["experiment"],
        stage=config["stage"],
        created=time.time(),
        elapsed_seconds=time.time() - started,
        config=config,
        config_sha256=digest(config_path),
        code_commit=commit,
        code_dirty=dirty,
        database_plays=int(database.shape[0]),
        curves_per_play=int(database.shape[1]),
        query_count=int(len(queries)),
        query_curve_count=int(len(rows)),
        exact_curve_metric="curveset.core.frechet (discrete Frechet)",
        source_files=dict(
            database_path=config["database_path"],
            test_path=config["test_path"],
            ground_truth_path=config["ground_truth_path"],
            database_sha256=digest(config["database_path"]),
            test_sha256=digest(config["test_path"]),
            ground_truth_sha256=digest(config["ground_truth_path"]),
        ),
        distributions_pct=dict(
            occurrence_at_r=stats([row["occurrence_at_r"] for row in rows]),
            occurrence_at_1_5r=stats([row["occurrence_at_1_5r"] for row in rows]),
            occurrence_at_2r=stats([row["occurrence_at_2r"] for row in rows]),
            occurrence_at_5ft=stats([row["occurrence_at_5ft"] for row in rows]),
            curve_fraction_at_r=stats([row["curve_fraction_at_r"] for row in rows]),
        ),
        play_count_distributions=dict(
            within_bottleneck_r=count_stats(play_hits_r),
            within_bottleneck_1_5r=count_stats(play_hits_15),
        ),
    )
    write_json(output / "summary.json", summary)
    write_json(output / "completion.json", dict(completed=time.time(), elapsed_seconds=time.time() - started))
    return summary


def main(args):
    run(Path(args.config).resolve(), Path(args.output).resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    main(parser.parse_args())
