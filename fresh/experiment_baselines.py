"""Quick HNSW baselines for multi-curve retrieval.

B1: one HNSW over flattened individual curves, then strict parent-set intersection across query curves.

B2: a permutation-invariant fixed-dimensional embedding built from SimHash-style random-hyperplane bucket
sums, then one HNSW over the multi-curve sketches. This sketch approximates a Chamfer-like set similarity;
it is only a candidate generator and does not model the paper's bijective Fréchet matching objective.
Every reported answer is still the exact-distance-verified best candidate.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numba
import numpy as np

from fresh.e1 import append_row, arrays_hash, monitor_worker, terminate_worker, write_json
from fresh.poc import dataset as synthetic_dataset, tuning_target
from fresh.regimes import timed_queries, verify, warm

try:
    import hnswlib as _hnswlib
except ImportError:  # pragma: no cover - exercised through the fallback path in tests
    _hnswlib = None
_HNSWLIB_SENTINEL = object()


BASELINES = ("per-curve-hnsw-intersection", "sketch-hnsw")
TOPK_GRID = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 20000)
SKETCH_BUCKET_BITS = (2, 3, 4)
EPS = 1e-8


def digest(path, algorithm="sha256"):
    handle = hashlib.new(algorithm)
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            handle.update(chunk)
    return handle.hexdigest()


def study_jobs(config):
    jobs = []
    for case in config["cases"]:
        seeds = config["nba_index_seeds"] if case["kind"] == "nba" else config["synthetic_index_seeds"]
        for seed in seeds:
            jobs.append(dict(case, case_id=case["id"], index_seed=seed, id=f"{case['id']}-seed{seed}"))
    return jobs


def flatten_curve_table(curves):
    """Curve vectors in a fixed point order; query-curve permutation is handled by the baselines themselves."""
    return np.asarray(curves, np.float32).reshape(curves.shape[0], -1)


class BruteForceANNIndex:
    """Testing fallback for environments without hnswlib."""

    backend_name = "brute-force"

    def __init__(self, space, dim, max_elements, seed, build_threads, M, ef_construction):
        if space not in ("l2", "ip"):
            raise ValueError(f"Unsupported brute-force space: {space}")
        self.space = space
        self.dim = dim
        self.build_threads = int(build_threads)
        self.vectors = np.empty((0, dim), np.float32)
        self.labels = np.empty(0, np.int64)
        self.ef = 0

    def add_items(self, vectors, labels):
        vectors = np.asarray(vectors, np.float32)
        labels = np.asarray(labels, np.int64)
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError("Unexpected brute-force vector shape")
        self.vectors = vectors.copy()
        self.labels = labels.copy()

    def set_num_threads(self, count):
        self.query_threads = int(count)

    def set_ef(self, value):
        self.ef = int(value)

    def get_current_count(self):
        return int(len(self.labels))

    def knn_query(self, vectors, k):
        queries = np.asarray(vectors, np.float32)
        if len(self.labels) == 0:
            raise ValueError("Cannot query an empty brute-force index")
        k = min(int(k), len(self.labels))
        if self.space == "l2":
            diffs = self.vectors[None, :, :] - queries[:, None, :]
            scores = np.einsum("qnd,qnd->qn", diffs, diffs)
            order = np.argsort(scores, axis=1)[:, :k]
            distances = np.take_along_axis(scores, order, axis=1)
        else:
            scores = queries @ self.vectors.T
            order = np.argsort(-scores, axis=1)[:, :k]
            distances = 1.0 - np.take_along_axis(scores, order, axis=1)
        return self.labels[order], distances

    def serialized_size_bytes(self, path):
        return int(self.vectors.nbytes + self.labels.nbytes)


class HnswANNIndex:
    backend_name = "hnswlib"

    def __init__(self, space, dim, max_elements, seed, build_threads, M, ef_construction):
        self.space = space
        self.dim = int(dim)
        self.build_threads = int(build_threads)
        self.index = _hnswlib.Index(space=space, dim=self.dim)
        self.index.init_index(max_elements=int(max_elements), ef_construction=int(ef_construction),
                              M=int(M), random_seed=int(seed))
        self.index.set_num_threads(self.build_threads)

    def add_items(self, vectors, labels):
        self.index.add_items(np.asarray(vectors, np.float32), np.asarray(labels, np.int64))

    def set_num_threads(self, count):
        self.index.set_num_threads(int(count))

    def set_ef(self, value):
        self._ef = int(value)
        self.index.set_ef(int(value))

    def get_current_count(self):
        return int(self.index.get_current_count())

    def knn_query(self, vectors, k):
        queries = np.asarray(vectors, np.float32)
        base_ef = getattr(self, "_ef", None)
        try:
            try:
                return self.index.knn_query(queries, k=int(k))
            except RuntimeError:
                pass
            # hnswlib cannot return k results when the beam is too small: widen it a few times. The cost
            # of these retries stays inside the measured query time, and the beam is restored afterwards.
            ef = max(int(k), int(base_ef or 0))
            for _ in range(3):
                ef *= 2
                self.ef_retries = getattr(self, "ef_retries", 0) + 1
                self.index.set_ef(ef)
                try:
                    return self.index.knn_query(queries, k=int(k))
                except RuntimeError:
                    continue
            # The graph may hold fewer than k reachable neighbors (many near-identical curves): return as
            # many as it can and count these queries so the shortfall is reported.
            want = int(k)
            while want > 1:
                want = max(1, want // 2)
                try:
                    result = self.index.knn_query(queries, k=want)
                    self.short_results = getattr(self, "short_results", 0) + 1
                    return result
                except RuntimeError:
                    continue
            return self.index.knn_query(queries, k=1)
        finally:
            if base_ef is not None:
                self.index.set_ef(base_ef)

    def serialized_size_bytes(self, path):
        path = Path(path)
        try:
            self.index.save_index(str(path))
            return int(path.stat().st_size)
        finally:
            if path.exists():
                path.unlink()


def make_ann_index(space, dim, max_elements, seed, build_threads, M, ef_construction,
                   allow_fallback=False, hnswlib_module=_HNSWLIB_SENTINEL):
    module = _hnswlib if hnswlib_module is _HNSWLIB_SENTINEL else hnswlib_module
    if module is not None:
        return HnswANNIndex(space, dim, max_elements, seed, build_threads, M, ef_construction)
    if allow_fallback:
        return BruteForceANNIndex(space, dim, max_elements, seed, build_threads, M, ef_construction)
    raise RuntimeError(
        "hnswlib is required for fresh.experiment_baselines. "
        "Install it into a separate target directory and export PYTHONPATH before running.")


class RandomHyperplaneSketch:
    """Permutation-invariant bucket-sum sketch for unordered sets of curves."""

    def __init__(self, curve_dim, bucket_bits, projection_dim, seed):
        self.curve_dim = int(curve_dim)
        self.bucket_bits = int(bucket_bits)
        self.bucket_count = 1 << self.bucket_bits
        self.seed = int(seed)
        rng = np.random.default_rng(self.seed)
        self.hyperplanes = rng.standard_normal((self.bucket_bits, self.curve_dim), dtype=np.float32)
        full_dim = self.bucket_count * self.curve_dim
        if projection_dim is not None and projection_dim < full_dim:
            signs = rng.choice(np.array([-1.0, 1.0], np.float32), size=(full_dim, int(projection_dim)))
            self.projection = signs / np.sqrt(float(projection_dim))
            self.output_dim = int(projection_dim)
        else:
            self.projection = None
            self.output_dim = full_dim

    def encode_one(self, multicurve):
        vectors = flatten_curve_table(multicurve)
        bits = (vectors @ self.hyperplanes.T) >= 0
        bucket_ids = bits.astype(np.int64) @ (1 << np.arange(self.bucket_bits, dtype=np.int64))
        summed = np.zeros((self.bucket_count, self.curve_dim), np.float32)
        for index, bucket in enumerate(bucket_ids):
            summed[bucket] += vectors[index]
        flat = summed.reshape(-1)
        if self.projection is not None:
            flat = flat @ self.projection
        norm = float(np.linalg.norm(flat))
        if norm > 0:
            flat = flat / norm
        return np.asarray(flat, np.float32)

    def encode_many(self, collection):
        out = np.empty((len(collection), self.output_dim), np.float32)
        for i, multicurve in enumerate(collection):
            out[i] = self.encode_one(multicurve)
        return out


class PerCurveHnsw:
    def __init__(self, X, config, seed, workdir, backend_factory=make_ann_index, allow_fallback=False):
        started = time.perf_counter()
        self.n, self.k = X.shape[:2]
        self.seed = int(seed)
        self.query_threads = 1
        curves = flatten_curve_table(X.reshape(self.n * self.k, *X.shape[2:]))
        self.parent_of_curve = np.repeat(np.arange(self.n, dtype=np.int64), self.k)
        labels = np.arange(len(curves), dtype=np.int64)
        order = np.random.default_rng(self.seed).permutation(len(curves))
        self.index = backend_factory("l2", curves.shape[1], len(curves), self.seed,
                                     config["hnsw_build_threads"], config["hnsw_M"],
                                     config["hnsw_ef_construction"], allow_fallback=allow_fallback)
        self.index.add_items(curves[order], labels[order])
        self.index.set_num_threads(self.query_threads)
        size_path = Path(workdir) / f"per-curve-hnsw-size-{self.seed}.bin"
        self.index_bytes = self.index.serialized_size_bytes(size_path)
        self.build_seconds = time.perf_counter() - started
        self.build_threads = int(self.index.build_threads)
        self.backend_name = self.index.backend_name

    def query(self, X, Q, config, top_k):
        started = time.perf_counter()
        if self.index.get_current_count() == 0:
            return dict(id=-1, distance=np.inf, timeout=False, candidates=0, verified=0,
                        seconds=time.perf_counter() - started, empty_answer=True, top_k=int(top_k))
        curve_vectors = flatten_curve_table(Q)
        retrieved = min(int(top_k), self.index.get_current_count())
        ef = max(retrieved, config["hnsw_query_ef_floor"])
        self.index.set_num_threads(self.query_threads)
        self.index.set_ef(ef)
        candidates = None
        for qvec in curve_vectors:
            labels, _ = self.index.knn_query(qvec[None, :], retrieved)
            parents = np.unique(self.parent_of_curve[labels[0]])
            candidates = parents if candidates is None else np.intersect1d(candidates, parents, assume_unique=True)
            if time.perf_counter() - started > config["query_deadline_seconds"]:
                return dict(id=-1, distance=np.inf, timeout=True, candidates=0, verified=0,
                            seconds=time.perf_counter() - started, empty_answer=False, top_k=int(top_k))
            if len(candidates) == 0:
                return dict(id=-1, distance=np.inf, timeout=False, candidates=0, verified=0,
                            seconds=time.perf_counter() - started, empty_answer=True, top_k=int(top_k))
        best_id, best_distance, verified, timeout = verify(X, Q, candidates.astype(np.int64), config, started)
        return dict(id=best_id, distance=best_distance, timeout=timeout, candidates=int(len(candidates)),
                    verified=verified, seconds=time.perf_counter() - started, empty_answer=False,
                    top_k=int(top_k))


class SketchHnsw:
    def __init__(self, X, config, seed, bucket_bits, workdir, backend_factory=make_ann_index,
                 allow_fallback=False):
        started = time.perf_counter()
        self.seed = int(seed)
        self.bucket_bits = int(bucket_bits)
        curve_dim = X.shape[2] * X.shape[3]
        self.sketcher = RandomHyperplaneSketch(curve_dim, self.bucket_bits,
                                               config.get("sketch_projection_dim"), self.seed)
        sketches = self.sketcher.encode_many(X)
        labels = np.arange(len(sketches), dtype=np.int64)
        order = np.random.default_rng(self.seed + 1).permutation(len(sketches))
        self.index = backend_factory(config["sketch_space"], sketches.shape[1], len(sketches), self.seed,
                                     config["hnsw_build_threads"], config["hnsw_M"],
                                     config["hnsw_ef_construction"], allow_fallback=allow_fallback)
        self.index.add_items(sketches[order], labels[order])
        self.index.set_num_threads(1)
        size_path = Path(workdir) / f"sketch-hnsw-size-b{self.bucket_bits}-seed{self.seed}.bin"
        self.index_bytes = self.index.serialized_size_bytes(size_path)
        self.build_seconds = time.perf_counter() - started
        self.build_threads = int(self.index.build_threads)
        self.backend_name = self.index.backend_name
        self.sketch_dim = sketches.shape[1]

    def query(self, X, Q, config, top_c):
        started = time.perf_counter()
        retrieved = min(int(top_c), self.index.get_current_count())
        ef = max(retrieved, config["hnsw_query_ef_floor"])
        self.index.set_num_threads(1)
        self.index.set_ef(ef)
        sketch = self.sketcher.encode_one(Q)
        labels, _ = self.index.knn_query(sketch[None, :], retrieved)
        candidates = np.asarray(labels[0], np.int64)
        best_id, best_distance, verified, timeout = verify(X, Q, candidates, config, started)
        return dict(id=best_id, distance=best_distance, timeout=timeout, candidates=int(len(candidates)),
                    verified=verified, seconds=time.perf_counter() - started, empty_answer=False,
                    top_c=int(top_c), bucket_bits=self.bucket_bits)


def summarize_rows(rows):
    seconds = np.asarray([row["seconds"] for row in rows], float)
    return dict(
        recall=float(sum(row["exact_nn"] for row in rows) / len(rows)) if rows else 0.0,
        median_ms=float(np.median(seconds) * 1000.0) if len(seconds) else None,
        p95_ms=float(np.percentile(seconds, 95) * 1000.0) if len(seconds) else None,
        empty_answer_rate=float(np.mean([row.get("empty_answer", False) for row in rows])) if rows else 0.0,
        mean_candidates=float(np.mean([row["candidates"] for row in rows])) if rows else 0.0,
    )


def select_smallest_qualifying(trials, parameter_key, target, fallback_value):
    complete = [trial for trial in trials if trial["complete"]]
    qualified = [trial for trial in complete if trial["tuning_recall"] >= target]
    best = max((trial["tuning_recall"] for trial in complete), default=None)
    if qualified:
        choice = min(qualified, key=lambda row: (row[parameter_key], row["tuning_median_seconds"]))
        return dict(status="qualified", choice=choice, best_tuning_recall=best)
    fallback = next((trial for trial in trials if trial[parameter_key] == fallback_value), None)
    if fallback is not None and fallback["complete"]:
        return dict(status="not reached", choice=fallback, best_tuning_recall=best)
    return dict(status="tuning timeout", choice=None, best_tuning_recall=best)


def select_sketch_trial(trials, target, fallback_c):
    complete = [trial for trial in trials if trial["complete"]]
    qualified = [trial for trial in complete if trial["tuning_recall"] >= target]
    best = max((trial["tuning_recall"] for trial in complete), default=None)
    if qualified:
        choice = min(qualified, key=lambda row: (row["top_c"], row["tuning_median_seconds"], row["bucket_bits"]))
        return dict(status="qualified", choice=choice, best_tuning_recall=best)
    fallback = [trial for trial in complete if trial["top_c"] == fallback_c]
    if fallback:
        choice = min(fallback, key=lambda row: (-row["tuning_recall"],
                                                row["tuning_median_seconds"], row["bucket_bits"]))
        return dict(status="not reached", choice=choice, best_tuning_recall=best)
    return dict(status="tuning timeout", choice=None, best_tuning_recall=best)


def load_dataset(config, case):
    if case["kind"] != "nba":
        X, Q, info = synthetic_dataset(dict(config, dataset_seed=case.get("dataset_seed", config["dataset_seed"])), case)
        name = case.get("reference_name", f"{case['kind']}-n{case['n']}-k{case['k']}")
        reference = Path(case.get("synthetic_reference", config["synthetic_reference"])) / f"{name}-seed{case.get('reference_seed', 1701)}"
        original = json.loads((reference / "dataset.json").read_text())
        if (original["dataset_sha256"] != info["dataset_sha256"]
                or original["queries_sha256"] != info["queries_sha256"]):
            raise ValueError("Synthetic baseline arrays differ from the frozen exact-reference arrays")
        with np.load(reference / "ground-truth.npz", allow_pickle=False) as source:
            if not np.array_equal(source["queries"], Q):
                raise ValueError("Synthetic reference query order changed")
            ids = source["ids"].copy()
            distances = source["distances"].copy()
        info.update(reference_ground_truth=str(reference / "ground-truth.npz"),
                    reference_dataset_json=str(reference / "dataset.json"))
        return X, Q, info, ids, distances

    root = Path(config["nba_data"])
    if digest(root / "dataset.json") != config["nba_dataset_sha256"]:
        raise ValueError("NBA baseline data must match the audited DTW-derived corpus")
    with np.load(root / "database.npz", allow_pickle=False) as source:
        X = source["curves"].copy()
    with np.load(root / "tune.npz", allow_pickle=False) as source:
        Qt = source["curves"].copy()
    with np.load(root / "test.npz", allow_pickle=False) as source:
        Qtest = source["curves"].copy()
    tune_path = root / "tune-ground-truth.npz"
    test_path = root / "test-ground-truth.npz"
    if digest(tune_path) != config["nba_reference_artifacts"][tune_path.name]:
        raise ValueError("Frozen NBA tuning ground truth changed")
    if digest(test_path) != config["nba_reference_artifacts"][test_path.name]:
        raise ValueError("Frozen NBA test ground truth changed")
    tune_ids, tune_distances = None, None
    test_ids, test_distances = None, None
    with np.load(tune_path, allow_pickle=False) as source:
        tune_ids = source["ids"].copy()
        tune_distances = source["distances"].copy()
    with np.load(test_path, allow_pickle=False) as source:
        test_ids = source["ids"].copy()
        test_distances = source["distances"].copy()
    warm = X[:config["warmup_queries"]].copy()
    Q = np.concatenate((Qt, Qtest, warm))
    ids = np.concatenate((tune_ids, test_ids, np.arange(len(warm), dtype=np.int64)))
    distances = np.concatenate((tune_distances, test_distances, np.zeros(len(warm))))
    info = dict(dataset_sha256=arrays_hash(X), queries_sha256=arrays_hash(Q), raw_curve_bytes=X.nbytes,
                nba_corpus_sha256=config["nba_dataset_sha256"],
                nba_reference_artifacts=config["nba_reference_artifacts"],
                warmup_source="first database multi-curves")
    return X, Q, info, ids, distances


def record_phase_rows(path, phase, manifest, info, seed, baseline, trial, optimum, rows):
    for q, row in enumerate(rows):
        append_row(path, dict(
            config_id=trial["config_id"], baseline=baseline, commit=manifest["commit"],
            config_sha256=manifest["config_sha256"], dataset_sha256=info["dataset_sha256"],
            queries_sha256=info["queries_sha256"], index_seed=seed, phase=phase, query=q,
            nn_distance=float(optimum[q]), **row))


def evaluate_trial(query, Qs, optimum, config, progress=None):
    rows, complete = timed_queries(query, Qs, optimum, config, progress)
    summary = summarize_rows(rows)
    tuning_recall = summary["recall"] if complete else None
    tuning_median_seconds = float(np.median([row["seconds"] for row in rows])) if complete and rows else None
    return rows, complete, summary, tuning_recall, tuning_median_seconds


def per_curve_baseline(config, case, output, manifest, info, X, Qt, Qtest, Qwarm, truth, seed):
    write_json(output / "active_configuration.json", dict(method="per-curve-hnsw-intersection",
                                                         seed=seed, phase="building"))
    signal.alarm(config["build_deadline_seconds"])
    try:
        index = PerCurveHnsw(X, config, seed, output,
                             allow_fallback=config.get("allow_bruteforce_fallback", False))
    finally:
        signal.alarm(0)
    trials = []
    for top_k in config["per_curve_topk_grid"]:
        config_id = f"b1-k{top_k}"
        query = lambda Q, kk=top_k: index.query(X, Q, config, kk)
        write_json(output / "active_configuration.json", dict(
            method="per-curve-hnsw-intersection", seed=seed, phase="tuning",
            config_id=config_id, top_k=top_k))
        progress = lambda count, kk=top_k, cfg=config_id: write_json(output / "active_configuration.json", dict(
            method="per-curve-hnsw-intersection", seed=seed, phase="tuning",
            config_id=cfg, top_k=kk, completed_queries=count))
        warm(query, Qwarm)
        rows, complete, summary, tuning_recall, tuning_median_seconds = evaluate_trial(
            query, Qt, truth[:len(Qt)], config, progress)
        trial = dict(config_id=config_id, baseline="per-curve-hnsw-intersection", top_k=int(top_k),
                     complete=complete, tuning_recall=tuning_recall, tuning_median_seconds=tuning_median_seconds,
                     tuning_empty_answer_rate=summary["empty_answer_rate"],
                     tuning_mean_candidates=summary["mean_candidates"])
        trials.append(trial)
        record_phase_rows(output / "tuning-queries.csv", "tuning", manifest, info, seed,
                          "per-curve-hnsw-intersection", trial, truth[:len(Qt)], rows)
    write_json(output / "per-curve-tuning.json", trials)
    selection = select_smallest_qualifying(trials, "top_k", tuning_target(config), config["per_curve_topk_grid"][-1])
    if selection["choice"] is None:
        return dict(baseline="per-curve-hnsw-intersection", index_seed=seed, test_status=selection["status"],
                    tuning_status=selection["status"], best_tuning_recall=selection["best_tuning_recall"],
                    backend=index.backend_name)
    choice = selection["choice"]
    query = lambda Q, kk=choice["top_k"]: index.query(X, Q, config, kk)
    write_json(output / "active_configuration.json", dict(
        method="per-curve-hnsw-intersection", seed=seed, phase="test",
        config_id=choice["config_id"], top_k=choice["top_k"]))
    warm(query, Qwarm)
    progress = lambda count: write_json(output / "active_configuration.json", dict(
        method="per-curve-hnsw-intersection", seed=seed, phase="test",
        config_id=choice["config_id"], top_k=choice["top_k"], completed_queries=count))
    rows, complete = timed_queries(query, Qtest, truth[len(Qt):len(Qt) + len(Qtest)], config, progress)
    record_phase_rows(output / "test-queries.csv", "test", manifest, info, seed,
                      "per-curve-hnsw-intersection", choice, truth[len(Qt):len(Qt) + len(Qtest)], rows)
    backend = getattr(index, "index", None)
    write_json(output / "hnsw-shortfall.json", dict(
        ef_retries=int(getattr(backend, "ef_retries", 0)), short_results=int(getattr(backend, "short_results", 0)),
        note="Counts over tuning and test: queries needing a wider beam, and queries answered with fewer than K neighbors."))
    summary = summarize_rows(rows) if complete else dict(recall=None, median_ms=None, p95_ms=None,
                                                         empty_answer_rate=None, mean_candidates=None)
    return dict(
        baseline="per-curve-hnsw-intersection", config_id=choice["config_id"], backend=index.backend_name,
        index_seed=seed, tuning_status=selection["status"], best_tuning_recall=selection["best_tuning_recall"],
        tuning_recall=choice["tuning_recall"], chosen_top_k=choice["top_k"], build_seconds=index.build_seconds,
        build_threads=index.build_threads, index_bytes=index.index_bytes, index_size_mb=index.index_bytes / 1e6,
        test_status="complete" if complete else "test timeout", test_queries=len(rows),
        test_recall_at_1=summary["recall"], median_latency_ms=summary["median_ms"],
        p95_latency_ms=summary["p95_ms"], empty_answer_rate=summary["empty_answer_rate"],
        mean_candidates=summary["mean_candidates"])


def sketch_baseline(config, case, output, manifest, info, X, Qt, Qtest, Qwarm, truth, seed):
    trials = []
    indexes = {}
    for bucket_bits in config["sketch_bucket_bits"]:
        write_json(output / "active_configuration.json", dict(method="sketch-hnsw", seed=seed,
                                                              phase="building", bucket_bits=bucket_bits))
        signal.alarm(config["build_deadline_seconds"])
        try:
            index = SketchHnsw(X, config, seed, bucket_bits, output,
                               allow_fallback=config.get("allow_bruteforce_fallback", False))
        finally:
            signal.alarm(0)
        indexes[bucket_bits] = index
        for top_c in config["sketch_topc_grid"]:
            config_id = f"b2-b{bucket_bits}-c{top_c}"
            query = lambda Q, bb=bucket_bits, cc=top_c: indexes[bb].query(X, Q, config, cc)
            write_json(output / "active_configuration.json", dict(
                method="sketch-hnsw", seed=seed, phase="tuning", config_id=config_id,
                bucket_bits=bucket_bits, top_c=top_c))
            progress = lambda count, bb=bucket_bits, cc=top_c, cfg=config_id: write_json(
                output / "active_configuration.json",
                dict(method="sketch-hnsw", seed=seed, phase="tuning", config_id=cfg,
                     bucket_bits=bb, top_c=cc, completed_queries=count))
            warm(query, Qwarm)
            rows, complete, summary, tuning_recall, tuning_median_seconds = evaluate_trial(
                query, Qt, truth[:len(Qt)], config, progress)
            trial = dict(config_id=config_id, baseline="sketch-hnsw", bucket_bits=int(bucket_bits),
                         top_c=int(top_c), complete=complete, tuning_recall=tuning_recall,
                         tuning_median_seconds=tuning_median_seconds,
                         tuning_empty_answer_rate=summary["empty_answer_rate"],
                         tuning_mean_candidates=summary["mean_candidates"],
                         sketch_dim=indexes[bucket_bits].sketch_dim)
            trials.append(trial)
            record_phase_rows(output / "tuning-queries.csv", "tuning", manifest, info, seed,
                              "sketch-hnsw", trial, truth[:len(Qt)], rows)
    write_json(output / "sketch-tuning.json", trials)
    selection = select_sketch_trial(trials, tuning_target(config), config["sketch_topc_grid"][-1])
    if selection["choice"] is None:
        return dict(baseline="sketch-hnsw", index_seed=seed, test_status=selection["status"],
                    tuning_status=selection["status"], best_tuning_recall=selection["best_tuning_recall"],
                    backend=next(iter(indexes.values())).backend_name)
    choice = selection["choice"]
    index = indexes[choice["bucket_bits"]]
    query = lambda Q, cc=choice["top_c"]: index.query(X, Q, config, cc)
    write_json(output / "active_configuration.json", dict(
        method="sketch-hnsw", seed=seed, phase="test", config_id=choice["config_id"],
        bucket_bits=choice["bucket_bits"], top_c=choice["top_c"]))
    warm(query, Qwarm)
    progress = lambda count: write_json(
        output / "active_configuration.json",
        dict(method="sketch-hnsw", seed=seed, phase="test", config_id=choice["config_id"],
             bucket_bits=choice["bucket_bits"], top_c=choice["top_c"], completed_queries=count))
    rows, complete = timed_queries(query, Qtest, truth[len(Qt):len(Qt) + len(Qtest)], config, progress)
    record_phase_rows(output / "test-queries.csv", "test", manifest, info, seed,
                      "sketch-hnsw", choice, truth[len(Qt):len(Qt) + len(Qtest)], rows)
    summary = summarize_rows(rows) if complete else dict(recall=None, median_ms=None, p95_ms=None,
                                                         empty_answer_rate=None, mean_candidates=None)
    return dict(
        baseline="sketch-hnsw", config_id=choice["config_id"], backend=index.backend_name, index_seed=seed,
        tuning_status=selection["status"], best_tuning_recall=selection["best_tuning_recall"],
        tuning_recall=choice["tuning_recall"], chosen_bucket_bits=choice["bucket_bits"],
        chosen_top_c=choice["top_c"], sketch_dim=index.sketch_dim, build_seconds=index.build_seconds,
        build_threads=index.build_threads, index_bytes=index.index_bytes, index_size_mb=index.index_bytes / 1e6,
        test_status="complete" if complete else "test timeout", test_queries=len(rows),
        test_recall_at_1=summary["recall"], median_latency_ms=summary["median_ms"],
        p95_latency_ms=summary["p95_ms"], empty_answer_rate=summary["empty_answer_rate"],
        mean_candidates=summary["mean_candidates"])


def worker(config, case, output):
    X, Q, info, ids, distances = load_dataset(config, case)
    h, s = config["tuning_queries"], config["test_queries"]
    Qt, Qtest, Qwarm = Q[:h], Q[h:h + s], Q[h + s:]
    np.savez(output / "ground-truth.npz", queries=Q, ids=ids, distances=distances)
    write_json(output / "dataset.json", dict(info, **case, metric=config["multicurve_metric"],
                                             tuning_queries=h, test_queries=s, warmup_queries=len(Qwarm)))
    manifest = json.loads((output.parent / "manifest.json").read_text())
    results = []
    results.append(per_curve_baseline(config, case, output, manifest, info, X, Qt, Qtest, Qwarm,
                                      distances, case["index_seed"]))
    if config.get("include_sketch", False):
        results.append(sketch_baseline(config, case, output, manifest, info, X, Qt, Qtest, Qwarm,
                                       distances, case["index_seed"]))
    for row in results:
        row.update(dataset=case["case_id"], job=case["id"], kind=case["kind"])
    write_json(output / "results.json", results)
    print(f"CASE {case['id']} complete", flush=True)


def main(args):
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    output = Path(args.output).resolve()
    if args.worker is not None:
        worker(config, study_jobs(config)[args.worker], output)
        return
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise RuntimeError("Run only from a clean frozen source tree")
    if numba.get_num_threads() != config["build_threads"] or int(os.environ.get("PIN_CORE", "-1")) != config["pin_core"]:
        raise ValueError("CPU settings differ from the frozen baseline protocol")
    output.mkdir(parents=True, exist_ok=False)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    write_json(output / "config.json", config)
    write_json(output / "manifest.json", dict(
        commit=commit, config_sha256=digest(config_path), started=time.time(), python=sys.version,
        numpy=np.__version__, numba=numba.__version__,
        hnswlib_importable=_hnswlib is not None, hnswlib_version=getattr(_hnswlib, "__version__", None),
        cpu_model=next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                       if line.startswith("model name")),
        metric=config["multicurve_metric"], pin_core=config["pin_core"], build_threads=config["build_threads"],
        hnsw_build_threads=config["hnsw_build_threads"],
        index_size_method="Serialized HNSW bytes measured via save_index(); brute-force fallback uses raw vector+label bytes"))
    started = time.monotonic()
    cases = []
    summary = []
    for i, case in enumerate(study_jobs(config)):
        remaining = config["run_cap_seconds"] - (time.monotonic() - started)
        if remaining <= 0:
            raise RuntimeError("Baseline study exceeded its run cap")
        folder = output / case["id"]
        folder.mkdir()
        command = [sys.executable, "-u", "-m", "fresh.experiment_baselines",
                   "--config", str(config_path), "--output", str(folder), "--worker", str(i)]
        with (folder / "worker.log").open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            try:
                code, interruption = monitor_worker(process, folder, config, remaining)
            finally:
                terminate_worker(process)
        status = interruption or ("complete" if code == 0 else "implementation or build failure")
        cases.append(dict(case=case["id"], case_id=case["case_id"], index_seed=case["index_seed"],
                          status=status, returncode=code))
        write_json(output / "cases.json", cases)
        if code or interruption:
            raise RuntimeError(f"Stopped at {folder}: {status}")
        summary.extend(json.loads((folder / "results.json").read_text()))
        write_json(output / "summary.json", summary)
        print(f"{case['id']}: {status}", flush=True)
    write_json(output / "completion.json", dict(completed=time.time(), elapsed_seconds=time.monotonic() - started))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--worker", type=int)
    main(parser.parse_args())
