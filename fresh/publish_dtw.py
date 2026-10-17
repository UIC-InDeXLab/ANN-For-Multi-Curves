"""Audit the frozen empirical DTW study and publish measured results only.

CLI: python -m fresh.publish_dtw --source RESULTS_ROOT --output OUTPUT_FOLDER
Use --nba-data PATH only to relocate byte-identical frozen NBA controls.

Measurement artifacts are read-only; output may use a distinct results subfolder.
In addition to the per-job test artifacts, the audit
requires tuning-queries.csv, ground-truth-cache/{case_id}.{json,npz}, the frozen
configuration, and the original synthetic/NBA data and Frechet controls.
No DTW kernel is imported or run; cached exact-truth provenance is audited, not
independently recomputed. Quality is averaged across seeds; timing and allocated
index memory use medians of seed summaries. Partial coverage is never averaged.
Cache initial-parent hashes come from each job's Frechet reference IDs, not its
DTW-optimal IDs. Receipts hash all global, cache, per-job, and external controls.
DTW/Frechet optimum ordering is a consistency check only. The measurement commit
identifies the frozen runner/kernels; publisher provenance uses its own hash.
"""
import argparse
import csv
import hashlib
import json
from math import factorial, prod
from pathlib import Path

import numpy as np

from fresh.e1 import arrays_hash, write_json


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
FROZEN_CONFIG = PACKAGE_ROOT / "configs" / "dtw-20261014.json"
METHODS = ("blocking", "probe_all", "store_all", "per_curve-intersection")
LABELS = {"blocking": "Blocking", "probe_all": "Probe-all",
          "store_all": "Store-all", "per_curve-intersection": "Intersection"}
CASES = (("common-n100000-k4", "conjunctive", 100000, 4),
         ("common-n100000-k8", "conjunctive", 100000, 8),
         ("rare-n100000-k4", "rare", 100000, 4),
         ("rare-n100000-k8", "rare", 100000, 8),
         ("nba-n23805-k5", "nba", 23805, 5))
CASE_LABELS = ("Common k=4", "Common k=8", "Rare k=4", "Rare k=8", "NBA k=5")
EPSILON = 1e-8
TEST_COLUMNS = {"config_id", "commit", "config_sha256", "dataset_sha256",
                "queries_sha256", "index_seed", "metric", "phase", "query",
                "nn_distance", "id", "distance", "timeout", "candidates",
                "verified", "probes", "postings", "seconds", "found", "exact_nn", "ratio"}
QUERY_COLUMNS = {"config_id", "query", "id", "distance", "timeout", "candidates",
                 "verified", "probes", "postings", "seconds", "found", "exact_nn", "ratio"}
QUALITY = ("recall_at_1", "success_at_1_1", "empty_rate", "timeout_rate")
MEASURED = ("median_ms", "p95_ms", "index_MB", "scratch_MB", "peak_process_MB")
RATIOS = ("mean_ratio", "median_ratio", "p90_ratio", "p95_ratio", "p99_ratio", "max_ratio")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def source_path(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PACKAGE_ROOT / path).resolve()


class Sources:
    def __init__(self, root, nba_data=None):
        self.root = root
        self.nba_data = Path(nba_data).resolve() if nba_data is not None else None
        self.paths = {}

    def check(self, path):
        path = Path(path).resolve()
        if not path.is_file():
            raise ValueError(f"Missing required source artifact: {path}")
        actual = digest(path)
        if path in self.paths and self.paths[path] != actual:
            raise ValueError(f"Source artifact changed during audit: {path}")
        self.paths[path] = actual
        return actual

    def json(self, path):
        self.check(path)
        return json.loads(Path(path).read_text())

    def arrays(self, path, names):
        self.check(path)
        with np.load(path, allow_pickle=False) as values:
            if not set(names) <= set(values.files):
                raise ValueError(f"Missing required arrays in {path}")
            return {name: values[name].copy() for name in names}

    def csv(self, path, columns):
        self.check(path)
        with Path(path).open(newline="") as handle:
            reader = csv.DictReader(handle)
            if (not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames)
                    or not columns <= set(reader.fieldnames)):
                raise ValueError(f"Missing or duplicate CSV columns in {path}")
            rows = list(reader)
        if any(None in row or any(value is None for value in row.values()) for row in rows):
            raise ValueError(f"Malformed CSV row in {path}")
        return rows

    def receipt(self):
        for path, expected in self.paths.items():
            if digest(path) != expected:
                raise ValueError(f"Source artifact changed during audit: {path}")
        return {str(path.relative_to(self.root)) if path.is_relative_to(self.root) else str(path): value
                for path, value in sorted(self.paths.items())}


def boolean(value, label):
    if value not in ("True", "False"):
        raise ValueError(f"Invalid boolean for {label}")
    return value == "True"


def number(value, label, minimum=0):
    result = float(value)
    if not np.isfinite(result) or result < minimum:
        raise ValueError(f"Invalid {label}")
    return result


def integer(value, label, minimum=0):
    result = int(value)
    if isinstance(value, bool) or str(result) != str(value) or result < minimum:
        raise ValueError(f"Invalid integer for {label}")
    return result


def equal_arrays(left, right, label):
    if (left.dtype != right.dtype or left.shape != right.shape
            or not np.array_equal(left, right)):
        raise ValueError(f"Different {label} arrays")


def json_ratio(value):
    return "infinity" if np.isinf(value) else float(value)


def ratio_quantile(values, quantile):
    ordered = np.sort(values)
    rank = quantile * (len(ordered) - 1)
    # Avoid infinity - infinity in NumPy's interpolating quantile.
    if np.isinf(ordered[int(np.ceil(rank))]):
        return "infinity"
    return float(np.quantile(ordered, quantile))


def query_metrics(rows, optimum, config, n, *, complete=True):
    expected = len(optimum)
    query_ids = [integer(row["query"], "query") for row in rows]
    required = expected if complete else len(rows)
    if (not rows or len(rows) != required or len(set(query_ids)) != len(rows)
            or set(query_ids) != set(range(required)) or required > expected):
        raise ValueError("Every query must appear exactly once in its declared split")
    exact_count = empty_count = timeout_count = 0
    factors = config["c_success_factors"]
    successes = {str(float(factor)): 0 for factor in factors}
    elapsed, ratios = [], []
    for row, q in zip(rows, query_ids):
        seconds = number(row["seconds"], "query seconds")
        if seconds == 0:
            raise ValueError("Measured query latency must be positive")
        timeout = boolean(row["timeout"], "timeout")
        if seconds > config["query_deadline_seconds"] and not timeout:
            raise ValueError("Query exceeded its deadline without a timeout")
        if complete and timeout:
            raise ValueError("A successful configuration cannot contain timeouts")
        found = boolean(row["found"], "found")
        parent = integer(row["id"], "returned parent", -1)
        if found != (parent >= 0) or parent >= n or timeout and found:
            raise ValueError("Invalid returned parent or timeout answer")
        counters = {field: integer(row[field], field)
                    for field in ("candidates", "verified", "probes", "postings")}
        if counters["verified"] > counters["candidates"] or found and counters["verified"] == 0:
            raise ValueError("Invalid candidate verification counts")
        if "nn_distance" in row and float(row["nn_distance"]) != float(optimum[q]):
            raise ValueError("Query NN distance differs from exact cached optimum")
        exact = False
        if found:
            distance = number(row["distance"], "returned distance")
            if distance + EPSILON < optimum[q]:
                raise ValueError("Returned distance is below the exact cached optimum")
            exact = bool(distance <= optimum[q] + EPSILON)
            ratio = distance / optimum[q] if optimum[q] > 0 else (1. if distance == 0 else np.inf)
            if float(row["ratio"]) != ratio:
                raise ValueError("Reported ratio differs from recomputed quality")
            ratios.append(ratio)
            for factor in factors:
                successes[str(float(factor))] += int(distance <= factor * optimum[q] + EPSILON)
        elif row["distance"] != "" or row["ratio"] != "":
            raise ValueError("An empty answer must have no distance or ratio")
        if boolean(row["exact_nn"], "exact_nn") != exact:
            raise ValueError("Reported exact-NN quality differs from tie-aware DTW success")
        exact_count += exact
        empty_count += not found
        timeout_count += timeout
        elapsed.append(seconds * 1000)
    result = dict(query_count=len(rows),exact_count=exact_count,
                  recall_at_1=exact_count / expected,
                  success_at_1_1=successes["1.1"] / expected,
                  c_success={key: value / expected for key, value in successes.items()},
                  empty_rate=empty_count / expected,timeout_rate=timeout_count / expected,
                  median_ms=float(np.median(elapsed)),p95_ms=float(np.quantile(elapsed, .95)),
                  returned_ratio_count=len(ratios),
                  unbounded_ratio_count=int(np.isinf(ratios).sum()))
    if ratios:
        result.update(mean_ratio=json_ratio(np.mean(ratios)),
                      median_ratio=ratio_quantile(ratios,.5),
                      p90_ratio=ratio_quantile(ratios,.9),p95_ratio=ratio_quantile(ratios,.95),
                      p99_ratio=ratio_quantile(ratios,.99),max_ratio=json_ratio(max(ratios)))
    else:
        result.update({field: None for field in RATIOS})
    return result


def study_jobs(config):
    declared = [(case["id"], case["kind"], case["n"], case["k"]) for case in config["cases"]]
    if declared != list(CASES):
        raise ValueError("DTW publication requires the five declared cases in protocol order")
    jobs = []
    for case in config["cases"]:
        seeds = config["nba_index_seeds"] if case["kind"] == "nba" else config["synthetic_index_seeds"]
        if len(seeds) != 3 or len(set(seeds)) != 3:
            raise ValueError("Every DTW case requires three distinct index seeds")
        jobs.extend(dict(case,case_id=case["id"],index_seed=seed,id=f"{case['id']}-seed{seed}")
                    for seed in seeds)
    return jobs


def reference_arrays(sources, config, job, dataset, queries):
    count = len(queries)
    if job["kind"] != "nba":
        folder = source_path(config["synthetic_reference"]) / f"{job['kind']}-n{job['n']}-k{job['k']}-seed1701"
        original = sources.json(folder / "dataset.json")
        if any(original[field] != dataset[field] for field in ("dataset_sha256", "queries_sha256")):
            raise ValueError("Synthetic data hashes differ from frozen Frechet controls")
        if original["raw_curve_bytes"] != dataset["raw_curve_bytes"]:
            raise ValueError("Synthetic raw data memory differs from the frozen arrays")
        path = folder / "ground-truth.npz"
        if (source_path(dataset["frechet_reference"]) != folder
                or sources.check(path) != dataset["frechet_reference_sha256"]):
            raise ValueError("Synthetic Frechet reference checksum or provenance changed")
        reference = sources.arrays(path, ("queries", "ids", "distances"))
        equal_arrays(reference["queries"], queries, "Frechet reference query")
        return reference
    folder = sources.nba_data if sources.nba_data is not None else source_path(config["nba_data"])
    sources.json(folder / "preparation-complete.json")
    if (sources.check(folder / "dataset.json") != config["nba_dataset_sha256"]
            or dataset["nba_corpus_sha256"] != config["nba_dataset_sha256"]):
        raise ValueError("NBA corpus provenance changed")
    corpus = sources.json(folder / "dataset.json")
    if (dataset["nba_source_artifacts"] != corpus["artifacts"]
            or dataset["nba_reference_artifacts"] != config["nba_reference_artifacts"]):
        raise ValueError("NBA source artifact provenance changed")
    for name, expected in corpus["artifacts"].items():
        if sources.check(folder / name) != expected:
            raise ValueError("NBA data artifact checksum changed")
    arrays = {}
    for name in ("database.npz", "tune.npz", "test.npz"):
        arrays[name] = sources.arrays(folder / name, ("curves",))["curves"]
    database = arrays["database.npz"]
    if (database.shape != (job["n"],job["k"],config["m"],config["d"])
            or arrays_hash(database) != dataset["dataset_sha256"]
            or database.nbytes != dataset["raw_curve_bytes"]):
        raise ValueError("NBA database arrays differ from the declared data hash")
    if (len(arrays["tune.npz"]) != config["tuning_queries"]
            or len(arrays["test.npz"]) != config["test_queries"]):
        raise ValueError("NBA query split identity changed")
    expected_queries = np.concatenate((arrays["tune.npz"],arrays["test.npz"],
                                       database[:config["warmup_queries"]]))
    equal_arrays(expected_queries, queries, "NBA data/query")
    ids, distances = [], []
    for split, size in (("tune",config["tuning_queries"]),("test",config["test_queries"])):
        name = f"{split}-ground-truth.npz"
        if sources.check(folder / name) != config["nba_reference_artifacts"][name]:
            raise ValueError("NBA Frechet reference checksum changed")
        reference = sources.arrays(folder / name, ("ids", "distances"))
        if reference["ids"].shape != (size,) or reference["distances"].shape != (size,):
            raise ValueError("NBA Frechet reference split changed")
        ids.extend(reference["ids"].tolist())
        distances.extend(reference["distances"].tolist())
    ids.extend(range(config["warmup_queries"]))
    distances.extend([0.] * config["warmup_queries"])
    return dict(ids=np.asarray(ids,np.int64),distances=np.asarray(distances),queries=queries)


def audit_truth(sources, config, manifest, job, folder, dataset):
    for field in ("id", "case_id", "kind", "n", "k", "index_seed"):
        if dataset[field] != job[field]:
            raise ValueError("Dataset identity differs from its declared case and seed")
    if dataset["metric"] != config["curve_metric"] or not isinstance(dataset["cached_ground_truth"], bool):
        raise ValueError("Dataset metric or cache provenance is invalid")
    truth = sources.arrays(folder / "ground-truth.npz",
                           ("queries", "ids", "distances", "frechet_parent_dtw"))
    count = config["tuning_queries"] + config["test_queries"] + config["warmup_queries"]
    queries = truth["queries"]
    if (queries.shape != (count,job["k"],config["m"],config["d"])
            or not np.isfinite(queries).all() or arrays_hash(queries) != dataset["queries_sha256"]):
        raise ValueError("Exact ground-truth query identity or data hash changed")
    integer(dataset["raw_curve_bytes"], "raw curve bytes", 1)
    reference = sources.arrays(folder / "frechet-reference.npz", ("ids", "distances"))
    original = reference_arrays(sources, config, job, dataset, queries)
    for field in ("ids", "distances"):
        equal_arrays(reference[field], original[field], f"frozen Frechet {field}")
    for values in (truth, reference):
        if (values["ids"].shape != (count,) or not np.issubdtype(values["ids"].dtype, np.integer)
                or np.any(values["ids"] < 0) or np.any(values["ids"] >= job["n"])
                or values["distances"].shape != (count,)
                or not np.isfinite(values["distances"]).all() or np.any(values["distances"] < 0)):
            raise ValueError("Invalid exact optimum or Frechet reference arrays")
    costs = truth["frechet_parent_dtw"]
    if (costs.shape != (count,) or not np.isfinite(costs).all()
            or np.any(costs + EPSILON < truth["distances"])):
        raise ValueError("Invalid DTW costs of Frechet reference parents")
    cache = sources.root / "ground-truth-cache" / job["case_id"]
    record = sources.json(cache.with_suffix(".json"))
    expected = dict(dataset_sha256=dataset["dataset_sha256"],queries_sha256=dataset["queries_sha256"],
                    initial_parents_sha256=arrays_hash(reference["ids"]),
                    dtw_kernel_sha256=manifest["dtw_kernel_sha256"],full_collection_size=job["n"],
                    array_sha256=sources.check(cache.with_suffix(".npz")))
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError("Exact DTW cache provenance or checksum changed")
    cached = sources.arrays(cache.with_suffix(".npz"), ("ids", "distances", "frechet_parent_dtw"))
    for field in cached:
        equal_arrays(truth[field], cached[field], f"cached exact truth {field}")
    inconsistent = np.flatnonzero(truth["distances"] < reference["distances"] - EPSILON)
    if len(inconsistent):
        raise ValueError(f"DTW/Frechet optimum consistency check failed for {job['id']}, "
                         f"query {int(inconsistent[0])}")
    h, s = config["tuning_queries"], config["test_queries"]
    fraction = float(np.mean(costs[h:h+s] > truth["distances"][h:h+s] + EPSILON))
    if (dataset["mean_test_dtw_optimum"] != float(truth["distances"][h:h+s].mean())
            or dataset["frechet_parent_not_dtw_optimal"] != fraction):
        raise ValueError("Dataset reported optimum or Frechet-parent quality changed")
    return truth, reference, fraction


def relevant(trials, method, k):
    if method == "per_curve-intersection":
        return [row for row in trials if row["method"] == method]
    rows = [row for row in trials if row["method"] == "blocking-family"]
    if method != "blocking":
        stored = 1 if method == "probe_all" else factorial(k)
        rows = [row for row in rows if row["stored"] == stored]
    return rows


def audit_tuning(sources, config, job, folder, optimum):
    trials = sources.json(folder / "tuning-configurations.json")
    by_id = {row["config_id"]: row for row in trials}
    if not trials or len(by_id) != len(trials):
        raise ValueError("Tuning configuration IDs must be unique")
    logs = sources.csv(folder / "tuning-queries.csv", QUERY_COLUMNS)
    grouped = {}
    for row in logs:
        if row["config_id"] not in by_id:
            raise ValueError("Query log names an undeclared tuning configuration")
        grouped.setdefault(row["config_id"], []).append(row)
    for trial in trials:
        if trial["index_seed"] != job["index_seed"] or trial["method"] not in ("blocking-family", "per_curve-intersection"):
            raise ValueError("Invalid tuning method or seed")
        if not isinstance(trial["swapped"], bool):
            raise ValueError("Invalid tuning orientation")
        if trial["method"] == "blocking-family":
            blocks = trial["blocks"]
            if not blocks or any(type(size) is not int or size < 1 for size in blocks) or sum(blocks) != job["k"]:
                raise ValueError("Invalid tuning block partition")
            stored = prod(factorial(size) for size in blocks)
            if trial["swapped"]:
                stored = factorial(job["k"]) // stored
            if trial["stored"] != stored or trial["probed"] != factorial(job["k"]) // stored:
                raise ValueError("Invalid tuning family permutation counts")
        if "tuning_complete" in trial:
            if not isinstance(trial["tuning_complete"], bool):
                raise ValueError("Invalid tuning completion flag")
            query_rows = grouped.get(trial["config_id"], [])
            measured = query_metrics(query_rows, optimum, config,
                                     job["n"], complete=trial["tuning_complete"])
            timed_out = [int(row["query"]) for row in query_rows
                         if boolean(row["timeout"], "tuning timeout")]
            if not trial["tuning_complete"] and timed_out != [len(query_rows) - 1]:
                raise ValueError("Incomplete tuning must end at its first actual timeout")
            if trial["tuning_recall"] != measured["recall_at_1"]:
                raise ValueError("Reported tuning quality differs from actual tuning queries")
            qualified = trial["tuning_complete"] and measured["recall_at_1"] >= config["tuning_acceptance_recall"]
            status = ("qualified" if qualified else "tuning recall below target"
                      if trial["tuning_complete"] else "tuning timeout")
            if trial["status"] != status:
                raise ValueError("Tuning qualification differs from actual query quality")
            if trial["tuning_complete"] and trial["tuning_median_seconds"] != measured["median_ms"] / 1000:
                if not np.isclose(trial["tuning_median_seconds"], measured["median_ms"] / 1000,
                                  rtol=1e-12, atol=0):
                    raise ValueError("Reported tuning latency differs from tuning queries")
        elif trial["status"] == "qualified" or trial["config_id"] in grouped:
            raise ValueError("A tuning-qualified configuration needs its actual query evidence")
        if trial["status"] == "qualified":
            if (integer(trial["index_bytes"], "allocated index bytes", 1) > config["index_budget_bytes"]
                    or not 1 <= integer(trial["L"], "table count", 1) <= config["Lmax"]
                    or trial["delta"] not in (config["nba_deltas"] if job["kind"] == "nba" else config["deltas"])):
                raise ValueError("Tuning-qualified configuration violates the frozen budget or grid")
            integer(trial["scratch_bytes"], "scratch bytes")
    chosen = {}
    for method in METHODS:
        options = [row for row in relevant(trials, method, job["k"]) if row["status"] == "qualified"]
        if not options:
            continue
        if method == "blocking":
            def key(row):
                cost = (row["index_bytes"] / config["cost_memory_scale_bytes"]
                        + config["blocking_main_latency_weight"] * row["tuning_median_seconds"]
                        / config["cost_latency_scale_seconds"])
                return cost, row["index_bytes"], row["config_id"]
        else:
            def key(row):
                return row["tuning_median_seconds"]
        chosen[method] = min(options, key=key)
    selections = sources.json(folder / "frozen-main-selections.json")
    if selections != chosen:
        raise ValueError("Frozen selections differ from actual tuning-qualified chosen configurations")
    return trials, selections


def combine_seeds(rows, seeds, target):
    if len(rows) != 3 or {row["index_seed"] for row in rows} != set(seeds):
        raise ValueError("All three required index seeds must be represented")
    available = [row for row in rows if row["test_status"] == "complete"]
    result = {key: rows[0][key] for key in ("case_id", "kind", "n", "k", "display_method")}
    result.update(seeds=rows,available_seed_count=len(available),required_seed_count=3,
                  test_status="complete" if len(available) == 3 else "unavailable in required seeds")
    if len(available) != 3:
        result["exclusion_reasons"] = sorted({reason for row in rows
                                             for reason in row.get("exclusion_reasons", [])})
        return result
    for metric in QUALITY + MEASURED + RATIOS:
        values = [row[metric] for row in rows]
        if any(value is None for value in values):
            result[metric] = None
            result[metric + "_seed_minmax"] = None
        else:
            numeric = [np.inf if value == "infinity" else value for value in values]
            result[metric] = json_ratio(np.mean(numeric)) if metric in QUALITY + RATIOS else float(np.median(numeric))
            result[metric + "_seed_minmax"] = [json_ratio(min(numeric)),json_ratio(max(numeric))]
    result["c_success"] = {factor: float(np.mean([row["c_success"][factor] for row in rows]))
                           for factor in rows[0]["c_success"]}
    result["c_success_seed_minmax"] = {factor: [min(row["c_success"][factor] for row in rows),
                                               max(row["c_success"][factor] for row in rows)]
                                      for factor in rows[0]["c_success"]}
    result["test_target_met"] = all(row["recall_at_1"] >= target for row in rows)
    return result


def audit(root, *, nba_data=None):
    root = Path(root).resolve()
    sources = Sources(root,nba_data)
    completion = sources.json(root / "completion.json")
    if not {"completed", "elapsed_seconds"} <= set(completion):
        raise ValueError("DTW study is not complete")
    number(completion["elapsed_seconds"], "study elapsed time")
    config = sources.json(root / "config.json")
    manifest = sources.json(root / "manifest.json")
    if (sources.json(FROZEN_CONFIG) != config or sources.check(FROZEN_CONFIG) != manifest["config_sha256"]):
        raise ValueError("Frozen configuration checksum changed")
    if (manifest["metric"] != config["curve_metric"] or manifest["pin_core"] != config["pin_core"]
            or manifest["build_threads"] != config["build_threads"] or not manifest["commit"]
            or len(manifest["dtw_kernel_sha256"]) != 64):
        raise ValueError("Frozen DTW manifest provenance changed")
    jobs = study_jobs(config)
    cases = sources.json(root / "cases.json")
    if (len(cases) != 15 or [row["case"] for row in cases] != [job["id"] for job in jobs]
            or any(row["status"] != "complete" or row["returncode"] != 0 for row in cases)):
        raise ValueError("Every configured case and seed must complete in sequential protocol order")
    groups, paired, references = {}, {}, {}
    for case, job in zip(cases, jobs):
        if case["case_id"] != job["case_id"] or case["index_seed"] != job["index_seed"]:
            raise ValueError("Case manifest has a different case or seed identity")
        folder = root / job["id"]
        dataset = sources.json(folder / "dataset.json")
        truth, reference, fraction = audit_truth(sources, config, manifest, job, folder, dataset)
        identity = (dataset["dataset_sha256"],dataset["queries_sha256"])
        if job["case_id"] in paired:
            previous_identity, previous_truth, previous_reference = paired[job["case_id"]]
            if identity != previous_identity:
                raise ValueError("All seeds must use identical data hashes")
            for field in truth:
                equal_arrays(truth[field],previous_truth[field],f"across-seed {field}")
            for field in reference:
                equal_arrays(reference[field],previous_reference[field],f"across-seed reference {field}")
        else:
            paired[job["case_id"]] = identity, truth, reference
        references.setdefault(job["case_id"], []).append(dict(index_seed=job["index_seed"],
                                                               nonoptimal_fraction=fraction))
        h, s = config["tuning_queries"], config["test_queries"]
        trials, selections = audit_tuning(sources, config, job, folder, truth["distances"][:h])
        results = sources.json(folder / "main-results.json")
        if len(results) != 4 or {row["display_method"] for row in results} != set(METHODS):
            raise ValueError("Every display method needs one main-results row; no rarest-first")
        logs = sources.csv(folder / "test-queries.csv", TEST_COLUMNS)
        selected_ids = {selection["config_id"] for selection in selections.values()}
        grouped = {}
        for row in logs:
            if (row["config_id"] not in selected_ids or row["commit"] != manifest["commit"]
                    or row["config_sha256"] != manifest["config_sha256"]
                    or row["dataset_sha256"] != dataset["dataset_sha256"]
                    or row["queries_sha256"] != dataset["queries_sha256"]
                    or integer(row["index_seed"], "index seed") != job["index_seed"]
                    or row["metric"] != "DTW" or row["phase"] != "test"):
                raise ValueError("Test query provenance differs from its frozen run")
            q = integer(row["query"], "test query")
            if q >= s:
                raise ValueError("Test query identity is outside the held-out split")
            if "query_sha256" in row and row["query_sha256"] != arrays_hash(truth["queries"][h+q]):
                raise ValueError("Test query array identity changed")
            grouped.setdefault(row["config_id"], []).append(row)
        measured = {cfg_id: query_metrics(grouped.get(cfg_id, []),truth["distances"][h:h+s],config,job["n"])
                    for cfg_id in selected_ids}
        resources = sources.json(folder / "resources.json")
        peak = integer(resources["peak_process_bytes"], "peak process memory", 1)
        if (peak > config["peak_ram_cap_bytes"] or resources["returncode"] != 0
                or resources.get("interruption") or resources["monitor_interval_seconds"] != config["monitor_interval_seconds"]):
            raise ValueError("Job resources violate the frozen completion or memory limits")
        for result in results:
            method = result["display_method"]
            row = dict(case_id=job["case_id"],kind=job["kind"],n=job["n"],k=job["k"],
                       index_seed=job["index_seed"],display_method=method)
            if result["index_seed"] != job["index_seed"]:
                raise ValueError("Main result has an incorrect index seed")
            if method in selections:
                selection = selections[method]
                if (result["test_status"] != "complete" or result["test_queries"] != s
                        or any(result.get(key) != value for key, value in selection.items() if key != "build_seconds")):
                    raise ValueError("Invalid or missing selected main result")
                number(result["build_seconds"], "final index build seconds")
                metrics = measured[selection["config_id"]]
                for key in QUALITY + RATIOS + ("c_success", "median_ms", "p95_ms"):
                    if key in result and result[key] != metrics[key]:
                        raise ValueError("Main result quality or timing differs from the test log")
                row.update(selection,test_status="complete",**metrics,
                           build_seconds=result["build_seconds"],tuning_build_seconds=selection["build_seconds"],
                           index_MB=selection["index_bytes"] / 1e6,
                           scratch_MB=selection["scratch_bytes"] / 1e6,peak_process_MB=peak / 1e6)
            else:
                reasons = sorted({trial["status"] for trial in relevant(trials,method,job["k"])})
                if (result["test_status"] != "no feasible DTW tuning configuration"
                        or not reasons or result.get("exclusion_reasons") != reasons
                        or "config_id" in result):
                    raise ValueError("An unavailable method needs its explicit actual tuning exclusions")
                row.update(test_status=result["test_status"],exclusion_reasons=reasons)
            groups.setdefault((job["case_id"], method), []).append(row)
    combined = []
    parent_reports = []
    for case in config["cases"]:
        seeds = config["nba_index_seeds"] if case["kind"] == "nba" else config["synthetic_index_seeds"]
        for method in METHODS:
            combined.append(combine_seeds(groups[(case["id"],method)],seeds,config["target_recall"]))
        values = references[case["id"]]
        fractions = [row["nonoptimal_fraction"] for row in values]
        if len(set(fractions)) != 1:
            raise ValueError("Frechet reference-parent fraction must be identical across seeds")
        parent_reports.append(dict(case_id=case["id"],nonoptimal_fraction=fractions[0],
                                   nonoptimal_fraction_seed_minmax=[min(fractions),max(fractions)],seeds=values))
    relocations = ({"nba_data": dict(declared=str(source_path(config["nba_data"])),
                                    audited=str(sources.nba_data))}
                   if sources.nba_data is not None else {})
    return dict(schema_version=1,config=config,manifest=manifest,main=combined,
                input_relocations=relocations,
                frechet_reference_parents=parent_reports,source_artifacts=sources.receipt(),
                aggregation=dict(quality="arithmetic mean of three seed statistics",
                                 timing_memory="median of three seed summaries",
                                 ratio_tails="mean of conditional per-seed tail statistics; not pooled quantiles",
                                 missing="no aggregate unless all three seeds are available"),
                definitions=dict(tie_tolerance=EPSILON,
                                 recall="returned DTW cost <= exact DTW optimum + 1e-8",
                                 success_at_1_1="returned DTW cost <= 1.1 * exact DTW optimum + 1e-8",
                                 ratios="conditional on a nonempty return; zero/zero=1; positive/zero=infinity",
                                 index_MB="allocated serving index bytes / 1000000; excludes raw curves and scratch",
                                 dtw_frechet_consistency="exact DTW optimum >= paired exact Frechet optimum - 1e-8; consistency only",
                                 frechet_reference_nonoptimal="reference parent DTW cost > exact DTW optimum + 1e-8"),
                notes=["Empirical DTW measurements only; no model-estimated timings or memory.",
                       "No Frechet collision or approximation guarantees are transferred.",
                       "DTW/Frechet optimum ordering is checked for artifact consistency only.",
                       "Exact truth is checked against the checksum-validated cache, not recomputed by this publisher.",
                       "Synthetic database identity is checked via frozen control hashes; NBA arrays are also hashed.",
                       "Unavailable methods retain per-seed tuning exclusions; resource estimates are not measurements."])


def tex_escape(value):
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
                    "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}"}
    return "".join(replacements.get(char,char) for char in str(value))


def compact_table(report):
    lines = [r"\begin{tabular}{lrrrrr}",r"\toprule",
             r"Case & \shortstack{Recall\\(\%)} & \shortstack{$1.1\times$\\(\%)} & "
             r"\shortstack{Time\\(ms)} & \shortstack{Index\\(MB)} & "
             r"\shortstack{Ref. nonopt.\\(\%)}\\",r"\midrule"]
    for case, label in zip(report["config"]["cases"], CASE_LABELS):
        row = next(row for row in report["main"] if row["case_id"] == case["id"] and row["display_method"] == "blocking")
        fraction = next(row["nonoptimal_fraction"] for row in report["frechet_reference_parents"]
                        if row["case_id"] == case["id"])
        if row["test_status"] == "complete":
            values = f"{100*row['recall_at_1']:.1f} & {100*row['success_at_1_1']:.1f} & {row['median_ms']:.2f} & {row['index_MB']:.2f}"
        else:
            values = r"\multicolumn{4}{c}{unavailable (see appendix)}"
        lines.append(f"{label} & {values} & {100*fraction:.1f}"+r"\\")
    return "\n".join(lines + [r"\bottomrule",r"\end{tabular}"]) + "\n"


def tex_value(value, factor=1):
    if value is None:
        return "--"
    if value == "infinity":
        return r"$\infty$"
    return f"{factor*value:.2f}"


def appendix_table(report):
    lines = [r"\begin{tabular}{llrrrrrrrr}",r"\toprule",
             r"Case & Method & \shortstack{Recall\\(\%)} & \shortstack{$1.1\times$\\(\%)} & "
             r"\shortstack{Empty\\(\%)} & \shortstack{Time\\(ms)} & "
             r"\shortstack{p95\\(ms)} & \shortstack{Index\\(MB)} & r95 & r99\\",
             r"\midrule"]
    for case, label in zip(report["config"]["cases"], CASE_LABELS):
        for method in METHODS:
            row = next(row for row in report["main"] if row["case_id"] == case["id"] and row["display_method"] == method)
            prefix = f"{label} & {LABELS[method]} & "
            if row["test_status"] == "complete":
                values = [(row[key],100 if key in QUALITY else 1) for key in
                          ("recall_at_1","success_at_1_1","empty_rate","median_ms","p95_ms","index_MB","p95_ratio","p99_ratio")]
                lines.append(prefix+" & ".join(tex_value(value,factor) for value,factor in values)+r"\\")
            else:
                lines.append(prefix+r"\multicolumn{8}{c}{unavailable}\\")
    lines.extend([r"\bottomrule",r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def results_text(report):
    fractions = [100*row["nonoptimal_fraction"] for row in report["frechet_reference_parents"]]
    return (
        r"DTW uses unconstrained alignment with summed Euclidean point costs, without length normalization. "
        r"Multi-curve distance minimizes, over curve bijections, the maximum pairwise DTW cost. "
        r"Recall counts a nonempty return with cost at most the exact DTW optimum plus $10^{-8}$; "
        r"$1.1\times$ success uses $1.1$ times that optimum plus the same tolerance. "
        r"Empty answers count as failures; ratio tails condition on nonempty returns. "
        r"The Fr\'echet reference parent is nonoptimal only when its measured DTW cost exceeds "
        r"the true DTW optimum by more than $10^{-8}$, not when the parent IDs differ. "
        f"The observed nonoptimal fractions range from {min(fractions):.1f}\\% to {max(fractions):.1f}\\% "
        r"over the five cases and are identical across the three seeds of each case. "
        r"Quality averages the three seed statistics. Reported median query time is the median of the "
        r"three per-seed query medians; allocated index MB is the median of the three index sizes "
        r"(MB denotes $10^6$ bytes, excluding raw curves and scratch). "
        r"This is empirical DTW validation only; no Fr\'echet collision or approximation guarantee is asserted."
        "\n")


def draw_figure(report, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from fresh.nba_paper_examples import configure_paper_font, save_paper_pdf
    from fresh.plot_styles import method_style

    configure_paper_font()
    with plt.rc_context({"font.size":8,"axes.titlesize":9,"axes.labelsize":8}):
        fig, axes = plt.subplots(2,1,figsize=(3.35,3.15))
        try:
            for method in METHODS:
                rows = [next(row for row in report["main"] if row["case_id"] == case["id"]
                             and row["display_method"] == method) for case in report["config"]["cases"]]
                for ax, metric in zip(axes, ("median_ms","index_MB")):
                    values = [row[metric] if row["test_status"] == "complete" else np.nan for row in rows]
                    ax.plot(range(5),values,**method_style(method),label=LABELS[method])
                    low = [row[metric+"_seed_minmax"][0] if row["test_status"] == "complete" else np.nan for row in rows]
                    high = [row[metric+"_seed_minmax"][1] if row["test_status"] == "complete" else np.nan for row in rows]
                    ax.vlines(range(5),low,high,color=method_style(method)["color"],linewidth=.5)
            for ax, title, ylabel in zip(axes, ("Query latency","Allocated index"),
                                         ("Time (ms)","Index (MB)")):
                ax.set_title(title)
                ax.set_ylabel(ylabel)
                ax.set_xticks(range(5), labels=("Common\nk=4","Common\nk=8","Rare\nk=4","Rare\nk=8","NBA\nk=5"))
                ax.set_yscale("log")
                ax.grid(axis="y",alpha=.2,linewidth=.5)
                ax.spines[["top","right"]].set_visible(False)
                ax.tick_params(labelsize=7)
            fig.legend(*axes[0].get_legend_handles_labels(),loc="upper center",ncol=2,
                       frameon=False,fontsize=8,columnspacing=.9,handlelength=2.5)
            fig.tight_layout(rect=(0,0,1,.86),pad=.35,h_pad=.7)
            save_paper_pdf(fig,Path(destination),pad_inches=.02)
        finally:
            plt.close(fig)


def publish(source, output, *, nba_data=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source:
        raise ValueError("Publication output cannot be the read-only results source root")
    report = audit(source,nba_data=nba_data)
    output.mkdir(parents=True,exist_ok=True)
    draw_figure(report,output / "compact-dtw.pdf")
    write_json(output / "dtw-summary.json",report)
    for name, text in (("compact-dtw-table.tex",compact_table(report)),
                       ("dtw-appendix-table.tex",appendix_table(report)),
                       ("dtw-results.tex",results_text(report))):
        (output / name).write_text(text)
    artifacts = {name: digest(output / name) for name in
                 ("dtw-summary.json","compact-dtw.pdf","compact-dtw-table.tex",
                  "dtw-appendix-table.tex","dtw-results.tex")}
    receipt = dict(schema_version=1,source=str(source),source_artifacts=report["source_artifacts"],
                   input_relocations=report["input_relocations"],
                   measurement_commit=report["manifest"]["commit"],
                   publisher_sha256=digest(Path(__file__)),outputs=artifacts,
                   all_15_jobs_complete=True,all_seed_arrays_identical=True,
                   exact_cache_arrays_checked=True,tuning_selections_rederived=True,
                   dtw_frechet_optimum_consistency_checked=True,
                   test_quality_recomputed=True,dtw_kernel_imported=False)
    write_json(output / "dtw-publication-receipt.json",receipt)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",required=True,type=Path,help="Read-only completed DTW results root")
    parser.add_argument("--output",required=True,type=Path,
                        help="Publication folder; may be a results subfolder, but not the source root")
    parser.add_argument("--nba-data",type=Path,
                        help="Explicit relocation of frozen NBA controls; all original checksums still required")
    args = parser.parse_args(argv)
    if args.nba_data is not None:
        print(f"Auditing relocated NBA controls at {args.nba_data.resolve()}; frozen configuration is unchanged.")
    publish(args.source,args.output,nba_data=args.nba_data)


if __name__ == "__main__":
    main()
