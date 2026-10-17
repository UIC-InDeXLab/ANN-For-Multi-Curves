"""One-thread construction-cost measurements for frozen selected configurations."""
import argparse
from collections import defaultdict
import gc
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import threading
import time

import numba
import numpy as np

from fresh.e1 import append_row, monitor_worker, terminate_worker, write_json
from fresh.nba_full import digest
from fresh.poc import Joint, PerCurve, dataset


METHODS = ("blocking", "probe_all", "per_curve-intersection", "store_all")
NBA_METHOD_MAP = {"blocking": "blocking", "probe_all": "probe_all",
                  "per_curve-intersection": "per_curve-intersection", "store_all": "store_all"}


def require_clean_tree():
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise RuntimeError("Run only from a clean frozen source tree")


def resolve_path(base, value):
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def current_rss():
    with Path("/proc/self/status").open() as handle:
        for line in handle:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("Linux process RSS is unavailable")


def median(values):
    return float(np.median(np.asarray(values, float)))


def worker_state(output, **payload):
    write_json(output / "active_configuration.json", payload)


def normalize_synthetic_selection(method, row, k):
    stored = row.get("stored")
    return dict(display_method=method, raw_method=row["method"], config_id=row["config_id"],
                delta=float(row["delta"]), L=int(row["L"]), blocks=None if row["blocks"] is None else tuple(row["blocks"]),
                swapped=bool(row["swapped"]), stored=int(k if stored is None else stored),
                source_index_bytes=int(row["index_bytes"]))


def normalize_nba_selection(method, row, k):
    stored = row.get("stored")
    return dict(display_method=method, raw_method=row["method"], config_id=row["config_id"],
                delta=float(row["delta"]), L=int(row["L"]), blocks=None if row["blocks"] is None else tuple(row["blocks"]),
                swapped=bool(row.get("swapped", False)), stored=int(k if stored is None else stored),
                source_index_bytes=int(row["index_bytes"]))


def verify_index_bytes(observed, expected, label):
    if int(observed) != int(expected):
        raise AssertionError(f"Frozen index_bytes mismatch for {label}: {observed} != {expected}")


def measured_build(factory, config):
    stop = threading.Event()
    samples = [current_rss()]
    def sample():
        while not stop.wait(config.get("rss_sampling_seconds", 0.05)):
            samples.append(current_rss())
    timer = threading.Thread(target=sample, name="build-time-rss", daemon=True)
    old_handler = signal.getsignal(signal.SIGALRM)
    old_timer = signal.getitimer(signal.ITIMER_REAL)
    started = time.perf_counter()
    ru_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    timer.start()
    try:
        signal.signal(signal.SIGALRM, lambda signum, frame: (_ for _ in ()).throw(TimeoutError("build deadline")))
        signal.setitimer(signal.ITIMER_REAL, config["build_deadline_seconds"])
        index = factory()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
        stop.set()
        timer.join()
        signal.setitimer(signal.ITIMER_REAL, *old_timer)
    samples.append(current_rss())
    ru_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    return index, dict(build_seconds=float(time.perf_counter() - started),
                       build_sampled_peak_rss_bytes=int(max(samples)),
                       build_peak_resident_delta_bytes=int(max(samples) - samples[0]),
                       process_peak_rss_delta_bytes=int(max(0, ru_after - ru_before)),
                       prebuild_resident_rss_bytes=int(samples[0]),
                       build_rss_sampling_seconds=float(config.get("rss_sampling_seconds", 0.05)),
                       constructor_reported_build_seconds=float(index.build_seconds),
                       index_bytes=int(index.index_bytes), scratch_bytes=int(index.scratch_bytes),
                       entries=int(index.entries), raw_entries=int(index.raw_entries))


def build_factory(X, selection, seed):
    if selection["display_method"] == "per_curve-intersection":
        return lambda: PerCurve(X, selection["delta"], selection["L"], seed)
    return lambda: Joint(X, selection["delta"], selection["L"], seed, selection["blocks"], selection["swapped"])


def compile_warmup(X, selections, seed):
    tiny = X[:min(len(X), 8)]
    if len(tiny) < 2:
        return
    seen = set()
    for selection in selections:
        key = (selection["display_method"], selection["delta"], selection["blocks"], selection["swapped"])
        if key in seen:
            continue
        seen.add(key)
        if selection["display_method"] == "per_curve-intersection":
            PerCurve(tiny, selection["delta"], 1, seed)
        else:
            Joint(tiny, selection["delta"], 1, seed, selection["blocks"], selection["swapped"])


def synthetic_cases(config):
    return [dict(source=entry["source"], **config["case_lookup"][entry["id"]]) for entry in config["synthetic_cases"]]


def study_jobs(config):
    jobs = [dict(case, case_id=case["id"], index_seed=seed, id=f"{case['id']}-seed{seed}")
            for case in synthetic_cases(config) for seed in config["synthetic_index_seeds"]]
    nba_case = dict(config["nba_case"], case_id=config["nba_case"]["id"])
    jobs.extend(dict(nba_case, index_seed=seed, id=f"{nba_case['id']}-seed{seed}") for seed in config["nba_index_seeds"])
    return jobs


def load_protocol(path):
    config_path = Path(path).resolve()
    raw = json.loads(config_path.read_text())
    expanded_path = resolve_path(config_path.parent, raw["expanded_config"])
    scaleup_path = resolve_path(config_path.parent, raw["scaleup_config"])
    expanded = json.loads(expanded_path.read_text())
    scaleup = json.loads(scaleup_path.read_text())
    case_lookup = {case["id"]: dict(case) for case in expanded["cases"] + scaleup["cases"]}
    config = dict(expanded)
    config.update(raw)
    config["config_path"] = str(config_path)
    config["case_lookup"] = case_lookup
    config["expanded_config"] = str(expanded_path)
    config["scaleup_config"] = str(scaleup_path)
    config["expanded_config_sha256"] = digest(expanded_path)
    config["scaleup_config_sha256"] = digest(scaleup_path)
    config["dataset_config_by_source"] = {"expanded": expanded, "scaleup": scaleup}
    config["expanded_results_root"] = str(resolve_path(config_path.parent, raw["expanded_results_root"]))
    config["scaleup_results_root"] = str(resolve_path(config_path.parent, raw["scaleup_results_root"]))
    config["nba_results_root"] = str(resolve_path(config_path.parent, raw["nba_results_root"]))
    config["nba_data_root"] = str(resolve_path(config_path.parent, raw["nba_data_root"]))
    config["methods"] = tuple(raw.get("methods", METHODS))
    config["synthetic_index_seeds"] = tuple(int(v) for v in raw["synthetic_index_seeds"])
    config["nba_index_seeds"] = tuple(int(v) for v in raw["nba_index_seeds"])
    return config


def synthetic_reference_root(config, source):
    return Path(config["expanded_results_root"] if source == "expanded" else config["scaleup_results_root"])


def load_synthetic_fixture(config, case):
    dataset_config = config.get("dataset_config_by_source", {}).get(case["source"], config)
    X, _, info = dataset(dataset_config, case)
    reference = synthetic_reference_root(config, case["source"]) / case["id"]
    original = json.loads((reference / "dataset.json").read_text())
    if info["dataset_sha256"] != original["dataset_sha256"] or info["queries_sha256"] != original["queries_sha256"]:
        raise ValueError(f"Synthetic arrays changed for {case['id']}")
    info.update(reference_dataset_json=str(reference / "dataset.json"),
                reference_dataset_sha256=digest(reference / "dataset.json"),
                reference_root=str(reference))
    selections = json.loads((reference / "frozen-main-selections.json").read_text())
    normalized = {}
    for method in METHODS:
        if method in selections:
            normalized[method] = normalize_synthetic_selection(method, selections[method], case["k"])
    return X, info, normalized


def load_nba_fixture(config, case):
    root = Path(config["nba_data_root"])
    with np.load(root / "database.npz", allow_pickle=False) as arrays:
        X = arrays["curves"].copy()
    k = int(X.shape[1])
    info = dict(dataset_sha256=digest(root / "database.npz"), curves_sha256=digest(root / "database.npz"),
                arrays_shape=list(X.shape), raw_curve_bytes=int(X.nbytes), reference_root=str(root))
    seed_root = Path(config["nba_results_root"]) / f"seed-{case['index_seed']}"
    frozen = json.loads((seed_root / "frozen-selections.json").read_text())["main"]
    normalized = {}
    for method, key in NBA_METHOD_MAP.items():
        if key in frozen:
            normalized[method] = normalize_nba_selection(method, frozen[key], k)
    resolved = dict(case, n=int(X.shape[0]), k=k)
    return X, info, normalized, resolved


def worker(config, case, output):
    if any(path.name != "worker.log" for path in output.iterdir()):
        raise ValueError(f"Worker output must start empty apart from worker.log: {output}")
    if numba.get_num_threads() != config["build_threads"]:
        raise ValueError("Build-time worker requires NUMBA_NUM_THREADS=1")
    manifest = json.loads((output.parent / "manifest.json").read_text())
    worker_state(output, method="preparation", seed=case["index_seed"], phase="loading")
    if case["kind"] == "nba":
        X, info, selections, resolved_case = load_nba_fixture(config, case)
        case = resolved_case
    else:
        X, info, selections = load_synthetic_fixture(config, case)
    write_json(output / "dataset.json", dict(info, **case))
    available = [selection for method, selection in selections.items() if method in config["methods"]]
    compile_warmup(X, available, case["index_seed"])
    results = []
    for method in config["methods"]:
        selection = selections.get(method)
        if selection is None:
            row = dict(case=case["id"], case_id=case["case_id"], kind=case["kind"], n=case["n"], k=case["k"],
                       index_seed=case["index_seed"], display_method=method, status="not built in frozen selection",
                       source_config_id=None, L=None, delta=None, repetitions=0, median_build_seconds=None,
                       min_build_seconds=None, max_build_seconds=None, index_bytes=None,
                       median_peak_resident_delta_bytes=None, note="Construction is static; inserts require rebuilding.",
                       commit=manifest["commit"], config_sha256=manifest["config_sha256"],
                       dataset_sha256=info["dataset_sha256"])
            append_row(output / "main-results.csv", row)
            results.append(row)
            continue
        runs = []
        for repetition in range(config["repetitions"]):
            worker_state(output, method=method, seed=case["index_seed"], phase="building",
                         repetition=repetition + 1, config_id=selection["config_id"])
            index, metrics = measured_build(build_factory(X, selection, case["index_seed"]), config)
            verify_index_bytes(index.index_bytes, selection["source_index_bytes"],
                               f"{case['id']} seed {case['index_seed']} {method}")
            run = dict(case=case["id"], case_id=case["case_id"], kind=case["kind"], n=case["n"], k=case["k"],
                       index_seed=case["index_seed"], display_method=method, repetition=repetition + 1,
                       source_config_id=selection["config_id"], L=selection["L"], delta=selection["delta"],
                       expected_index_bytes=selection["source_index_bytes"], commit=manifest["commit"],
                       config_sha256=manifest["config_sha256"], dataset_sha256=info["dataset_sha256"], **metrics)
            append_row(output / "build-runs.csv", run)
            runs.append(run)
            del index
            gc.collect()
        build_seconds = [run["build_seconds"] for run in runs]
        peak_deltas = [run["build_peak_resident_delta_bytes"] for run in runs]
        summary = dict(case=case["id"], case_id=case["case_id"], kind=case["kind"], n=case["n"], k=case["k"],
                       index_seed=case["index_seed"], display_method=method, status="complete",
                       source_config_id=selection["config_id"], L=selection["L"], delta=selection["delta"],
                       repetitions=len(runs), median_build_seconds=median(build_seconds),
                       min_build_seconds=float(min(build_seconds)), max_build_seconds=float(max(build_seconds)),
                       index_bytes=selection["source_index_bytes"],
                       median_peak_resident_delta_bytes=median(peak_deltas),
                       note="Construction is static; inserts require rebuilding.", commit=manifest["commit"],
                       config_sha256=manifest["config_sha256"], dataset_sha256=info["dataset_sha256"])
        append_row(output / "main-results.csv", summary)
        results.append(summary)
    write_json(output / "main-results.json", results)
    write_json(output / "completion.json", dict(completed=time.time(), case=case["id"], rows=len(results)))


def collect_rows(output, config):
    rows = []
    for job in study_jobs(config):
        rows.extend(json.loads((output / job["id"] / "main-results.json").read_text()))
    return rows


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        if row["status"] == "complete":
            grouped[(row["case_id"], row["display_method"])].append(row)
    summary = []
    for (case_id, method), group in sorted(grouped.items()):
        build_seconds = np.asarray([row["median_build_seconds"] for row in group], float)
        index_bytes = np.asarray([row["index_bytes"] for row in group], float)
        peak_delta = np.asarray([row["median_peak_resident_delta_bytes"] for row in group], float)
        summary.append(dict(case_id=case_id, display_method=method, seed_count=len(group),
                            median_build_seconds=float(np.median(build_seconds)),
                            min_build_seconds=float(build_seconds.min()),
                            max_build_seconds=float(build_seconds.max()),
                            median_index_bytes=float(np.median(index_bytes)),
                            min_index_bytes=float(index_bytes.min()),
                            max_index_bytes=float(index_bytes.max()),
                            median_peak_resident_delta_bytes=float(np.median(peak_delta)),
                            note="Construction is static; inserts require rebuilding."))
    return summary


def write_top_level_summaries(output, config):
    rows = collect_rows(output, config)
    for row in rows:
        append_row(output / "all-results.csv", row)
    write_json(output / "all-results.json", rows)
    summary_rows = summarize(rows)
    write_json(output / "summary.json", dict(summary=summary_rows,
                                              excluded=[row for row in rows if row["status"] != "complete"],
                                              note="Construction is static sorted-table indexing; insertion currently means rebuilding."))


def main(args):
    config = load_protocol(args.config)
    output = Path(args.output).resolve()
    if args.worker is not None:
        worker(config, study_jobs(config)[args.worker], output)
        return
    require_clean_tree()
    if numba.get_num_threads() != config["build_threads"] or int(os.environ.get("PIN_CORE", "-1")) != config["pin_core"]:
        raise ValueError("CPU settings differ from the frozen one-thread build protocol")
    output.mkdir(parents=True, exist_ok=False)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    write_json(output / "config.json", config)
    write_json(output / "manifest.json",
               dict(commit=commit, config_sha256=digest(Path(args.config).resolve()), started=time.time(),
                    python=sys.version, numpy=np.__version__, numba=numba.__version__,
                    cpu_model=next(line.split(":", 1)[1].strip()
                                   for line in Path("/proc/cpuinfo").read_text().splitlines()
                                   if line.startswith("model name")),
                    build_threads=numba.get_num_threads(), pin_core=config["pin_core"],
                    expanded_config_sha256=config["expanded_config_sha256"],
                    scaleup_config_sha256=config["scaleup_config_sha256"],
                    implementation_sha256=digest(__file__)))
    started = time.monotonic()
    cases = []
    for i, case in enumerate(study_jobs(config)):
        remaining = config["run_cap_seconds"] - (time.monotonic() - started)
        if remaining <= 0:
            raise RuntimeError("Build-time study exceeded its run cap")
        folder = output / case["id"]
        folder.mkdir()
        with (folder / "worker.log").open("w") as log:
            process = subprocess.Popen([sys.executable, "-u", "-m", "fresh.experiment_build_time",
                                        "--config", str(Path(args.config).resolve()),
                                        "--output", str(folder), "--worker", str(i)],
                                       stdout=log, stderr=subprocess.STDOUT)
            try:
                code, interruption = monitor_worker(process, folder, config, remaining)
            finally:
                terminate_worker(process)
        status = interruption or ("complete" if code == 0 else "implementation or build failure")
        cases.append(dict(case=case["id"], case_id=case["case_id"], index_seed=case["index_seed"],
                          status=status, returncode=code))
        write_json(output / "cases.json", cases)
        print(f"{case['id']}: {status}", flush=True)
        if code or interruption:
            raise RuntimeError(f"Stopped at {folder}: {status}")
    write_top_level_summaries(output, config)
    write_json(output / "completion.json", dict(completed=time.time(), elapsed_seconds=time.monotonic() - started))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--worker", type=int)
    main(parser.parse_args())
