"""Ten-player NBA plays (offense and defense): the single-blocking comparison of fresh.poc on real tracking data."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numba
import numpy as np

from fresh import poc
from fresh.e1 import arrays_hash, monitor_worker, terminate_worker, write_json
from fresh.experiment_first_entry import require_clean_tree, resolve_path
from fresh.nba_full import digest
from fresh.regimes import ground_truth as exact_ground_truth


def load_protocol(path):
    config_path = Path(path).resolve()
    raw = json.loads(config_path.read_text())
    source_path = resolve_path(config_path.parent, raw["synthetic_config"])
    config = dict(json.loads(source_path.read_text()))
    config.update(raw)
    config["config_path"] = str(config_path)
    config["source_config_sha256"] = digest(source_path)
    config["data_root"] = str(resolve_path(config_path.parent, raw["data_root"]))
    return config


def study_jobs(config):
    case = dict(config["case"], kind="nba")
    return [dict(case, case_id=case["id"], index_seed=seed, id=f"{case['id']}-seed{seed}") for seed in config["index_seeds"]]


def load_arrays(config):
    root = Path(config["data_root"])
    with np.load(root / "database.npz", allow_pickle=False) as a:
        X = a["curves"].copy()
    with np.load(root / "tune.npz", allow_pickle=False) as a:
        Qt = a["curves"].copy()
    with np.load(root / "test.npz", allow_pickle=False) as a:
        Qs = a["curves"].copy()
    return X, Qt, Qs


def nba_dataset(config, case):
    X, Qt, Qs = load_arrays(config)
    h, s = config["tuning_queries"], config["test_queries"]
    Qt, Qs = Qt[:h], Qs[:s]
    warm = X[:config["warmup_queries"]].copy()
    Q = np.concatenate((Qt, Qs, warm))
    info = dict(dataset_sha256=arrays_hash(X), queries_sha256=arrays_hash(Q), raw_curve_bytes=X.nbytes,
                nba_corpus="offense and defense of the five-player corpus", database=len(X))
    return X, Q, info


def cached_ground_truth(X, Q, radius):
    """Exact nearest neighbors of the tuning and test queries, computed once and shared by the seeds."""
    path = Path(os.environ["NBA10_GROUND_TRUTH"])
    if path.is_file():
        with np.load(path, allow_pickle=False) as a:
            if len(a["ids"]) == len(Q):
                return a["ids"].copy(), a["distances"].copy(), a["relevant"].copy()
    ids, distances, relevant = exact_ground_truth(X, Q, 0.0)
    if np.any(ids < 0) or not np.isfinite(distances).all():
        raise AssertionError("Exact nearest-neighbor search returned invalid distances")
    temporary = path.with_name(path.name + f".{os.getpid()}.npz")
    np.savez(temporary, ids=ids, distances=distances, relevant=relevant)
    temporary.replace(path)
    return ids, distances, relevant


def run_worker(config, case, output):
    poc.dataset = nba_dataset
    poc.ground_truth = cached_ground_truth
    os.environ.setdefault("NBA10_GROUND_TRUTH", str(Path(config["output_root"]) / "ground-truth.npz"))
    poc.worker(config, case, output)


def main(args):
    config = load_protocol(args.config)
    output = Path(args.output).resolve()
    config["output_root"] = str(output if args.worker is None else output.parent)
    if args.worker is not None:
        run_worker(config, study_jobs(config)[args.worker], output)
        return
    require_clean_tree()
    if numba.get_num_threads() != config["build_threads"] or int(os.environ.get("PIN_CORE", "-1")) != config["pin_core"]:
        raise ValueError("CPU settings differ from the frozen protocol")
    output.mkdir(parents=True, exist_ok=False)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    write_json(output / "config.json", config)
    write_json(output / "manifest.json", dict(commit=commit, config_sha256=digest(Path(args.config).resolve()),
                                              started=time.time(), python=sys.version, numpy=np.__version__,
                                              numba=numba.__version__, pin_core=config["pin_core"],
                                              build_threads=numba.get_num_threads()))
    started = time.monotonic()
    cases = []
    for i, case in enumerate(study_jobs(config)):
        remaining = config["run_cap_seconds"] - (time.monotonic() - started)
        if remaining <= 0:
            raise RuntimeError("Run cap exceeded")
        folder = output / case["id"]
        folder.mkdir()
        with (folder / "worker.log").open("w") as log:
            process = subprocess.Popen([sys.executable, "-u", "-m", "fresh.experiment_nba_k10", "--config",
                                        str(Path(args.config).resolve()), "--output", str(folder), "--worker", str(i)],
                                       stdout=log, stderr=subprocess.STDOUT)
            try:
                code, interruption = monitor_worker(process, folder, config, remaining)
            finally:
                terminate_worker(process)
        status = interruption or ("complete" if code == 0 else "implementation or build failure")
        cases.append(dict(case=case["id"], case_id=case["case_id"], index_seed=case["index_seed"], status=status, returncode=code))
        write_json(output / "cases.json", cases)
        print(f"{case['id']}: {status}", flush=True)
        if code or interruption:
            raise RuntimeError(f"Stopped at {folder}: {status}")
    write_json(output / "completion.json", dict(completed=time.time(), elapsed_seconds=time.monotonic() - started))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--worker", type=int)
    main(parser.parse_args())
