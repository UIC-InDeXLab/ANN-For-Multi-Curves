"""Prepare portable copies of protocols and run individual studies sequentially."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
STUDIES = {
    "smoke": ("poc", "smoke.json", "smoke"),
    "synthetic": ("poc", "synthetic-expanded-20261012.json", "fresh-synthetic-expanded-20261012"),
    "scaling": ("poc", "synthetic-scaleup-20261013.json", "fresh-synthetic-scaleup-20261013"),
    "selectivity": ("poc", "selectivity-expanded-20261013.json", "fresh-selectivity-expanded-20261013"),
    "collision": ("experiment_collision", "collision-20261015.json", "collision-20261015"),
    "approximation": ("e1", "e1-full-20261012.json", "fresh-e1-20261012-full"),
    "approximation-k10": ("e1", "approximation-k10-20261012.json", "fresh-approximation-k10-20261012"),
    "frontier": ("experiment_frontier", "frontier-20261016.json", "frontier-20261016"),
    "matched-recall": ("experiment_matched_recall", "matched-recall-20261015.json", "matched-recall-20261015"),
    "build-time": ("experiment_build_time", "build-time-20261015.json", "build-time-20261015"),
    "budget": ("feasibility", "feasibility-20261014.json", "fresh-feasibility-20261014"),
    "budget-256mb-1ms": ("feasibility", "feasibility-256mb-1ms-20261015.json", "feasibility-256mb-1ms-20261015"),
    "budget-16gb-10ms": ("feasibility", "feasibility-16gb-10ms-20261015.json", "feasibility-16gb-10ms-20261015"),
    "budget-16gb-100ms": ("feasibility", "feasibility-16gb-100ms-20261015.json", "feasibility-16gb-100ms-20261015"),
    "replication-b": ("poc", "dataset-seed-b-20261015.json", "dataset-seed-b-20261015"),
    "replication-c": ("poc", "dataset-seed-c-20261015.json", "dataset-seed-c-20261015"),
    "sensitivity-m16": ("poc", "sensitivity-m16-20261015.json", "sensitivity-m16-20261015"),
    "sensitivity-m64": ("poc", "sensitivity-m64-20261015.json", "sensitivity-m64-20261015"),
    "sensitivity-r025": ("poc", "sensitivity-rho025-20261015.json", "sensitivity-rho025-20261015"),
    "sensitivity-r1": ("poc", "sensitivity-rho100-20261015.json", "sensitivity-rho100-20261015"),
    "first-hit-a": ("experiment_first_entry", "first-entry-a-20261016.json", "first-entry-a-20261016"),
    "first-hit-b": ("experiment_first_entry", "first-entry-b-20261016.json", "first-entry-b-20261016"),
    "first-hit-1m": ("experiment_first_entry", "first-entry-1m-20261016.json", "first-entry-1m-20261016"),
    "first-hit-selectivity": ("experiment_first_entry", "first-entry-selectivity-20261016.json", "first-entry-selectivity-20261016"),
    "hnsw": ("experiment_baselines", "baselines-20261015.json", "baselines-20261015"),
    "hnsw-ksweep": ("experiment_baselines", "baselines-ksweep-20261015.json", "baselines-ksweep-20261015"),
    "hnsw-scaling": ("experiment_baselines", "baselines-scale-20261015.json", "baselines-scale-20261015"),
    "hnsw-selectivity": ("experiment_baselines", "baselines-selectivity-20261015.json", "baselines-selectivity-20261015"),
    "dtw": ("experiment_dtw", "dtw-20261014.json", "dtw-20261014"),
    "nba-selectivity": ("experiment_nba_selectivity", "nba-selectivity-20261015.json", "nba-selectivity-20261015"),
}
CONFIG_KEYS = {"synthetic_config", "expanded_config", "scaleup_config"}
DATA_KEYS = {"nba_data", "nba_data_root"}


def prepare_config(name, core, threads, generated):
    source = ROOT / "configs" / name
    config = json.loads(source.read_text())
    for key, value in list(config.items()):
        if not isinstance(value, str):
            continue
        if key in CONFIG_KEYS:
            config[key] = str(prepare_config(Path(value).name, core, threads, generated))
        elif key.endswith(("_root", "_path", "_reference")) or key in DATA_KEYS:
            if key in DATA_KEYS:
                value = "data/nba-full-20261012"
            elif value.startswith("data/nba-dtw-20261014"):
                value = value.replace("data/nba-dtw-20261014", "data/nba-full-20261012", 1)
            if not value.startswith(("http://", "https://")):
                config[key] = str((ROOT / value.removeprefix("../")).resolve())
    if "pin_core" in config:
        config["pin_core"] = core
    if "build_threads" in config:
        config["build_threads"] = threads
    if "hnsw_build_threads" in config:
        config["hnsw_build_threads"] = threads
    if "nba_data" in config:
        from fresh.nba_full import digest
        corpus = ROOT / "data/nba-full-20261012"
        config["nba_dataset_sha256"] = digest(corpus / "dataset.json")
        config["nba_reference_artifacts"] = {
            f"{split}-ground-truth.npz": digest(corpus / f"{split}-ground-truth.npz")
            for split in ("tune", "test")}
    target = generated / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(config, indent=2) + "\n")
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", choices=sorted(STUDIES))
    parser.add_argument("--core", type=int, default=min(os.sched_getaffinity(0)))
    parser.add_argument("--threads", type=int, default=min(48, len(os.sched_getaffinity(0))))
    parser.add_argument("--dry-run", action="store_true", help="Print commands without constructing or timing indexes")
    args = parser.parse_args()
    if args.core not in os.sched_getaffinity(0) or not 1 <= args.threads <= len(os.sched_getaffinity(0)):
        raise ValueError("Choose an available CPU and a thread count within the CPU affinity mask")
    module, name, folder = STUDIES[args.study]
    if args.dry_run:
        print(f"Protocol: configs/{name}; output: results/{folder}")
        print("Dependencies: see docs/REPRODUCIBILITY.md (no output generated)")
        print(shlex.join([sys.executable, "-m", f"fresh.{module}", "--config",
                          f"results/protocols/{name}", "--output", f"results/{folder}"]))
        return
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError("Run this entry point from the repository root")
    config = prepare_config(name, args.core, args.threads, ROOT / "results/protocols")
    env = dict(os.environ, NUMBA_NUM_THREADS=str(args.threads), PIN_CORE=str(args.core),
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    output = ROOT / "results" / folder
    commands = [[sys.executable, "-u", "-m", f"fresh.{module}", "--config", str(config), "--output", str(output)]]
    if module == "poc" and args.study != "smoke":
        commands.extend([[sys.executable, "-m", "fresh.plot_poc", str(output)],
                         [sys.executable, "-m", "fresh.aggregate_replications", str(output)]])
        if args.study == "selectivity":
            commands.append([sys.executable, "-m", "fresh.plot_selectivity", str(output)])
    for command in commands:
        print(shlex.join(command), flush=True)
        subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
