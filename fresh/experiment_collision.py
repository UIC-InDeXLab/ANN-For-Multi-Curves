"""Empirical multi-curve collision rates versus the slot-wise lower bound p^k."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numba
import numpy as np
from numba import njit, prange

from curveset import core, data
from fresh.e1 import arrays_hash, write_json


PAIR_MULT = np.uint64(0x9E3779B97F4A7C15)
TRIAL_MULT = np.uint64(0xBF58476D1CE4E5B9)
SLOT_MULT = np.uint64(0x94D049BB133111EB)
AXIS_MULT = np.uint64(0xD6E8FEB86659FD93)
U53 = 1.0 / 9007199254740992.0


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


def case_seed(base, k, p_index):
    return int(np.uint64(base) + np.uint64(10_007 * k + 104_729 * (p_index + 1)))


def shift_seed(base, k, p_index):
    return int(np.uint64(base) + np.uint64(65_537 * k + 1_000_003 * (p_index + 1)))


def pair_collection(seed, pairs, k, m, d, extent, step_sd, radius):
    rng = np.random.default_rng(seed)
    P = data.collection(rng, pairs, k, m, d, extent=extent, step=step_sd)
    Q = data.jitter(rng, P, radius)
    return P, Q


@njit(cache=True, inline="always")
def shift_coordinate(base_seed, pair_index, trial_index, slot_index, axis, delta):
    x = np.uint64(base_seed)
    x += np.uint64(pair_index + 1) * PAIR_MULT
    x += np.uint64(trial_index + 1) * TRIAL_MULT
    x += np.uint64(slot_index + 1) * SLOT_MULT
    x += np.uint64(axis + 1) * AXIS_MULT
    x = core.mix64(x)
    return delta * np.float64(x >> np.uint64(11)) * U53


def shift_vector(base_seed, pair_index, trial_index, slot_index, d, delta):
    return np.array(
        [shift_coordinate(base_seed, pair_index, trial_index, slot_index, axis, delta) for axis in range(d)],
        dtype=np.float64,
    )


@njit(cache=True, parallel=True)
def measure_collision_trials(P, Q, delta, trials, base_seed):
    pairs, k, _, d = P.shape
    joint = np.zeros((pairs, trials), np.bool_)
    slot_counts = np.zeros((pairs, k), np.int64)
    for pair_index in prange(pairs):
        shift = np.empty(d, np.float64)
        for trial_index in range(trials):
            all_hit = True
            for slot_index in range(k):
                for axis in range(d):
                    shift[axis] = shift_coordinate(base_seed, pair_index, trial_index, slot_index, axis, delta)
                left = core.curve_fp(P[pair_index, slot_index], shift, delta)
                right = core.curve_fp(Q[pair_index, slot_index], shift, delta)
                hit = left == right
                slot_counts[pair_index, slot_index] += int(hit)
                all_hit = all_hit and hit
            joint[pair_index, trial_index] = all_hit
    return joint, slot_counts


def frequency_stats(values):
    values = np.asarray(values, dtype=np.float64)
    return dict(
        mean=float(values.mean()),
        min=float(values.min()),
        p05=float(np.quantile(values, 0.05)),
        max=float(values.max()),
    )


def any_hit_probabilities(joint, max_u):
    pairs, trials = joint.shape
    values = []
    for u in range(1, max_u + 1):
        batches = trials // u
        window = joint[:, : batches * u].reshape(pairs, batches, u)
        pair_freq = window.any(axis=2).mean(axis=1)
        values.append(pair_freq)
    return values


def first_u_reaching(values, target):
    for offset, value in enumerate(values, start=1):
        if value >= target:
            return offset
    return None


def violation_summary(frequencies, bound, trials, z):
    frequencies = np.asarray(frequencies, dtype=np.float64)
    margin = z * math.sqrt(bound * (1.0 - bound) / trials)
    deficits = bound - frequencies
    raw = deficits > 0
    significant = deficits > margin
    return dict(
        bound=float(bound),
        monte_carlo_sigma_reference=float(math.sqrt(bound * (1.0 - bound) / trials)),
        z_threshold=float(z),
        raw_below_bound=int(raw.sum()),
        significant_below_bound=int(significant.sum()),
        worst_raw_gap=float(deficits.max()),
        worst_frequency=float(frequencies.min()),
        margin=float(margin),
    )


def collision_case(config, k, p_value, p_index):
    delta = 2.0 * config["d"] * config["m"] * config["r"] / (1.0 - p_value)
    planted_seed = case_seed(config["dataset_seed"], k, p_index)
    seeded_shift = shift_seed(config["shift_seed"], k, p_index)
    P, Q = pair_collection(
        planted_seed,
        config["pairs"],
        k,
        config["m"],
        config["d"],
        config["extent"],
        config["step_sd"],
        config["r"],
    )
    joint, slot_counts = measure_collision_trials(P, Q, delta, config["trials"], seeded_shift)
    slot_freqs = slot_counts / config["trials"]
    slot_pair_means = slot_freqs.mean(axis=1)
    joint_freqs = joint.mean(axis=1)
    any_hit = any_hit_probabilities(joint, config["max_tables_u"])
    joint_bound = p_value ** k
    theory_any = [1.0 - (1.0 - joint_bound) ** u for u in range(1, config["max_tables_u"] + 1)]
    empirical_mean_any = [float(v.mean()) for v in any_hit]
    empirical_min_any = [float(v.min()) for v in any_hit]
    empirical_p05_any = [float(np.quantile(v, 0.05)) for v in any_hit]
    return dict(
        k=k,
        p=float(p_value),
        delta=float(delta),
        pairs=config["pairs"],
        trials=config["trials"],
        pair_seed=planted_seed,
        shift_seed=seeded_shift,
        dataset_sha256=arrays_hash(P, Q),
        joint_bound=float(joint_bound),
        single_curve_bound=float(p_value),
        joint_pair_frequency=frequency_stats(joint_freqs),
        slot_pair_mean_frequency=frequency_stats(slot_pair_means),
        slot_all_frequency=frequency_stats(slot_freqs.reshape(-1)),
        any_table_collision=dict(
            u_values=list(range(1, config["max_tables_u"] + 1)),
            empirical_mean=empirical_mean_any,
            empirical_min=empirical_min_any,
            empirical_p05=empirical_p05_any,
            theory=theory_any,
            u95_empirical_mean=first_u_reaching(empirical_mean_any, 0.95),
            u95_theory=first_u_reaching(theory_any, 0.95),
        ),
        guarantee_validation=dict(
            joint=violation_summary(joint_freqs, joint_bound, config["trials"], config["report_violation_z"]),
            single_curve=violation_summary(
                slot_freqs.reshape(-1), p_value, config["trials"], config["report_violation_z"]
            ),
        ),
    )


def csv_rows(summary):
    for case in summary["cases"]:
        joint = case["joint_pair_frequency"]
        slot = case["slot_pair_mean_frequency"]
        yield dict(
            k=case["k"],
            p=case["p"],
            delta=case["delta"],
            pairs=case["pairs"],
            trials=case["trials"],
            joint_bound=case["joint_bound"],
            joint_mean=joint["mean"],
            joint_min=joint["min"],
            joint_p05=joint["p05"],
            slot_bound=case["single_curve_bound"],
            slot_mean=slot["mean"],
            slot_min=slot["min"],
            slot_p05=slot["p05"],
            u95_empirical_mean=case["any_table_collision"]["u95_empirical_mean"],
            u95_theory=case["any_table_collision"]["u95_theory"],
            joint_raw_below_bound=case["guarantee_validation"]["joint"]["raw_below_bound"],
            joint_significant_below_bound=case["guarantee_validation"]["joint"]["significant_below_bound"],
            slot_raw_below_bound=case["guarantee_validation"]["single_curve"]["raw_below_bound"],
            slot_significant_below_bound=case["guarantee_validation"]["single_curve"]["significant_below_bound"],
        )


def run(config_path, output_path):
    started = time.time()
    workspace = Path(__file__).resolve().parents[1]
    config = json.loads(Path(config_path).read_text())
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=False)
    numba.set_num_threads(min(config["numba_threads"], numba.get_num_threads()))
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
    cases = []
    for k in config["k_values"]:
        for p_index, p_value in enumerate(config["p_values"]):
            print(f"collision-case k={k} p={p_value} start", flush=True)
            cases.append(collision_case(config, k, p_value, p_index))
            print(f"collision-case k={k} p={p_value} done", flush=True)
    summary = dict(
        experiment=config["experiment"],
        stage=config["stage"],
        created=time.time(),
        elapsed_seconds=time.time() - started,
        config=config,
        config_sha256=digest(config_path),
        code_commit=commit,
        code_dirty=dirty,
        cases=cases,
    )
    with (output / "summary.csv").open("w", newline="") as handle:
        rows = list(csv_rows(summary))
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
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
