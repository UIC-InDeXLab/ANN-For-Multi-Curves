"""Canonical-collision checks and resource-bounded fresh E1 evaluation."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
from itertools import product
from math import factorial

import numpy as np
from numba import njit, prange

from curveset import core, data
from curveset.bench import pinned
from curveset.costmodel import CostModel
from curveset.implicit import fill_table
from curveset.index import Tables, _probe_orders, index_bytes, make_index, theory_params
from curveset.orders import all_orders, balanced_blocking, blocking_families, blocking_points, partitions


@njit(cache=True)
def canonical_cells(C, shift, delta):
    cells = np.empty((C.shape[0], C.shape[1]), np.int64)
    count = 0
    for j in range(C.shape[0]):
        same = count > 0
        for a in range(C.shape[1]):
            value = int(np.floor((C[j, a] - shift[a]) / delta + 0.5))
            cells[count, a] = value
            if count > 0 and value != cells[count - 1, a]:
                same = False
        if not same:
            count += 1
    return cells[:count]


@njit(cache=True)
def canonical_equal(A, B, shift, delta):
    a = canonical_cells(A, shift, delta)
    b = canonical_cells(B, shift, delta)
    return a.shape[0] == b.shape[0] and np.array_equal(a, b)


@njit(cache=True, parallel=True)
def collision_samples(P, Q, independent, shared, delta):
    pairs, trials, k, _ = independent.shape
    joint = np.zeros((pairs, trials), np.bool_)
    shared_joint = np.zeros((pairs, trials), np.bool_)
    slots = np.zeros((pairs, trials, k), np.bool_)
    mismatch = np.zeros((pairs, trials), np.int64)
    for q in prange(pairs):
        for u in range(trials):
            hit, common = True, True
            for i in range(k):
                eq = canonical_equal(P[q, i], Q[q, i], independent[q, u, i], delta)
                slots[q, u, i] = eq
                hit = hit and eq
                fp = core.curve_fp(P[q, i], independent[q, u, i], delta)
                fq = core.curve_fp(Q[q, i], independent[q, u, i], delta)
                mismatch[q, u] += int((fp == fq) != eq)
                common = common and canonical_equal(P[q, i], Q[q, i], shared[q, u], delta)
            joint[q, u], shared_joint[q, u] = hit, common
    return joint, shared_joint, slots, mismatch


def star_pair(k, m, d, delta, r, phase=0.0):
    components = 2 * (m // 3)
    if components < 2 or 2 * components * r >= delta:
        raise ValueError("Star residues require at least two separated components and 2*s*r < delta")
    P, Q = [], []
    for i in range(k):
        a, b = [], []
        for j in range(components):
            center = np.zeros(d)
            center[0] = phase + i * 256 * delta + j * (4 * delta + delta / components)
            left, right = center.copy(), center.copy()
            left[0] -= r
            right[0] += r
            if j % 2 == 0:
                a.append(center)
                b.extend((left, right))
            else:
                a.extend((left, right))
                b.append(center)
        a.extend([a[-1]] * (m - len(a)))
        b.extend([b[-1]] * (m - len(b)))
        P.append(a)
        Q.append(b)
    return np.asarray(P), np.asarray(Q), 1 - 2 * components * r / delta


def matched_pairs(rng, workload, count, k, m, d, delta, r):
    if workload == "random":
        P = data.collection(rng, count, k, m, d, 100 * delta, 0.05 * delta)
        return P, data.jitter(rng, P, r), None
    if workload == "aligned":
        base = data.random_walks(rng, count, m, d, 100 * delta, 0.05 * delta)
        other = data.jitter(rng, base, r)
        offsets = np.zeros((k, d))
        offsets[:, 0] = np.arange(k) * 256 * delta
        return base[:, None] + offsets[None, :, None], other[:, None] + offsets[None, :, None], None
    if workload == "separated-star":
        values = [star_pair(k, m, d, delta, r, rng.uniform(0, delta)) for _ in range(count)]
        return np.stack([v[0] for v in values]), np.stack([v[1] for v in values]), values[0][2]
    raise ValueError(f"Unknown workload: {workload}")


@njit(cache=True)
def factorization_counts(stored, probed):
    k = stored.shape[1]
    facts = np.ones(k + 1, np.int64)
    for i in range(1, k + 1):
        facts[i] = facts[i - 1] * i
    seen = np.zeros(facts[k], np.int64)
    pi = np.empty(k, np.int64)
    for a in range(stored.shape[0]):
        for b in range(probed.shape[0]):
            for i in range(k):
                pi[probed[b, i]] = stored[a, i]
            rank = 0
            for i in range(k):
                smaller = 0
                for j in range(i + 1, k):
                    smaller += int(pi[j] < pi[i])
                rank += smaller * facts[k - 1 - i]
            seen[rank] += 1
    return seen.min(), seen.max()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def arrays_hash(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def write_rows(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def append_row(path, row):
    path = Path(path)
    new = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)


def resident_bytes(n, a, b, k, d, L):
    return index_bytes(n * a, L) + (a + b) * k + L * k * d * 8 + 2 * k * k


def selected_blocking(model, config, n, k, d, L):
    candidates = []
    for a, b, sizes, swapped in blocking_points(k):
        memory = resident_bytes(n, a, b, k, d, L)
        seconds = model.time(n, L, a, b)
        if memory <= config["index_budget_bytes"] and seconds <= config["query_deadline_seconds"]:
            cost = memory / config["index_budget_bytes"] + seconds / config["query_deadline_seconds"]
            candidates.append((cost, a, b, sizes, swapped))
    return min(candidates) if candidates else None


def audit_factorizations(limit):
    rows = []
    for k in range(2, limit + 1):
        for sizes in partitions(k):
            for swapped in (False, True):
                S, T = blocking_families(sizes, swapped)
                low, high = factorization_counts(S, T)
                if low != 1 or high != 1 or S.shape[0] * T.shape[0] != factorial(k):
                    raise AssertionError(f"Factorization failed: {k}, {sizes}, {swapped}")
                rows.append(dict(k=k, blocks="+".join(map(str, sizes)), swapped=swapped,
                                 stored=len(S), probed=len(T), min_multiplicity=low, max_multiplicity=high))
    return rows


def calibrate(config, path, d=2):
    Q = np.cumsum(np.random.default_rng(901).normal(size=(8, config["m"], d)), axis=1)
    shifts = np.random.default_rng(902).uniform(0, 8, (1, 8, d))
    family = all_orders(8)
    need = np.ones((8, 8), np.bool_)
    values = []
    for N in config["calibration_sizes"]:
        table = Tables(1, N)
        keys = np.random.default_rng(N).integers(0, 2**63, N, dtype=np.uint64)
        fill_table(keys, 1, table.B, table.keys, table.ids, table.dir)
        mark = np.zeros(N, np.int32)
        cand = np.empty(N, np.int64)
        _probe_orders(Q, shifts, 8, family, need, table.keys, table.ids, table.dir,
                      table.N, table.B, 0, 1, False, mark, 1, cand, 0)
        samples = []
        with pinned():
            for repeat in range(config["calibration_repetitions"]):
                t0 = time.perf_counter()
                for inner in range(5):
                    _probe_orders(Q, shifts, 8, family, need, table.keys, table.ids, table.dir,
                                  table.N, table.B, 0, 1, False, mark, 2 + repeat * 5 + inner, cand, 0)
                samples.append((time.perf_counter() - t0) / (5 * len(family)))
        values.append(float(np.median(samples)))
    write_json(path, dict(entries=config["calibration_sizes"], seconds_per_probe=values,
                          note="Fresh E1 selector; includes prefix hashing and empty-bucket lookup, not verification",
                          d=d, k=8, m=config["m"],
                          samples_per_size=config["calibration_repetitions"]))
    return CostModel(config["calibration_sizes"], values)


def collection(config, workload, k, d, delta):
    rng = np.random.default_rng(config["dataset_seed"] + 100 * d + k)
    n, nq = config["background_n"], config["test_queries"]
    X, paired, _ = matched_pairs(rng, workload, n, k, config["m"], d, delta, config["r"])
    targets = rng.choice(n, nq, replace=False)
    Q = np.stack([paired[p, rng.permutation(k)] for p in targets])
    X = data.add_near_misses(rng, X, targets, config["near_miss_radii"])
    return X, Q, targets


def build_index(index, X):
    t0 = time.perf_counter()
    n, a = len(X), len(index.stored)
    index.tables = T = Tables(index.L, n * a)
    for u in range(index.L):
        F = core.slot_fps(X, index.shifts[u], index.delta, index.need_s)
        keys = core.order_keys(F, index.stored).ravel()
        fill_table(keys, a, T.B, T.keys[u * T.N:(u + 1) * T.N], T.ids[u * T.N:(u + 1) * T.N],
                   T.dir[u * (T.B + 1):(u + 1) * (T.B + 1)])
    index._buffers(n)
    index.build_seconds = time.perf_counter() - t0
    return index


def query(index, X, Q, mode, deadline, chunk, r=1.0):
    t0 = time.perf_counter()
    T, stamp = index.tables, index._next_stamp()
    ncand, probes = 0, 0
    for u in range(index.L):
        for start in range(0, len(index.probed), chunk):
            hit, done, ncand = _probe_orders(
                Q, index.shifts, index.delta, index.probed[start:start + chunk], index.need_q,
                T.keys, T.ids, T.dir, T.N, T.B, u, u + 1, mode == "first",
                index.mark, stamp, index.cand, ncand)
            probes += done
            if time.perf_counter() - t0 > deadline:
                return dict(id=-1, candidates=ncand, probes=probes, seconds=time.perf_counter() - t0,
                            timeout=True, verified=0)
            if mode == "first" and hit >= 0:
                return dict(id=int(hit), candidates=1, probes=probes, seconds=time.perf_counter() - t0,
                            timeout=False, verified=0)
    best, result, verified, relevant = np.inf, -1, 0, 0
    if mode == "verify":
        for p in index.cand[:ncand]:
            if time.perf_counter() - t0 > deadline:
                return dict(id=-1, candidates=ncand, probes=probes, seconds=time.perf_counter() - t0,
                            timeout=True, verified=verified)
            if core.endpoint_lb(X[p], Q) <= max(best, r):
                dd = core.dmatch(X[p], Q)
                verified += 1
                relevant += int(dd <= r)
                if dd < best:
                    best, result = dd, int(p)
    elapsed = time.perf_counter() - t0
    return dict(id=result if elapsed <= deadline else -1, candidates=ncand, probes=probes, seconds=elapsed,
                timeout=elapsed > deadline, verified=verified, relevant_found=relevant)


@njit(cache=True, parallel=True)
def exact_ground_truth(X, Qs, r):
    distances = np.empty(len(Qs))
    counts = np.zeros(len(Qs), np.int64)
    for q in prange(len(Qs)):
        best = np.inf
        for p in range(len(X)):
            if core.endpoint_lb(X[p], Qs[q]) <= max(best, r):
                value = core.dmatch(X[p], Qs[q])
                counts[q] += int(value <= r)
                best = min(best, value)
        distances[q] = best
    return distances, counts


def worker(config, output, workload, k, d):
    tp = theory_params(d, config["m"], config["r"], config["p"], k, config["eta"])
    X, Qs, targets = collection(config, workload, k, d, tp["delta"])
    for q, p in enumerate(targets):
        if core.dmatch(X[p], Qs[q]) > config["r"] + 1e-8:
            raise AssertionError("Planted query is not within r")
    gt, nrel = exact_ground_truth(X, Qs, config["r"])
    np.savez(output / "dataset-ground-truth.npz", X=X, Q=Qs, targets=targets, nn_distance=gt, relevant=nrel)
    manifest = json.loads((output.parent / "manifest.json").read_text())
    dataset_hash, query_hash = arrays_hash(X), arrays_hash(Qs)
    write_json(output / "dataset.json", dict(dataset_sha256=dataset_hash, queries_sha256=query_hash,
                                            ground_truth_sha256=arrays_hash(gt, nrel),
                                            parameters=tp, background_n=config["background_n"], total_n=len(X)))
    calibration = output.parent / f"calibration-d{d}.json"
    if not calibration.exists() and config["stage"] != "full":
        calibration = output.parent / "calibration.json"
    model = CostModel.load(calibration)
    choices = [("probe_all", None, False, 1, factorial(k)),
               ("store_all", None, False, factorial(k), 1)]
    a, b, sizes, swap = balanced_blocking(k)
    choices.append(("balanced", sizes, swap, a, b))
    selected = selected_blocking(model, config, len(X), k, d, tp["L"])
    if selected is not None:
        choices.append(("cost-model", selected[3], selected[4], selected[1], selected[2]))
    requested=config.get("methods")
    if requested is not None:
        choices=[choice for choice in choices if choice[0] in requested]
        if not choices:
            raise ValueError("None of the requested E1 methods has a feasible family")
    statuses = []
    for seed in config["index_seeds"]:
        for name, sizes, swapped, a, b in choices:
            table_bytes = index_bytes(len(X) * a, tp["L"])
            mem = resident_bytes(len(X), a, b, k, d, tp["L"])
            base = dict(commit=manifest["commit"], config_sha256=manifest["config_sha256"],
                        dataset_sha256=dataset_hash, queries_sha256=query_hash,
                        stage=config["stage"], workload=workload, k=k, d=d, n=len(X), seed=seed, method=name,
                        blocks="+".join(map(str, sizes)) if sizes else "", swapped=swapped,
                        stored=a, probed=b, L=tp["L"], delta=tp["delta"], index_bytes=mem,
                        table_bytes=table_bytes, modeled_table_bytes=13 * len(X) * a * tp["L"],
                        curve_store_bytes=X.nbytes, query_scratch_bytes=len(X) * 12)
            cap = config.get("resident_index_cap_bytes", config.get("pilot_index_cap_bytes", config["index_budget_bytes"]))
            if mem > config["index_budget_bytes"] or mem > cap:
                status = "over index memory (computed)" if mem > config["index_budget_bytes"] else "not run: configured index cap"
                statuses.append(dict(base, mode="both", status=status))
                write_json(output / "configurations.json", statuses)
                continue
            if mem + X.nbytes + len(X) * a * 32 > config["peak_ram_cap_bytes"]:
                statuses.append(dict(base, mode="both", status="not run: peak RAM/capacity"))
                write_json(output / "configurations.json", statuses)
                continue
            method = name if name in ("probe_all", "store_all") else "blocking"
            print(f"BUILD {workload} d={d} k={k} {name} seed={seed} bytes={mem}", flush=True)
            write_json(output / "active_configuration.json", dict(base, phase="building"))
            signal.alarm(config["build_deadline_seconds"])
            try:
                idx = build_index(make_index(method, k, d, tp["delta"], tp["L"],
                                             sizes=sizes, swapped=swapped, seed=seed), X)
            finally:
                signal.alarm(0)
            actual = (idx.index_bytes + idx.stored.nbytes + idx.probed.nbytes + idx.shifts.nbytes
                      + idx.need_s.nbytes + idx.need_q.nbytes)
            if actual != mem:
                raise AssertionError(f"Index allocation mismatch {actual} != {mem}")
            for mode in config.get("query_modes",("first","verify")):
                estimated = tp["L"] * b * model.tau(len(X) * a) * len(Qs)
                estimated_cap = config["configuration_estimated_query_cap_seconds"]
                if mode == "verify" and estimated_cap is not None and estimated > estimated_cap:
                    statuses.append(dict(base, mode=mode, status="not run: estimated configured query time",
                                         estimated_configuration_seconds=estimated))
                    write_json(output / "configurations.json", statuses)
                    continue
                complete = True
                write_json(output / "active_configuration.json", dict(base, phase="warm-up", mode=mode))
                with pinned():
                    for q in range(config["warmup_queries"]):
                        query(idx, X, X[q % config["background_n"]], mode,
                              config["query_deadline_seconds"], config["probe_chunk"])
                    for q, Q in enumerate(Qs):
                        if q % 25 == 0:
                            write_json(output / "active_configuration.json",
                                       dict(base, phase="test", mode=mode, completed_queries=q))
                        res = query(idx, X, Q, mode, config["query_deadline_seconds"], config["probe_chunk"])
                        p = res["id"]
                        distance = float(core.dmatch(X[p], Q)) if p >= 0 else None
                        relevant_found = (int(p >= 0 and distance <= config["r"]) if mode == "first"
                                          else res.get("relevant_found", 0))
                        if distance is not None and distance > tp["c"] * config["r"] + 1e-7:
                            raise AssertionError(f"Certificate violation: {distance}")
                        row = dict(base, phase="test", mode=mode, query=q,
                                         query_sha256=arrays_hash(Q), seconds=res["seconds"],
                                         probes=res["probes"], candidates=res["candidates"], verified=res["verified"],
                                         found=p >= 0, returned_distance=distance, nn_distance=float(gt[q]),
                                         relevant=int(nrel[q]), relevant_found=relevant_found,
                                         recall=relevant_found / nrel[q] if nrel[q] else None,
                                         precision=relevant_found / res["candidates"] if res["candidates"] else None,
                                         within_cr=p >= 0,
                                         exact_nn=p >= 0 and distance <= gt[q] + 1e-8,
                                         status="over query time (measured)" if res["timeout"] else "ok")
                        append_row(output / "queries.csv", row)
                        if res["timeout"]:
                            complete = False
                            break
                statuses.append(dict(base, mode=mode, build_seconds=idx.build_seconds,
                                     status="complete" if complete else "over query time (measured)"))
                write_json(output / "configurations.json", statuses)
                print(f"CONFIG {workload} d={d} k={k} {name} seed={seed} mode={mode} "
                      f"{statuses[-1]['status']}", flush=True)
            del idx
            write_json(output / "configurations.json", statuses)
    write_json(output / "configurations.json", statuses)


def collisions(config, output):
    rows = []
    manifest = json.loads((output / "manifest.json").read_text())
    raw = output / "collision-samples"
    raw.mkdir()
    for d in config["dimensions"]:
        for k in config["collision_k"]:
            tp = theory_params(d, config["m"], config["r"], config["p"], k, config["eta"])
            for workload in config["workloads"]:
                P, Q, expected = matched_pairs(
                    np.random.default_rng(config["dataset_seed"] + d * 100 + k), workload,
                    config["collision_pairs"], k, config["m"], d, tp["delta"], config["r"])
                for seed in config["index_seeds"]:
                    shape = (len(P), config["shifts_per_pair"], k, d)
                    rng = np.random.default_rng(seed)
                    independent = rng.uniform(0, tp["delta"], shape)
                    shared = rng.uniform(0, tp["delta"], shape[:2] + (d,))
                    joint, common, slots, mismatch = collision_samples(P, Q, independent, shared, tp["delta"])
                    if mismatch.any():
                        raise AssertionError("Fingerprints and canonical cells disagree")
                    label = f"{workload}-d{d}-k{k}-seed{seed}"
                    np.savez_compressed(raw / f"{label}.npz", independent=joint, shared=common, slots=slots)
                    rate = float(joint.mean())
                    rows.append(dict(commit=manifest["commit"], config_sha256=manifest["config_sha256"],
                                     dataset_sha256=arrays_hash(P, Q), phase="collision diagnostic", stage=config["stage"],
                                     workload=workload, d=d, k=k, seed=seed, pairs=len(P),
                                     trials_per_pair=config["shifts_per_pair"], delta=tp["delta"],
                                     slot_rate=float(slots.mean()), joint_rate=rate, shared_rate=float(common.mean()),
                                     bound=config["p"]**k, bound_ratio=rate / config["p"]**k,
                                     construction_slot_probability=expected,
                                     construction_joint_probability=expected**k if expected is not None else None,
                                     fingerprint_mismatches=int(mismatch.sum())))
                    write_rows(output / "collisions.csv", rows)
            print(f"COLLISIONS d={d} k={k} complete", flush=True)


def terminate_worker(process):
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def monitor_worker(process, folder, config, remaining):
    started = time.monotonic()
    peak, sampled = 0, {}
    last_report = started
    interruption = None
    while process.poll() is None:
        try:
            status = Path(f"/proc/{process.pid}/status").read_text()
        except FileNotFoundError:
            if process.poll() is not None:
                break
            raise
        memory = {}
        for line in status.splitlines():
            if line.startswith(("VmRSS:", "VmHWM:")):
                key, amount, unit = line.split()
                if unit != "kB":
                    raise RuntimeError(f"Unexpected process-memory unit: {unit}")
                memory[key.rstrip(":")] = int(amount) * 1024
        rss = memory.get("VmRSS", 0)
        peak = max(peak, memory.get("VmHWM", rss))
        active_path = folder / "active_configuration.json"
        active = json.loads(active_path.read_text()) if active_path.exists() else {}
        label = f"{active.get('method', 'preparation')}-seed{active.get('seed', 'none')}"
        sampled[label] = max(sampled.get(label, 0), rss)
        if peak > config["peak_ram_cap_bytes"]:
            interruption = "incomplete: measured peak RAM limit"
        elif time.monotonic() - started > remaining:
            interruption = "incomplete: run cap"
        else:
            minimum_available = config.get("minimum_available_ram_bytes", 0)
            if minimum_available:
                available = next(int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines()
                                 if line.startswith("MemAvailable:"))
                if available < minimum_available:
                    interruption = "incomplete: server memory headroom"
        if interruption is not None:
            terminate_worker(process)
            write_json(folder / "interruption.json", dict(status=interruption, active=active))
            break
        if time.monotonic() - last_report >= 30:
            write_json(folder / "resources.json", dict(peak_process_bytes=peak,
                                                       sampled_peak_rss_by_configuration=sampled,
                                                       monitor_interval_seconds=config.get("monitor_interval_seconds", 1),
                                                       elapsed_seconds=time.monotonic() - started))
            print(f"PROGRESS {folder.name} {active.get('method', 'preparation')} "
                  f"seed={active.get('seed', '-')} phase={active.get('phase', 'ground-truth')} "
                  f"mode={active.get('mode', '-')} q={active.get('completed_queries', '-')} "
                  f"peak_GB={peak / 1e9:.2f}", flush=True)
            last_report = time.monotonic()
        time.sleep(config.get("monitor_interval_seconds", 1))
    code = process.wait()
    write_json(folder / "resources.json", dict(peak_process_bytes=peak,
                                               sampled_peak_rss_by_configuration=sampled,
                                               monitor_interval_seconds=config.get("monitor_interval_seconds", 1),
                                               elapsed_seconds=time.monotonic() - started, returncode=code,
                                               interruption=interruption))
    return code, interruption


def main(args):
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    output = Path(args.output).resolve()
    if args.worker:
        worker(config, output, args.workload, args.k, args.d)
        return
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_bytes(config_path.read_bytes())
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise RuntimeError("Freeze a clean experiment-code commit before running")
    import numba
    import scipy
    cpu = next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
               if line.startswith("model name"))
    memory = {line.split()[0].rstrip(":"): int(line.split()[1]) * 1024
              for line in Path("/proc/meminfo").read_text().splitlines()
              if line.startswith(("MemTotal:", "MemAvailable:"))}
    write_json(output / "manifest.json", dict(
        commit=commit, config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        python=sys.version, numpy=np.__version__, numba=numba.__version__, scipy=scipy.__version__,
        machine=platform.uname()._asdict(), cpu_model=cpu, memory_at_start=memory,
        affinity=sorted(os.sched_getaffinity(0)), actual_numba_threads=numba.get_num_threads(),
        pin_core=config["pin_core"], build_threads=config["build_threads"], started=time.time(),
        stage=config["stage"], background_n=config["background_n"], test_queries=config["test_queries"],
        model_scope="E1 probe-only family selector; not a candidate-aware C6 performance validation"))
    os.environ["PIN_CORE"] = str(config["pin_core"])
    started = time.monotonic()
    write_rows(output / "factorizations.csv", audit_factorizations(config["factorization_max_k"]))
    for d in config.get("calibration_dimensions", [2]):
        calibrate(config, output / f"calibration-d{d}.json", d)
        if d == 2:
            (output / "calibration.json").write_bytes((output / "calibration-d2.json").read_bytes())
    collisions(config, output)
    results = []
    for k, d, workload in product(config["index_k"], config["dimensions"], config["workloads"]):
        remaining = config["run_cap_seconds"] - (time.monotonic() - started)
        if remaining <= 0:
            results.append(dict(d=d, k=k, workload=workload, status="not run: run cap"))
            write_json(output / "cases.json", results)
            continue
        folder = output / f"{workload}-d{d}-k{k}"
        folder.mkdir()
        command = [sys.executable, "-m", "fresh.e1", "--worker", "--config", str(config_path),
                   "--output", str(folder), "--workload", workload, "--k", str(k), "--d", str(d)]
        with (folder / "worker.log").open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            try:
                code, interruption = monitor_worker(process, folder, config, remaining)
            finally:
                terminate_worker(process)
        status = interruption or ("complete" if code == 0 else "build deadline" if code == -signal.SIGALRM else "implementation failure")
        results.append(dict(d=d, k=k, workload=workload, status=status, returncode=code))
        write_json(output / "cases.json", results)
        print(f"CASE {workload} d={d} k={k} {status}", flush=True)
        if interruption is not None:
            raise RuntimeError(f"Worker interrupted: {status}; inspect {folder}")
        if code != 0:
            raise RuntimeError(f"Worker stopped with {status}; inspect {folder / 'worker.log'}")
    write_json(output / "completion.json", dict(elapsed_seconds=time.monotonic() - started,
                                               completed=time.time(), stage=config["stage"]))
    print(f"E1 {config['stage']} finished; E2 was not launched", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--workload")
    parser.add_argument("--k", type=int)
    parser.add_argument("--d", type=int)
    main(parser.parse_args())
