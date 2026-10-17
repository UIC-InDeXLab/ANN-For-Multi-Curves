from itertools import permutations
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from curveset import core
from curveset.index import make_index, theory_params
from curveset.costmodel import CostModel
from fresh.e1 import (build_index, canonical_cells, canonical_equal, collision_samples, exact_ground_truth,
                      factorization_counts, monitor_worker, query, resident_bytes, selected_blocking,
                      star_pair, worker, write_json, write_rows)
from fresh.analyze_e1 import summarize


def reference_cells(C, shift, delta):
    points = [tuple(np.floor((v - shift) / delta + 0.5).astype(int)) for v in C]
    return tuple(p for i, p in enumerate(points) if i == 0 or p != points[i - 1])


def reference_frechet(A, B):
    table = np.full((len(A), len(B)), np.inf)
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            prior = 0 if i == j == 0 else min(
                table[i - 1, j] if i else np.inf,
                table[i, j - 1] if j else np.inf,
                table[i - 1, j - 1] if i and j else np.inf)
            table[i, j] = max(prior, np.linalg.norm(a - b))
    return table[-1, -1]


@pytest.mark.parametrize("d", [2, 3])
def test_star_vertices_distance_and_exact_collision_probability(d):
    tp = theory_params(d, 32, 1, 0.75, 4, 0.01)
    P, Q, expected = star_pair(4, 32, d, tp["delta"], 1)
    assert P.shape == Q.shape == (4, 32, d)
    for i in range(4):
        assert reference_frechet(P[i], Q[i]) == pytest.approx(1)
    assert core.dmatch(P, Q) == pytest.approx(1)
    shifts = np.zeros((1, 8192, 4, d))
    shifts[..., 0] = (np.arange(8192)[None, :, None] + 0.314159) * tp["delta"] / 8192
    independent, shared, slots, mismatch = collision_samples(
        P[None], Q[None], shifts, shifts[:, :, 0], tp["delta"])
    assert not mismatch.any()
    assert slots.mean() == pytest.approx(expected, abs=20 / 8192)
    assert independent.mean() == shared.mean()
    assert expected > 0.75


def test_canonical_keys_match_independent_reference():
    rng = np.random.default_rng(30)
    for _ in range(80):
        C = rng.normal(size=(int(rng.integers(1, 20)), 3))
        shift = rng.uniform(0, 2, 3)
        assert tuple(map(tuple, canonical_cells(C, shift, 2))) == reference_cells(C, shift, 2)
        assert canonical_equal(C, C, shift, 2)


@pytest.mark.parametrize("method,sizes,swapped", [
    ("probe_all", None, False), ("store_all", None, False),
    ("blocking", (2, 1), False), ("blocking", (2, 1), True)])
def test_query_candidate_sets_match_canonical_reference(method, sizes, swapped):
    rng = np.random.default_rng(111)
    X = rng.normal(size=(10, 3, 5, 2))
    Q = X[2, [2, 0, 1]] + 0.02
    index = build_index(make_index(method, 3, 2, 8, 2, sizes=sizes, swapped=swapped), X)
    expected = set()
    for u in range(index.L):
        stored = {
            tuple(reference_cells(X[p, s[i]], index.shifts[u, i], 8) for i in range(3)): p
            for p in range(len(X)) for s in index.stored
        }
        # Multiple parents can share a canonical key.
        for p in range(len(X)):
            keys = {tuple(reference_cells(X[p, s[i]], index.shifts[u, i], 8) for i in range(3))
                    for s in index.stored}
            probes = {tuple(reference_cells(Q[t[i]], index.shifts[u, i], 8) for i in range(3))
                      for t in index.probed}
            if keys & probes:
                expected.add(p)
        assert stored
    res = query(index, X, Q, "verify", 60, 2)
    assert not res["timeout"]
    assert set(index.cand[:res["candidates"]]) == expected
    assert core.dmatch(X[res["id"]], Q) == pytest.approx(min(core.dmatch(X[p], Q) for p in expected))
    first = query(index, X, Q, "first", 60, 2)
    assert first["id"] in expected
    timed = query(index, X, Q, "verify", 0, 2)
    assert timed["timeout"]


def test_ground_truth_counts_ties_and_radius_boundary():
    rng = np.random.default_rng(90)
    X = rng.normal(size=(8, 2, 4, 2))
    X[1] = X[0]
    Qs = X[:2].copy()
    distances, counts = exact_ground_truth(X, Qs, 0)
    assert np.array_equal(distances, [0, 0])
    assert np.array_equal(counts, [2, 2])
    for q, Q in enumerate(Qs):
        values = []
        for P in X:
            distances_by_pair = np.array([[reference_frechet(a, b) for b in P] for a in Q])
            values.append(min(max(distances_by_pair[i, p[i]] for i in range(2)) for p in permutations(range(2))))
        assert distances[q] == min(values)
        assert counts[q] == sum(value <= 0 for value in values)


def test_factorization_detects_duplicate_orders():
    stored = np.array([[0, 1], [0, 1]], np.int8)
    probed = np.array([[0, 1]], np.int8)
    assert factorization_counts(stored, probed) == (0, 2)


@pytest.mark.parametrize("stage", ["pilot", "full"])
def test_worker_logs_fresh_provenance_and_complete_queries(tmp_path, stage):
    config_path = Path(__file__).resolve().parents[1] / "configs" / "e1-pilot-20261012.json"
    config = json.loads(config_path.read_text())
    config.update(background_n=8, test_queries=4, index_seeds=[17], m=8, warmup_queries=1, stage=stage)
    write_json(tmp_path / "manifest.json", dict(commit="unit-fixture", config_sha256="unit-fixture"))
    name = "calibration-d2.json" if stage == "full" else "calibration.json"
    write_json(tmp_path / name, dict(entries=[1000], seconds_per_probe=[1e-8]))
    folder = tmp_path / "separated-star-d2-k2"
    folder.mkdir()
    worker(config, folder, "separated-star", 2, 2)
    statuses = json.loads((folder / "configurations.json").read_text())
    assert len(statuses) == 8 and all(row["status"] == "complete" for row in statuses)
    import csv
    with (folder / "queries.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 32
    assert all(row["commit"] == "unit-fixture" and len(row["dataset_sha256"]) == 64 for row in rows)
    assert all(row["within_cr"] == "True" and row["status"] == "ok" for row in rows)
    write_json(tmp_path / "config.json", config)
    write_rows(tmp_path / "collisions.csv", [dict(
        workload="separated-star", d=2, k=2, seed=17, bound=0.75**2,
        construction_joint_probability=None, shared_rate=1, fingerprint_mismatches=0)])
    raw = tmp_path / "collision-samples"
    raw.mkdir()
    np.savez(raw / "separated-star-d2-k2-seed17.npz", independent=np.ones((4, 2), bool))
    write_rows(tmp_path / "factorizations.csv", [dict(k=2, min_multiplicity=1, max_multiplicity=1)])
    result = summarize(tmp_path)
    assert result["stage"] == stage and not result["evaluation_finished"]
    assert len(result["configurations"]) == 8
    assert all(record["certificate_violations"] == 0 for record in result["configurations"])
    write_rows(folder / "queries.csv", rows[:-1])
    with pytest.raises(AssertionError, match="Incomplete query log"):
        summarize(tmp_path)


@pytest.mark.parametrize("limit,remaining,expected", [
    (1, 5, "incomplete: measured peak RAM limit"),
    (10**9, 0.02, "incomplete: run cap")])
def test_worker_monitor_stops_only_its_owned_process(tmp_path, limit, remaining, expected):
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    config = dict(peak_ram_cap_bytes=limit, monitor_interval_seconds=0.01)
    try:
        code, reason = monitor_worker(process, tmp_path, config, remaining)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    assert code != 0 and reason == expected
    resource = json.loads((tmp_path / "resources.json").read_text())
    assert resource["interruption"] == expected
    assert process.poll() is not None


def test_full_configuration_does_not_use_pilot_exclusions():
    path = Path(__file__).resolve().parents[1] / "configs" / "e1-full-20261012.json"
    config = json.loads(path.read_text())
    assert config["stage"] == "full" and config["test_queries"] == 200
    assert config["background_n"] == 10000 and config["index_k"] == list(range(2, 11))
    assert config["configuration_estimated_query_cap_seconds"] is None
    assert config["resident_index_cap_bytes"] == config["index_budget_bytes"] == 24e9


def test_block_selector_accounts_for_resident_auxiliaries():
    model = CostModel([1000], [1e-8])
    config = dict(index_budget_bytes=1, query_deadline_seconds=60)
    assert selected_blocking(model, config, 8, 4, 2, 2) is None
    config["index_budget_bytes"] = 10**6
    choice = selected_blocking(model, config, 8, 4, 2, 2)
    assert choice is not None
    _, a, b, _, _ = choice
    assert resident_bytes(8, a, b, 4, 2, 2) <= config["index_budget_bytes"]
