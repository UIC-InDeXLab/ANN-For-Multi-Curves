"""Read the partition-frontier runs: one point per block configuration, the J_lambda selections, and the count model."""
import json
from math import factorial
from pathlib import Path

import numpy as np

FOLDER = "frontier-20261016"
K = 8


def _ranks(values):
    values = np.asarray(values, float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values))
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(x, y):
    rx, ry = _ranks(x), _ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _key(row):
    return ("+".join(map(str, row["blocks"])), bool(row["swapped"]))


def load(results, seeds_required=3):
    """Return {case_id: dict(points, stars, missing)} from the completed frontier study."""
    root = Path(results) / FOLDER
    summary_path = root / "summary.json"
    if not summary_path.is_file():
        return {}
    jobs = json.loads(summary_path.read_text())["jobs"]
    cases = {}
    for job in jobs:
        cases.setdefault(job["case_id"], []).append(job)
    out = {}
    for case_id, group in cases.items():
        per_config = {}
        for job in group:
            best = {}
            for row in job["rows"]:
                if row["status"] != "qualified":
                    continue
                key = _key(row)
                if key not in best or (row["tuning_median_seconds"], row["index_bytes"]) < (
                        best[key]["tuning_median_seconds"], best[key]["index_bytes"]):
                    best[key] = row
            for key, row in best.items():
                per_config.setdefault(key, []).append(row)
        points, missing = [], []
        for (blocks, swapped), rows in sorted(per_config.items()):
            if len(rows) < min(seeds_required, len(group)):
                missing.append(dict(blocks=blocks, swapped=swapped, seeds=len(rows)))
                continue
            points.append(dict(
                blocks=blocks, swapped=swapped, a=rows[0]["stored"], b=rows[0]["probed"],
                index_mb=float(np.median([r["index_bytes"] for r in rows]) / 1e6),
                median_ms=float(np.median([r["test_median_ms"] for r in rows])),
                recall=float(np.mean([r["test_recall"] for r in rows])),
                tables=sorted({r["L"] for r in rows}), deltas=sorted({r["delta"] for r in rows})))
        stars = {}
        for job in group:
            for weight, row in job["selections"].items():
                stars.setdefault(weight, []).append(row)
        star_points = {}
        for weight, rows in stars.items():
            star_points[weight] = dict(
                index_mb=float(np.median([r["index_bytes"] for r in rows]) / 1e6),
                median_ms=float(np.median([r["test_median_ms"] for r in rows])),
                recall=float(np.mean([r["test_recall"] for r in rows])),
                configs=sorted({f"{'+'.join(map(str, r['blocks']))}{'T' if r['swapped'] else ''}@{int(r['delta'])}/L{r['L']}" for r in rows}))
        out[case_id] = dict(points=points, stars=star_points, missing=missing,
                            excluded=sorted({(_key(r), r["status"]) for job in group for r in job["rows"]
                                             if r["status"] not in ("qualified", "tuning recall below target")}))
    return out


def count_model(points):
    """Index size grows with the stored orders a and latency with the probes b, scaled to the measured endpoints."""
    probe_all = min(points, key=lambda p: (p["a"], p["index_mb"]))
    largest = max(points, key=lambda p: (p["a"], p["index_mb"]))
    a = np.geomspace(probe_all["a"], largest["a"], 60)
    b = factorial(K) / a
    size = probe_all["index_mb"] + (largest["index_mb"] - probe_all["index_mb"]) * (a - probe_all["a"]) / (largest["a"] - probe_all["a"])
    latency = largest["median_ms"] + (probe_all["median_ms"] - largest["median_ms"]) * (b - largest["b"]) / (probe_all["b"] - largest["b"])
    return size, latency


def statistics(data):
    """Rank correlations of a with index size and of b with latency, and the largest rank disagreements."""
    report = {}
    for case_id, d in data.items():
        pts = d["points"]
        a = [p["a"] for p in pts]
        b = [p["b"] for p in pts]
        size = [p["index_mb"] for p in pts]
        latency = [p["median_ms"] for p in pts]
        rank = lambda v: _ranks(v)
        size_gap = np.abs(rank(a) - rank(size))
        lat_gap = np.abs(rank(b) - rank(latency))
        report[case_id] = dict(
            configurations=len(pts), spearman_a_size=spearman(a, size), spearman_b_latency=spearman(b, latency),
            size_range_mb=(min(size), max(size)), latency_range_ms=(min(latency), max(latency)),
            worst_size=[(pts[i]["blocks"], pts[i]["swapped"], pts[i]["a"], round(pts[i]["index_mb"], 1), float(size_gap[i]))
                        for i in np.argsort(-size_gap)[:5]],
            worst_latency=[(pts[i]["blocks"], pts[i]["swapped"], pts[i]["b"], round(pts[i]["median_ms"], 3), float(lat_gap[i]))
                           for i in np.argsort(-lat_gap)[:5]])
    return report
