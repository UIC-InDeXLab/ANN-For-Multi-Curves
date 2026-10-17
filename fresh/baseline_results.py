"""Read the audited graph-based baseline runs for the paper tables and figures."""
import csv
import json
from pathlib import Path

import numpy as np

FOLDERS = ("baselines-20261015", "baselines-ksweep-20261015", "baselines-selectivity-20261015",
           "baselines-scale-20261015")
BASELINES = ("per-curve-hnsw-intersection",)


def _quality(path, baseline):
    """Mean excess and success within 1.05 and 1.1 of the optimum, with empty answers as failures."""
    rows = [r for r in csv.DictReader(path.open()) if r["baseline"] == baseline and r["phase"] == "test"]
    if not rows:
        raise ValueError(f"No test rows for {baseline} in {path}")
    answered = [r for r in rows if int(r["id"]) >= 0]
    best = np.array([float(r["nn_distance"]) for r in answered])
    value = np.array([float(r["distance"]) for r in answered])
    n = len(rows)
    return dict(c105=float(np.sum(value <= 1.05 * best + 1e-8)) / n,
                c110=float(np.sum(value <= 1.1 * best + 1e-8)) / n,
                excess=float(np.mean(np.maximum(0.0, value - best))) if answered else None,
                ratio=float(np.mean(value / np.where(best > 0, best, 1.0))) if answered else None,
                empty=1 - len(answered) / n)


def load(results):
    """Return {(dataset, baseline): [per-seed dicts]} for every complete baseline run found."""
    found = {}
    for folder in FOLDERS:
        root = Path(results) / folder
        summary = root / "summary.json"
        if not (root / "completion.json").is_file() or not summary.is_file():
            continue
        for row in json.loads(summary.read_text()):
            if row["baseline"] not in BASELINES:
                continue
            if row["test_status"] != "complete":
                raise ValueError(f"Incomplete baseline row {row['job']} {row['baseline']}")
            quality = _quality(root / row["job"] / "test-queries.csv", row["baseline"])
            found.setdefault((row["dataset"], row["baseline"]), []).append(dict(
                recall=row["test_recall_at_1"], median_ms=row["median_latency_ms"], p95_ms=row["p95_latency_ms"],
                index_mb=row["index_size_mb"], build_s=row["build_seconds"],
                top=row.get("chosen_top_k", row.get("chosen_top_c")), seed=row["index_seed"], **quality))
    for key, seeds in found.items():
        if len(seeds) != 3:
            raise ValueError(f"Expected three seeds for {key}, found {len(seeds)}")
    return found


def _mean(values):
    present = [v for v in values if v is not None]
    return float(np.mean(present)) if present else None


def aggregate(seeds):
    """Mean recall and the other medians over the three index seeds; worst seed for recall and latency."""
    return dict(
        recall=float(np.mean([s["recall"] for s in seeds])), recall_min=min(s["recall"] for s in seeds),
        median_ms=float(np.median([s["median_ms"] for s in seeds])), median_ms_max=max(s["median_ms"] for s in seeds),
        p95_ms=float(np.median([s["p95_ms"] for s in seeds])), index_mb=float(np.median([s["index_mb"] for s in seeds])),
        c105=float(np.mean([s["c105"] for s in seeds])), c110=float(np.mean([s["c110"] for s in seeds])),
        excess=_mean([s["excess"] for s in seeds]), ratio=_mean([s["ratio"] for s in seeds]),
        empty=float(np.mean([s["empty"] for s in seeds])), build_s=float(np.median([s["build_s"] for s in seeds])))
