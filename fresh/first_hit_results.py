"""Read the first-hit (one entry per key) index runs for the paper tables and figures."""
import csv
import json
from pathlib import Path

import numpy as np

FOLDERS = ("first-entry-a-20261016", "first-entry-b-20261016", "first-entry-1m-20261016",
           "first-entry-selectivity-20261016", "first-entry-nba-20261016")
METHODS = ("blocking", "probe_all", "store_all")


def _quality(path, method):
    rows = [r for r in csv.DictReader(path.open()) if r["display_method"] == method]
    answered = [r for r in rows if r["found"] == "True"]
    best = np.array([float(r["nn_distance"]) for r in answered])
    value = np.array([float(r["distance"]) for r in answered])
    n = len(rows)
    ratio = np.where(best > 0, value / np.where(best > 0, best, 1.0), 1.0)
    return dict(c105=float(np.sum(ratio <= 1.05 + 1e-8)) / n, c110=float(np.sum(ratio <= 1.1 + 1e-8)) / n,
                excess=float(np.mean(np.maximum(0.0, value - best))) if len(answered) else None,
                ratio=float(np.mean(ratio)) if len(answered) else None, empty=1 - len(answered) / n)


def load(results):
    """Return {(case_id, method): [per-seed dicts]} from every completed first-hit run found."""
    found = {}
    for folder in FOLDERS:
        root = Path(results) / folder
        if not (root / "completion.json").is_file() and not (root / "summary.json").is_file():
            continue
        for job in sorted(root.iterdir()):
            path = job / "main-results.json"
            if not job.is_dir() or not path.is_file():
                continue
            for row in json.loads(path.read_text()):
                if row.get("status") != "complete" or row["display_method"] not in METHODS:
                    continue
                quality = _quality(job / "test-queries.csv", row["display_method"])
                found.setdefault((row["case_id"], row["display_method"]), []).append(dict(
                    recall=row["recall_at_1"], median_ms=row["median_ms"], p95_ms=row["p95_ms"],
                    index_mb=row["index_mb"], delta=row["delta"], tables=row["target_L"], seed=row["index_seed"],
                    build_s=row["build_seconds"], **quality))
    return found


def aggregate(seeds):
    present = lambda key: [s[key] for s in seeds if s[key] is not None]
    mean = lambda values: float(np.mean(values)) if values else None
    return dict(
        recall=float(np.mean([s["recall"] for s in seeds])), recall_min=min(s["recall"] for s in seeds),
        median_ms=float(np.median([s["median_ms"] for s in seeds])), median_ms_max=max(s["median_ms"] for s in seeds),
        median_ms_min=min(s["median_ms"] for s in seeds), p95_ms=float(np.median([s["p95_ms"] for s in seeds])),
        index_mb=float(np.median([s["index_mb"] for s in seeds])), c105=mean(present("c105")), c110=mean(present("c110")),
        excess=mean(present("excess")), ratio=mean(present("ratio")), empty=mean(present("empty")),
        deltas=sorted({s["delta"] for s in seeds}), tables=sorted({s["tables"] for s in seeds}), seeds=len(seeds))
