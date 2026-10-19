"""Distance-quality reporting from actual frozen query returns; never retunes."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import re

import numpy as np

JOINT = ("blocking", "probe_all", "store_all")
METHODS = (*JOINT, "per_curve-intersection")
FH_ROOTS = ("first-entry-a-20261016", "first-entry-b-20261016",
            "first-entry-1m-20261016", "first-entry-selectivity-20261016", "first-entry-nba-20261016")
POC_ROOTS = {
    "fresh-synthetic-expanded-20261012": "comparison",
    "fresh-synthetic-scaleup-20261013": "comparison",
    "fresh-selectivity-expanded-20261013": "selectivity",
    "dataset-seed-b-20261015": "replication-b",
    "dataset-seed-c-20261015": "replication-c",
    "sensitivity-m16-20261015": "sensitivity-m16",
    "sensitivity-m64-20261015": "sensitivity-m64",
    "sensitivity-rho025-20261015": "sensitivity-rho025",
    "sensitivity-rho100-20261015": "sensitivity-rho100",
    "dtw-20261014": "dtw",
}


def flag(value):
    if value in (True, "True", "true", 1, "1"):
        return True
    if value in (False, "False", "false", 0, "0", None, ""):
        return False
    raise ValueError(f"Invalid Boolean measurement {value!r}")


def metrics(rows):
    if not rows:
        raise ValueError("No measured queries")
    optimum, distance, seconds, returned, candidates, probes, postings = [], [], [], [], [], [], []
    for row in rows:
        best = float(row["nn_distance"] if "nn_distance" in row else row["optimal_feet"])
        elapsed = float(row["seconds"])
        if not np.isfinite(best) or best < 0 or not np.isfinite(elapsed) or elapsed < 0:
            raise ValueError("Invalid reference distance or query time")
        found = int(row["id"]) >= 0 and not flag(row.get("timeout"))
        value = float(row["distance"]) if found else np.inf
        if found and (not np.isfinite(value) or value < best - 1e-7):
            raise ValueError("Invalid returned distance relative to exact reference")
        optimum.append(best)
        distance.append(value)
        seconds.append(elapsed)
        returned.append(found)
        for field, target in (("candidates", candidates), ("probes", probes), ("postings", postings)):
            value = row.get(field)
            if value not in (None, ""):
                target.append(float(value))
    optimum, distance, returned = np.array(optimum), np.array(distance), np.array(returned)
    ratios = []
    for value, best in zip(distance[returned], optimum[returned]):
        ratios.append(float(value / best) if best > 0 else 1.0 if value == 0 else float("inf"))
    out = dict(queries=len(rows), returned=int(returned.sum()),
               ratios=[v if np.isfinite(v) else "infinity" for v in ratios],
               empty=1 - float(returned.mean()),
               a110=float(np.mean(returned & (distance <= 1.1 * optimum + 1e-8))),
               a125=float(np.mean(returned & (distance <= 1.25 * optimum + 1e-8))),
               median_ms=float(np.median(seconds)) * 1000,
               p95_ms=float(np.quantile(seconds, .95)) * 1000,
               candidates=float(np.mean(candidates)) if candidates else None,
               probes=float(np.mean(probes)) if probes else None,
               postings=float(np.mean(postings)) if postings else None)
    return out


def summarize(records):
    if not records:
        return None
    if len({r["seed"] for r in records}) != len(records):
        raise ValueError("Duplicate seed records in a reported comparison")
    ratios = [float("inf") if v == "infinity" else v for r in records for v in r["ratios"]]
    finite = all(np.isfinite(v) for v in ratios)
    out = dict(
        seeds=len(records), seed_ids=sorted(r["seed"] for r in records),
        queries=sum(r["queries"] for r in records), returned=sum(r["returned"] for r in records),
        median_ms=float(np.median([r["median_ms"] for r in records])),
        latency_range=[min(r["median_ms"] for r in records), max(r["median_ms"] for r in records)],
        index_mb=float(np.median([r["index_mb"] for r in records])),
        index_range=[min(r["index_mb"] for r in records), max(r["index_mb"] for r in records)],
        mean_ratio=(float(np.mean(ratios)) if finite else "infinity") if ratios else None,
        max_ratio=(float(max(ratios)) if finite else "infinity") if ratios else None,
    )
    for field in ("a110", "a125", "empty"):
        out[field] = sum(r[field] * r["queries"] for r in records) / out["queries"]
    out["min_a125"] = min(r["a125"] for r in records)
    for field in ("candidates", "probes", "postings"):
        values = [r[field] for r in records if r[field] is not None]
        out[field] = float(np.mean(values)) if values else None
    for field, q in (("median_ratio", .5), ("p95_ratio", .95), ("p99_ratio", .99)):
        value = np.quantile(ratios, q, method="linear" if finite else "inverted_cdf") if ratios else None
        out[field] = float(value) if value is not None and np.isfinite(value) else "infinity" if value is not None else None
    return out


def csv_rows(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def seed_of(folder):
    match = re.search(r"seed-?(\d+)$", folder.name)
    if match is None:
        raise ValueError(f"Unrecognized seed directory: {folder}")
    return int(match.group(1))


def add_record(out, study, case, variant, method, seed, index_bytes, rows, **metadata):
    case = re.sub(r"^common-", "conjunctive-", case)
    if len(rows) != 1000 or sorted(int(r["query"]) for r in rows) != list(range(1000)):
        raise ValueError(f"Expected exactly 1000 distinct test rows: {study}/{case}/{variant}/{method}/{seed}")
    out.append(dict(study=study, case=case, variant=variant, method=method, seed=seed,
                    index_mb=float(index_bytes) / 1e6, **metadata, **metrics(rows)))


def load(root):
    root = Path(root)
    out = []
    for folder, study in POC_ROOTS.items():
        source = root / folder
        for job in sorted(source.glob("*seed*")):
            if ".interrupted" in job.name:
                continue
            if not (job / "main-results.json").is_file() or not (job / "test-queries.csv").is_file():
                continue
            info = json.loads((job / "dataset.json").read_text())
            case = info.get("case_id", job.name.rsplit("-seed", 1)[0])
            main = json.loads((job / "main-results.json").read_text())
            groups = defaultdict(list)
            for row in csv_rows(job / "test-queries.csv"):
                if row.get("phase", "test") == "test":
                    groups[row["config_id"]].append(row)
            for row in main:
                method = row["display_method"]
                if method not in METHODS or row.get("test_status") != "complete":
                    continue
                add_record(out, study, case, "cl", method, seed_of(job), row["index_bytes"],
                           groups[row["config_id"]], source=str(job))
            weights = job / "frozen-weight-selections.json"
            if weights.is_file() and info["k"] == 8:
                for weight, row in json.loads(weights.read_text()).items():
                    add_record(out, "lambda", case, "cl", "blocking", seed_of(job), row["index_bytes"],
                               groups[row["config_id"]], weight=float(weight), source=str(job))
    for folder in FH_ROOTS:
        for job in sorted((root / folder).glob("*seed*")):
            main = job / "main-results.json"
            if not main.is_file():
                continue
            rows = csv_rows(job / "test-queries.csv")
            for row in json.loads(main.read_text()):
                method = row.get("display_method")
                if method not in JOINT or row.get("status") != "complete":
                    continue
                selected = [r for r in rows if r["display_method"] == method and r.get("phase", "test") == "test"]
                study = "nba" if row["case_id"].startswith("nba") else "selectivity" if row["case_id"].startswith("frequency") else "comparison"
                add_record(out, study, row["case_id"], "fh", method, seed_of(job),
                           row["index_bytes"] if "index_bytes" in row else row["index_mb"] * 1e6,
                           selected, source=str(job))
    for job in sorted((root / "nba-full-20261012").glob("seed-*")):
        path = job / "main-results.json"
        if not path.exists():
            continue
        for method, row in json.loads(path.read_text()).items():
            method = {"one_block": "per_curve-intersection", "rarest": "per_curve-rarest"}.get(method, method)
            if method not in METHODS or row["status"] != "complete":
                continue
            add_record(out, "nba", "nba-n23805-k5", "cl", method, seed_of(job),
                       row["configuration"]["index_bytes"], row["queries"], source=str(job))
    for folder in ("baselines-20261015", "baselines-ksweep-20261015", "baselines-scale-20261015", "baselines-selectivity-20261015"):
        summary = root / folder / "summary.json"
        if not summary.exists():
            continue
        for row in json.loads(summary.read_text()):
            if row["baseline"] != "per-curve-hnsw-intersection" or row["test_status"] != "complete":
                continue
            case = row["dataset"]
            study = "nba" if case.startswith("nba") else "selectivity" if case.startswith("frequency") else "comparison"
            key = (study, case, "cl", "hnsw", row["index_seed"])
            if any((r["study"], r["case"], r["variant"], r["method"], r["seed"]) == key for r in out):
                continue
            rows = [r for r in csv_rows(root / folder / row["job"] / "test-queries.csv")
                    if r["phase"] == "test" and r["baseline"] == "per-curve-hnsw-intersection"]
            add_record(out, study, case, "cl", "hnsw", row["index_seed"], row["index_bytes"], rows,
                       source=str(root / folder / row["job"]))
    for job in sorted((root / "matched-recall-20261015").glob("*seed*")):
        path = job / "main-results.json"
        if not path.exists():
            continue
        groups = defaultdict(list)
        for row in csv_rows(job / "test-queries.csv"):
            groups[(row["display_method"], int(row["target_L"]))].append(row)
        for row in json.loads(path.read_text()):
            if row.get("status") != "complete":
                continue
            add_record(out, "matched", row["case_id"], "cl", row["display_method"], seed_of(job),
                       row["index_bytes"], groups[(row["display_method"], row["target_L"])],
                       L=int(row["target_L"]), source=str(job))
    follow_jobs = [(job, JOINT) for job in sorted((root / "first-hit-quality-followup-20261017").glob("*seed*"))]
    follow_jobs += [(job, ("probe_all", "store_all")) for job in sorted(
        (root / "joint-first-hit-baselines-seed1-20261017").glob("*seed*"))]
    for job, allowed_methods in follow_jobs:
        if ".interrupted" in job.name:
            continue
        if not (job / "main-results.json").exists() or not (job / "test-queries.csv").exists():
            continue
        if job.name.startswith("budget-"):
            continue
        groups = defaultdict(list)
        for row in csv_rows(job / "test-queries.csv"):
            groups[row["config_id"]].append(row)
        group = next((name for name in ("replication-b", "replication-c", "sensitivity-m16", "sensitivity-m64",
                                        "sensitivity-rho025", "sensitivity-rho100", "dtw", "lambda",
                                        "matched", "scaleup", "frontier") if job.name.startswith(name + "-")), None)
        if group is None:
            raise ValueError(f"Unknown follow-up study: {job.name}")
        for row in json.loads((job / "main-results.json").read_text()):
            if group != "matched" and not row["selected_table_count"]:
                continue
            common = dict(source=str(job), family=row["family"], L=row["L"], metric=row["metric"],
                          job_complete=(job / "completion.json").exists())
            labels = row["labels"]
            if group in ("lambda", "scaleup") and "blocking" in allowed_methods:
                for label in labels:
                    if label.startswith("lambda="):
                        add_record(out, "lambda", row["case_id"], "fh", "blocking", row["index_seed"],
                                   row["index_bytes"], groups[row["config_id"]], weight=float(label.split("=")[1]), **common)
                if group == "lambda":
                    continue
            study = "comparison" if group == "scaleup" else group
            for method in (("blocking",) if group == "frontier" else allowed_methods):
                if group != "frontier" and method not in labels:
                    continue
                case = re.sub(r"^common-", "conjunctive-", row["case_id"])
                key = (study, case, "fh", method, row["index_seed"])
                present = [r for r in out if (r["study"], r["case"], r["variant"], r["method"], r["seed"]) == key
                           and (group != "matched" or r.get("L") == row["L"])]
                if present and group != "frontier":
                    continue
                add_record(out, study, row["case_id"], "fh", method, row["index_seed"],
                           row["index_bytes"], groups[row["config_id"]], **common)
    revised = []
    for job in sorted((root / "submission-fh-quality-20261017").glob("*seed*")):
        if ".interrupted" in job.name or not (job / "completion.json").exists():
            continue
        rows = csv_rows(job / "test-queries.csv")
        for row in json.loads((job / "main-results.json").read_text()):
            if row["status"] != "complete":
                continue
            revised.append((job, row, [r for r in rows if r["display_method"] == row["display_method"]]))
    replaced = {(row["study"], re.sub(r"^common-", "conjunctive-", row["case_id"]),
                 row["display_method"]) for _, row, _ in revised}
    retained = []
    for row in out:
        if row["variant"] == "fh" and (row["study"], row["case"], row["method"]) in replaced:
            if row["study"] == "nba":
                retained.append(dict(row, study="nba-legacy"))
        else:
            retained.append(row)
    out = retained
    for job, row, rows in revised:
        add_record(out, row["study"], row["case_id"], "fh", row["display_method"], row["index_seed"],
                   row["index_bytes"], rows, source=str(job), quality_tuned=True,
                   delta=row["delta"], L=row["L"], blocks=row["blocks"], swapped=row["swapped"])
    return out


def select(records, study, case, variant="fh", method="blocking", **filters):
    group = [r for r in records if (r["study"], r["case"], r["variant"], r["method"]) ==
             (study, case, variant, method) and all(r.get(k) == v for k, v in filters.items())]
    return summarize(group)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = load(args.results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
    print(f"Loaded {len(records)} measured seed/configuration records")
