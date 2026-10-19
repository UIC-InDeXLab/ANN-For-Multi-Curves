"""Completed FH partition frontiers, rescored from query records and separated by quality."""
import json
from pathlib import Path

from fresh import quality_paper_results as quality
from fresh.frontier_results import spearman
from fresh.nba_full import digest


def point_summary(points):
    if not points:
        raise ValueError("A completed frontier requires measured points")
    useful = [p for p in points if p["a125"] >= .9]
    if len(useful) < 2:
        raise ValueError("At least two quality-qualified points are required for the cost model")
    probe = min((p for p in useful if p["a"] == 1), key=lambda p: p["index_mb"])
    fastest = min(useful, key=lambda p: p["median_ms"])
    largest = max(useful, key=lambda p: (p["a"], p["index_mb"]))
    return dict(
        measured=len(points), qualified=len(useful),
        spearman_a_size=spearman([p["a"] for p in useful], [p["index_mb"] for p in useful]),
        spearman_b_latency=spearman([p["b"] for p in useful], [p["median_ms"] for p in useful]),
        probe=probe, fastest=fastest, largest_stored=largest,
        speedup=probe["median_ms"] / fastest["median_ms"],
        qualified_a125_range=[min(p["a125"] for p in useful), max(p["a125"] for p in useful)],
        index_range_mb=[min(p["index_mb"] for p in useful), max(p["index_mb"] for p in useful)],
        latency_range_ms=[min(p["median_ms"] for p in useful), max(p["median_ms"] for p in useful)])


def load(root, records):
    result = {}
    for kind in ("conjunctive", "rare"):
        case = f"{kind}-n100000-k8"
        folder = Path(root) / f"frontier-{case}-seed1701"
        completion = folder / "completion.json"
        if not completion.exists():
            continue
        for name, expected in json.loads(completion.read_text())["artifacts"].items():
            if digest(folder / name) != expected:
                raise ValueError(f"Frontier artifact changed: {folder / name}")
        points = []
        for row in json.loads((folder / "main-results.json").read_text()):
            if not row["selected_table_count"] or row["status"] != "complete":
                continue
            selected = [r for r in records if r["study"] == "frontier" and r["case"] == case
                        and r["seed"] == 1701 and r["family"] == row["family"]]
            stats = quality.summarize(selected)
            if stats is None or stats["queries"] != 1000:
                raise ValueError(f"Frontier configuration lacks 1000 test queries: {case}/{row['family']}")
            if abs(stats["a125"] - row["success_125"]) > 1e-12:
                raise ValueError("Rescored frontier quality disagrees with recorded returns")
            points.append(dict(stats, family=row["family"], blocks=row["blocks"], swapped=row["swapped"],
                               a=row["stored"], b=row["probed"], delta=row["delta"], L=row["L"]))
        excluded = {}
        for row in json.loads((folder / "excluded.json").read_text()):
            excluded.setdefault(row["family"], row["status"])
        built = {p["family"] for p in points}
        if len(built) != len(points) or built & set(excluded) or len(built | set(excluded)) != 42:
            raise ValueError("The frontier must account for exactly 42 distinct canonical families")
        stars = {}
        for weight in (.1, 1., 10., 100.):
            stats = quality.select(records, "lambda", case, weight=weight)
            if stats is None or stats["seeds"] != 1:
                raise ValueError("FH shortlist preferences must use the same first-seed cohort")
            stars[str(weight)] = stats
        result[case] = dict(points=points, stars=stars, excluded=excluded, summary=point_summary(points))
    return result
