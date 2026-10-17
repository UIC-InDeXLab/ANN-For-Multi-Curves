"""Print the quantities that the text of the frontier subsection reports."""
import json
import sys
from pathlib import Path

import numpy as np

from fresh import frontier_results as fr


def main(results):
    data = fr.load(Path(results))
    stats = fr.statistics(data)
    for case_id, d in data.items():
        pts = d["points"]
        print(f"== {case_id}: {len(pts)} configurations, missing {len(d['missing'])}")
        size = np.array([p["index_mb"] for p in pts]); lat = np.array([p["median_ms"] for p in pts])
        print(f"   size range {size.min():.1f}-{size.max():.0f} MB (x{size.max() / size.min():.0f}); latency range {lat.min():.3f}-{lat.max():.2f} ms (x{lat.max() / lat.min():.0f})")
        s = stats[case_id]
        print(f"   Spearman(a, size)={s['spearman_a_size']:.3f}  Spearman(b, latency)={s['spearman_b_latency']:.3f}")
        # tables explain the residuals
        tables = [np.median(p["tables"]) for p in pts]
        a = np.array([p["a"] for p in pts]); b = np.array([p["b"] for p in pts])
        print(f"   with the tuned tables: Spearman(a*L, size)={fr.spearman(a * tables, size):.3f}  Spearman(b*L, latency)={fr.spearman(b * tables, lat):.3f}")
        # configurations with the same a
        groups = {}
        for p in pts:
            groups.setdefault(p["a"], []).append(p)
        spreads = [(max(q["index_mb"] for q in g) / min(q["index_mb"] for q in g), max(q["median_ms"] for q in g) / min(q["median_ms"] for q in g), a_) for a_, g in groups.items() if len(g) > 1]
        if spreads:
            print("   same-a groups:", len(spreads), "median size spread x%.2f, latency spread x%.2f; max size spread x%.2f (a=%d), max latency spread x%.2f (a=%d)" % (
                np.median([x[0] for x in spreads]), np.median([x[1] for x in spreads]), max(spreads)[0], max(spreads)[2],
                max(spreads, key=lambda x: x[1])[1], max(spreads, key=lambda x: x[1])[2]))
        print("   worst size misorderings (blocks, swapped, a, MB, rank gap):", s["worst_size"][:3])
        print("   worst latency misorderings (blocks, swapped, b, ms, rank gap):", s["worst_latency"][:3])
        probe = [p for p in pts if p["a"] == 1][0]
        print(f"   Probe-all: {probe['index_mb']:.1f} MB, {probe['median_ms']:.2f} ms;  largest a={max(p['a'] for p in pts)}")
        for w, st in sorted(d["stars"].items(), key=lambda t: float(t[0])):
            print(f"   lambda={w}: {st['index_mb']:.1f} MB, {st['median_ms']:.3f} ms  {st['configs']}")
        print("   excluded:", d["excluded"][:4], "...")
        pareto = [p for p in pts if not any(q["index_mb"] <= p["index_mb"] and q["median_ms"] < p["median_ms"] * 0.999 and q is not p for q in pts)]
        print("   Pareto-optimal configs:", len(pareto), "of", len(pts))
    json.dump(stats, open(Path(results) / fr.FOLDER / "frontier-statistics.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main(sys.argv[1])
