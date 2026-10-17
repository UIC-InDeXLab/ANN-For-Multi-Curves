"""Render the experiment-section tables from the audited result summaries.

Every table body is derived from a summary whose published checksum is
verified by the existing readers; nothing is re-measured or re-tuned.
"""
import argparse
import json

import numpy as np
from pathlib import Path
from statistics import median

from fresh.nba_full import digest
from fresh.plot_paper_experiments import read_report, read_scaleup_report
from fresh import baseline_results, first_hit_results, frontier_results, publish_feasibility

LABELS = {"blocking": "Blocking", "probe_all": "Probe-all", "store_all": "Store-all",
          "per_curve-intersection": "Intersection", "brute-force": "Exact scan"}
REGIME_NAMES = {"conjunctive": "Common", "rare": "Rare", "repeated": "Repeated"}
SEEDS = (1701, 1702, 1703)


def number(value, digits=2):
    if digits == 2 and 0 < value < 0.005:
        return r"$<$0.01"
    return f"{value:,.{digits}f}".replace(",", "{,}")


def percent(value, digits=1):
    return number(100 * value, digits)


def tabular(columns, header, rows):
    lines = [r"\begin{tabular}{" + columns + "}", r"\toprule", header + r"\\", r"\midrule"]
    lines.extend(rows)
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"



BASELINE_LABELS = {"per-curve-hnsw-intersection": "Per-curve HNSW"}


def baseline_stats(found, dataset, baseline):
    seeds = found.get((dataset, baseline)) if found else None
    return baseline_results.aggregate(seeds) if seeds else None


FIRST_HIT = {}
FIRST_HIT_LABELS = {"blocking": "Blocking, first hit", "probe_all": "Probe-all, first hit",
                    "store_all": "Store-all, first hit"}


def first_hit_stats(case, method):
    seeds = FIRST_HIT.get((case, method))
    return first_hit_results.aggregate(seeds) if seeds else None


def first_hit_cells(case, method):
    f = first_hit_stats(case, method)
    return r" & \multicolumn{3}{c}{---}" if f is None else f" & {number(f['median_ms'])} & {number(f['index_mb'], 1)} & {percent(f['recall'])}"


def row_of(report, kind, n, k, method):
    rows = [r for r in report["main"] if (r["kind"], r["n"], r["k"], r["display_method"]) == (kind, n, k, method)]
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one {kind} n={n} k={k} {method} row")
    return rows[0]


def nba_table(report, scan=None, external=None):
    rows = []
    for method in ("blocking", "probe_all", "store_all", "per_curve-intersection"):
        found = [r for r in report["main"] if r["method"] == method]
        if len(found) != 1 or found[0]["status"] != "complete":
            raise ValueError("The NBA table requires every method to be complete")
        r = found[0]
        rows.append(f"{LABELS[method]} & {number(r['index_MB'])} & {number(r['median_ms'])} & "
                    f"{number(r['p95_ms'])} & {percent(r['exact_recall'])} & {percent(r['c_success']['1.1'])} & "
                    f"{number(r['mean_excess_feet'], 3)}" + r"\\")
    for baseline, label in BASELINE_LABELS.items():
        b = baseline_stats(external, "nba-n23805-k5", baseline)
        if b is not None:
            rows.append(f"{label} & {number(b['index_mb'])} & {number(b['median_ms'])} & {number(b['p95_ms'])} & "
                        f"{percent(b['recall'])} & {percent(b['c110'])} & {number(b['excess'], 3) if b['excess'] is not None else '---'}" + r"\\")
    if scan is not None:
        if scan["exact_agreement_count"] != scan["total_queries"] or scan["timeout_count"]:
            raise ValueError("The exact scan must agree with the audited ground truth on every query")
        rows.append(f"Exact scan & --- & {number(scan['median_ms'])} & {number(scan['p95_ms'])} & 100.0 & 100.0 & 0.000" + r"\\")
    header = (r"Method & \shortstack{Index\\(MB)} & \shortstack{Median\\(ms)} & \shortstack{p95\\(ms)} & "
              r"\shortstack{Recall\\(\%)} & \shortstack{$\le1.1d^*$\\(\%)} & \shortstack{Excess\\(ft)}")
    return tabular("lrrrrrr", header, rows)


def first_hit_table(report):
    if [r["d"] for r in report["by_dimension"]] != [2, 3]:
        raise ValueError("The first-hit table requires both dimensions")
    rows = []
    for r in report["by_dimension"]:
        rows.append(f"$\\R^{r['d']}$ & {number(r['theoretical_c'], 0)} & {number(r['mean_optimal_distance'])} & "
                    f"{number(r['mean_first_hit_distance'])} & {number(r['mean_ratio'])} & "
                    f"{number(r['mean_uniform_random_tuple_distance'], 0)}" + r"\\")
    return tabular("lrrrrr", r"Space & Bound $c$ & $d^*$ & First hit & $d/d^*$ & Random return", rows)


def regimes_table(report, k=8, n=100000, external=None):
    rows = []
    for kind in ("conjunctive", "rare", "repeated"):
        for index, method in enumerate(("blocking", "probe_all", "store_all", "per_curve-intersection")):
            label = REGIME_NAMES[kind] if index == 0 else ""
            found = [r for r in report["main"] if (r["kind"], r["n"], r["k"], r["display_method"]) == (kind, n, k, method)]
            if len(found) != 1:
                raise ValueError("Every regime and method needs one k=8 row")
            r = found[0]
            if r["test_status"] != "complete":
                if method != "store_all":
                    raise ValueError("Only Store-all may be unavailable at k=8")
                rows.append(f"{label} & {LABELS[method]} & \\multicolumn{{7}}{{c}}{{not built (construction memory)}}" + r"\\")
                continue
            index = number(r["index_MB"])
            rows.append(f"{label} & {LABELS[method]} & {index} & {number(r['median_ms'])} & "
                        f"{number(r['p95_ms'])} & {percent(r['recall_at_1'])} & {percent(r['c_success']['1.05'])} & "
                        f"{number(r['mean_ratio'])} & {percent(r['empty_rate'])}" + r"\\")
        for method in ("blocking", "probe_all", "store_all"):
            f = first_hit_stats(f"{kind}-n{n}-k{k}", method)
            if f is not None:
                rows.insert(len(rows) - 1, f" & {FIRST_HIT_LABELS[method]} & {number(f['index_mb'])} & "
                            f"{number(f['median_ms'])} & {number(f['p95_ms'])} & {percent(f['recall'])} & "
                            f"{percent(f['c105'])} & {number(f['ratio']) if f['ratio'] is not None else '---'} & "
                            f"{percent(f['empty'])}" + r"\\")
        for baseline, label in BASELINE_LABELS.items():
            b = baseline_stats(external, f"{kind}-n{n}-k{k}", baseline)
            if b is not None:
                rows.insert(len(rows) - 1, f" & {label} & {number(b['index_mb'])} & {number(b['median_ms'])} & "
                            f"{number(b['p95_ms'])} & {percent(b['recall'])} & {percent(b['c105'])} & "
                            f"{number(b['ratio']) if b['ratio'] is not None else '---'} & {percent(b['empty'])}" + r"\\")
        if kind != "repeated":
            rows[-1] += r"\addlinespace[2pt]"
    return tabular("llrrrrrrr", r"Paths & Method & Index (MB) & Median (ms) & p95 (ms) & Recall@1 (\%) & $\le1.05d^*$ (\%) & Mean $d/d^*$ & Empty (\%)", rows)


def build_seconds(root, case, method):
    values = []
    for seed in SEEDS:
        folder = root / f"{case}-seed{seed}"
        rows = [r for r in json.loads((folder / "main-results.json").read_text()) if r["display_method"] == method]
        if len(rows) != 1 or rows[0].get("build_seconds") is None:
            raise ValueError(f"Missing build time for {case} {method} seed {seed}")
        values.append(rows[0]["build_seconds"])
    return median(values)


def baseline_cells(external, dataset):
    cells = ""
    for baseline in BASELINE_LABELS:
        b = baseline_stats(external, dataset, baseline)
        if b is None:
            cells += r" & \multicolumn{3}{c}{not run}"
        else:
            cells += f" & {number(b['median_ms'])} & {number(b['index_mb'], 1)} & {percent(b['recall'])}"
    return cells


def scaling_table(original, extension, original_root, extension_root, external=None):
    sources = [("conjunctive", 100000, original, original_root), ("conjunctive", 1000000, original, original_root),
               ("conjunctive", 2000000, extension, extension_root), ("conjunctive", 5000000, extension, extension_root),
               ("rare", 1000000, original, original_root), ("rare", 5000000, extension, extension_root)]
    rows = []
    for kind, n, report, root in sources:
        cells = []
        for method in ("blocking", "probe_all", "per_curve-intersection"):
            r = row_of(report, kind, n, 8, method)
            if r["test_status"] != "complete":
                raise ValueError("Every plotted scaling method must be complete")
            cells.append((r["median_ms"], r["index_MB"], r["recall_at_1"]))
        block, probe, inter = cells
        label = REGIME_NAMES[kind]
        case = f"{kind}-n{n}-k8"
        rows.append(f"{label} & {n / 1e6:g} & {number(block[0])} & {number(block[1], 1)} & {percent(block[2])} & "
                    f"{number(probe[0])} & {number(probe[1], 1)} & {number(inter[0])} & {number(inter[1], 1)} & "
                    f"{percent(inter[2])} & {number(inter[0] / block[0], 1)} & "
                    f"{number(build_seconds(root, case, 'blocking'))}" + first_hit_cells(case, "blocking") + first_hit_cells(case, "probe_all") + baseline_cells(external, case) + r"\\")
    header = (r"Paths & $n$ (M) & \multicolumn{3}{c}{Blocking} & \multicolumn{2}{c}{Probe-all} & "
              r"\multicolumn{3}{c}{Intersection} & Int./Block. & Build & "
              r"\multicolumn{3}{c}{Blocking, first hit} & \multicolumn{3}{c}{Probe-all, first hit} & "
              r"\multicolumn{3}{c}{Per-curve HNSW}")
    sub = (r" &  & ms & MB & Rec. & ms & MB & ms & MB & Rec. & time & (s) & ms & MB & Rec. & ms & MB & Rec. & ms & MB & Rec.")
    lines = [r"\begin{tabular}{lrrrrrrrrrrrrrrrrrrrr}", r"\toprule", header + r"\\",
             r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}\cmidrule(lr){8-10}\cmidrule(lr){13-15}\cmidrule(lr){16-18}\cmidrule(lr){19-21}", sub + r"\\", r"\midrule"]
    lines.extend(rows)
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def selectivity_table(report, external=None):
    comparisons = sorted(report["comparisons"], key=lambda r: r["realized_theoretical"])
    if len(comparisons) != 6 or any(r["status"] != "complete" for r in comparisons):
        raise ValueError("The selectivity table requires all six completed settings")
    rows = []
    names = ["005", "020", "050", "180", "400", "650"]
    for r, name in zip(comparisons, names):
        extra = ""
        for method in ("blocking", "probe_all"):
            f = first_hit_stats(f"frequency-{name}-n100000-k8", method)
            extra += (f" & {number(f['median_ms'])} & {percent(f['recall'])}" if f else r" & \multicolumn{2}{c}{---}")
        for baseline in BASELINE_LABELS:
            b = baseline_stats(external, f"frequency-{name}-n100000-k8", baseline)
            extra += (f" & {number(b['median_ms'])} & {percent(b['recall'])}" if b else r" & \multicolumn{2}{c}{not run}")
        rows.append(f"{number(100 * r['realized_theoretical'], 1)} & {number(r['blocking_latency_ms'])} & "
                    f"{number(r['best_pc_latency_ms'])} & {number(r['speedup_best_pc_over_blocking'])} & "
                    f"{number(r['blocking_index_MB'], 1)} & {number(r['best_pc_index_MB'], 1)} & "
                    f"{percent(r['blocking_recall'])} & {percent(r['best_pc_recall'])} & "
                    f"{number(r['blocking_mean_postings'], 0)} & {number(r['best_pc_mean_postings'], 0)} & "
                    f"{number(r['mean_relevant_sets'], 1)}" + extra + r"\\")
    header = (r"Occurrence (\%) & \multicolumn{2}{c}{Latency (ms)} & Int./Block. & \multicolumn{2}{c}{Index (MB)} & "
              r"\multicolumn{2}{c}{Recall@1 (\%)} & \multicolumn{2}{c}{Entries read} & Relevant & "
              r"\multicolumn{2}{c}{Blocking, first hit} & \multicolumn{2}{c}{Probe-all, first hit} & "
              r"\multicolumn{2}{c}{Per-curve HNSW}")
    sub = r" & Block. & Int. & ratio & Block. & Int. & Block. & Int. & Block. & Int. & sets & ms & Rec. & ms & Rec. & ms & Rec."
    lines = [r"\begin{tabular}{rrrrrrrrrrrrrrrrr}", r"\toprule", header + r"\\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}\cmidrule(lr){12-13}\cmidrule(lr){14-15}\cmidrule(lr){16-17}", sub + r"\\", r"\midrule"]
    lines.extend(rows)
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def ablation_rows(report, kind, n, k=8):
    rows = sorted((r for r in report["ablation"] if (r["kind"], r["n"], r["k"]) == (kind, n, k)),
                  key=lambda r: r["latency_weight"])
    if [r["latency_weight"] for r in rows] != [0.1, 1.0, 10.0, 100.0] or any(r["test_status"] != "complete" for r in rows):
        raise ValueError("The weight ablation needs all four completed preferences")
    return rows


def ablation_table(original, extension, frontier=None):
    cases = [("conjunctive", 100000, original), ("rare", 100000, original), ("repeated", 100000, original),
             ("conjunctive", 2000000, extension), ("conjunctive", 5000000, extension), ("rare", 5000000, extension)]
    rows = []
    recalls = []
    for kind, n, report in cases:
        stars = (frontier or {}).get(f"{kind}-n{n}-k8", {}).get("stars")
        if stars:
            values = [dict(median_ms=stars[w]["median_ms"], index_MB=stars[w]["index_mb"], recall_at_1=stars[w]["recall"])
                      for w in sorted(stars, key=float)]
        else:
            values = ablation_rows(report, kind, n)
        recalls.extend(r["recall_at_1"] for r in values)
        cells = " & ".join(f"{number(r['median_ms'])} & {number(r['index_MB'], 1)}" for r in values)
        rows.append(f"{REGIME_NAMES[kind]} & {n / 1e6:g} & {cells}" + r"\\")
    header = (r"Paths & $n$ (M) & \multicolumn{2}{c}{$\lambda=0.1$} & \multicolumn{2}{c}{$\lambda=1$} & "
              r"\multicolumn{2}{c}{$\lambda=10$} & \multicolumn{2}{c}{$\lambda=100$}")
    sub = r" &  & ms & MB & ms & MB & ms & MB & ms & MB"
    lines = [r"\begin{tabular}{lrrrrrrrrr}", r"\toprule", header + r"\\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}", sub + r"\\", r"\midrule"]
    lines.extend(rows)
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n", (min(recalls), max(recalls))


def feasibility_tables(report):
    groups = {(g["k"], g["method"]): g for g in report["groups"]}
    methods = publish_feasibility.METHODS
    summary = []
    for method in methods:
        largest = report["largest_feasible_tested_k"][method]
        if largest is None:
            summary.append(f"{LABELS[method]} & -- & \\multicolumn{{3}}{{c}}{{none}}" + r"\\")
            continue
        g = groups[(largest, method)]
        summary.append(f"{LABELS[method]} & {largest} & {number(g['index_MB'], 0)} & {number(g['median_ms'])} & "
                       f"{percent(g['success_rate'])}" + r"\\")
    summary_table = tabular("lrrrr", r"Method & Largest $k$ & Index (MB) & Median (ms) & On-time NN (\%)", summary)
    codes = {"excluded: integer capacity": "capacity", "excluded: construction RAM estimate": "RAM",
             "excluded: measured serving-index storage": "size",
             "excluded: mandatory order arrays exceed serving-index budget": "orders",
             "rejected: on-time tuning recall": "late"}
    ks = sorted({g["k"] for g in report["groups"]})
    rows = []
    for k in ks:
        cells = []
        for method in methods:
            g = groups[(k, method)]
            if g["feasible"]:
                cells.append(f"{number(g['index_MB'], 0)}\\,MB, {number(g['median_ms'])}\\,ms, {percent(g['success_rate'])}\\%")
            else:
                reasons = sorted({codes[reason] for seed in g["seeds"] for reason in seed.get("reasons", [])})
                if g["all_seeds_measured"]:
                    reasons = ["recall or size"]
                cells.append("--- (" + ", ".join(reasons) + ")" if reasons else "---")
        rows.append(f"{k} & " + " & ".join(cells) + r"\\")
    detail = tabular("lllll", r"$k$ & " + " & ".join(LABELS[m] for m in methods), rows)
    return summary_table, detail


def dtw_table(report):
    cases = [c["id"] for c in report["config"]["cases"]]
    names = {"common-n100000-k4": "Common, $k=4$", "common-n100000-k8": "Common, $k=8$",
             "rare-n100000-k4": "Rare, $k=4$", "rare-n100000-k8": "Rare, $k=8$", "nba-n23805-k5": "NBA, $k=5$"}
    fractions = {r["case_id"]: r["nonoptimal_fraction"] for r in report["frechet_reference_parents"]}
    rows = []
    for case in cases:
        cells = []
        for method in ("blocking", "probe_all", "store_all", "per_curve-intersection"):
            found = [r for r in report["main"] if r["case_id"] == case and r["display_method"] == method]
            if len(found) != 1:
                raise ValueError("Every DTW case and method needs one row")
            r = found[0]
            if r["test_status"] == "complete":
                cells.append(f"{percent(r['recall_at_1'])} & {number(r['median_ms'])} & {number(r['index_MB'], 1)}")
            else:
                cells.append(r"\multicolumn{3}{c}{not built}")
        rows.append(f"{names[case]} & " + " & ".join(cells) + f" & {percent(fractions[case])}" + r"\\")
    header = (r"Case & \multicolumn{3}{c}{Blocking} & \multicolumn{3}{c}{Probe-all} & "
              r"\multicolumn{3}{c}{Store-all} & \multicolumn{3}{c}{Intersection} & Fr\'echet")
    sub = r" & Rec. & ms & MB & Rec. & ms & MB & Rec. & ms & MB & Rec. & ms & MB & non-opt. (\%)"
    lines = [r"\begin{tabular}{lrrrrrrrrrrrrr}", r"\toprule", header + r"\\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}\cmidrule(lr){11-13}", sub + r"\\", r"\midrule"]
    lines.extend(rows)
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def dtw_compact_table(report):
    names = {"common-n100000-k4": "Common, $k=4$", "common-n100000-k8": "Common, $k=8$",
             "rare-n100000-k4": "Rare, $k=4$", "rare-n100000-k8": "Rare, $k=8$", "nba-n23805-k5": "NBA, $k=5$"}
    methods = ("blocking", "probe_all", "per_curve-intersection")
    rows = []
    for case in [c["id"] for c in report["config"]["cases"]]:
        cells = []
        for method in methods:
            found = [r for r in report["main"] if r["case_id"] == case and r["display_method"] == method]
            if len(found) != 1 or found[0]["test_status"] != "complete":
                raise ValueError("Every compact DTW cell needs one complete row")
            cells.append(f"{percent(found[0]['recall_at_1'])} & {number(found[0]['median_ms'])}")
        rows.append(f"{names[case]} & " + " & ".join(cells) + r"\\")
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Case & \multicolumn{2}{c}{Blocking} & \multicolumn{2}{c}{Probe-all} & \multicolumn{2}{c}{Intersection}\\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
             r" & Rec. (\%) & ms & Rec. (\%) & ms & Rec. (\%) & ms\\", r"\midrule"]
    lines.extend(rows)
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def probability(value):
    return f"{value:.3g}"


def collision_table(summary):
    ks = sorted({case["k"] for case in summary["cases"]})
    ps = sorted({case["p"] for case in summary["cases"]})
    if ps != [0.5, 0.75]:
        raise ValueError("The collision table expects p in {0.5, 0.75}")
    rows = []
    for k in ks:
        cells = []
        for p in ps:
            found = [c for c in summary["cases"] if c["k"] == k and c["p"] == p]
            if len(found) != 1:
                raise ValueError("Every k and p needs exactly one collision case")
            c = found[0]
            if c["guarantee_validation"]["joint"]["significant_below_bound"]:
                raise ValueError("A measured collision frequency falls significantly below its proved bound")
            theory = "$>64$" if c["any_table_collision"]["u95_theory"] is None else str(c["any_table_collision"]["u95_theory"])
            cells.append(f"{probability(c['joint_bound'])} & {number(c['joint_pair_frequency']['mean'], 3)} & "
                         f"{number(c['joint_pair_frequency']['min'], 3)} & {c['any_table_collision']['u95_empirical_mean']}/{theory}")
        rows.append(f"{k} & " + " & ".join(cells) + r"\\")
    header = (r"$k$ & \multicolumn{4}{c}{$p=0.5$ ($\delta=128$)} & \multicolumn{4}{c}{$p=0.75$ ($\delta=256$)}")
    sub = r" & $p^k$ & Mean & Min & Tables & $p^k$ & Mean & Min & Tables"
    lines = [r"\begin{tabular}{rrrrrrrrr}", r"\toprule", header + r"\\", r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
             sub + r"\\", r"\midrule"]
    lines.extend(rows)
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def nba_selectivity_table(summary):
    names = (("occurrence_at_r", "$r(Q)$"), ("occurrence_at_1_5r", "$1.5\\,r(Q)$"), ("occurrence_at_2r", "$2\\,r(Q)$"),
             ("occurrence_at_5ft", "$5$\\,ft"))
    rows = []
    for key, label in names:
        d = summary["distributions_pct"][key]
        digits = 2 if key == "occurrence_at_5ft" else 1
        rows.append(f"{label} & {number(d['p10'], digits)} & {number(d['median'], digits)} & {number(d['mean'], digits)} & "
                    f"{number(d['p90'], digits)}" + r"\\")
    return tabular("lrrrr", r"Radius & P10 & Median & Mean & P90", rows)


def replicated(root):
    path = Path(root) / "replicated-summary.json"
    return json.loads(path.read_text()) if path.is_file() else None


def regime_cells(report, kind, n=100000, k=8):
    """Blocking, Intersection and Probe-all values for one regime of a replicated report."""
    cells = {}
    for method in ("blocking", "per_curve-intersection", "probe_all"):
        found = [r for r in report["main"] if (r["kind"], r["n"], r["k"], r["display_method"]) == (kind, n, k, method)]
        if len(found) != 1:
            raise ValueError(f"Missing {kind} {method} row")
        cells[method] = found[0]
    return cells


def replication_table(reports):
    """reports: list of (label, replicated summary) for independent dataset seeds."""
    rows = []
    for kind in ("conjunctive", "rare"):
        for index, (label, report) in enumerate(reports):
            c = regime_cells(report, kind)
            for r in c.values():
                if r["test_status"] != "complete":
                    raise ValueError("Replication rows must be complete")
            b, i = c["blocking"], c["per_curve-intersection"]
            first = REGIME_NAMES[kind] if index == 0 else ""
            rows.append(f"{first} & {label} & {number(b['median_ms'])} & {number(b['index_MB'], 1)} & {percent(b['recall_at_1'])} & "
                        f"{number(i['median_ms'])} & {number(i['index_MB'], 1)} & {percent(i['recall_at_1'])} & "
                        f"{number(c['probe_all']['median_ms'])} & "
                        f"{number(i['median_ms'] / b['median_ms'])}" + r"\\")
        if kind == "conjunctive":
            rows[-1] += r"\addlinespace[2pt]"
    header = (r"Paths & Dataset & \multicolumn{3}{c}{Blocking} & \multicolumn{3}{c}{Intersection} & Probe-all & Int./Block.")
    sub = r" &  & ms & MB & Rec. & ms & MB & Rec. & ms & ratio"
    lines = [r"\begin{tabular}{llrrrrrrrr}", r"\toprule", header + r"\\", r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
             sub + r"\\", r"\midrule"]
    lines.extend(rows)
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def sensitivity_table(variants):
    """variants: list of (parameter label, replicated summary or None); None marks a variant with no feasible index."""
    rows = []
    for label, report in variants:
        cells = []
        for kind in ("conjunctive", "rare"):
            if report is None:
                cells.append(r"\multicolumn{4}{c}{no feasible configuration}")
                continue
            c = regime_cells(report, kind)
            b, i = c["blocking"], c["per_curve-intersection"]
            if b["test_status"] != "complete" or i["test_status"] != "complete":
                cells.append(r"\multicolumn{4}{c}{no feasible configuration}")
                continue
            cells.append(f"{number(b['median_ms'])} & {percent(b['recall_at_1'])} & {number(i['median_ms'])} & "
                         f"{number(i['median_ms'] / b['median_ms'])}")
        rows.append(f"{label} & " + " & ".join(cells) + r"\\")
    header = r"Parameter & \multicolumn{4}{c}{Common paths} & \multicolumn{4}{c}{Rare paths}"
    sub = r" & Block. ms & Rec. & Int. ms & Int./Block. & Block. ms & Rec. & Int. ms & Int./Block."
    lines = [r"\begin{tabular}{lrrrrrrrr}", r"\toprule", header + r"\\", r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
             sub + r"\\", r"\midrule"]
    lines.extend(rows)
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def budget_table(reports):
    """reports: list of (label, feasibility audit report) for different budget pairs."""
    rows = []
    for method in publish_feasibility.METHODS:
        values = []
        for _, report in reports:
            largest = report["largest_feasible_tested_k"][method]
            values.append("--" if largest is None else str(largest))
        rows.append(f"{LABELS[method]} & " + " & ".join(values) + r"\\")
    header = r"Method & " + " & ".join(label for label, _ in reports)
    return tabular("l" + "r" * len(reports), header, rows)


def matched_recall_table(summary, case_ids=("conjunctive-n100000-k8", "rare-n100000-k8")):
    names = {"conjunctive-n100000-k8": "Common", "rare-n100000-k8": "Rare"}
    targets = sorted({row["target_recall"] for row in summary["matched_recall"]})
    rows = []
    for case in case_ids:
        for index, method in enumerate(("blocking", "probe_all", "per_curve-intersection")):
            cells = []
            for target in targets:
                found = [r for r in summary["matched_recall"]
                         if (r["case_id"], r["display_method"], r["target_recall"]) == (case, method, target)]
                if len(found) != 1:
                    raise ValueError("Missing matched-recall row")
                r = found[0]
                cells.append(f"{number(r['median_seed_median_ms'])} ({r['target_L']})" if r["status"] == "reached" else "---")
            rows.append(f"{names[case] if index == 0 else ''} & {LABELS[method]} & " + " & ".join(cells) + r"\\")
        if case != case_ids[-1]:
            rows[-1] += r"\addlinespace[2pt]"
    header = r"Paths & Method & " + " & ".join(f"{percent(t, 0)}\\%" for t in targets)
    return tabular("llrrrr", header, rows)


def build_time_table(summary):
    names = (("conjunctive-n100000-k8", "Common, $n=10^5$"), ("conjunctive-n1000000-k8", "Common, $n=10^6$"),
             ("rare-n1000000-k8", "Rare, $n=10^6$"), ("nba-full", "NBA"))
    methods = ("blocking", "probe_all", "per_curve-intersection", "store_all")
    rows = []
    for case, label in names:
        cells = []
        for method in methods:
            found = [r for r in summary["summary"] if r["case_id"] == case and r["display_method"] == method]
            cells.append(number(found[0]["median_build_seconds"]) if found else "---")
        peaks = []
        for method in ("blocking", "per_curve-intersection"):
            found = [r for r in summary["summary"] if r["case_id"] == case and r["display_method"] == method]
            if len(found) != 1:
                raise ValueError("Every case needs Blocking and Intersection build rows")
            peaks.append(number(found[0]["median_peak_resident_delta_bytes"] / 1e6, 0))
        rows.append(f"{label} & " + " & ".join(cells) + " & " + " & ".join(peaks) + r"\\")
    header = (r"Case & \multicolumn{4}{c}{Build time (s)} & \multicolumn{2}{c}{Peak extra RAM (MB)}")
    sub = r" & Block. & Probe & Int. & Store & Block. & Int."
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule", header + r"\\", r"\cmidrule(lr){2-5}\cmidrule(lr){6-7}",
             sub + r"\\", r"\midrule"]
    lines.extend(rows)
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def ksweep_table(comparison, external):
    rows = []
    for kind in ("conjunctive", "rare", "repeated"):
        for k in (4, 6, 8, 10):
            case = f"{kind}-n100000-k{k}"
            cells = []
            for method in ("blocking", "probe_all", "store_all", "per_curve-intersection"):
                found = [r for r in comparison["main"] if (r["kind"], r["n"], r["k"], r["display_method"]) == (kind, 100000, k, method)]
                if len(found) != 1:
                    raise ValueError("Missing comparison row")
                r = found[0]
                cells.append(f"{number(r['median_ms'])} & {percent(r['recall_at_1'])}" if r["test_status"] == "complete"
                             else r"\multicolumn{2}{c}{---}")
                if method != "per_curve-intersection":
                    f = first_hit_stats(case, method)
                    cells.append(f"{number(f['median_ms'])} & {percent(f['recall'])}" if f else r"\multicolumn{2}{c}{---}")
            b = baseline_stats(external, case, "per-curve-hnsw-intersection")
            cells.append(f"{number(b['median_ms'])} & {percent(b['recall'])}" if b else r"\multicolumn{2}{c}{---}")
            rows.append(f"{REGIME_NAMES[kind] if k == 4 else ''} & {k} & " + " & ".join(cells) + r"\\")
        if kind != "repeated":
            rows[-1] += r"\addlinespace[2pt]"
    header = (r"Paths & $k$ & \multicolumn{2}{c}{Blocking} & \multicolumn{2}{c}{Blocking, FH} & \multicolumn{2}{c}{Probe-all} & "
              r"\multicolumn{2}{c}{Probe-all, FH} & \multicolumn{2}{c}{Store-all} & \multicolumn{2}{c}{Store-all, FH} & "
              r"\multicolumn{2}{c}{Intersection} & \multicolumn{2}{c}{Per-curve HNSW}")
    sub = r" &  & ms & Rec. (\%) & ms & Rec. (\%) & ms & Rec. (\%) & ms & Rec. (\%) & ms & Rec. (\%) & ms & Rec. (\%) & ms & Rec. (\%) & ms & Rec. (\%)"
    cm = "".join(rf"\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}" for i in range(8))
    lines = [r"\begin{tabular}{ll" + "rr" * 8 + "}", r"\toprule", header + r"\\", cm, sub + r"\\", r"\midrule"]
    lines.extend(rows)
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def main(args):
    paper, results = args.paper.resolve(), args.results.resolve()
    output = paper / "experiments" / "tables"
    output.mkdir(exist_ok=True)
    approximation, _ = read_report(paper, results, "approximation-results.tex", "completion.json")
    comparison, _ = read_report(paper, results, "comparison-results.tex", "completion.json")
    nba, _ = read_report(paper, results, "nba-full-results.tex", "complete.json")
    selectivity, _ = read_report(paper, results, "selectivity-results.tex", "completion.json")
    extension, _ = read_scaleup_report(paper, results)
    feasibility = publish_feasibility.audit(results / "fresh-feasibility-20261014")
    dtw_root = results / "dtw-20261014" / "paper"
    receipt = json.loads((dtw_root / "dtw-publication-receipt.json").read_text())
    if digest(dtw_root / "dtw-summary.json") != receipt["outputs"]["dtw-summary.json"]:
        raise ValueError("The DTW summary differs from its publication receipt")
    dtw = json.loads((dtw_root / "dtw-summary.json").read_text())
    original_root = results / "fresh-synthetic-expanded-20261012"
    extension_root = results / "fresh-synthetic-scaleup-20261013"
    ablation, recall_range = ablation_table(comparison, extension)
    summary, detail = feasibility_tables(feasibility)
    scan_path = results / "nba-exact-scan-20261015" / "summary.json"
    scan = json.loads(scan_path.read_text()) if scan_path.is_file() else None
    external = baseline_results.load(results)
    FIRST_HIT.update(first_hit_results.load(results))
    tables = {"nba": nba_table(nba, None, external), "first-hit": first_hit_table(approximation),
              "regimes": regimes_table(comparison, external=external),
              "scaling": scaling_table(comparison, extension, original_root, extension_root, external),
              "selectivity": selectivity_table(selectivity, external), "lambda": ablation,
              "ksweep": ksweep_table(comparison, external),
              "feasibility-summary": summary, "feasibility-detail": detail, "dtw": dtw_table(dtw),
              "dtw-compact": dtw_compact_table(dtw)}
    extra = results / "collision-20261015" / "summary.json"
    if extra.is_file():
        tables["collision"] = collision_table(json.loads(extra.read_text()))
    extra = results / "nba-selectivity-20261015" / "summary.json"
    if extra.is_file():
        tables["nba-selectivity"] = nba_selectivity_table(json.loads(extra.read_text()))
    seeds = [("A", comparison)]
    for tag, label in (("b", "B"), ("c", "C")):
        report = replicated(results / f"dataset-seed-{tag}-20261015")
        if report is not None:
            seeds.append((label, report))
    if len(seeds) == 3:
        tables["replication"] = replication_table(seeds)
    variants = []
    for label, name in (("$m=16$", "sensitivity-m16"), ("$m=32$ (base)", None), ("$m=64$", "sensitivity-m64"),
                        ("$r=0.25$", "sensitivity-rho025"), ("$r=0.5$ (base)", None), ("$r=1$", "sensitivity-rho100")):
        report = comparison if name is None else replicated(results / f"{name}-20261015")
        if report is not None:
            variants.append((label, report))
    if len(variants) == 6:
        tables["sensitivity"] = sensitivity_table(variants)
    budgets = [(r"1\,GB, 10\,ms", feasibility)]
    for label, folder, config_name, not_run in (
            (r"256\,MB, 1\,ms", "feasibility-256mb-1ms-20261015", "feasibility-256mb-1ms-20261015.json", ()),
            (r"16\,GB, 10\,ms", "feasibility-16gb-10ms-20261015", "feasibility-16gb-10ms-20261015.json", ()),
            (r"16\,GB, 100\,ms", "feasibility-16gb-100ms-20261015", "feasibility-16gb-100ms-20261015.json", (12,))):
        root = results / folder
        if (root / "completion.json").is_file() or not_run and (root / "cases.json").is_file():
            budgets.append((label, publish_feasibility.audit(
                root, Path(__file__).resolve().parents[1] / "configs" / config_name, not_run)))
    if len(budgets) > 1:
        tables["feasibility-budgets"] = budget_table(budgets)
    build_path = results / "build-time-20261015" / "summary.json"
    if build_path.is_file():
        tables["build-time"] = build_time_table(json.loads(build_path.read_text()))
    matched_path = results / "matched-recall-20261015" / "summary.json"
    if matched_path.is_file():
        tables["matched-recall"] = matched_recall_table(json.loads(matched_path.read_text()))
    for name, text in tables.items():
        (output / f"{name}.tex").write_text("% Generated by fresh.render_experiment_tables from audited summaries.\n" + text)
    print("lambda sweep recall range:", number(100 * recall_range[0]), "to", number(100 * recall_range[1]))
    print("Rendered", len(tables), "tables")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", type=Path, required=True)
    parser.add_argument("--results", type=Path, default=Path("results"))
    main(parser.parse_args())
