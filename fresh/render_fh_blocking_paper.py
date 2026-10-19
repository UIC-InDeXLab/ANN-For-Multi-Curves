"""FH joint-key publication with explicit seed cohorts and unchanged verified baselines."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from fresh import quality_paper_results as quality
from fresh.plot_experiment_figures import _paper_context, _make_figure, _save_close
from fresh.plot_styles import method_style
from fresh import fh_frontier_results
from fresh import budget_quality_results
from fresh.plot_experiment_figures import frontier as draw_frontier

NAMES = {"blocking": "Blocking", "probe_all": "Probe-all", "store_all": "Store-all",
         "per_curve-intersection": "Intersection", "hnsw": "Per-curve HNSW"}
REGIMES = {"conjunctive": "Common", "rare": "Rare", "repeated": "Repeated"}
TABLE2_CASES = [
    ("Common, $k=4$", "comparison", "conjunctive-n100000-k4"),
    ("Common, $k=6$", "comparison", "conjunctive-n100000-k6"),
    ("Common, $k=8$", "comparison", "conjunctive-n100000-k8"),
    ("Common, $k=10$", "comparison", "conjunctive-n100000-k10"),
    ("Rare, $k=8$", "comparison", "rare-n100000-k8"),
    ("Rare, $k=10$", "comparison", "rare-n100000-k10"),
    ("Repeated, $k=8$", "comparison", "repeated-n100000-k8"),
    ("0.5\\% occurrence", "selectivity", "frequency-005-n100000-k8"),
    ("17.9\\% occurrence", "selectivity", "frequency-180-n100000-k8"),
    ("40.3\\% occurrence", "selectivity", "frequency-400-n100000-k8"),
    ("65.6\\% occurrence", "selectivity", "frequency-650-n100000-k8"),
    ("Common, $n=10^6$", "comparison", "conjunctive-n1000000-k8"),
    ("Rare, $n=10^6$", "comparison", "rare-n1000000-k8")]


def number(value, digits=2):
    if value is None:
        return "---"
    if value == "infinity":
        return r"$\infty$"
    return f"{value:,.{digits}f}".replace(",", "{,}")


def latency(value):
    return r"$<$0.01" if 0 < value < .005 else number(value)


def percent(value):
    return number(100 * value, 1)


def tabular(columns, header, rows):
    return "\n".join([r"\begin{tabular}{" + columns + "}", r"\toprule", header + r"\\",
                      r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"]) + "\n"


def method_label(method, variant, stats):
    return NAMES[method] + (" (CL)" if method == "blocking" and variant == "cl" else "")


def first_seed_joint_view(records):
    return [r for r in records if r["method"] not in quality.JOINT or r["seed"] % 100 == 1
            or r.get("quality_tuned") or r.get("study") == "nba-legacy"]


def completed_collection_reference(root, case):
    path = root / case / "summary.json"
    if not path.exists():
        return None
    reference = json.loads(path.read_text())
    if not reference["complete"]:
        return None
    if (reference["case"] != case or reference["queries"] != 1000 or
            reference["evaluated_queries"] != 1000 or reference["evaluated_database_entries"] != "all"):
        raise ValueError(f"Reference must evaluate the entire database for all 1000 queries: {case}")
    fraction = reference["mean_database_within_125_fraction"]
    if not 0 <= fraction <= 1:
        raise ValueError(f"Invalid full-database approximation-success fraction: {case}")
    return reference


def completed_database_reference(full_root, near_root, case):
    full = completed_collection_reference(full_root, case)
    near = completed_collection_reference(near_root, case)
    if full is not None and near is not None:
        if (full["dataset_sha256"] != near["dataset_sha256"] or
                full["queries_sha256"] != near["queries_sha256"] or
                abs(full["mean_database_within_125_fraction"] -
                    near["mean_database_within_125_fraction"]) > 1e-12):
            raise ValueError(f"Full-distance and threshold-count references disagree: {case}")
    return full if full is not None else near


def paired_comparison_table(records, full_root, near_root):
    rows = []
    for label, study, case in TABLE2_CASES:
        k = int(case.rsplit("-k", 1)[1])
        case_label = label.split(", $k=", 1)[0]
        cl = quality.select(records, study, case, "cl")
        fh = quality.select(records, study, case)
        if cl is None or fh is None:
            raise ValueError(f"The main paired comparison needs both variants: {case}")
        reference = completed_database_reference(full_root, near_root, case)
        background = (number(100 * reference["mean_database_within_125_fraction"],
                             6 if reference["n"] <= 100000 else 7) if reference else "---")
        rows.append(" & ".join([case_label, str(k), latency(cl["median_ms"]), number(cl["mean_ratio"], 3),
                                latency(fh["median_ms"]), number(fh["mean_ratio"], 3),
                                percent(fh["a110"]), percent(fh["a125"]), background]) + r"\\")
    header = (r" & & \multicolumn{2}{c}{CL} & \multicolumn{4}{c}{FH} & DB\\"
              r"Case & $k$ & ms & Mean & ms & Mean & $A_{1.1}$ & $A_{1.25}$ & Near (\%)")
    return tabular("lrrrrrrrr", header, rows)


def write_collection_reference(records, root, paper):
    rows = []
    cases = [
        ("Common, $k=4$", "comparison", "conjunctive-n100000-k4"),
        ("Common, $k=8$", "comparison", "conjunctive-n100000-k8"),
        ("Repeated, $k=8$", "comparison", "repeated-n100000-k8"),
        ("65.6\\% occurrence", "selectivity", "frequency-650-n100000-k8")]
    for label, study, case in cases:
        reference = completed_collection_reference(root, case)
        if reference is None:
            continue
        fh = quality.select(records, study, case)
        if fh is None:
            raise ValueError(f"Full-database reference has no corresponding FH measurement: {case}")
        rows.append(" & ".join([
            label, number(reference["mean_database_ratio"], 1), number(fh["mean_ratio"], 3),
            number(100 * reference["mean_database_within_125_fraction"], 4), percent(fh["a125"])]) + r"\\")
    if rows:
        text = tabular("lrrrr", r"Case & DB mean & Mean & DB Near (\%) & $A_{1.25}$ (\%)", rows)
        (paper / "experiments/tables/collection-reference.tex").write_text(
            "% Exact distances to every database multi-curve, over 1000 held-out queries.\n" + text)
    else:
        print("Full-collection reference is still running; no incomplete averages were published.")


def write_fh_frontier(records, root, paper, preview):
    data = fh_frontier_results.load(root, records)
    if len(data) != 2:
        return
    draw_frontier(data, paper / "experiments/figures/fig-frontier.pdf",
                  preview=preview / "fh-frontier.png", first_hit=True)
    header = (r"Blocks & $a$ & $b$ & $\delta$ & $L$ & MB & ms & Mean $d/d^*$ & "
              r"$A_{1.1}$ & $A_{1.25}$ & Empty")
    summaries = {}
    for kind in ("conjunctive", "rare"):
        case = f"{kind}-n100000-k8"
        rows = []
        for p in sorted(data[case]["points"], key=lambda p: (p["a"], p["family"])):
            label = "+".join(map(str, p["blocks"])) + (" (swap)" if p["swapped"] else "")
            rows.append(" & ".join([label, number(p["a"], 0), number(p["b"], 0),
                                    number(p["delta"], 0), str(p["L"]), number(p["index_mb"], 1),
                                    number(p["median_ms"], 3), number(p["mean_ratio"], 3),
                                    percent(p["a110"]), percent(p["a125"]), percent(p["empty"])]) + r"\\")
        (paper / f"experiments/tables/frontier-fh-{kind}.tex").write_text(
            "% One measured first-seed FH point per built canonical family.\n" +
            tabular("lrrrrrrrrrr", header, rows))
        summaries[kind] = data[case]["summary"]
    common, rare = summaries["conjunctive"], summaries["rare"]
    intermediate = next(p for p in data["conjunctive-n100000-k8"]["points"]
                        if tuple(p["blocks"]) == (4, 2, 1, 1) and not p["swapped"])
    if intermediate["a125"] < .9:
        raise ValueError("The illustrative intermediate partition must meet the figure's quality threshold")
    text = [
        "Intermediate partitions give fast lookup with modest index sizes. "
        "On common paths, blocks $4+2+1+1$ use "
        f"{number(intermediate['index_mb'], 0)}\\,MB and answer in "
        f"{number(intermediate['median_ms'], 3)}\\,ms, compared with "
        f"{number(common['probe']['median_ms'], 2)}\\,ms for Probe-all. "
        "Some order enumeration has moved into construction, so lookup examines fewer "
        "allocations without giving up the joint key.",
        "Stars mark the block configurations "
        "selected by $J_\\lambda$ in Equation~\\eqref{eq:jlambda}; the annotations give "
        "the corresponding values of $\\lambda$. Larger $\\lambda$ assigns more weight "
        "to query time, favoring more storage to reduce query work. "
        "For memory-rich deployments, configurations using roughly 10\\,GB reduce lookup "
        f"time by {number(rare['speedup'], 0)}$\\times$--{number(common['speedup'], 0)}$\\times$ "
        "relative to Probe-all.",
        "The dashed curve fits index size linearly to stored-order count and latency "
        "linearly to allocation count, using the measured endpoints of the stored-order range. "
        "These counts correctly predict the direction of the trade-off. "
        "Actual costs also depend on the number of tables, the merging of equal hash keys, "
        "and how soon lookup finds its first collision. These effects explain why the measured "
        "points do not lie exactly on the count-based curve. The two hollow common-path points "
        "fail the figure's quality threshold and are excluded from the speedup comparison. "
        "Appendix~\\ref{app:exp-ablation} gives the configuration accounting and selected operating points."
    ]
    (paper / "experiments/frontier-fh-summary.tex").write_text("\n\n".join(text) + "\n")
    (root / "paper-frontier-summary.json").write_text(json.dumps(summaries, indent=2, allow_nan=False) + "\n")


def write_budget_results(root, paper, preview):
    report = budget_quality_results.load(root)
    if not report["cases"]:
        return
    grouped = budget_quality_results.group_outcomes(report["records"])
    methods = tuple(method for method in budget_quality_results.METHODS if method.startswith("fh_"))
    names = {method: NAMES[method.removeprefix("fh_")] for method in methods}
    k, budget = 10, "16gb-100ms"
    if k not in report["cases"]:
        raise ValueError("The fixed-budget illustration requires a completed k=10 measurement")
    rows = []
    for method in methods:
        group = grouped[k, budget, method]
        limits = group["limits"]
        if limits["index_budget_bytes"] != 16_000_000_000 or limits["query_deadline_seconds"] != .1:
            raise ValueError("The paper's operating point must use the measured 16 GB / 100 ms budget")
        stats = group["metrics"]
        cells = [names[method], group["status"]]
        cells.extend([
            number(stats["index_mb"] / 1000, 2), number(stats["median_ms"], 2),
            number(stats["p95_ms"], 2), number(stats["mean_ratio"], 3),
            percent(stats["a125"]), percent(stats["empty"])
        ] if stats else ["---"] * 6)
        rows.append(" & ".join(cells) + r"\\")
    (paper / "experiments/tables/feasibility-operating-point.tex").write_text(
        "% Validated n=1 million, k=10 outcomes under the shared 16 GB / 100 ms budget.\n" +
        tabular("llrrrrrr",
                r"Method & Feasible & GB & Median ms & p95 ms & Mean & $A_{1.25}$ (\%) & Miss (\%)",
                rows))
    selections = []
    for row in report["records"]:
        if row["k"] != k or row["budget"] != budget or not row["tested"] or row["method"] not in methods:
            continue
        choice = row["selection"]
        blocks = "+".join(map(str, choice["blocks"])) + (" (swap)" if choice["swapped"] else "")
        selections.append(
            f"{names[row['method']]} uses blocks ${blocks}$, "
            f"$\\delta={choice['delta']:g}$ and $L={choice['L']}$")
    parameter_note = ("\\noindent " + "; ".join(dict.fromkeys(selections)) + ".\n") if selections else ""
    (paper / "experiments/appendix/approx-budget-tables.tex").write_text(
        "\\begin{table*}[t]\n\\centering\n"
        "\\caption{Feasibility at $n=10^6,k=10$ under a shared 16\\,GB index budget and "
        "100\\,ms query deadline. NQ means no qualifying configuration in the bounded tuning search, "
        "not zero measured test accuracy. Mean is the distance ratio for timely answers; "
        "success and missing-answer rates count all queries.}\n"
        "\\label{tab:exp-budget-operating-point}\n\\small\n\\setlength{\\tabcolsep}{4pt}\n"
        "\\input{experiments/tables/feasibility-operating-point}\n"
        "\\end{table*}\n" + parameter_note)

    text = []
    point = grouped.get((10, "16gb-100ms", "fh_blocking"))
    if point and point["feasible"]:
        stats = point["metrics"]
        endpoints = [r for r in report["records"] if r["k"] == 10 and r["budget"] == "16gb-100ms"
                     and r["method"] in ("fh_probe_all", "fh_store_all")]
        endpoint_note = (
            "Neither factorial endpoint finds a qualifying configuration in the same bounded search. "
            "Dividing the order search therefore makes this collection searchable under "
            "constraints that the tested endpoints cannot meet."
            if all(not r["tested"] for r in endpoints) else
            "The detailed tables compare the factorial endpoints under the same constraints.")
        text.append(
            f"Blocking answers {percent(stats['a125'])}\\% of queries on time and within "
            f"the required distance. Its index uses {number(stats['index_mb'] / 1000, 1)}\\,GB "
            f"and median lookup takes {number(stats['median_ms'])}\\,ms. " + endpoint_note)
        if stats["max_ratio"] is not None and stats["max_ratio"] != "infinity" and stats["max_ratio"] <= 1 + 1e-8:
            text[0] += " Every answer returned on time in this case is optimal."
    (paper / "experiments/budget-fh-summary.tex").write_text("\n\n".join(text) + "\n")
    serializable = [dict(k=k, budget=budget, method=method, **value)
                    for (k, budget, method), value in grouped.items()]
    (Path(root) / "paper-budget-summary.json").write_text(
        json.dumps(dict(cases=report["cases"], outcomes=serializable), indent=2, allow_nan=False) + "\n")


def write_study_narratives(records, paper):
    nba = quality.select(records, "nba", "nba-n23805-k5")
    cl = quality.select(records, "nba", "nba-n23805-k5", "cl")
    intersection = quality.select(records, "nba", "nba-n23805-k5", "cl", "per_curve-intersection")
    hnsw = quality.select(records, "nba", "nba-n23805-k5", "cl", "hnsw")
    tuned = any(r["study"] == "nba" and r.get("quality_tuned") for r in records)
    if nba is None or cl is None or not tuned:
        raise ValueError("NBA narrative needs measured first-hit and candidate-list Blocking")
    text = [
        "For NBA queries requiring close geometric matches, we use "
        "independently tuned candidate-list Blocking (CL). It answers "
        f"{percent(cl['a125'])}\\% of queries within $1.25d^*$ in "
        f"{number(cl['median_ms'])}\\,ms. Verification selects the closest play among "
        "the colliding candidates instead of accepting the first one.",
        "A hash collision bounds distance at the grid's resolution, but does not rank "
        "the colliding plays by distance to the query. Real possessions are not generated "
        "as small perturbations of shared templates, so a first collision need not fall "
        "within a tight tolerance of the optimum. CL is useful here because it resolves "
        "this remaining choice by distance, after joint-key retrieval.",
    ]
    if hnsw and cl["a125"] >= .9 and hnsw["a125"] >= .9:
        text.append(
            "Joint indexing remains valuable with verification. At comparable approximation "
            f"success, CL Blocking is about {number(hnsw['median_ms'] / cl['median_ms'], 0)}$\\times$ "
            f"faster than Per-curve HNSW and uses "
            f"{number(hnsw['index_mb'] / cl['index_mb'], 0)}$\\times$ less index space."
            + (" It also has lower latency than Intersection."
               if intersection and cl["median_ms"] < intersection["median_ms"] else ""))
    text.append(
        "Candidate verification is an optional step after joint-key retrieval, not a "
        "different treatment of unknown correspondence. We use it here for tighter "
        "geometric accuracy; first hit remains the default elsewhere, avoiding this "
        "extra work where the first answer is already close.")
    (paper / "experiments/nba-quality-summary.tex").write_text("\n\n".join(text) + "\n")

    cases = [f"{kind}-n100000-k{k}" for kind in ("conjunctive", "rare") for k in (4, 8)]
    dtw = [quality.select(records, "dtw", case) for case in cases]
    if not all(dtw):
        raise ValueError("DTW applicability needs all four metric-specific measured cases")
    text = [
        "To test whether the same hash remains useful under a different curve distance, "
        "we replace Fr\\'echet distance by DTW for tuning and answer evaluation. "
        "The curves and block structures are unchanged, but nearest neighbors are "
        "recomputed under DTW rather than inherited from the Fr\\'echet experiment.",
        f"Across the four controlled workloads (Table~\\ref{{tab:exp-dtw-compact}} and Figure~\\ref{{fig:exp-dtw-applicability}}), $A_{{1.25}}$ ranges from "
        f"{percent(min(s['a125'] for s in dtw))}\\% to {percent(max(s['a125'] for s in dtw))}\\%. "
        f"The slowest median lookup is {number(max(s['median_ms'] for s in dtw))}\\,ms. "
        "Thus the joint hash also retrieves close DTW matches without computing DTW "
        "during lookup. Appendix~\\ref{app:exp-dtw} gives the tuning protocol and distance tails."
    ]
    (paper / "experiments/dtw-applicability-summary.tex").write_text("\n\n".join(text) + "\n")


def long_rows(records, study, cases, methods, tails=False):
    rows = []
    for case_label, case in cases:
        first = True
        for variant, method in methods:
            stats = quality.select(records, study, case, variant, method)
            if stats is None:
                continue
            cells = [case_label if first else "", method_label(method, variant, stats),
                     latency(stats["median_ms"]), number(stats["index_mb"], 1),
                     number(stats["mean_ratio"], 3)]
            if tails:
                cells.extend(number(stats[k], 3) for k in ("p95_ratio", "p99_ratio", "max_ratio"))
            cells.extend(percent(stats[k]) for k in ("a110", "a125", "empty"))
            rows.append(" & ".join(cells) + r"\\")
            first = False
        if rows:
            rows[-1] += r"\addlinespace[2pt]"
    return rows


def write_tables(records, paper, reference_results, near_results):
    dest = paper / "experiments/tables"
    common_methods = [("fh", "blocking"), ("fh", "probe_all"),
                      ("fh", "store_all"), ("cl", "per_curve-intersection"), ("cl", "hnsw")]
    normal_header = r"Case & Method & ms & MB & Mean $d/d^*$ & $A_{1.1}$ (\%) & $A_{1.25}$ (\%) & Empty (\%)"
    tail_header = (r"Case & Method & ms & MB & Mean $d/d^*$ & p95 & p99 & Max & "
                   r"$A_{1.1}$ (\%) & $A_{1.25}$ (\%) & Empty (\%)")
    tables = {}
    cases = [(REGIMES[kind], f"{kind}-n100000-k8") for kind in REGIMES]
    tables["regimes"] = tabular("llrrrrrr", normal_header,
                               long_rows(records, "comparison", cases, common_methods))
    cases = [(f"{REGIMES[kind]}, $n={n / 1e6:g}$M", f"{kind}-n{n}-k8")
             for kind, n in (("conjunctive", 100000), ("conjunctive", 1000000),
                             ("conjunctive", 2000000), ("conjunctive", 5000000),
                             ("rare", 1000000), ("rare", 5000000))]
    tables["scaling"] = tabular("llrrrrrr", normal_header,
                               long_rows(records, "comparison", cases, common_methods))
    cases = [(f"{f:g}", f"frequency-{key}-n100000-k8") for key, f in
             (("005", .5), ("020", 2), ("050", 5), ("180", 17.9), ("400", 40.3), ("650", 65.6))]
    tables["selectivity"] = tabular("llrrrrrr", normal_header.replace("Case", r"Occurrence (\%)"),
                                  long_rows(records, "selectivity", cases, [
                                      ("fh", "blocking"),
                                      ("fh", "probe_all"), ("fh", "store_all"),
                                      ("cl", "per_curve-intersection"), ("cl", "hnsw")]))
    rows = []
    nba_methods = [("cl", "blocking"), ("fh", "blocking"), *common_methods[1:]]
    for variant, method in nba_methods:
        stats = quality.select(records, "nba", "nba-n23805-k5", variant, method)
        if stats:
            rows.append(" & ".join([method_label(method, variant, stats), latency(stats["median_ms"]),
                                    number(stats["index_mb"], 1), number(stats["mean_ratio"], 3),
                                    percent(stats["a110"]), percent(stats["a125"]), percent(stats["empty"])]) + r"\\")
    tables["nba"] = tabular("lrrrrrr", r"Method & ms & MB & Mean $d/d^*$ & $A_{1.1}$ & $A_{1.25}$ & Empty", rows)
    rows = []
    for kind in REGIMES:
        for k in (4, 6, 8, 10):
            cells = [REGIMES[kind] if k == 4 else "", str(k)]
            for variant, method in [("fh", "blocking"), ("fh", "probe_all"), ("fh", "store_all"),
                                    ("cl", "per_curve-intersection"), ("cl", "hnsw")]:
                s = quality.select(records, "comparison", f"{kind}-n100000-k{k}", variant, method)
                cells.extend([latency(s["median_ms"]), number(s["mean_ratio"], 3), percent(s["a110"]),
                              percent(s["a125"])] if s else ["---"] * 4)
            rows.append(" & ".join(cells) + r"\\")
    header = r"Paths & $k$ & " + " & ".join(r"\multicolumn{4}{c}{" + name + "}" for name in
                                           ("Blocking", "Probe-all", "Store-all", "Intersection", "Per-curve HNSW"))
    header += r"\\ & & " + " & ".join([r"ms & Mean & $A_{1.1}$ & $A_{1.25}$"] * 5)
    tables["ksweep"] = tabular("ll" + "rrrr" * 5, header, rows)
    tables["first-hit-comparison"] = paired_comparison_table(records, reference_results, near_results)
    dtw_cases = [(f"{name}, $k={k}$", f"{kind}-n100000-k{k}") for kind, name in
                 (("conjunctive", "Common"), ("rare", "Rare")) for k in (4, 8)]
    paired_dtw = []
    for label, case in dtw_cases:
        stats = quality.select(records, "dtw", case)
        if stats is None:
            continue
        paired_dtw.append(" & ".join(
            [label, latency(stats["median_ms"]), number(stats["index_mb"], 1),
             number(stats["mean_ratio"], 3), number(stats["p95_ratio"], 3),
             number(stats["p99_ratio"], 3), number(stats["max_ratio"], 3),
             percent(stats["a110"]), percent(stats["a125"]), percent(stats["empty"])]) + r"\\")
    tables["dtw"] = tabular("lrrrrrrrrr", tail_header.replace(" & Method", ""), paired_dtw)
    rows = []
    for label, case in dtw_cases:
        s = quality.select(records, "dtw", case)
        if s:
            rows.append(" & ".join([label,
                                    latency(s["median_ms"]), number(s["mean_ratio"], 3),
                                    percent(s["a110"]), percent(s["a125"]), percent(s["empty"])]) + r"\\")
    tables["dtw-compact"] = tabular("lrrrrr", r"Case & ms & Mean $d/d^*$ & $A_{1.1}$ & $A_{1.25}$ & Empty", rows)
    rows = []
    for tag, study in (("A", "comparison"), ("B", "replication-b"), ("C", "replication-c")):
        for kind in ("conjunctive", "rare"):
            case = f"{kind}-n100000-k8"
            rows.extend(long_rows(records, study, [(f"{REGIMES[kind]}, {tag}", case)],
                                  [("fh", "blocking"), ("fh", "probe_all"), ("fh", "store_all"),
                                   ("cl", "per_curve-intersection")]))
    tables["replication"] = tabular("llrrrrrr", normal_header, rows)
    rows = []
    for label, study in (("$m=16$", "sensitivity-m16"), ("$m=32$", "comparison"),
                         ("$m=64$", "sensitivity-m64"), ("$r=0.25$", "sensitivity-rho025"),
                         ("$r=0.5$", "comparison"), ("$r=1$", "sensitivity-rho100")):
        for kind in ("conjunctive", "rare"):
            rows.extend(long_rows(records, study, [(f"{REGIMES[kind]}, {label}", f"{kind}-n100000-k8")],
                                  [("fh", "blocking"), ("fh", "probe_all"), ("fh", "store_all"),
                                   ("cl", "per_curve-intersection")]))
    tables["sensitivity"] = tabular("llrrrrrr", normal_header, rows)
    rows = []
    settings = [("Dataset " + tag, study) for tag, study in
                (("A", "comparison"), ("B", "replication-b"), ("C", "replication-c"))]
    settings += [(label, study) for label, study in
                 (("$m=16$", "sensitivity-m16"), ("$m=64$", "sensitivity-m64"),
                  ("$r=0.25$", "sensitivity-rho025"), ("$r=1$", "sensitivity-rho100"))]
    for label, study in settings:
        for kind in ("conjunctive", "rare"):
            case = f"{kind}-n100000-k8"
            blocking = quality.select(records, study, case)
            intersection = quality.select(records, study, case, "cl", "per_curve-intersection")
            if blocking is None and intersection is None:
                continue
            cells = [label, REGIMES[kind]]
            cells.extend([latency(blocking["median_ms"]), number(blocking["mean_ratio"], 3),
                          percent(blocking["a125"])] if blocking else ["---"] * 3)
            cells.extend([latency(intersection["median_ms"]), percent(intersection["a125"])]
                         if intersection else ["---"] * 2)
            rows.append(" & ".join(cells) + r"\\")
    tables["robustness"] = tabular("llrrrrr",
        r"Setting & Paths & \multicolumn{3}{c}{Blocking} & \multicolumn{2}{c}{Intersection}\\"
        r" & & ms & Mean & $A_{1.25}$ (\%) & ms & $A_{1.25}$ (\%)", rows)
    # Frozen CL preferences remain separately available; FH rows use the same chosen blocks.
    weights = (.1, 1., 10., 100.)
    rows = []
    for kind, n in (("conjunctive", 100000), ("rare", 100000)):
        case = f"{kind}-n{n}-k8"
        data = [quality.select(records, "lambda", case, weight=w) for w in weights]
        if not all(data):
            continue
        for w, s in zip(weights, data):
            rows.append(" & ".join([REGIMES[kind], f"{w:g}", latency(s["median_ms"]),
                                    number(s["index_mb"], 1), number(s["mean_ratio"], 3),
                                    percent(s["a125"]), percent(s["empty"])]) + r"\\")
    tables["lambda-fh"] = tabular("lrrrrrr",
                                  r"Paths & $\lambda$ & ms & MB & Mean & $A_{1.25}$ (\%) & Empty (\%)", rows)
    rows = []
    for kind in ("conjunctive", "rare"):
        case = f"{kind}-n100000-k8"
        for factor, metric in ((1.1, "a110"), (1.25, "a125")):
            for variant, method in [("fh", "blocking"),
                                    ("fh", "probe_all"), ("cl", "per_curve-intersection")]:
                cells = []
                for target in (.90, .95, .97, .99):
                    candidates = []
                    for L in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64):
                        s = quality.select(records, "matched", case, variant, method, L=L)
                        if s and s[metric] >= target:
                            candidates.append((L, s))
                    if candidates:
                        L, s = min(candidates, key=lambda pair: pair[0])
                        cells.append(latency(s["median_ms"]) + f" ({L})")
                    else:
                        cells.append("---")
                rows.append(" & ".join([REGIMES[kind], f"$A_{{{factor:g}}}$",
                                        method_label(method, variant, quality.select(
                                            records, "matched", case, variant, method, L=1)), *cells]) + r"\\")
    tables["matched-recall"] = tabular("lllrrrr",
                                       r"Paths & Quality & Method & 90\% & 95\% & 97\% & 99\%", rows)
    rows = []
    for label, case in [(REGIMES[kind], f"{kind}-n100000-k8") for kind in REGIMES] + [
            ("Common, $n=10^6$", "conjunctive-n1000000-k8")]:
        stats = quality.select(records, "comparison", case)
        if stats is None:
            raise ValueError(f"The appendix quality examples require a measured result: {case}")
        rows.append(" & ".join([label, *[number(stats[key], 3) for key in
                                        ("mean_ratio", "p95_ratio", "p99_ratio", "max_ratio")],
                                percent(stats["empty"])]) + r"\\")
    tables["blocking-quality-tails"] = tabular("lrrrrr",
        r"Case & Mean $d/d^*$ & p95 & p99 & Max & Empty (\%)", rows)
    tables["nba-quality-tails"] = tabular("llrrrrrrrrr", tail_header, long_rows(
        records, "nba", [("NBA", "nba-n23805-k5")], [("fh", "blocking"), ("cl", "blocking")], True))
    rows = []
    for row in records:
        if row["study"] != "nba" or row["method"] != "blocking" or not row.get("quality_tuned"):
            continue
        blocks = "+".join(map(str, row["blocks"])) + (" (swap)" if row["swapped"] else "")
        rows.append(" & ".join([str(row["seed"]), blocks, number(row["delta"], 0), str(row["L"]),
                                latency(row["median_ms"]), number(row["index_mb"], 1),
                                number(quality.summarize([row])["mean_ratio"], 3),
                                percent(row["a110"]), percent(row["a125"]), percent(row["empty"])]) + r"\\")
    if rows:
        tables["nba-quality-configurations"] = tabular("llrrrrrrrr",
            r"Seed & Blocks & $\delta$ & $L$ & ms & MB & Mean & $A_{1.1}$ & $A_{1.25}$ & Empty", rows)
    for name, text in tables.items():
        (dest / f"{name}.tex").write_text("% Generated from frozen per-query distance and timing records.\n" + text)
    return tables


def write_applicability_figures(records, paper, preview):
    context = _paper_context()
    try:
        cases = ["conjunctive-n100000-k4", "conjunctive-n100000-k8",
                 "rare-n100000-k4", "rare-n100000-k8"]
        fig, axes = _make_figure(3.31, 1.8, 2)
        stats = [quality.select(records, "dtw", case) for case in cases]
        if not all(stats):
            raise ValueError("DTW applicability figure requires the four measured cases")
        axes[0].bar(range(4), [s["median_ms"] for s in stats], width=.6,
                    color=method_style("blocking")["color"])
        for offset, (field, label, color) in enumerate((
                ("a110", r"$A_{1.1}$", method_style("blocking")["color"]),
                ("a125", r"$A_{1.25}$", method_style("probe_all")["color"]))):
            positions = [x + (offset - .5) * .34 for x in range(4)]
            axes[1].bar(positions, [100 * s[field] for s in stats], width=.32, color=color, label=label)
        for ax in axes:
            ax.set_xticks(range(4), labels=["C4", "C8", "R4", "R8"])
        axes[0].set_yscale("log")
        axes[0].set_ylabel("Query time (ms)")
        axes[1].set_ylabel("Success (%)")
        axes[1].set_ylim(0, 105)
        fig.legend(*axes[1].get_legend_handles_labels(), loc="upper center", ncol=2, frameon=False, fontsize=6.5)
        fig.subplots_adjust(left=.18, right=.99, bottom=.20, top=.76, wspace=.48)
        _save_close(fig, paper / "experiments/figures/fig-dtw-applicability.pdf",
                    preview=preview / "dtw-applicability.png")
    finally:
        context.__exit__(None, None, None)


def draw_series(ax, records, study, cases, methods, field):
    for variant, method in methods:
        points = []
        for x, case in cases:
            s = quality.select(records, study, case, variant, method)
            if s and s[field] is not None and s[field] != "infinity":
                points.append((x, s))
        if not points:
            continue
        style = "hnsw_curve" if method == "hnsw" else method
        options = method_style(style)
        label = NAMES[method]
        y = [100 * s[field] if field.startswith("a1") else s[field] for _, s in points]
        ax.plot([x for x, _ in points], y, **options, label=label)
        for (x, s), value in zip(points, y):
            if s["min_a125"] < .9:
                ax.plot(x, value, linestyle="none", marker=options["marker"],
                        markerfacecolor="white", markeredgecolor=options["color"], markersize=3.8)
        if field in ("median_ms", "index_mb"):
            key = "latency_range" if field == "median_ms" else "index_range"
            ax.fill_between([x for x, _ in points], [s[key][0] for _, s in points],
                            [s[key][1] for _, s in points], alpha=.07, color=options["color"], linewidth=0)


def write_figures(records, paper, preview):
    dest = paper / "experiments/figures"
    methods = [("fh", "blocking"), ("fh", "probe_all"), ("fh", "store_all"),
               ("cl", "per_curve-intersection"), ("cl", "hnsw")]
    context = _paper_context()
    try:
        fig, axes = _make_figure(6.96, 1.65, 4)
        for ax, kind in zip(axes[:3], REGIMES):
            cases = [(k, f"{kind}-n100000-k{k}") for k in (4, 6, 8, 10)]
            draw_series(ax, records, "comparison", cases, methods, "median_ms")
            ax.set_title(REGIMES[kind] + " paths")
            ax.set_yscale("log")
            ax.set_xticks([4, 6, 8, 10])
        draw_series(axes[3], records, "comparison", [(k, f"repeated-n100000-k{k}") for k in (4, 6, 8, 10)],
                    methods, "mean_ratio")
        axes[3].set_title("Repeated-path quality")
        axes[3].set_yscale("log")
        axes[3].set_xticks([4, 6, 8, 10])
        axes[0].set_ylabel("Query time (ms)")
        axes[3].set_ylabel(r"Mean $d/d^*$")
        fig.supxlabel(r"Curves per multi-curve $k$", y=.02, fontsize=7)
        fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=5, frameon=False,
                   fontsize=6.7, columnspacing=.7, handlelength=1.4)
        fig.subplots_adjust(left=.075, right=.995, bottom=.25, top=.70, wspace=.44)
        _save_close(fig, dest / "fig-regimes.pdf", preview=preview / "fh-regimes.png")
        fig, axes = _make_figure(3.31, 1.64, 2)
        cases = [(f, f"frequency-{name}-n100000-k8") for name, f in
                 (("005", .5), ("020", 2), ("050", 5), ("180", 17.9), ("400", 40.3), ("650", 65.6))]
        selected = [("fh", "blocking"), ("fh", "probe_all"), ("cl", "per_curve-intersection"), ("cl", "hnsw")]
        draw_series(axes[0], records, "selectivity", cases, selected, "median_ms")
        draw_series(axes[1], records, "selectivity", cases, selected, "a125")
        for ax in axes:
            ax.set_xscale("log")
            ax.set_xticks([.5, 5, 18, 65], labels=["0.5", "5", "18", "65"])
            ax.set_xlabel("Template occurrence (%)")
        axes[0].set_yscale("log")
        axes[0].set_ylabel("Query time (ms)")
        axes[1].set_ylabel(r"$A_{1.25}$ (%)")
        axes[1].set_ylim(-5, 105)
        fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2, frameon=False,
                   fontsize=6.2, columnspacing=.6, handlelength=1.2)
        fig.subplots_adjust(left=.18, right=.99, bottom=.25, top=.66, wspace=.48)
        _save_close(fig, dest / "fig-selectivity.pdf", preview=preview / "fh-selectivity.png")
        fig, axes = _make_figure(3.31, 1.64, 2)
        cases = [(n / 1e6, f"conjunctive-n{n}-k8") for n in (100000, 1000000, 2000000, 5000000)]
        selected = [("fh", "blocking"), ("fh", "probe_all"), ("cl", "per_curve-intersection"), ("cl", "hnsw")]
        draw_series(axes[0], records, "comparison", cases, selected, "median_ms")
        draw_series(axes[1], records, "comparison", cases, selected, "index_mb")
        for ax in axes:
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xticks([.1, 1, 2, 5], labels=["0.1", "1", "2", "5"])
            ax.set_xlabel(r"Collection size $n$ (millions)")
            ax.minorticks_off()
        axes[0].set_ylabel("Query time (ms)")
        axes[1].set_ylabel("Index size (MB)")
        fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2, frameon=False,
                   fontsize=6.5, columnspacing=.8, handlelength=1.4)
        fig.subplots_adjust(left=.18, right=.99, bottom=.25, top=.66, wspace=.47)
        _save_close(fig, dest / "fig-scaling.pdf", preview=preview / "fh-scaling.png")
        fig, axes = _make_figure(3.31, 1.72, 2)
        for ax, kind in zip(axes, ("conjunctive", "rare")):
            for variant, method in [("fh", "blocking"), ("fh", "probe_all"), ("cl", "per_curve-intersection")]:
                points = [quality.select(records, "matched", f"{kind}-n100000-k8",
                                         variant, method, L=L) for L in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)]
                points = [s for s in points if s]
                if not points:
                    continue
                ax.plot([s["median_ms"] for s in points], [100 * s["a125"] for s in points],
                        **method_style(method), label=NAMES[method])
            ax.set_xscale("log")
            ax.set_ylim(50, 101)
            ax.axhline(90, color=".5", linewidth=.5, linestyle=":")
            ax.axhline(95, color=".5", linewidth=.5, linestyle=":")
            ax.set_title(REGIMES[kind] + " paths")
            ax.set_xlabel("Query time (ms)")
        axes[0].set_ylabel(r"$A_{1.25}$ (%)")
        fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False,
                   fontsize=6.2, columnspacing=.6, handlelength=1.2)
        fig.subplots_adjust(left=.16, right=.99, bottom=.25, top=.72, wspace=.33)
        _save_close(fig, dest / "fig-matched-recall.pdf", preview=preview / "fh-matched.png")
    finally:
        context.__exit__(None, None, None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--paper", type=Path, required=True)
    parser.add_argument("--preview", type=Path, required=True)
    parser.add_argument("--reference-results", type=Path, default=Path("results/collection-reference-20261017"))
    parser.add_argument("--near-results", type=Path, default=Path("results/database-near-20261017"))
    parser.add_argument("--frontier-results", type=Path, default=Path("results/first-hit-quality-followup-20261017"))
    args = parser.parse_args()
    records = json.loads(args.records.read_text())
    records = first_seed_joint_view(records)
    args.preview.mkdir(parents=True, exist_ok=True)
    write_tables(records, args.paper, args.reference_results, args.near_results)
    write_figures(records, args.paper, args.preview)
    write_collection_reference(records, args.reference_results, args.paper)
    write_study_narratives(records, args.paper)
    write_applicability_figures(records, args.paper, args.preview)
    write_fh_frontier(records, args.frontier_results, args.paper, args.preview)
    if all((folder / "raw-test-queries.jsonl").exists()
           for folder in Path(args.frontier_results).glob("budget-*seed*") if (folder / "completion.json").exists()):
        write_budget_results(args.frontier_results, args.paper, args.preview)
    else:
        print("Skipping the hard-budget tables: raw budget query logs are not part of the compact reference results.")
