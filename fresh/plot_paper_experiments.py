"""Compact manuscript figures and tables derived from the frozen result reports."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, NullFormatter, ScalarFormatter

from fresh.e1 import write_json
from fresh.nba_full import digest
from fresh.nba_paper_examples import configure_paper_font, save_paper_pdf
from fresh.plot_poc import COLORS, LABELS, MAIN, REGIMES
from fresh.plot_styles import LINE_WIDTH, MARKERS, method_style


REGIME_TITLES = ["Common paths", "Rare paths", "Repeated paths"]
COMPARISON_METHODS = MAIN


COMPACT_ROW_HEIGHT = 1.65


def checked_report(results, recorded_source, expected_hash, marker):
    source = Path(recorded_source)
    source = results/source.parent.name/source.name
    if not (source.parent/marker).is_file():
        raise ValueError(f"Result report is not complete: {source}")
    if digest(source) != expected_hash:
        raise ValueError(f"Result report differs from the published measurements: {source}")
    return json.loads(source.read_text()), source


def read_report(paper, results, fragment, marker):
    metadata = json.loads((paper/"experiments"/"generated-results.json").read_text())
    record = metadata[fragment]
    return checked_report(results, record["source"], record["source_sha256"], marker)


def read_scaleup_report(paper, results):
    metadata = json.loads((paper/"experiments"/"scaleup-provenance.json").read_text())
    record = metadata["extension_source"]
    return checked_report(results, record["file"], record["sha256"], "completion.json")


def row_figure(count, *, height=2.1):
    fig, axes = plt.subplots(1, count, figsize=(7, height))
    for ax in axes:
        ax.grid(alpha=.2, linewidth=.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
    return fig, axes


def save_row(fig, destination, *, top=.86):
    try:
        fig.tight_layout(rect=(0, 0, 1, top), pad=.3, w_pad=.6)
        save_paper_pdf(fig, destination.with_suffix(".pdf"),pad_inches=.02)
        fig.savefig(destination.with_suffix(".png"), dpi=220,bbox_inches="tight",pad_inches=.02)
    finally:
        plt.close(fig)


def first_hit(report, destination):
    fig, axes = row_figure(4, height=COMPACT_ROW_HEIGHT)
    panels = ((2, "mean_ratio", "2-D mean"), (2, "p90_ratio", "2-D p90"),
              (3, "mean_ratio", "3-D mean"), (3, "p90_ratio", "3-D p90"))
    for ax, (dimension, field, title) in zip(axes, panels):
        for workload, label, style in (("random", "Random paths", "blocking"),
                                       ("aligned", "Translated copies", "probe_all"),
                                       ("separated-star", "Alternating segments", "store_all")):
            rows = sorted((r for r in report["cases"] if r["workload"] == workload
                           and r["d"] == dimension), key=lambda r: r["k"])
            if [r["k"] for r in rows] != list(range(2, 11)):
                raise ValueError("First-hit plots require every object count in both dimensions")
            ax.plot([r["k"] for r in rows], [r[field] for r in rows],
                    **method_style(style), label=label)
        ax.set_title(title)
        ax.set_xlabel("Objects per set")
        ax.set_xticks([2, 4, 6, 8, 10])
        ax.set_yscale("log")
        ax.set_yticks([2, 5, 10, 20, 40])
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
    axes[0].set_ylabel("Distance ratio")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3,
               frameon=False, fontsize=8)
    save_row(fig, destination, top=.82)


def regimes(report, destination):
    fig, axes = row_figure(4, height=COMPACT_ROW_HEIGHT)
    panels = [(kind, title, "median_ms") for kind, title in zip(REGIMES, REGIME_TITLES)]
    panels.append(("repeated", "Repeated candidates", "mean_candidates"))
    for ax, (kind, title, field) in zip(axes, panels):
        for method in COMPARISON_METHODS:
            rows = sorted((r for r in report["main"] if r["kind"] == kind
                           and r["n"] == 100000 and r["display_method"] == method
                           and r["test_status"] == "complete"), key=lambda r: r["k"])
            if not rows:
                continue
            x = [r["k"] for r in rows]
            color=COLORS[method]
            label={"per_curve-intersection":"Intersection"}.get(method,LABELS[method])
            ax.plot(x,[r[field] for r in rows],**method_style(method),label=label)
            ax.fill_between(x, [r[field+"_seed_minmax"][0] for r in rows],
                            [r[field+"_seed_minmax"][1] for r in rows],
                            color=color, alpha=.07, linewidth=0)
            for r in rows:
                if not r["test_target_met"]:
                    ax.plot(r["k"],r[field],marker=MARKERS[method],linestyle="none",
                            markersize=3.5,markeredgewidth=.55,
                            markerfacecolor="white",markeredgecolor=color)
        ax.set_title(title)
        ax.set_xlabel("Objects per set")
        ax.set_xticks([4, 6, 8, 10])
        ax.set_yscale("log")
    axes[0].set_ylabel("Query time (ms)")
    axes[3].set_ylabel("Candidates / query", fontsize=8)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=5,
               frameon=False, fontsize=8, columnspacing=.8, handlelength=2.5)
    save_row(fig, destination, top=.82)


def selectivity(report, destination):
    rows = sorted(report["comparisons"], key=lambda r: r["realized_theoretical"])
    if len(rows) != 6 or any(r["status"] != "complete" for r in rows):
        raise ValueError("Selectivity plots require all six completed settings")
    fig, axes = row_figure(4)
    x = [100*r["realized_theoretical"] for r in rows]
    axes[0].plot(x,[r["speedup_best_pc_over_blocking"] for r in rows],**method_style("blocking"))
    axes[0].axhline(1, color="black", linestyle=":", linewidth=.8)
    for key,method,label in (("blocking","blocking","Blocking"),
                            ("best_pc","per_curve-intersection","Intersection")):
        for ax, field, scale in ((axes[1], "index_MB", 1),
                                (axes[2], "recall", 100), (axes[3], "mean_postings", 1)):
            ax.plot(x,[scale*r[f"{key}_{field}"] for r in rows],
                    **method_style(method),label=label)
    for ax, title, label in zip(axes, ("Time ratio", "Index size", "Recall", "Entries read"),
                                ("Per-curve / blocking", "Index size (MB)", "Recall (%)", "Entries / query")):
        ax.set_title(title)
        ax.set_ylabel(label)
        ax.set_xscale("log")
        ax.set_xticks([.5, 5, 18, 65], labels=["0.5", "5", "18", "65"])
        ax.set_xlabel("Path occurrence (%)", fontsize=8)
    axes[1].set_yscale("log")
    axes[2].set_ylim(90, 101)
    axes[3].set_yscale("log")
    fig.legend(*axes[1].get_legend_handles_labels(), loc="upper center", ncol=2,
               frameon=False, fontsize=8,handlelength=2.5)
    save_row(fig, destination)


def ablation(report, destination, *, scaleup):
    fig, axes = row_figure(4, height=COMPACT_ROW_HEIGHT)
    panels = [(report, kind, 100000, title) for kind, title in zip(
        REGIMES, (r"Common: $10^5$", r"Rare: $10^5$", r"Repeated: $10^5$"))]
    panels.append((scaleup, "conjunctive", 5000000, r"Common: $5\times10^6$"))
    for ax, (source, kind, n, title) in zip(axes, panels):
        rows = sorted((r for r in source["ablation"] if r["kind"] == kind
                       and r["n"] == n and r["k"] == 8), key=lambda r: r["latency_weight"])
        if [r["latency_weight"] for r in rows] != [.1, 1, 10, 100] or any(
                r["test_status"] != "complete" for r in rows):
            raise ValueError("Weight ablation requires all four completed preferences")
        ax.plot([r["index_MB"] for r in rows],[r["median_ms"] for r in rows],
                "-",color=COLORS["blocking"],linewidth=LINE_WIDTH)
        for r in rows:
            ax.plot(r["index_MB"],r["median_ms"],
                    marker={.1:"o",1:"s",10:"^",100:"D"}[r["latency_weight"]],
                    linestyle="none",markersize=3,markeredgewidth=.55,color=COLORS["blocking"])
        groups = {}
        for r in rows:
            groups.setdefault((r["index_MB"], r["median_ms"]), []).append(r["latency_weight"])
        for (memory, latency), weights in groups.items():
            offset = -10 if any(other_memory == memory and 1 < other_latency/latency < 1.4
                                for other_memory, other_latency in groups) else 5
            ax.annotate(", ".join(f"{w:g}" for w in weights), (memory, latency),
                        xytext=(3, offset), textcoords="offset points", fontsize=8)
        main = next(r for r in rows if r["latency_weight"] == 1)
        ax.plot(main["index_MB"], main["median_ms"], "*", color="black", markersize=6,
                label=r"Default ($\lambda=1$)")
        ax.set_title(title)
        ax.set_xlabel("Index size (MB)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(min(r["index_MB"] for r in rows)*.65, max(r["index_MB"] for r in rows)*2.5)
        ax.set_ylim(min(r["median_ms"] for r in rows)*.75, max(r["median_ms"] for r in rows)*1.7)
        ax.yaxis.set_minor_formatter(NullFormatter())
        low, high = min(r["median_ms"] for r in rows), max(r["median_ms"] for r in rows)
        if high/low < 1.1:
            padding = max((high-low)/2, high*.01)
            ax.set_ylim(low-padding, high+padding)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.yaxis.set_major_formatter(ScalarFormatter())
    axes[0].set_ylabel("Query time (ms)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", frameon=False, fontsize=8)
    save_row(fig, destination, top=.82)


def synthetic_rows(report):
    lines = []
    for kind, title in zip(REGIMES, ("Common", "Rare", "Repeated")):
        for method,name in (("blocking","Blocking"),("per_curve-intersection","Intersection")):
            rows = [r for r in report["main"] if r["kind"] == kind and r["n"] == 100000
                    and r["k"] == 8 and r["display_method"] == method]
            if len(rows) != 1 or rows[0]["test_status"] != "complete":
                raise ValueError("Synthetic summary requires both methods in every regime")
            r = rows[0]
            lines.append(f"{title} & {name} & {r['index_MB']:.2f} & "
                         f"{r['median_ms']:.2f} & {100*r['recall_at_1']:.1f}" + r"\\")
    return "\n".join(lines) + "\n"


def nba_rows(report):
    lines = []
    for method, name in (("blocking", "Blocking"), ("per_curve-intersection", "Intersection"),
                         ("probe_all", "Probe all"),
                         ("store_all", "Store all")):
        rows = [r for r in report["main"] if r["method"] == method]
        if len(rows) != 1 or rows[0]["status"] != "complete":
            raise ValueError("NBA summary requires every baseline and all index seeds")
        r = rows[0]
        lines.append(f"{name} & {r['index_MB']:.2f} & {r['median_ms']:.2f} & "
                     f"{100*r['exact_recall']:.1f} & {100*r['c_success']['1.1']:.1f}" + r"\\")
    return "\n".join(lines) + "\n"


def preserved_panel_count(metadata, name, source_key, source, destination):
    recorded = metadata["sources"][source_key]
    relative = f"figures/compact-{name}.pdf"
    if (recorded["sha256"] != digest(source) or
            metadata["outputs"][relative] != digest(destination)):
        raise ValueError(f"Cannot preserve {name}: its source or published figure changed")
    return metadata["panel_counts"][name]


def main(args):
    configure_paper_font()
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9})
    paper, results = args.paper.resolve(), args.results.resolve()
    reports, sources = {}, {}
    for key, fragment, marker in (("approximation", "approximation-results.tex", "completion.json"),
                                  ("comparison", "comparison-results.tex", "completion.json"),
                                  ("nba", "nba-full-results.tex", "complete.json"),
                                  ("selectivity", "selectivity-results.tex", "completion.json")):
        reports[key], sources[key] = read_report(paper, results, fragment, marker)
    reports["scaleup"], sources["scaleup"] = read_scaleup_report(paper, results)
    output = paper/"experiments"
    figures = output/"figures"
    panel_counts = {"first-hit": 4, "regimes": 4, "selectivity": 4, "ablation": 4}
    metadata_path = output/"compact-results.json"
    previous = json.loads(metadata_path.read_text()) if metadata_path.is_file() else None
    for name, source_key in (("first-hit", "approximation"), ("regimes", "comparison"),
                             ("selectivity", "selectivity"), ("ablation", "comparison")):
        if name not in args.figures:
            if previous is None:
                raise ValueError("Selective regeneration requires existing compact-figure provenance")
            panel_counts[name] = preserved_panel_count(
                previous, name, source_key, sources[source_key], figures/f"compact-{name}.pdf")
    generated = []
    for name, draw, report in (("first-hit", first_hit, reports["approximation"]),
                                ("regimes", regimes, reports["comparison"]),
                                ("selectivity", selectivity, reports["selectivity"]),
                                ("ablation", ablation, reports["comparison"])):
        destination = figures/f"compact-{name}"
        if name in args.figures:
            if name == "ablation":
                draw(report, destination, scaleup=reports["scaleup"])
            else:
                draw(report, destination)
        generated.append(destination.with_suffix(".pdf"))
    for name, text, columns, heading in (
            ("synthetic", synthetic_rows(reports["comparison"]), "llrrr",
             r"Paths & Method & MB & Time (ms) & Recall (\%)"),
            ("nba", nba_rows(reports["nba"]), "lrrrr",
             r"Method & MB & Time (ms) & Recall (\%) & $1.10$ (\%)")):
        path = output/f"compact-{name}-table.tex"
        path.write_text(
            "% Derived from the frozen report; regenerate with fresh.plot_paper_experiments.\n"
            + r"\begin{tabular}{" + columns + "}\n"
            + "\\toprule\n" + heading + "\\\\\n\\midrule\n" + text
            + "\\bottomrule\n\\end{tabular}\n")
        generated.append(path)
    write_json(output/"compact-results.json", dict(
        sources={key: dict(file=str(path), sha256=digest(path)) for key, path in sources.items()},
        outputs={str(path.relative_to(output)): digest(path) for path in generated},
        panel_counts=panel_counts,
        changed_measurements=False, retuned=False))
    print("Rendered selected compact figure rows and two tables from unchanged completed reports.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", type=Path, required=True)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--figures", nargs="+",
                        choices=("first-hit", "regimes", "selectivity", "ablation"),
                        default=("first-hit", "regimes", "selectivity", "ablation"))
    main(parser.parse_args())
