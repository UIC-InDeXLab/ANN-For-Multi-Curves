"""Rebuild tables and plots from the included aggregate measurements, without timing indexes."""
import argparse
import json
from pathlib import Path
import tempfile

from fresh import baseline_results, first_hit_results, frontier_results
from fresh import plot_experiment_figures as figures
from fresh import render_experiment_tables as tables
from fresh.publish_scaleup import scale_rows
from reproducibility.verify_reference import verify


def render(root, output, include_figures=False):
    root, output = Path(root), Path(output)
    verify(root)
    output.mkdir(parents=True, exist_ok=True)

    def read(name):
        return json.loads((root / f"{name}.json").read_text())

    comparison, scaling, nba = read("comparison"), read("scaling"), read("nba")
    auxiliary = read("auxiliary")
    external = {(r["case"], r["method"]): r["seeds"] for r in auxiliary["baselines"]}
    tables.FIRST_HIT.update({(r["case"], r["method"]): r["seeds"] for r in auxiliary["first_hit"]})
    figures.BASELINES_FOUND.update(external)
    feasibility = read("feasibility")
    summary, detail = tables.feasibility_tables(feasibility)
    ablation, _ = tables.ablation_table(comparison, scaling)
    budget_reports = [
        (r"1\,GB, 10\,ms", feasibility),
        (r"256\,MB, 1\,ms", read("budget-256mb-1ms")),
        (r"16\,GB, 10\,ms", read("budget-16gb-10ms")),
        (r"16\,GB, 100\,ms", read("budget-16gb-100ms")),
    ]
    replications = [("A", comparison), ("B", read("dataset-seed-b-20261015")),
                    ("C", read("dataset-seed-c-20261015"))]
    sensitivity = [
        ("$m=16$", read("sensitivity-m16-20261015")), ("$m=32$ (base)", comparison),
        ("$m=64$", read("sensitivity-m64-20261015")),
        ("$r=0.25$", read("sensitivity-rho025-20261015")), ("$r=0.5$ (base)", comparison),
        ("$r=1$", read("sensitivity-rho100-20261015")),
    ]
    rendered = {
        "first-hit": tables.first_hit_table(read("approximation")),
        "nba": tables.nba_table(nba, None, external),
        "regimes": tables.regimes_table(comparison, external=external),
        "selectivity": tables.selectivity_table(read("selectivity"), external),
        "lambda": ablation,
        "ksweep": tables.ksweep_table(comparison, external),
        "dtw": tables.dtw_table(read("dtw")),
        "dtw-compact": tables.dtw_compact_table(read("dtw")),
        "collision": tables.collision_table(read("collision")),
        "matched-recall": tables.matched_recall_table(read("matched-recall")),
        "build-time": tables.build_time_table(read("build-time")),
        "nba-selectivity": tables.nba_selectivity_table(read("nba-selectivity")),
        "replication": tables.replication_table(replications),
        "sensitivity": tables.sensitivity_table(sensitivity),
        "feasibility-summary": summary,
        "feasibility-detail": detail,
        "feasibility-budgets": tables.budget_table(budget_reports),
    }
    for name, text in rendered.items():
        (output / f"{name}.tex").write_text("% Regenerated from reference_results.\n" + text)
    # Scaling's Build column also needs the original selected seed build records.
    # Its released table is retained verbatim rather than inventing those timings.
    (output / "scaling.tex").write_bytes((root / "tables/scaling.tex").read_bytes())
    print(f"Wrote {len(rendered)} regenerated tables and the recorded scaling table to {output}")
    if include_figures:
        dest = output / "figures"
        dest.mkdir(exist_ok=True)
        figures.first_hit(read("approximation"), dest / "fig-first-hit.pdf")
        figures.regimes(comparison, dest / "fig-regimes.pdf")
        figures.scaling(scale_rows(comparison, scaling), dest / "fig-scaling.pdf")
        figures.selectivity(read("selectivity"), dest / "fig-selectivity.pdf")
        figures.ablation(comparison, scaling, dest / "fig-ablation.pdf")
        figures.feasibility(feasibility, dest / "fig-feasibility.pdf")
        figures.matched_recall(read("matched-recall"), dest / "fig-matched-recall.pdf")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / frontier_results.FOLDER
            path.mkdir()
            (path / "summary.json").write_bytes((root / "frontier.json").read_bytes())
            frontier = frontier_results.load(Path(temporary))
        for kind in ("conjunctive", "rare"):
            frontier[f"{kind}-n100000-k8"]["stars"] = {
                str(r["latency_weight"]): dict(index_mb=r["index_MB"], median_ms=r["median_ms"])
                for r in figures._complete_ablation_rows(comparison, 100000, kind)}
        figures.frontier(frontier, dest / "fig-frontier.pdf")
        print(f"Wrote eight plot PDFs to {dest}")
    return rendered


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, default=Path("reference_results"))
    parser.add_argument("--output", type=Path, default=Path("outputs/reference"))
    parser.add_argument("--figures", action="store_true", help="Requires the paper's TeX font dependencies")
    args = parser.parse_args()
    render(args.references, args.output, args.figures)
