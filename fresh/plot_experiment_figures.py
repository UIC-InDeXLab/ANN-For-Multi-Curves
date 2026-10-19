"""Render compact, print-safe experiment figures from frozen audited reports."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import NullFormatter, ScalarFormatter

from fresh import baseline_results, first_hit_results, frontier_results
from fresh.nba_paper_examples import configure_paper_font, save_paper_pdf
from fresh.plot_paper_experiments import read_report, read_scaleup_report
from fresh.plot_styles import COLORS, LINE_WIDTH, MARKERS, method_style
from fresh.publish_feasibility import LABELS as FEASIBILITY_LABELS
from fresh.publish_feasibility import METHODS as FEASIBILITY_METHODS
from fresh.publish_feasibility import audit as audit_feasibility
from fresh.publish_scaleup import DISPLAY_NAMES as SCALEUP_DISPLAY_NAMES
from fresh.publish_scaleup import checked_reports as checked_scaleup_reports
from fresh.publish_scaleup import scale_rows


BASELINES_FOUND = {}
FIRST_HIT = {}
FIRST_HIT_NAMES = {"blocking": "Blocking, first hit", "probe_all": "Probe-all, first hit",
                   "store_all": "Store-all, first hit"}
BASELINE_SERIES = (("per-curve-hnsw-intersection", "hnsw_curve", "Per-curve HNSW"),)

SINGLE_COLUMN_WIDTH = 3.35
FULL_WIDTH = 7.0
FIRST_HIT_HEIGHT = 1.72
REGIMES_HEIGHT = 1.55
SCALING_HEIGHT = 1.68
SELECTIVITY_HEIGHT = 1.68
ABLATION_HEIGHT = 1.68
FEASIBILITY_HEIGHT = 1.62
FRONTIER_HEIGHT = 1.85
PDF_PAD_INCHES = 0.02
FEASIBILITY_WIDTH_NUDGE = 0.011

PAPER_FONTS = {
    "font.size": 7.2,
    "axes.titlesize": 7.3,
    "axes.labelsize": 7.2,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "legend.fontsize": 7.0,
}

PREVIEW_DIR = Path(
    "outputs/previews"
)

WORKLOAD_SERIES = (
    ("random", "Random paths", "blocking"),
    ("aligned", "Translated copies", "probe_all"),
    ("separated-star", "Alternating segments", "store_all"),
)
SCALING_METHODS = ("blocking", "probe_all", "per_curve-intersection")
SELECTIVITY_SERIES = (
    ("blocking", "blocking_latency_ms", "blocking_mean_postings", "Blocking"),
    ("per_curve-intersection", "best_pc_latency_ms", "best_pc_mean_postings", "Intersection"),
)
ABLATION_SERIES = (
    ("conjunctive", "Common paths", "blocking"),
    ("rare", "Rare paths", "per_curve-intersection"),
)
REGIMES_METHODS = (
    ("blocking", "Blocking"),
    ("probe_all", "Probe-all"),
    ("store_all", "Store-all"),
    ("per_curve-intersection", "Intersection"),
)
REGIMES_PANELS = (
    ("conjunctive", "Common paths", "median_ms", "Query time (ms)"),
    ("rare", "Rare paths", "median_ms", None),
    ("repeated", "Repeated paths", "median_ms", None),
    ("repeated", "Repeated candidates", "mean_candidates", "Candidates / query"),
)


def _paper_context():
    context = plt.rc_context()
    context.__enter__()
    configure_paper_font()
    plt.rcParams.update(PAPER_FONTS)
    return context


def _canvas_size(width, height):
    return width - 2 * PDF_PAD_INCHES, height - 2 * PDF_PAD_INCHES


def _style_axis(ax):
    ax.grid(True, alpha=0.24, linewidth=0.5, color="#b9b9b9")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=0.6, length=2.6, pad=1.4)


def _make_figure(width, height, columns, *, sharey=False):
    fig, axes = plt.subplots(1, columns, figsize=(width, height), sharey=sharey)
    if columns == 1:
        axes = [axes]
    for ax in axes:
        _style_axis(ax)
    return fig, list(axes)


def _pin_pdf_bounds(fig):
    fig.add_artist(
        Rectangle(
            (0, 0),
            1,
            1,
            transform=fig.transFigure,
            fill=False,
            linewidth=0.01,
            edgecolor=(0, 0, 0, 0),
        )
    )


def _finalize(fig, dest: Path, *, preview: Path | None = None):
    _pin_pdf_bounds(fig)
    dest.parent.mkdir(parents=True, exist_ok=True)
    save_paper_pdf(fig, dest, pad_inches=PDF_PAD_INCHES)
    if preview is not None:
        preview.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview, dpi=200, bbox_inches=None, pad_inches=PDF_PAD_INCHES)


def _save_close(fig, dest: Path, *, preview: Path | None = None):
    try:
        _finalize(fig, dest, preview=preview)
    finally:
        plt.close(fig)


def _require_exact_keys(rows, key, expected, message):
    observed = [row[key] for row in rows]
    if observed != list(expected):
        raise ValueError(message)


def _major_log_plain(ax):
    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    ax.yaxis.set_major_formatter(formatter)
    ax.yaxis.set_minor_formatter(NullFormatter())


def _highlight_under_target(ax, x, y, method):
    ax.plot(
        x,
        y,
        marker=MARKERS[method],
        linestyle="none",
        markersize=3.5,
        markeredgewidth=0.55,
        markerfacecolor="white",
        markeredgecolor=COLORS[method],
    )


def _complete_ablation_rows(source, n, kind):
    rows = sorted(
        (
            row
            for row in source["ablation"]
            if row["n"] == n and row["k"] == 8 and row["kind"] == kind
        ),
        key=lambda row: row["latency_weight"],
    )
    if [row["latency_weight"] for row in rows] != [0.1, 1, 10, 100] or any(
        row["test_status"] != "complete" for row in rows
    ):
        raise ValueError("Ablation plots require all four completed lambda settings")
    return rows


def _annotate_weight_groups(ax, rows, *, common_paths):
    groups = defaultdict(list)
    max_memory = max(row["index_MB"] for row in rows)
    for row in rows:
        groups[(row["index_MB"], row["median_ms"])].append(row["latency_weight"])
    above, below = (4, 3), (-4, -3)
    for (memory, latency), weights in sorted(groups.items()):
        weights = sorted(weights)
        if weights == [1]:
            continue
        label = ", ".join(f"{weight:g}" for weight in weights)
        xshift, yshift = above if common_paths else below
        ax.annotate(
            label,
            (memory, latency),
            xytext=(xshift, yshift),
            textcoords="offset points",
            fontsize=6.7,
            ha="right" if xshift < 0 else "left",
            va="top" if yshift < 0 else "bottom",
        )


def first_hit(report, dest, *, preview: Path | None = None):
    context = _paper_context()
    try:
        fig, axes = _make_figure(*_canvas_size(SINGLE_COLUMN_WIDTH, FIRST_HIT_HEIGHT), 2, sharey=True)
        for ax, dimension, title in zip(axes, (2, 3), ("(a) 2-D", "(b) 3-D")):
            for workload, label, style in WORKLOAD_SERIES:
                rows = sorted(
                    (
                        row
                        for row in report["cases"]
                        if row["workload"] == workload and row["d"] == dimension
                    ),
                    key=lambda row: row["k"],
                )
                _require_exact_keys(
                    rows,
                    "k",
                    range(2, 11),
                    "First-hit plots require every k from 2 through 10 in both dimensions",
                )
                ax.plot(
                    [row["k"] for row in rows],
                    [row["mean_ratio"] for row in rows],
                    **method_style(style),
                    label=label if dimension == 2 else "_nolegend_",
                )
            ax.set_title(title, pad=4)
            ax.set_xticks([2, 4, 6, 8, 10])
            ax.set_yscale("log")
            ax.set_xlim(1.8, 10.2)
            ax.set_ylim(1.8, 16)
            ax.set_yticks([2, 3, 5, 10, 15], labels=["2", "3", "5", "10", "15"])
            _major_log_plain(ax)
        axes[0].set_ylabel(r"Mean ratio $d/d^*$")
        fig.supxlabel(r"Objects per set $k$", y=0.02, fontsize=7.2)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=3,
            frameon=False,
            handlelength=1.7,
            handletextpad=0.5,
            columnspacing=0.8,
        )
        fig.subplots_adjust(left=0.17, right=0.995, bottom=0.25, top=0.78, wspace=0.16)
        _save_close(fig, Path(dest), preview=preview)
    finally:
        context.__exit__(None, None, None)


def _baseline_points(dataset_of, xs, baseline, field="median_ms"):
    """Medians and seed ranges of one graph baseline at each x where it was run."""
    points = []
    for x in xs:
        seeds = BASELINES_FOUND.get((dataset_of(x), baseline))
        if seeds:
            values = [seed[field] for seed in seeds]
            points.append((x, sorted(values)[1], min(values), max(values)))
    return points


def _draw_first_hit(ax, case_of, xs, methods, *, label_it=True, scale=1.0, field="median_ms"):
    """First-hit variants: same color and marker as the method, dotted line, hollow marker."""
    for method in methods:
        points = []
        for x in xs:
            seeds = FIRST_HIT.get((case_of(x), method))
            if seeds and sum(seed["recall"] for seed in seeds) / len(seeds) >= 0.9:
                values = [seed[field] for seed in seeds]
                points.append((x * scale, sorted(values)[1], min(values), max(values)))
        if not points:
            continue
        style = method_style(method)
        style.update(linestyle=(0, (1.2, 1.2)), markerfacecolor="white", markeredgecolor=COLORS[method])
        ax.plot([p[0] for p in points], [p[1] for p in points], **style,
                label=FIRST_HIT_NAMES[method] if label_it else "_nolegend_")
        ax.fill_between([p[0] for p in points], [p[2] for p in points], [p[3] for p in points],
                        color=COLORS[method], alpha=0.05, linewidth=0)


def _draw_baselines(ax, dataset_of, xs, *, label_it=True, scale=1.0):
    for baseline, style, label in BASELINE_SERIES:
        points = _baseline_points(dataset_of, xs, baseline)
        if not points:
            continue
        x = [p[0] * scale for p in points]
        ax.plot(x, [p[1] for p in points], **method_style(style), label=label if label_it else "_nolegend_")
        for point in points:
            recall = sum(seed["recall"] for seed in BASELINES_FOUND[(dataset_of(point[0]), baseline)]) / 3
            if recall < 0.9:
                _highlight_under_target(ax, point[0] * scale, point[1], style)
        ax.fill_between(x, [p[2] for p in points], [p[3] for p in points], color=COLORS[style], alpha=0.07, linewidth=0)


def regimes(report, dest, *, preview: Path | None = None):
    context = _paper_context()
    try:
        fig, axes = _make_figure(*_canvas_size(FULL_WIDTH, REGIMES_HEIGHT), 4)
        for ax, (kind, title, field, ylabel) in zip(axes, REGIMES_PANELS):
            for method, label in REGIMES_METHODS:
                rows = sorted(
                    (
                        row
                        for row in report["main"]
                        if row["kind"] == kind
                        and row["n"] == 100000
                        and row["display_method"] == method
                        and row["test_status"] == "complete"
                    ),
                    key=lambda row: row["k"],
                )
                expected = [4, 6] if method == "store_all" else [4, 6, 8, 10]
                if [row["k"] for row in rows] != expected:
                    raise ValueError("Regimes plots require complete baseline rows at every plotted k")
                x = [row["k"] for row in rows]
                y = [row[field] for row in rows]
                ax.plot(x, y, **method_style(method), label=label if title == "Common paths" else "_nolegend_")
                limits = [row[f"{field}_seed_minmax"] for row in rows]
                ax.fill_between(
                    x,
                    [low for low, _ in limits],
                    [high for _, high in limits],
                    color=COLORS[method],
                    alpha=0.07,
                    linewidth=0,
                )
                for row in rows:
                    if not row["test_target_met"]:
                        _highlight_under_target(ax, row["k"], row[field], method)
            if field == "median_ms":
                _draw_first_hit(ax, lambda k, kind=kind: f"{kind}-n100000-k{k}", [4, 6, 8, 10],
                                ("blocking", "probe_all", "store_all"), label_it=title == "Common paths")
                _draw_baselines(ax, lambda k, kind=kind: f"{kind}-n100000-k{k}", [4, 6, 8, 10],
                                label_it=title == "Common paths")
            ax.set_title(title, pad=2)
            ax.set_xticks([4, 6, 8, 10])
            ax.set_yscale("log")
            if ylabel is not None:
                ax.set_ylabel(ylabel)
        fig.supxlabel(r"Objects per set $k$", y=0.02, fontsize=7.2)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=6,
            frameon=False,
            handlelength=1.5,
            handletextpad=0.4,
            columnspacing=0.6,
        )
        fig.subplots_adjust(left=0.075, right=0.995, bottom=0.24, top=0.72, wspace=0.4)
        _save_close(fig, Path(dest), preview=preview)
    finally:
        context.__exit__(None, None, None)


def scaling(rows, dest, *, preview: Path | None = None):
    context = _paper_context()
    try:
        fig, axes = _make_figure(*_canvas_size(SINGLE_COLUMN_WIDTH, SCALING_HEIGHT), 2)
        sizes = [100000, 1000000, 2000000, 5000000]
        for method in SCALING_METHODS:
            series = sorted(
                (
                    row
                    for row in rows
                    if row["display_method"] == method and row["test_status"] == "complete"
                ),
                key=lambda row: row["n"],
            )
            _require_exact_keys(
                series,
                "n",
                sizes,
                "Scaling plots require complete rows for every size and plotted method",
            )
            x = [row["n"] / 1e6 for row in series]
            label = SCALEUP_DISPLAY_NAMES[method]
            axes[0].plot(x, [row["median_ms"] for row in series], **method_style(method), label=label)
            limits = [row["median_ms_seed_minmax"] for row in series]
            axes[0].fill_between(
                x,
                [low for low, _ in limits],
                [high for _, high in limits],
                color=COLORS[method],
                alpha=0.07,
                linewidth=0,
            )
            if method != "brute-force":
                axes[1].plot(
                    x,
                    [row["index_MB"] for row in series],
                    **method_style(method),
                    label=label if method != "blocking" else "Blocking",
                )
                limits = [row["index_MB_seed_minmax"] for row in series]
                axes[1].fill_between(
                    x,
                    [low for low, _ in limits],
                    [high for _, high in limits],
                    color=COLORS[method],
                    alpha=0.07,
                    linewidth=0,
                )
            for point, row in zip(x, series):
                if not row["test_target_met"]:
                    _highlight_under_target(axes[0], point, row["median_ms"], method)
        _draw_first_hit(axes[0], lambda n: f"conjunctive-n{int(n)}-k8", [100000, 1000000],
                        ("blocking", "probe_all"), scale=1e-6)
        _draw_first_hit(axes[1], lambda n: f"conjunctive-n{int(n)}-k8", [100000, 1000000],
                        ("blocking", "probe_all"), scale=1e-6, label_it=False, field="index_mb")
        _draw_baselines(axes[0], lambda n: f"conjunctive-n{int(n)}-k8", [100000, 1000000], scale=1e-6)
        for baseline, style, label in BASELINE_SERIES:
            points = _baseline_points(lambda n: f"conjunctive-n{int(n)}-k8", [100000, 1000000], baseline, "index_mb")
            if points:
                axes[1].plot([p[0] / 1e6 for p in points], [p[1] for p in points], **method_style(style), label=label)
        for ax, title, ylabel in zip(
            axes,
            ("(a) Median query latency", "(b) Allocated index size"),
            ("Latency (ms)", "Allocated index (MB)"),
        ):
            ax.set_title(title, pad=2)
            ax.set_xlabel(r"Collection size $n$ (millions)")
            ax.set_ylabel(ylabel)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xticks([0.1, 1, 2, 5], labels=["0.1", "1", "2", "5"])
            ax.minorticks_off()
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=2,
            frameon=False,
            handlelength=1.45,
            handletextpad=0.45,
            columnspacing=0.45,
            fontsize=6.6,
        )
        fig.subplots_adjust(left=0.17, right=0.99, bottom=0.24, top=0.70, wspace=0.42)
        _save_close(fig, Path(dest), preview=preview)
    finally:
        context.__exit__(None, None, None)


def selectivity(report, dest, *, preview: Path | None = None):
    rows = sorted(report["comparisons"], key=lambda row: row["realized_theoretical"])
    if len(rows) != 6 or any(row["status"] != "complete" for row in rows):
        raise ValueError("Selectivity plots require all six completed settings")
    context = _paper_context()
    try:
        fig, axes = _make_figure(*_canvas_size(SINGLE_COLUMN_WIDTH, SELECTIVITY_HEIGHT), 2)
        x = [100 * row["realized_theoretical"] for row in rows]
        for method, latency_field, postings_field, label in SELECTIVITY_SERIES:
            axes[0].plot(x, [row[latency_field] for row in rows], **method_style(method), label=label)
            axes[1].plot(
                x,
                [row[postings_field] for row in rows],
                **method_style(method),
                label=label,
            )
        names = ["005", "020", "050", "180", "400", "650"]
        by_x = dict(zip(x, names))
        _draw_first_hit(axes[0], lambda value: f"frequency-{by_x[value]}-n100000-k8", x, ("blocking", "probe_all"))
        _draw_baselines(axes[0], lambda value: f"frequency-{by_x[value]}-n100000-k8", x)
        for ax, title, ylabel in zip(
            axes,
            ("(a) Median query latency", "(b) Mean postings read"),
            ("Latency (ms)", "Postings / query"),
        ):
            ax.set_title(title, pad=2)
            ax.set_xlabel("Individual-path occurrence (%)")
            ax.set_ylabel(ylabel)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xticks([0.5, 5, 18, 65], labels=["0.5", "5", "18", "65"])
            ax.set_xlim(0.45, 78)
            ax.minorticks_off()
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=3,
            frameon=False,
            handlelength=1.3,
            handletextpad=0.4,
            columnspacing=0.55,
            fontsize=6.6,
        )
        fig.subplots_adjust(left=0.18, right=0.985, bottom=0.24, top=0.76, wspace=0.34)
        _save_close(fig, Path(dest), preview=preview)
    finally:
        context.__exit__(None, None, None)


def ablation(comparison_report, scaleup_report, dest, *, preview: Path | None = None):
    context = _paper_context()
    try:
        fig, axes = _make_figure(*_canvas_size(SINGLE_COLUMN_WIDTH, ABLATION_HEIGHT), 2)
        panels = (
            (axes[0], comparison_report, 100000, "(a) n = 100k"),
            (axes[1], scaleup_report, 5000000, "(b) n = 5M"),
        )
        for ax, source, n, title in panels:
            for kind, label, style in ABLATION_SERIES:
                rows = _complete_ablation_rows(source, n, kind)
                x = [row["index_MB"] for row in rows]
                y = [row["median_ms"] for row in rows]
                ax.plot(x, y, **method_style(style), label=label if n == 100000 else "_nolegend_")
                _annotate_weight_groups(ax, rows, common_paths=(kind == "conjunctive"))
                default = next(row for row in rows if row["latency_weight"] == 1)
                ax.plot(
                    default["index_MB"],
                    default["median_ms"],
                    marker="*",
                    linestyle="none",
                    markersize=6.5,
                    markerfacecolor="white",
                    markeredgecolor="black",
                    markeredgewidth=0.75,
                    color="black",
                    label="Default $\\lambda=1$" if n == 100000 and kind == "conjunctive" else "_nolegend_",
                )
            ax.set_title(title, pad=2)
            ax.set_xlabel("Allocated index (MB)")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.margins(x=0.14, y=0.2)
        axes[0].set_ylabel("Median latency (ms)")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=3,
            frameon=False,
            handlelength=1.4,
            handletextpad=0.4,
            columnspacing=0.7,
            fontsize=6.7,
        )
        fig.subplots_adjust(left=0.18, right=0.985, bottom=0.24, top=0.69, wspace=0.24)
        _save_close(fig, Path(dest), preview=preview)
    finally:
        context.__exit__(None, None, None)


def frontier(data, dest, *, preview: Path | None = None, first_hit=False):
    """Index size versus median latency of every block partition, with the J_lambda selections and the count model."""
    panels = (("conjunctive-n100000-k8", "Common paths"), ("rare-n100000-k8", "Rare paths"))
    context = _paper_context()
    try:
        fig, axes = _make_figure(*_canvas_size(SINGLE_COLUMN_WIDTH, FRONTIER_HEIGHT), 2)
        for ax, (case_id, title) in zip(axes, panels):
            d = data[case_id]
            points = d["points"]
            others = [p for p in points if p["a"] != 1]
            probe_all = [p for p in points if p["a"] == 1]
            first = ax is axes[0]
            filled = [p for p in others if not first_hit or p["a125"] >= .9]
            hollow = [p for p in others if first_hit and p["a125"] < .9]
            ax.plot([p["index_mb"] for p in filled], [p["median_ms"] for p in filled], linestyle="none", marker="o",
                    markersize=2.6, markerfacecolor=COLORS["blocking"], markeredgecolor="black", markeredgewidth=0.35,
                    alpha=0.85, label="Configuration" if first else "_nolegend_")
            if hollow:
                ax.plot([p["index_mb"] for p in hollow], [p["median_ms"] for p in hollow],
                        linestyle="none", marker="o", markersize=3.2, markerfacecolor="white",
                        markeredgecolor=COLORS["blocking"], markeredgewidth=.7)
            fit = [p for p in points if not first_hit or p["a125"] >= .9]
            size, latency = frontier_results.count_model(fit)
            ax.plot(size, latency, linestyle=(0, (3, 1.6)), color="#555555", linewidth=0.7,
                    label="Count model" if first else "_nolegend_")
            ax.plot([p["index_mb"] for p in probe_all], [p["median_ms"] for p in probe_all],
                    **dict(method_style("probe_all"), linestyle="none", markersize=4.2),
                    label="Probe-all" if first else "_nolegend_")
            merged = {}
            for weight in sorted(d["stars"], key=float):
                star = d["stars"][weight]
                merged.setdefault((star["index_mb"], star["median_ms"]), []).append(f"{float(weight):g}")
            for number, ((x, y), weights) in enumerate(merged.items()):
                ax.plot(x, y, marker="*", linestyle="none", markersize=7,
                        markerfacecolor="white", markeredgecolor="black", markeredgewidth=0.75,
                        label=r"Selected by $J_\lambda$" if first and number == 0 else "_nolegend_")
                ax.annotate(", ".join(weights), (x, y), xytext=(3.5, 2.5), textcoords="offset points", fontsize=5.8)
            ax.set_title(title, pad=2)
            ax.set_xlabel("Index size (MB)")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.margins(x=0.1, y=0.12)
        axes[0].set_ylabel("Median latency (ms)")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=4, frameon=False,
                   handlelength=1.2, handletextpad=0.3, columnspacing=0.6, fontsize=6.2)
        fig.subplots_adjust(left=0.17, right=0.985, bottom=0.25, top=0.82, wspace=0.28)
        _save_close(fig, Path(dest), preview=preview)
    finally:
        context.__exit__(None, None, None)


def feasibility(report, dest, *, preview: Path | None = None):
    context = _paper_context()
    try:
        fig, axes = _make_figure(
            *_canvas_size(FULL_WIDTH - FEASIBILITY_WIDTH_NUDGE, FEASIBILITY_HEIGHT), 3
        )
        config = report["config"]
        ks = sorted({case["k"] for case in config["cases"]})
        labels = {method: FEASIBILITY_LABELS[method] +
                  (" (CL)" if method != "per_curve-intersection" else "")
                  for method in FEASIBILITY_METHODS}
        for index, method in enumerate(FEASIBILITY_METHODS):
            groups = sorted(
                (row for row in report["groups"] if row["method"] == method),
                key=lambda row: row["k"],
            )
            _require_exact_keys(
                groups,
                "k",
                ks,
                "Feasibility plots require every tested k for each method",
            )
            label = labels[method]
            for row in groups:
                axes[0].plot(
                    row["k"],
                    index,
                    MARKERS[method] if row["feasible"] else "x",
                    color=COLORS[method] if row["feasible"] else "black",
                    markersize=4.8 if row["feasible"] else 4.4,
                    markeredgewidth=0.9,
                )
            measured = [row for row in groups if row["all_seeds_measured"]]
            if measured:
                x = [row["k"] for row in measured]
                for ax, metric in ((axes[1], "index_MB"), (axes[2], "median_ms")):
                    ax.plot(
                        x,
                        [row[metric] for row in measured],
                        **method_style(method),
                        label=label,
                    )
                    limits = [row[f"{metric}_minmax"] for row in measured]
                    ax.fill_between(
                        x,
                        [low for low, _ in limits],
                        [high for _, high in limits],
                        color=COLORS[method],
                        alpha=0.07,
                        linewidth=0,
                    )
        axes[0].set_title("(a) Verified-index feasibility", pad=2)
        axes[0].set_yticks(range(len(FEASIBILITY_METHODS)), labels=[labels[m] for m in FEASIBILITY_METHODS])
        axes[0].invert_yaxis()
        axes[0].set_xlabel("Curves per multi-curve $k$")
        axes[1].set_title("(b) Allocated index size", pad=2)
        axes[1].set_xlabel("Curves per multi-curve $k$")
        axes[1].set_ylabel("Allocated index (MB)")
        axes[1].set_yscale("log")
        axes[1].axhline(1000, color="black", linestyle=":", linewidth=0.7)
        axes[2].set_title("(c) Median query time", pad=2)
        axes[2].set_xlabel("Curves per multi-curve $k$")
        axes[2].set_ylabel("Latency (ms)")
        axes[2].set_yscale("log")
        axes[2].axhline(10, color="black", linestyle=":", linewidth=0.7)
        for ax in axes:
            ax.set_xticks(ks)
            ax.set_xlim(ks[0] - .4, ks[-1] + .4)
        handles, labels = axes[1].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.005),
            ncol=4,
            frameon=False,
            handlelength=1.35,
            handletextpad=0.45,
            columnspacing=0.65,
            fontsize=6.8,
        )
        fig.subplots_adjust(left=0.095, right=0.995, bottom=0.24, top=0.78, wspace=0.24)
        _save_close(fig, Path(dest), preview=preview)
    finally:
        context.__exit__(None, None, None)


MATCHED_PANELS = (
    ("conjunctive-n100000-k8", "(a) Common paths"),
    ("rare-n100000-k8", "(b) Rare paths"),
)
MATCHED_METHODS = (
    ("blocking", "Blocking"),
    ("probe_all", "Probe-all"),
    ("per_curve-intersection", "Intersection"),
)
MATCHED_HEIGHT = 1.72


def matched_recall(summary, dest, *, preview: Path | None = None):
    aggregates = summary["aggregates"]
    context = _paper_context()
    try:
        fig, axes = _make_figure(*_canvas_size(SINGLE_COLUMN_WIDTH, MATCHED_HEIGHT), 2, sharey=True)
        for ax, (case_id, title) in zip(axes, MATCHED_PANELS):
            for method, label in MATCHED_METHODS:
                rows = sorted(
                    (r for r in aggregates if r["case_id"] == case_id and r["display_method"] == method),
                    key=lambda r: r["target_L"],
                )
                if not rows:
                    raise ValueError(f"Matched-recall plots need every method for {case_id}")
                ax.plot(
                    [r["median_seed_median_ms"] for r in rows],
                    [100 * r["mean_recall_at_1"] for r in rows],
                    **method_style(method),
                    label=label,
                )
            for target in (90, 95):
                ax.axhline(target, color="black", linestyle=":", linewidth=0.6)
            ax.set_title(title, pad=2)
            ax.set_xlabel("Median latency (ms)")
            ax.set_xscale("log")
            ax.set_ylim(75, 100.5)
            ax.minorticks_off()
        axes[0].set_ylabel("Recall@1 (%)")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=3, frameon=False,
                   handlelength=1.7, handletextpad=0.5, columnspacing=0.9)
        fig.subplots_adjust(left=0.18, right=0.985, bottom=0.24, top=0.76, wspace=0.12)
        _save_close(fig, Path(dest), preview=preview)
    finally:
        context.__exit__(None, None, None)


def main(args):
    paper = Path(args.paper).resolve()
    results = Path(args.results).resolve()
    figures_dir = paper / "experiments" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = Path(getattr(args, "preview_dir", PREVIEW_DIR)).resolve()
    preview_dir.mkdir(parents=True, exist_ok=True)

    approximation_report, _ = read_report(
        paper, results, "approximation-results.tex", "completion.json"
    )
    comparison_report, _ = read_report(
        paper, results, "comparison-results.tex", "completion.json"
    )
    selectivity_report, _ = read_report(
        paper, results, "selectivity-results.tex", "completion.json"
    )
    scaleup_report, _ = read_scaleup_report(paper, results)
    original_root = results / "fresh-synthetic-expanded-20261012"
    extension_root = results / "fresh-synthetic-scaleup-20261013"
    original, extension = checked_scaleup_reports(original_root, extension_root)
    scaling_rows = scale_rows(original, extension)
    feasibility_report = audit_feasibility(results / "fresh-feasibility-20261014")
    BASELINES_FOUND.update(baseline_results.load(results))

    outputs = [
        (first_hit, approximation_report, figures_dir / "fig-first-hit.pdf"),
        (regimes, comparison_report, figures_dir / "fig-regimes.pdf"),
        (scaling, scaling_rows, figures_dir / "fig-scaling.pdf"),
        (selectivity, selectivity_report, figures_dir / "fig-selectivity.pdf"),
        (
            ablation,
            (comparison_report, scaleup_report),
            figures_dir / "fig-ablation.pdf",
        ),
        (feasibility, feasibility_report, figures_dir / "fig-feasibility.pdf"),
    ]
    frontier_data = frontier_results.load(results)
    if {"conjunctive-n100000-k8", "rare-n100000-k8"} <= set(frontier_data):
        for kind in ("conjunctive", "rare"):
            frontier_data[f"{kind}-n100000-k8"]["stars"] = {
                str(row["latency_weight"]): dict(index_mb=row["index_MB"], median_ms=row["median_ms"])
                for row in _complete_ablation_rows(comparison_report, 100000, kind)}
        outputs.append((frontier, frontier_data, figures_dir / "fig-frontier.pdf"))
    matched_path = results / "matched-recall-20261015" / "summary.json"
    if matched_path.is_file():
        outputs.append((matched_recall, json.loads(matched_path.read_text()), figures_dir / "fig-matched-recall.pdf"))
    for draw, source, dest in outputs:
        preview = preview_dir / f"{dest.stem}.png"
        if draw is ablation:
            draw(source[0], source[1], dest, preview=preview)
        else:
            draw(source, dest, preview=preview)
    return [dest for _, _, dest in outputs]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", type=Path, required=True)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--preview-dir", type=Path, default=PREVIEW_DIR)
    main(parser.parse_args())
