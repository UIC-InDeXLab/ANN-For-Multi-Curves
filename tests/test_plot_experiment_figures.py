import importlib.util
import shutil
from copy import deepcopy
from pathlib import Path

import pytest


if importlib.util.find_spec("matplotlib") is None:
    pytest.skip("experiment figures require matplotlib", allow_module_level=True)
if shutil.which("kpsewhich") is None:
    pytest.skip("experiment figures require kpsewhich", allow_module_level=True)

import matplotlib.pyplot as plt

from fresh.plot_experiment_figures import (
    ABLATION_HEIGHT,
    PDF_PAD_INCHES,
    FEASIBILITY_WIDTH_NUDGE,
    FEASIBILITY_HEIGHT,
    FIRST_HIT_HEIGHT,
    FULL_WIDTH,
    REGIMES_HEIGHT,
    SCALING_HEIGHT,
    SELECTIVITY_HEIGHT,
    SINGLE_COLUMN_WIDTH,
    ablation,
    feasibility,
    first_hit,
    regimes,
    scaling,
    selectivity,
)


def canvas(width, height):
    return [width - 2 * PDF_PAD_INCHES, height - 2 * PDF_PAD_INCHES]


@pytest.fixture(autouse=True)
def closed_figures():
    plt.close("all")
    yield
    plt.close("all")
    assert plt.get_fignums() == []


@pytest.fixture
def capture_plot(monkeypatch):
    captured = {}

    def fake_configure():
        return None

    def fake_save(fig, path, pad_inches=0.02):
        captured["fig"] = fig
        captured["path"] = path
        captured["pad_inches"] = pad_inches

    monkeypatch.setattr("fresh.plot_experiment_figures.configure_paper_font", fake_configure)
    monkeypatch.setattr("fresh.plot_experiment_figures.save_paper_pdf", fake_save)
    return captured


@pytest.fixture
def approximation_report():
    rows = []
    for dimension, base in ((2, 2.0), (3, 3.0)):
        for offset, workload in enumerate(("random", "aligned", "separated-star")):
            for k in range(2, 11):
                rows.append(
                    dict(
                        workload=workload,
                        d=dimension,
                        k=k,
                        mean_ratio=base + offset + k / 10,
                    )
                )
    return dict(cases=rows)


@pytest.fixture
def scaling_rows_fixture():
    rows = []
    for n in (100000, 1000000, 2000000, 5000000):
        for method, factor in (
            ("blocking", 1.0),
            ("probe_all", 2.0),
            ("per_curve-intersection", 3.0),
            ("brute-force", 8.0),
        ):
            base = n / 100000
            rows.append(
                dict(
                    display_method=method,
                    n=n,
                    test_status="complete",
                    median_ms=factor * base / 10,
                    median_ms_seed_minmax=[factor * base / 11, factor * base / 9],
                    index_MB=factor * base,
                    index_MB_seed_minmax=[factor * base / 1.1, factor * base / 0.9],
                    test_target_met=not (method == "blocking" and n == 100000),
                )
            )
    return rows


@pytest.fixture
def selectivity_report():
    rows = []
    for fraction, latency, postings in (
        (0.005, 0.5, 12),
        (0.02, 0.8, 18),
        (0.05, 1.2, 30),
        (0.18, 2.0, 70),
        (0.40, 3.5, 120),
        (0.65, 6.0, 240),
    ):
        rows.append(
            dict(
                status="complete",
                realized_theoretical=fraction,
                blocking_latency_ms=latency,
                best_pc_latency_ms=latency / 2,
                blocking_mean_postings=postings,
                best_pc_mean_postings=postings * 10,
            )
        )
    return dict(comparisons=rows)


@pytest.fixture
def comparison_report():
    main_rows = []
    for kind, base_time, base_candidates in (
        ("conjunctive", 1.0, 80),
        ("rare", 0.4, 30),
        ("repeated", 5.0, 1000),
    ):
        for method, factor in (
            ("blocking", 1.0),
            ("probe_all", 2.0),
            ("per_curve-intersection", 3.0),
            ("brute-force", 4.0),
        ):
            for k in (4, 6, 8, 10):
                main_rows.append(
                    dict(
                        kind=kind,
                        n=100000,
                        k=k,
                        display_method=method,
                        test_status="complete",
                        median_ms=base_time * factor * k / 4,
                        median_ms_seed_minmax=[base_time * factor * k / 5, base_time * factor * k / 3.5],
                        mean_candidates=base_candidates * factor * k,
                        mean_candidates_seed_minmax=[
                            base_candidates * factor * k * 0.9,
                            base_candidates * factor * k * 1.1,
                        ],
                        test_target_met=not (method == "blocking" and kind == "repeated" and k == 8),
                    )
                )
        for k in (4, 6):
            main_rows.append(
                dict(
                    kind=kind,
                    n=100000,
                    k=k,
                    display_method="store_all",
                    test_status="complete",
                    median_ms=base_time * k / 8,
                    median_ms_seed_minmax=[base_time * k / 10, base_time * k / 6],
                    mean_candidates=base_candidates * k,
                    mean_candidates_seed_minmax=[base_candidates * k * 0.85, base_candidates * k * 1.15],
                    test_target_met=True,
                )
            )
    ablation_rows = []
    for kind, base_mem, base_lat in (("conjunctive", 20, 8), ("rare", 15, 5)):
        for weight, mem, lat in (
            (0.1, base_mem, base_lat),
            (1, base_mem, base_lat),
            (10, base_mem * 4, base_lat / 4),
            (100, base_mem * 10, base_lat / 10),
        ):
            ablation_rows.append(
                dict(
                    kind=kind,
                    n=100000,
                    k=8,
                    latency_weight=weight,
                    test_status="complete",
                    index_MB=mem,
                    median_ms=lat,
                )
            )
    return dict(main=main_rows, ablation=ablation_rows)


@pytest.fixture
def scaleup_report():
    ablation_rows = []
    for kind, base_mem, base_lat in (("conjunctive", 200, 20), ("rare", 130, 11)):
        for weight, mem, lat in (
            (0.1, base_mem, base_lat),
            (1, base_mem, base_lat),
            (10, base_mem * 6, base_lat / 4),
            (100, base_mem * 18, base_lat / 10),
        ):
            ablation_rows.append(
                dict(
                    kind=kind,
                    n=5000000,
                    k=8,
                    latency_weight=weight,
                    test_status="complete",
                    index_MB=mem,
                    median_ms=lat,
                )
            )
    return dict(ablation=ablation_rows)


@pytest.fixture
def feasibility_report():
    groups = []
    ks = list(range(4, 13))
    feasible_caps = {
        "blocking": 9,
        "store_all": 4,
        "probe_all": 8,
        "per_curve-intersection": 5,
    }
    for method, scale in (
        ("blocking", 100),
        ("store_all", 300),
        ("probe_all", 30),
        ("per_curve-intersection", 60),
    ):
        for k in ks:
            measured = k <= feasible_caps[method]
            row = dict(k=k, method=method, all_seeds_measured=measured, feasible=measured)
            if measured:
                row.update(
                    index_MB=scale * k,
                    index_MB_minmax=[scale * k * 0.9, scale * k * 1.1],
                    median_ms=0.2 * k,
                    median_ms_minmax=[0.18 * k, 0.22 * k],
                )
            groups.append(row)
    return dict(
        groups=groups,
        config=dict(
            cases=[dict(k=k) for k in ks],
            index_budget_bytes=1_000_000_000,
            query_deadline_seconds=0.01,
        ),
    )


def test_first_hit_layout_and_y_data(capture_plot, approximation_report):
    first_hit(approximation_report, Path("fixture.pdf"))

    fig = capture_plot["fig"]
    assert capture_plot["pad_inches"] == pytest.approx(0.02)
    assert len(fig.axes) == 2
    assert fig.get_size_inches().tolist() == canvas(SINGLE_COLUMN_WIDTH, FIRST_HIT_HEIGHT)
    assert [ax.get_title() for ax in fig.axes] == ["(a) 2-D", "(b) 3-D"]
    assert [tick.get_text() for tick in fig.axes[0].get_yticklabels()] == ["2", "3", "5", "10", "15"]
    assert fig.axes[0].get_ylim() == pytest.approx((1.8, 16.0))
    expected = {
        (2, "random"): [2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0],
        (2, "aligned"): [3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0],
        (2, "separated-star"): [4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0],
        (3, "random"): [3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0],
        (3, "aligned"): [4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0],
        (3, "separated-star"): [5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0],
    }
    for axis, dimension in zip(fig.axes, (2, 3)):
        for line, workload in zip(axis.lines, ("random", "aligned", "separated-star")):
            assert list(line.get_ydata()) == expected[(dimension, workload)]


def test_scaling_layout_y_data_and_hollow_marker(capture_plot, scaling_rows_fixture):
    original = deepcopy(scaling_rows_fixture)
    scaling(scaling_rows_fixture, Path("fixture.pdf"))

    fig = capture_plot["fig"]
    assert len(fig.axes) == 2
    assert fig.get_size_inches().tolist() == canvas(SINGLE_COLUMN_WIDTH, SCALING_HEIGHT)
    assert [ax.get_title() for ax in fig.axes] == [
        "(a) Median query latency",
        "(b) Allocated index size",
    ]
    assert fig.legends[0]._ncols == 2
    latency = {line.get_label(): list(line.get_ydata()) for line in fig.axes[0].lines if line.get_label() != "_nolegend_"}
    memory = {line.get_label(): list(line.get_ydata()) for line in fig.axes[1].lines if line.get_label() != "_nolegend_"}
    assert latency["Blocking"] == [0.1, 1.0, 2.0, 5.0]
    assert latency["Probe-all"] == [0.2, 2.0, 4.0, 10.0]
    assert latency["Intersection"] == [0.3, 3.0, 6.0, 15.0]
    assert "Exact scan" not in latency
    assert "Exact scan" not in memory
    assert memory["Blocking"] == [1.0, 10.0, 20.0, 50.0]
    assert any(line.get_markerfacecolor() == "white" for line in fig.axes[0].lines)
    assert scaling_rows_fixture == original


def test_regimes_layout_y_data_and_labels(capture_plot, comparison_report):
    original = deepcopy(comparison_report)
    regimes(comparison_report, Path("fixture.pdf"))

    fig = capture_plot["fig"]
    assert len(fig.axes) == 4
    assert fig.get_size_inches().tolist() == canvas(FULL_WIDTH, REGIMES_HEIGHT)
    assert [ax.get_title() for ax in fig.axes] == [
        "Common paths",
        "Rare paths",
        "Repeated paths",
        "Repeated candidates",
    ]
    labels = [line.get_label() for line in fig.axes[0].lines if line.get_label() != "_nolegend_"]
    assert labels == ["Blocking", "Probe-all", "Store-all", "Intersection"]
    latency = {line.get_label(): list(line.get_ydata()) for line in fig.axes[0].lines if line.get_label() != "_nolegend_"}
    repeated_candidates = [list(line.get_ydata()) for line in fig.axes[3].lines if line.get_markerfacecolor() != "white"]
    assert latency["Blocking"] == [1.0, 1.5, 2.0, 2.5]
    assert latency["Store-all"] == [0.5, 0.75]
    assert "Exact scan" not in latency
    assert [4000, 6000, 8000, 10000] in repeated_candidates
    assert [4000, 6000] in repeated_candidates
    assert any(line.get_markerfacecolor() == "white" for line in fig.axes[2].lines)
    assert comparison_report == original


def test_selectivity_layout_and_y_data(capture_plot, selectivity_report):
    selectivity(selectivity_report, Path("fixture.pdf"))

    fig = capture_plot["fig"]
    assert len(fig.axes) == 2
    assert fig.get_size_inches().tolist() == canvas(SINGLE_COLUMN_WIDTH, SELECTIVITY_HEIGHT)
    assert [ax.get_title() for ax in fig.axes] == [
        "(a) Median query latency",
        "(b) Mean postings read",
    ]
    assert fig.axes[1].get_xlim() == pytest.approx((0.45, 78.0))
    latency = {line.get_label(): list(line.get_ydata()) for line in fig.axes[0].lines}
    postings = {line.get_label(): list(line.get_ydata()) for line in fig.axes[1].lines}
    assert latency["Blocking"] == [0.5, 0.8, 1.2, 2.0, 3.5, 6.0]
    assert latency["Intersection"] == [0.25, 0.4, 0.6, 1.0, 1.75, 3.0]
    assert postings["Blocking"] == [12, 18, 30, 70, 120, 240]
    assert postings["Intersection"] == [120, 180, 300, 700, 1200, 2400]


def test_ablation_layout_and_y_data(capture_plot, comparison_report, scaleup_report):
    ablation(comparison_report, scaleup_report, Path("fixture.pdf"))

    fig = capture_plot["fig"]
    assert len(fig.axes) == 2
    assert fig.get_size_inches().tolist() == canvas(SINGLE_COLUMN_WIDTH, ABLATION_HEIGHT)
    assert [ax.get_title() for ax in fig.axes] == ["(a) n = 100k", "(b) n = 5M"]
    left = {line.get_label(): list(line.get_ydata()) for line in fig.axes[0].lines if line.get_label() != "_nolegend_"}
    right_series = [line for line in fig.axes[1].lines if line.get_marker() != "*"]
    assert left["Common paths"] == [8, 8, 2, 0.8]
    assert left["Rare paths"] == [5, 5, 1.25, 0.5]
    assert left["Default $\\lambda=1$"] == [8]
    assert list(right_series[0].get_ydata()) == [20, 20, 5, 2]
    assert list(right_series[1].get_ydata()) == [11, 11, 2.75, 1.1]
    annotations = {text.get_text() for ax in fig.axes for text in ax.texts}
    assert "1" not in annotations
    assert "0.1, 1" in annotations
    assert {"10", "100"} <= annotations


def test_feasibility_layout_and_y_data(capture_plot, feasibility_report):
    feasibility(feasibility_report, Path("fixture.pdf"))

    fig = capture_plot["fig"]
    assert len(fig.axes) == 3
    assert fig.get_size_inches().tolist() == canvas(
        FULL_WIDTH - FEASIBILITY_WIDTH_NUDGE, FEASIBILITY_HEIGHT
    )
    assert [ax.get_title() for ax in fig.axes] == [
        "(a) Verified-index feasibility",
        "(b) Allocated index size",
        "(c) Median query time",
    ]
    memory = {line.get_label(): list(line.get_ydata()) for line in fig.axes[1].lines if line.get_label() != "_nolegend_"}
    latency = {line.get_label(): list(line.get_ydata()) for line in fig.axes[2].lines if line.get_label() != "_nolegend_"}
    assert memory["Blocking (CL)"] == [400, 500, 600, 700, 800, 900]
    assert memory["Store-all (CL)"] == [1200]
    assert memory["Probe-all (CL)"] == [120, 150, 180, 210, 240]
    assert memory["Intersection"] == [240, 300]
    assert latency["Blocking (CL)"] == pytest.approx([0.8, 1.0, 1.2, 1.4, 1.6, 1.8])
    assert sum(line.get_marker() == "x" for line in fig.axes[0].lines) > 0


def test_completeness_errors(approximation_report, scaling_rows_fixture, selectivity_report, comparison_report):
    approximation_report["cases"] = [row for row in approximation_report["cases"] if not (row["d"] == 2 and row["k"] == 10)]
    with pytest.raises(ValueError, match="every k from 2 through 10"):
        first_hit(approximation_report, Path("fixture.pdf"))

    broken_scaling = [row for row in scaling_rows_fixture if not (row["display_method"] == "probe_all" and row["n"] == 5000000)]
    with pytest.raises(ValueError, match="every size and plotted method"):
        scaling(broken_scaling, Path("fixture.pdf"))

    comparison_report["main"] = [
        row for row in comparison_report["main"] if not (row["display_method"] == "probe_all" and row["kind"] == "conjunctive" and row["k"] == 10)
    ]
    with pytest.raises(ValueError, match="complete baseline rows at every plotted k"):
        regimes(comparison_report, Path("fixture.pdf"))

    selectivity_report["comparisons"][0]["status"] = "missing"
    with pytest.raises(ValueError, match="all six completed settings"):
        selectivity(selectivity_report, Path("fixture.pdf"))

    comparison_report["ablation"] = [row for row in comparison_report["ablation"] if row["latency_weight"] != 100]
    with pytest.raises(ValueError, match="all four completed lambda settings"):
        ablation(comparison_report, dict(ablation=[]), Path("fixture.pdf"))
