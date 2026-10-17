import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

if importlib.util.find_spec("matplotlib") is None:
    raise unittest.SkipTest("Compact paper figures use the local graphics environment")

import matplotlib.pyplot as plt

from fresh.nba_paper_examples import configure_paper_font
from fresh.e1 import star_pair
from fresh.plot_paper_experiments import (
    COMPACT_ROW_HEIGHT, ablation, first_hit, nba_rows, preserved_panel_count,
    read_report, read_scaleup_report, regimes, selectivity, synthetic_rows,
)
from fresh.plot_poc import MAIN, REGIMES
from fresh.nba_full import digest


class CompactPaperTests(unittest.TestCase):
    def test_legacy_star_input_is_collinear_with_alternating_endpoint_visits(self):
        import numpy as np
        stored,query,_=star_pair(2,32,2,512,1)
        self.assertTrue(np.all(stored[:,:,1:]==0))
        self.assertTrue(np.all(query[:,:,1:]==0))
        self.assertEqual(stored[0,0,0],np.mean(query[0,:2,0]))
        self.assertEqual(query[0,2,0],np.mean(stored[0,1:3,0]))
        self.assertEqual(query[0,1,0]-query[0,0,0],2)
        self.assertEqual(stored[0,2,0]-stored[0,1,0],2)

    def comparison(self):
        rows = []
        for kind in REGIMES:
            for method in MAIN + ["per_curve-rarest"]:
                for k in (4, 6, 8, 10):
                    rows.append(dict(kind=kind, display_method=method, n=100000, k=k,
                                     test_status="excluded" if method == "store_all" and k >= 8 else "complete",
                                     median_ms=50.5 if kind == "repeated" else k/10,
                                     median_ms_seed_minmax=[.5, 55], index_MB=20,
                                     mean_candidates=1000*k,
                                     mean_candidates_seed_minmax=[900*k,1100*k],
                                     recall_at_1=.96, test_target_met=True))
        alternatives = [dict(kind=kind, n=100000, k=8, latency_weight=weight,
                             test_status="complete", index_MB=10*weight,
                             median_ms=50.5 if kind == "repeated" else 1/weight)
                        for kind in REGIMES for weight in (.1, 1, 10, 100)]
        return dict(main=rows, ablation=alternatives)

    def test_synthetic_table_keeps_exact_report_values_and_all_regimes(self):
        text = synthetic_rows(self.comparison())
        self.assertEqual(len(text.splitlines()), 6)
        self.assertNotIn("Rarest",text)
        self.assertIn("Rare & Blocking & 20.00 & 0.80 & 96.0", text)
        self.assertIn("Repeated & Intersection & 20.00 & 50.50 & 96.0", text)

    def test_summary_tables_refuse_missing_or_incomplete_methods(self):
        report = self.comparison()
        report["main"] = [r for r in report["main"] if r["kind"] != "rare"]
        with self.assertRaisesRegex(ValueError, "every regime"):
            synthetic_rows(report)
        with self.assertRaisesRegex(ValueError, "every baseline"):
            nba_rows(dict(main=[]))

    def test_nba_table_uses_unconditional_success_and_omits_removed_baseline(self):
        report = dict(main=[dict(method=method, status="complete", index_MB=9.52,
                                 median_ms=1.36, exact_recall=.944, c_success={"1.1": .989})
                            for method in ("blocking", "per_curve-intersection", "per_curve-rarest",
                                           "probe_all", "store_all")])
        text = nba_rows(report)
        self.assertEqual(len(text.splitlines()), 4)
        self.assertNotIn("Rarest",text)
        self.assertIn("Blocking & 9.52 & 1.36 & 94.4 & 98.9",text)

    def test_report_refuses_unpublished_source_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/"paper"/"experiments").mkdir(parents=True)
            (root/"results"/"completed").mkdir(parents=True)
            (root/"results"/"completed"/"completion.json").write_text("{}")
            (root/"results"/"completed"/"report.json").write_text("{}")
            (root/"paper"/"experiments"/"generated-results.json").write_text(json.dumps({
                "comparison-results.tex": dict(source="/original/completed/report.json", source_sha256="changed")
            }))
            with self.assertRaisesRegex(ValueError, "published measurements"):
                read_report(root/"paper", root/"results", "comparison-results.tex", "completion.json")

    @unittest.skipUnless(shutil.which("kpsewhich"), "Compact figures use the paper fonts")
    def test_figures_have_four_nonoverlapping_panels_without_smaller_fonts(self):
        configure_paper_font()
        plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9})
        workloads = ("random", "aligned", "separated-star")
        quality = dict(cases=[dict(workload=workload, d=d, k=k,
                                  mean_ratio=d+workloads.index(workload)+k/10,
                                  p90_ratio=d+6+workloads.index(workload)+k)
                              for workload in workloads
                              for d in (2, 3) for k in range(2, 11)])
        crossover = dict(comparisons=[
            dict(status="complete", realized_theoretical=f, speedup_best_pc_over_blocking=2,
                 blocking_index_MB=10, best_pc_index_MB=20, blocking_recall=.96, best_pc_recall=.98,
                 blocking_mean_postings=5, best_pc_mean_postings=100000)
            for f in (.005, .02, .05, .18, .40, .65)])
        report = self.comparison()
        scaleup = dict(ablation=[dict(kind="conjunctive",n=5000000,k=8,
                                     latency_weight=weight,test_status="complete",
                                     index_MB=100*weight,median_ms=10/weight)
                                for weight in (.1,1,10,100)])
        for draw, data in ((first_hit, quality), (regimes, report),
                           (selectivity, crossover), (ablation, report)):
            with self.subTest(draw=draw.__name__):
                def check(fig, path, **kwargs):
                    fig.canvas.draw()
                    self.assertEqual(len(fig.axes),4)
                    expected_height = 2.1 if draw == selectivity else COMPACT_ROW_HEIGHT
                    self.assertEqual(fig.get_size_inches().tolist(),[7,expected_height])
                    self.assertTrue(all(ax.xaxis.label.get_fontsize()>=8 for ax in fig.axes))
                    self.assertTrue(all(tick.get_fontsize()>=8 for ax in fig.axes for tick in ax.get_xticklabels()))
                    self.assertEqual(kwargs["pad_inches"],.02)
                    axes = [ax.get_position() for ax in fig.axes]
                    self.assertTrue(all(abs(a.y0-axes[0].y0) < 1e-8 for a in axes))
                    self.assertTrue(all(a.x1 < b.x0 for a, b in zip(axes, axes[1:])))
                    boxes = [ax.get_tightbbox(fig.canvas.get_renderer()) for ax in fig.axes]
                    self.assertFalse(any(a.overlaps(b) for i, a in enumerate(boxes) for b in boxes[i+1:]))
                    legend=fig.legends[0].get_window_extent(fig.canvas.get_renderer())
                    gap=(legend.y0-max(box.y1 for box in boxes))/fig.dpi
                    self.assertGreaterEqual(gap,0)
                    self.assertLess(gap,.12)
                    if draw == first_hit:
                        self.assertEqual([ax.get_title() for ax in fig.axes],
                                         ["2-D mean","2-D p90","3-D mean","3-D p90"])
                        for ax,(dimension,field) in zip(fig.axes,((2,"mean_ratio"),(2,"p90_ratio"),
                                                                 (3,"mean_ratio"),(3,"p90_ratio"))):
                            for line,workload in zip(ax.lines,workloads):
                                rows=sorted((row for row in quality["cases"]
                                             if row["d"]==dimension and row["workload"]==workload),
                                            key=lambda row:row["k"])
                                self.assertEqual(list(line.get_ydata()),[row[field] for row in rows])
                    if draw == regimes:
                        self.assertEqual(fig.axes[1].get_title(),"Rare paths")
                        store = next(line for line in fig.axes[0].lines if line.get_label() == "Store all")
                        self.assertEqual(list(store.get_xdata()), [4, 6])
                        lines=[line for line in fig.axes[0].lines if line.get_label() in
                               ("Blocking","Probe all","Store all","Intersection","Brute force")]
                        self.assertEqual(len({line.get_marker() for line in lines}),len(lines))
                        self.assertEqual(len({repr(line._unscaled_dash_pattern) for line in lines}),len(lines))
                        self.assertTrue(all(line.get_linewidth()<=.7 for line in lines))
                        self.assertFalse(any("Rarest" in line.get_label() for line in fig.axes[0].lines))
                        self.assertEqual(fig.axes[3].get_ylabel(),"Candidates / query")
                        candidates=next(line for line in fig.axes[3].lines if line.get_label()=="Blocking")
                        self.assertEqual(list(candidates.get_ydata()),[4000,6000,8000,10000])
                    if draw == ablation:
                        self.assertEqual(list(fig.axes[3].lines[0].get_ydata()),[100,10,1,.1])
                with patch("fresh.plot_paper_experiments.save_paper_pdf", side_effect=check), \
                        patch.object(plt.Figure, "savefig"):
                    if draw == ablation:
                        draw(data,Path("unused"),scaleup=scaleup)
                    else:
                        draw(data,Path("unused"))

    def test_large_ablation_panel_refuses_missing_measurements(self):
        with self.assertRaisesRegex(ValueError,"all four completed preferences"):
            ablation(self.comparison(),Path("unused"),scaleup=dict(ablation=[]))
        plt.close("all")

    def test_large_ablation_source_refuses_stale_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            (root/"paper"/"experiments").mkdir(parents=True)
            folder=root/"results"/"scaleup";folder.mkdir(parents=True)
            (folder/"completion.json").write_text("{}")
            (folder/"report.json").write_text("{}")
            (root/"paper"/"experiments"/"scaleup-provenance.json").write_text(json.dumps({
                "extension_source":dict(file="/original/scaleup/report.json",sha256="changed")}))
            with self.assertRaisesRegex(ValueError,"published measurements"):
                read_scaleup_report(root/"paper",root/"results")

    def test_preserved_figure_requires_unchanged_source_and_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            source=root/"report.json";source.write_text("{}")
            destination=root/"figure.pdf";destination.write_bytes(b"published figure")
            metadata=dict(sources=dict(selectivity=dict(sha256=digest(source))),
                          outputs={"figures/compact-selectivity.pdf":digest(destination)},
                          panel_counts=dict(selectivity=4))
            self.assertEqual(preserved_panel_count(
                metadata,"selectivity","selectivity",source,destination),4)
            source.write_text('{"changed": true}')
            with self.assertRaisesRegex(ValueError,"Cannot preserve"):
                preserved_panel_count(metadata,"selectivity","selectivity",source,destination)
            source.write_text("{}");destination.write_bytes(b"edited figure")
            with self.assertRaisesRegex(ValueError,"Cannot preserve"):
                preserved_panel_count(metadata,"selectivity","selectivity",source,destination)

    def test_first_hit_refuses_to_hide_missing_object_counts(self):
        with self.assertRaisesRegex(ValueError, "every object count"):
            first_hit(dict(cases=[]), Path("unused"))
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
