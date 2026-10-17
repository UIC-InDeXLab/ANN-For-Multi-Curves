import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import MagicMock,patch

if importlib.util.find_spec("matplotlib") is None or importlib.util.find_spec("ijson") is None:
    raise unittest.SkipTest("NBA graphics summary validates in its local optional-dependency environment")

import numpy as np
from fresh import summarize_nba_full as MODULE
from fresh.summarize_nba_full import combine,comparison_figure,game_interval,quality_figure
from fresh.plot_styles import method_style


class FullNBASummaryTests(unittest.TestCase):
    def test_display_filters_rarest_without_changing_measurements_and_uses_print_safe_styles(self):
        import matplotlib.pyplot as plt
        methods=["blocking","per_curve-intersection","probe_all","store_all","per_curve-rarest"]
        rows=[dict(method=method,status="complete",median_ms=i+1,index_MB=10*(i+1),exact_recall=.95,
                   median_seed_ms=[i+.9,i+1,i+1.1],index_seed_MB=[9*(i+1),10*(i+1),11*(i+1)],
                   recall_query_interval=[.93,.97],mean_excess_feet=.1*(i+1),
                   c_success={str(float(c)):.8+c/10 for c in [1,1.05,1.1,1.25,1.5,2]})
              for i,method in enumerate(methods)]
        frozen=json.dumps(rows)
        comparison=comparison_figure(rows);quality=quality_figure(rows)
        try:
            for ax in comparison.axes+[quality.axes[1]]:
                self.assertEqual(len(ax.patches),4)
                self.assertEqual([tick.get_text() for tick in ax.get_xticklabels()],
                                 ["Blocking","Intersection","Probe-all","Store-all"])
                self.assertEqual(len({bar.get_hatch() for bar in ax.patches}),4)
                for bar in ax.patches:
                    self.assertEqual(bar.get_edgecolor(),(0,0,0,1))
                    self.assertEqual(bar.get_linewidth(),.55)
            self.assertEqual([bar.get_height() for bar in comparison.axes[0].patches],[1,2,3,4])
            self.assertEqual(len(quality.axes[0].lines),4)
            styles=set()
            for line,method in zip(quality.axes[0].lines,methods):
                expected=method_style(method)
                self.assertEqual(line.get_marker(),expected["marker"])
                self.assertEqual(line.get_linestyle(),expected["linestyle"])
                self.assertEqual(line.get_linewidth(),.7)
                self.assertEqual(line.get_markersize(),3)
                self.assertEqual(line.get_markeredgewidth(),.55)
                styles.add((line.get_marker(),line.get_linestyle()))
            self.assertEqual(len(styles),4)
            self.assertEqual(json.dumps(rows),frozen)
        finally:
            plt.close(comparison);plt.close(quality)

    def test_frozen_render_only_writes_two_pdfs_and_checks_receipt(self):
        root=Path("results");data=Path("data");output=Path("paper/figures")
        report=dict(main=[dict(method="blocking",status="complete"),
                          dict(method="per_curve-rarest",status="complete")])
        receipt=dict(summary_sha256="summary",dataset_sha256="dataset")
        payloads={root/"replicated-summary.json":json.dumps(report),
                  root/"analysis-complete.json":json.dumps(receipt)}
        comparison=MagicMock();quality=MagicMock()
        checksums={root/"replicated-summary.json":"summary",data/"dataset.json":"dataset"}
        with patch.object(Path,"exists",return_value=True), \
                patch.object(Path,"read_text",autospec=True,side_effect=lambda path:payloads[path]), \
                patch.object(Path,"write_text") as write_text,patch.object(MODULE,"write_json") as write_json, \
                patch.object(MODULE,"digest",side_effect=lambda path:checksums[path]), \
                patch.object(MODULE,"comparison_figure",return_value=comparison) as comparison_plot, \
                patch.object(MODULE,"quality_figure",return_value=quality) as quality_plot, \
                patch.object(MODULE.plt,"close") as close:
            MODULE.render_frozen_figures(root,data,output)
            comparison.savefig.assert_called_once_with(output/"nba-full-comparison.pdf",bbox_inches="tight")
            quality.savefig.assert_called_once_with(output/"nba-full-quality.pdf",bbox_inches="tight")
            write_text.assert_not_called();write_json.assert_not_called()
            self.assertEqual(close.call_count,2)
            checksums[root/"replicated-summary.json"]="changed"
            with self.assertRaisesRegex(ValueError,"analysis receipt"):
                MODULE.render_frozen_figures(root,data,output)
            self.assertEqual(comparison_plot.call_count,1)
            self.assertEqual(quality_plot.call_count,1)

    def test_latency_bars_have_zero_baseline_and_quality_intervals_are_drawn(self):
        import matplotlib.pyplot as plt
        row=dict(method="blocking",median_ms=1.4,index_MB=10,exact_recall=.95,
                 median_seed_ms=[1.3,1.4,1.5],index_seed_MB=[9,10,11],
                 recall_query_interval=[.93,.97])
        fig=comparison_figure([row])
        try:
            self.assertEqual(fig.axes[0].get_yscale(),"linear")
            self.assertEqual(fig.axes[0].get_ylim()[0],0)
            self.assertEqual(fig.axes[1].get_yscale(),"log")
            self.assertTrue(fig.axes[2].collections)
        finally:
            plt.close(fig)

    def test_game_resampling_keeps_correlated_queries_together(self):
        outcomes=np.array([[1,1,1],[1,1,1],[0,0,0],[0,0,0]],float)
        low,high=game_interval(outcomes,["game-A","game-A","game-B","game-B"])
        self.assertEqual(low,0)
        self.assertEqual(high,1)
        with self.assertRaises(ValueError):
            game_interval(outcomes,["game-A"])
    def test_partial_seed_is_not_summarized_as_complete(self):
        row=dict(method="blocking",seed=1,status="complete",queries=[])
        result=combine([row],[1,2,3])
        self.assertNotEqual(result["status"],"complete")
        self.assertNotIn("median_ms",result)

    def test_paired_results_keep_empty_answers_and_absolute_excess(self):
        rows=[]
        for seed in (1,2,3):
            queries=[dict(query=0,id=1,distance=2.,optimal_feet=2.,seconds=.001),
                     dict(query=1,id=-1,distance=None,optimal_feet=4.,seconds=.002)]
            rows.append(dict(method="one_block",seed=seed,status="complete",
                             configuration=dict(index_bytes=1000,delta=32,L=4),queries=queries))
        result=combine(rows,[1,2,3])
        self.assertEqual(result["method"],"per_curve-intersection")
        self.assertEqual(result["coverage"],.5)
        self.assertEqual(result["exact_recall"],.5)
        self.assertEqual(result["c_success"]["1.1"],.5)
        self.assertEqual(result["mean_excess_feet"],0)
        self.assertEqual(result["mean_ratio"],1)
        rows[2]["queries"][1]["optimal_feet"]=8
        with self.assertRaises(ValueError):
            combine(rows,[1,2,3])


if __name__=="__main__":
    unittest.main()
