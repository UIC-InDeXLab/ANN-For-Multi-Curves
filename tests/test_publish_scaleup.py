import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fresh.publish_scaleup import METHODS,ablation_table,appendix_table,checked_reports,draw_scaling,scale_rows


def row(n,method="blocking",kind="conjunctive"):
    return dict(kind=kind,n=n,k=8,display_method=method,test_status="complete",
                index_MB=n/100000,median_ms=n/1000000,p95_ms=n/500000,
                recall_at_1=.97,test_target_met=True,
                index_MB_seed_minmax=[n/110000,n/90000],
                median_ms_seed_minmax=[n/1100000,n/900000],
                recall_at_1_seed_minmax=[.96,.98])


class ScaleupPublicationTests(unittest.TestCase):
    def test_scaling_keeps_every_size_and_rejects_missing_baselines(self):
        original=dict(main=[row(n,method) for n in (100000,1000000) for method in METHODS])
        extension=dict(main=[row(n,method) for n in (2000000,5000000) for method in METHODS])
        self.assertEqual(len(scale_rows(original,extension)),20)
        extension["main"].pop()
        with self.assertRaisesRegex(ValueError,"every declared size"):
            scale_rows(original,extension)

    def test_incomplete_runs_cannot_be_published(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            with self.assertRaisesRegex(ValueError,"not complete"):
                checked_reports(root,root)
            (root/"completion.json").write_text("{}")
            config=dict(index_seeds=[1701,1702,1703],cases=[
                dict(id="conjunctive-n2000000-k8",kind="conjunctive",n=2000000,k=8)])
            (root/"config.json").write_text(json.dumps(config))
            (root/"cases.json").write_text("[]")
            with self.assertRaisesRegex(ValueError,"Every configured case"):
                checked_reports(root,root)

    def test_appendix_keeps_accuracy_and_explicit_resource_exclusions(self):
        report=dict(main=[row(5000000),
                          dict(kind="conjunctive",n=5000000,k=8,display_method="store_all",
                               test_status="incomplete across required seeds",
                               seed_statuses=[dict(reasons=["not run: integer capacity"]),
                                              dict(reasons=["not run: peak RAM estimate"])])])
        text=appendix_table(report)
        self.assertIn("97.0",text)
        self.assertIn("50.00",text)
        self.assertIn("construction RAM or integer capacity",text)
        report["main"][1]["seed_statuses"]=[dict(reasons=["not run: integer capacity"])]
        self.assertIn("not built: integer capacity",appendix_table(report))
        report["main"][1]["seed_statuses"][0]["reasons"]=["unknown"]
        with self.assertRaisesRegex(ValueError,"explicit construction"):
            appendix_table(report)

    def test_compact_scaling_plots_only_completed_runs_and_keeps_low_recall_markers(self):
        rows=[row(n,method) for n in (100000,1000000,2000000,5000000) for method in METHODS]
        for item in rows:
            if item["display_method"]=="store_all":
                item["test_status"]="excluded"
        rows[0].update(recall_at_1=.89,test_target_met=False)

        def inspect(fig,destination):
            self.assertEqual(len(fig.axes),3)
            self.assertEqual([ax.get_title() for ax in fig.axes],
                             ["Query time","Index size","Exact-NN recall"])
            labels={line.get_label() for line in fig.axes[0].lines}
            self.assertNotIn("Store-all",labels)
            self.assertIn("Exact scan",labels)
            self.assertNotIn("Rarest-first",labels)
            memory_labels={line.get_label() for line in fig.axes[1].lines}
            self.assertNotIn("Exact scan",memory_labels)
            self.assertTrue(any(line.get_markerfacecolor()=="white" for line in fig.axes[0].lines))
            self.assertEqual(fig.axes[2].get_ylim(),(85,100.5))
            self.assertEqual(list(fig.axes[0].get_xticks()),[.1,1,2,5])

        unchanged=copy.deepcopy(rows)
        with patch("fresh.publish_scaleup.configure_paper_font"), \
                patch("fresh.publish_scaleup.save_row",side_effect=inspect):
            draw_scaling(rows,Path("fixture"))
        self.assertEqual(rows,unchanged)

    def test_weight_table_labels_ablation_without_treating_it_as_a_main_method(self):
        record=dict(row(5000000),latency_weight=10)
        report=dict(ablation=[record])
        text=ablation_table(report)
        self.assertIn("not additional main methods",text)
        self.assertIn("5 & 10 & 50.00 & 5.00 & 97.0",text)
        record["test_status"]="incomplete"
        with self.assertRaisesRegex(ValueError,"partial weight"):
            ablation_table(report)


if __name__=="__main__":
    unittest.main()
