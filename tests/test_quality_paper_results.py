import csv
import json
import tempfile
import unittest
from pathlib import Path

from fresh.quality_paper_results import flag, load, metrics, seed_of, summarize


class QualityPaperResultsTests(unittest.TestCase):
    def test_means_condition_on_returns_and_success_counts_all_queries(self):
        rows = [
            dict(query=0, id=1, distance=1.05, nn_distance=1, seconds=.001, timeout=False),
            dict(query=1, id=2, distance=1.2, nn_distance=1, seconds=.003, timeout=False),
            dict(query=2, id=-1, distance="", nn_distance=1, seconds=.002, timeout=False),
        ]
        result = summarize([dict(metrics(rows), seed=1701, index_mb=10)])
        self.assertAlmostEqual(result["mean_ratio"], 1.125)
        self.assertAlmostEqual(result["a110"], 1 / 3)
        self.assertAlmostEqual(result["a125"], 2 / 3)
        self.assertAlmostEqual(result["empty"], 1 / 3)
        self.assertEqual(result["median_ms"], 2)

    def test_late_answers_are_not_successes_and_zero_optimum_is_explicit(self):
        rows = [
            dict(query=0, id=1, distance=1, nn_distance=1, seconds=2, timeout="True"),
            dict(query=1, id=2, distance=2, nn_distance=0, seconds=.001, timeout="False"),
        ]
        result = summarize([dict(metrics(rows), seed=1701, index_mb=10)])
        self.assertEqual(result["a125"], 0)
        self.assertEqual(result["mean_ratio"], "infinity")
        self.assertEqual(result["empty"], .5)

    def test_duplicate_seeds_and_invalid_ground_truth_fail(self):
        one = dict(metrics([dict(id=0, distance=1, nn_distance=1, seconds=.001)]),
                   seed=1701, index_mb=1)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            summarize([one, one])
        with self.assertRaises(ValueError):
            metrics([dict(id=0, distance=.5, nn_distance=1, seconds=.001)])
        self.assertFalse(flag("False"))

    def test_both_seed_naming_conventions(self):
        self.assertEqual(seed_of(Path("common-seed1701")), 1701)
        self.assertEqual(seed_of(Path("seed-2301")), 2301)

    def test_first_seed_view_does_not_reduce_intersection_or_hnsw_seeds(self):
        from fresh.render_fh_blocking_paper import first_seed_joint_view
        records = [dict(method=method, seed=seed) for method in
                   ("blocking", "probe_all", "store_all", "per_curve-intersection", "hnsw")
                   for seed in (1701, 1702, 1703)]
        filtered = first_seed_joint_view(records)
        for method in ("blocking", "probe_all", "store_all"):
            self.assertEqual([r["seed"] for r in filtered if r["method"] == method], [1701])
        for method in ("per_curve-intersection", "hnsw"):
            self.assertEqual([r["seed"] for r in filtered if r["method"] == method], [1701, 1702, 1703])
        updated = [dict(method="blocking", seed=2302, study="nba", quality_tuned=True),
                   dict(method="blocking", seed=2302, study="nba-legacy")]
        self.assertEqual(first_seed_joint_view(updated), updated)

    def test_paired_table_has_explicit_k_for_every_workload(self):
        from fresh.render_fh_blocking_paper import TABLE2_CASES, paired_comparison_table
        records = []
        for _, study, case in TABLE2_CASES:
            for variant in ("cl", "fh"):
                records.append(dict(metrics([dict(query=0, id=1, distance=1, nn_distance=1, seconds=.001)]),
                                    study=study, case=case, variant=variant, method="blocking",
                                    seed=1701, index_mb=1))
        with tempfile.TemporaryDirectory() as folder:
            text = paired_comparison_table(records, Path(folder), Path(folder))
        self.assertIn("Case & $k$ &", text)
        for k in (4, 6, 8, 10):
            self.assertIn(f"Common & {k} &", text)
        self.assertIn("0.5\\% occurrence & 8 &", text)
        self.assertIn("Common, $n=10^6$ & 8 &", text)

    def test_publication_confines_cl_to_the_paired_and_nba_comparisons(self):
        from fresh.render_fh_blocking_paper import TABLE2_CASES, write_study_narratives, write_tables
        records = []
        for _, study, case in TABLE2_CASES:
            for variant in ("cl", "fh"):
                records.append(dict(metrics([dict(query=0, id=1, distance=1.2, nn_distance=1, seconds=.001)]),
                                    study=study, case=case, variant=variant, method="blocking",
                                    seed=1701, index_mb=1))
        for study, case in [("nba", "nba-n23805-k5")] + [
                ("dtw", f"{kind}-n100000-k{k}") for kind in ("conjunctive", "rare") for k in (4, 8)]:
            records.append(dict(metrics([dict(query=0, id=1, distance=1.2, nn_distance=1, seconds=.001)]),
                                study=study, case=case, variant="fh", method="blocking", quality_tuned=True,
                                seed=2301, index_mb=1, blocks=[4], swapped=False, delta=16, L=1))
        records.append(dict(metrics([dict(query=0, id=1, distance=1, nn_distance=1, seconds=.003)]),
                            study="nba", case="nba-n23805-k5", variant="cl", method="blocking",
                            seed=2301, index_mb=.5))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "experiments/tables").mkdir(parents=True)
            tables = write_tables(records, root, root, root)
            self.assertIn("{CL}", tables["first-hit-comparison"])
            self.assertIn("Blocking (CL)", tables["nba"])
            self.assertLess(tables["nba"].index("Blocking (CL)"), tables["nba"].index("Blocking &"))
            self.assertIn("Blocking (CL)", tables["nba-quality-tails"])
            for name, text in tables.items():
                if name not in ("first-hit-comparison", "nba", "nba-quality-tails"):
                    self.assertNotIn("CL", text, name)
                if name != "first-hit-comparison":
                    self.assertNotIn("FH", text, name)
                self.assertNotIn("$^{(", text, name)
            write_study_narratives(records, root)
            nba = (root / "experiments/nba-quality-summary.tex").read_text()
            for phrase in ("FH", "historical", "legacy", "seed"):
                self.assertNotIn(phrase, nba)
            self.assertIn("candidate-list Blocking (CL)", nba)
            self.assertIn("100.0\\%", nba)
            self.assertIn("independently tuned", nba)
            self.assertIn("1.200", tables["nba"])
            self.assertIn("optional step after joint-key retrieval", nba)

    def test_database_fraction_publishes_only_complete_full_query_sets(self):
        from fresh.render_fh_blocking_paper import completed_collection_reference, number
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            case = "conjunctive-n100000-k4"
            target = root / case
            target.mkdir()
            path = target / "summary.json"
            reference = dict(case=case, complete=False, queries=1000, evaluated_queries=752,
                             evaluated_database_entries="all", mean_database_within_125_fraction=.00013368)
            path.write_text(json.dumps(reference))
            self.assertIsNone(completed_collection_reference(root, case))
            reference.update(complete=True, evaluated_queries=1000)
            path.write_text(json.dumps(reference))
            self.assertEqual(number(100 * completed_collection_reference(root, case)[
                "mean_database_within_125_fraction"], 4), "0.0134")
            self.assertIsNone(completed_collection_reference(root, "not-yet-measured"))
            reference["evaluated_database_entries"] = "sample"
            path.write_text(json.dumps(reference))
            with self.assertRaises(ValueError):
                completed_collection_reference(root, case)

    def test_original_fh_baseline_returns_are_loaded_without_cl_substitution(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            job = root / "first-entry-a-20261016" / "conjunctive-n100000-k4-seed1701"
            job.mkdir(parents=True)
            main = [dict(display_method=method, status="complete", case_id="conjunctive-n100000-k4",
                         index_bytes=1e6) for method in ("probe_all", "store_all")]
            (job / "main-results.json").write_text(json.dumps(main))
            row = dict(display_method="probe_all", query=0, id=1, nn_distance=1, distance=1.2, seconds=.001)
            with (job / "test-queries.csv").open("w") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
                for method in ("probe_all", "store_all"):
                    for query in range(1000):
                        writer.writerow(dict(row, display_method=method, query=query))
            records = load(root)
            self.assertEqual({r["method"] for r in records}, {"probe_all", "store_all"})
            self.assertTrue(all(r["variant"] == "fh" and r["queries"] == 1000 for r in records))
            self.assertTrue(all(r["a110"] == 0 and r["a125"] == 1 for r in records))

    def test_completed_configuration_in_live_scale_job_has_separate_lambda_record(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            job = root / "first-hit-quality-followup-20261017" / "scaleup-common-n2000000-k8-seed1701"
            job.mkdir(parents=True)
            main = dict(config_id="complete", selected_table_count=True, status="complete",
                        family="3+1+1+1+1+1", L=2, metric="frechet", index_bytes=1e6,
                        case_id="common-n2000000-k8", index_seed=1701, labels=["blocking", "lambda=1"])
            (job / "main-results.json").write_text(json.dumps([main]))
            row = dict(config_id="complete", query=0, id=1, nn_distance=1, distance=1.2,
                       seconds=.001)
            with (job / "test-queries.csv").open("w") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
                for query in range(1000):
                    writer.writerow(dict(row, query=query))
                writer.writerow(dict(row, config_id="unfinished"))
            records = load(root)
            self.assertEqual(len(records), 2)
            self.assertEqual({r["study"] for r in records}, {"comparison", "lambda"})
            self.assertTrue(all(r["case"] == "conjunctive-n2000000-k8" for r in records))
            self.assertTrue(all(not r["job_complete"] for r in records))
            self.assertTrue(all(r["queries"] == 1000 and r["a125"] == 1 for r in records))

    def test_quality_tuned_seed_replaces_legacy_cohort_without_mixing_old_seeds(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for seed in (2301, 2302):
                old = root / "first-entry-nba-20261016" / f"nba-n23805-k5-seed{seed}"
                old.mkdir(parents=True)
                (old / "main-results.json").write_text(json.dumps([
                    dict(display_method="blocking", status="complete", case_id="nba-n23805-k5", index_bytes=1e6)]))
                with (old / "test-queries.csv").open("w") as handle:
                    row = dict(display_method="blocking", query=0, id=1, nn_distance=1, distance=1.5, seconds=.001)
                    writer = csv.DictWriter(handle, fieldnames=list(row))
                    writer.writeheader()
                    for q in range(1000):
                        writer.writerow(dict(row, query=q))
            new = root / "submission-fh-quality-20261017" / "nba-n23805-k5-frechet-seed2301"
            new.mkdir(parents=True)
            (new / "completion.json").write_text("{}")
            (new / "main-results.json").write_text(json.dumps([
                dict(study="nba", metric="frechet", case_id="nba-n23805-k5", index_seed=2301,
                     display_method="blocking", status="complete", index_bytes=2e6,
                     delta=16, L=32, blocks=[3, 2], swapped=False)]))
            with (new / "test-queries.csv").open("w") as handle:
                row = dict(display_method="blocking", query=0, id=1, nn_distance=1, distance=1.2, seconds=.001)
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
                for q in range(1000):
                    writer.writerow(dict(row, query=q))
            records = load(root)
            active = [r for r in records if r["study"] == "nba"]
            self.assertEqual(len(active), 1)
            self.assertTrue(active[0]["quality_tuned"])
            self.assertEqual(active[0]["a125"], 1)
            self.assertEqual({r["seed"] for r in records if r["study"] == "nba-legacy"}, {2301, 2302})


if __name__ == "__main__":
    unittest.main()
