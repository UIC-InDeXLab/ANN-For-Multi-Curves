import copy
import csv
import hashlib
import json
from math import factorial
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from fresh.e1 import arrays_hash, write_json
from fresh import publish_dtw as publisher


def read_json(path):
    return json.loads(path.read_text())


def change_json(path, change):
    value = read_json(path)
    change(value)
    write_json(path, value)


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields=None):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def query_row(q, optimum, distance, seconds):
    found = distance is not None
    ratio = (distance / optimum if optimum > 0 else
             (1. if distance == 0 else np.inf)) if found else None
    return dict(query=q,nn_distance=float(optimum),id=9 if found else -1,
                distance=distance,timeout=False,candidates=10,verified=2,
                probes=4,postings=10,seconds=seconds,found=found,
                exact_nn=bool(found and distance <= optimum + publisher.EPSILON),ratio=ratio)


def make_fixture(base):
    """Small arrays, but all fifteen canonical case/seed jobs and real hashes."""
    root = base / "results"
    root.mkdir()
    config = read_json(publisher.FROZEN_CONFIG)
    config.update(m=1,d=1,tuning_queries=2,test_queries=4,warmup_queries=1)
    synthetic, nba = base / "synthetic", base / "nba"
    synthetic.mkdir()
    nba.mkdir()
    config.update(synthetic_reference=str(synthetic),nba_data=str(nba))
    truth_distances = np.asarray([1.,2.,1.,2.,0.,4.,0.])
    parent_costs = np.asarray([1.,2.,1.,2.5,0.,5.,0.])
    truth_ids = np.zeros(7,np.int64)
    reference_ids = np.ones(7,np.int64)
    reference_ids[-1] = 0
    frechet_distances = truth_distances / 2
    identities, queries_by_case = {}, {}
    for case in config["cases"]:
        queries = np.arange(7*case["k"],dtype=np.float64).reshape(7,case["k"],1,1)
        queries[-1] = 0
        queries_by_case[case["id"]] = queries
        if case["kind"] != "nba":
            dataset_hash = hashlib.sha256(case["id"].encode()).hexdigest()
            folder = synthetic / f"{case['kind']}-n{case['n']}-k{case['k']}-seed1701"
            folder.mkdir()
            write_json(folder / "dataset.json",
                       dict(dataset_sha256=dataset_hash,queries_sha256=arrays_hash(queries),
                            raw_curve_bytes=case["n"]*case["k"]*8))
            np.savez(folder / "ground-truth.npz",queries=queries,ids=reference_ids,distances=frechet_distances)
            extra = dict(frechet_reference=str(folder),
                         frechet_reference_sha256=publisher.digest(folder / "ground-truth.npz"))
        else:
            database = np.zeros((case["n"],case["k"],1,1))
            dataset_hash = arrays_hash(database)
            np.savez(nba / "database.npz",curves=database)
            np.savez(nba / "tune.npz",curves=queries[:2])
            np.savez(nba / "test.npz",curves=queries[2:6])
            artifacts = {name: publisher.digest(nba / name) for name in
                         ("database.npz","tune.npz","test.npz")}
            write_json(nba / "dataset.json",dict(artifacts=artifacts))
            write_json(nba / "preparation-complete.json",{})
            config["nba_dataset_sha256"] = publisher.digest(nba / "dataset.json")
            for split, selection in (("tune",slice(0,2)),("test",slice(2,6))):
                np.savez(nba / f"{split}-ground-truth.npz",
                         ids=reference_ids[selection],distances=frechet_distances[selection])
            config["nba_reference_artifacts"] = {
                name: publisher.digest(nba / name) for name in
                ("tune-ground-truth.npz","test-ground-truth.npz")}
            extra = dict(nba_corpus_sha256=config["nba_dataset_sha256"],
                         nba_source_artifacts=artifacts,nba_reference_artifacts=config["nba_reference_artifacts"])
        identities[case["id"]] = dataset_hash, extra
    frozen = base / "frozen.json"
    write_json(frozen, config)
    write_json(root / "config.json", config)
    manifest = dict(commit="audited-fixture-commit",config_sha256=publisher.digest(frozen),
                    dtw_kernel_sha256="a"*64,metric=config["curve_metric"],
                    pin_core=config["pin_core"],build_threads=config["build_threads"])
    write_json(root / "manifest.json", manifest)
    write_json(root / "completion.json",dict(completed=123,elapsed_seconds=100))
    cache = root / "ground-truth-cache"
    cache.mkdir()
    for case in config["cases"]:
        dataset_hash, _ = identities[case["id"]]
        queries = queries_by_case[case["id"]]
        np.savez(cache / f"{case['id']}.npz",ids=truth_ids,distances=truth_distances,frechet_parent_dtw=parent_costs)
        write_json(cache / f"{case['id']}.json",
                   dict(dataset_sha256=dataset_hash,queries_sha256=arrays_hash(queries),
                        initial_parents_sha256=arrays_hash(reference_ids),dtw_kernel_sha256=manifest["dtw_kernel_sha256"],
                        array_sha256=publisher.digest(cache / f"{case['id']}.npz"),full_collection_size=case["n"],
                        preparation_seconds=1))
    cases = []
    jobs = publisher.study_jobs(config)
    for job in jobs:
        folder = root / job["id"]
        folder.mkdir()
        seeds = config["nba_index_seeds"] if job["kind"] == "nba" else config["synthetic_index_seeds"]
        seed_number = seeds.index(job["index_seed"])
        scale = (1,2,6)[seed_number]
        queries = queries_by_case[job["case_id"]]
        dataset_hash, extra = identities[job["case_id"]]
        write_json(folder / "dataset.json",
                   dict(job,**extra,dataset_sha256=dataset_hash,queries_sha256=arrays_hash(queries),
                        raw_curve_bytes=job["n"]*job["k"]*8,metric=config["curve_metric"],
                        cached_ground_truth=seed_number > 0,mean_test_dtw_optimum=1.75,
                        frechet_parent_not_dtw_optimal=.5))
        np.savez(folder / "ground-truth.npz",queries=queries,ids=truth_ids,
                 distances=truth_distances,frechet_parent_dtw=parent_costs)
        np.savez(folder / "frechet-reference.npz",ids=reference_ids,distances=frechet_distances)
        delta = config["nba_deltas"][0] if job["kind"] == "nba" else config["deltas"][0]
        trials = []
        for name, blocks, memory in (("probe",[1]*job["k"],10*scale),
                                     ("middle",[2,job["k"]-2],10*scale+5),
                                     ("store",[job["k"]],10*scale+200),
                                     ("intersection",None,10*scale+2)):
            trial = dict(config_id=name,method="blocking-family" if blocks else "per_curve-intersection",
                         blocks=blocks,swapped=False,delta=delta,L=1,index_seed=job["index_seed"])
            if blocks:
                trial.update(stored=np.prod([factorial(size) for size in blocks]).item(),
                             probed=factorial(job["k"])//np.prod([factorial(size) for size in blocks]).item())
            if name == "store" and job["k"] == 8:
                trial["status"] = "not run: peak RAM estimate"
            else:
                trial.update(status="qualified",index_bytes=memory*1000000,scratch_bytes=1000,
                             tuning_complete=True,tuning_recall=1.,tuning_median_seconds=.0001*scale,
                             build_seconds=.1)
            trials.append(trial)
        write_json(folder / "tuning-configurations.json",trials)
        tuning_rows = []
        for trial in trials:
            if trial["status"] != "qualified":
                continue
            for q in range(2):
                row = query_row(q,truth_distances[q],float(truth_distances[q]),.0001*scale)
                row.pop("nn_distance")
                tuning_rows.append(dict(config_id=trial["config_id"],**row))
        write_csv(folder / "tuning-queries.csv", tuning_rows)
        selections = dict(blocking=trials[0],probe_all=trials[0],
                          **{"per_curve-intersection":trials[3]})
        if trials[2]["status"] == "qualified":
            selections["store_all"] = trials[2]
        write_json(folder / "frozen-main-selections.json",selections)
        test_rows = []
        answers = [1.+publisher.EPSILON/2,2.2,0.,8. if seed_number == 0 else None if seed_number == 1 else 4.]
        for cfg_id in {trial["config_id"] for trial in selections.values()}:
            for q in range(4):
                row = query_row(q,truth_distances[2+q],answers[q],.001*(q+1)*scale)
                test_rows.append(dict(config_id=cfg_id,commit=manifest["commit"],config_sha256=manifest["config_sha256"],
                                      dataset_sha256=dataset_hash,queries_sha256=arrays_hash(queries),
                                      index_seed=job["index_seed"],metric="DTW",phase="test",**row))
        write_csv(folder / "test-queries.csv", test_rows)
        main = []
        for method in publisher.METHODS:
            if method in selections:
                main.append(dict(selections[method],display_method=method,test_status="complete",
                                 test_queries=4,build_seconds=.2))
            else:
                main.append(dict(display_method=method,index_seed=job["index_seed"],
                                 test_status="no feasible DTW tuning configuration",
                                 exclusion_reasons=["not run: peak RAM estimate"]))
        write_json(folder / "main-results.json",main)
        write_json(folder / "resources.json",
                   dict(peak_process_bytes=1000000000,returncode=0,interruption=None,
                        monitor_interval_seconds=config["monitor_interval_seconds"]))
        cases.append(dict(case=job["id"],case_id=job["case_id"],index_seed=job["index_seed"],
                          status="complete",returncode=0))
    write_json(root / "cases.json",cases)
    return root, frozen


class DTWPublisherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root, frozen = make_fixture(self.base)
        self.frozen_patch = patch.object(publisher,"FROZEN_CONFIG",frozen)
        self.frozen_patch.start()
        self.addCleanup(self.frozen_patch.stop)
        self.first = self.root / "common-n100000-k4-seed1701"

    def audit(self):
        return publisher.audit(self.root)

    def change_test(self, change, folder=None):
        path = (folder or self.first) / "test-queries.csv"
        rows = read_csv(path)
        change(rows)
        write_csv(path,rows)

    def change_frechet_optima(self, case_id, distances):
        config = read_json(self.root / "config.json")
        jobs = [job for job in publisher.study_jobs(config) if job["case_id"] == case_id]
        dataset = read_json(self.root / jobs[0]["id"] / "dataset.json")
        if jobs[0]["kind"] == "nba":
            for name,selection in (("tune-ground-truth.npz",slice(0,2)),
                                   ("test-ground-truth.npz",slice(2,6))):
                path = Path(config["nba_data"]) / name
                with np.load(path,allow_pickle=False) as original:
                    values = {key: original[key].copy() for key in original.files}
                values["distances"] = distances[selection].copy()
                np.savez(path,**values)
                config["nba_reference_artifacts"][name] = publisher.digest(path)
            write_json(publisher.FROZEN_CONFIG,config)
            write_json(self.root / "config.json",config)
            config_hash = publisher.digest(publisher.FROZEN_CONFIG)
            change_json(self.root / "manifest.json",lambda value: value.update(config_sha256=config_hash))
            for job in publisher.study_jobs(config):
                path = self.root / job["id"] / "test-queries.csv"
                rows = read_csv(path)
                for row in rows:
                    row["config_sha256"] = config_hash
                write_csv(path,rows)
            provenance = dict(nba_reference_artifacts=config["nba_reference_artifacts"])
        else:
            path = Path(dataset["frechet_reference"]) / "ground-truth.npz"
            with np.load(path,allow_pickle=False) as original:
                values = {key: original[key].copy() for key in original.files}
            values["distances"] = distances.copy()
            np.savez(path,**values)
            provenance = dict(frechet_reference_sha256=publisher.digest(path))
        for job in jobs:
            folder = self.root / job["id"]
            path = folder / "frechet-reference.npz"
            with np.load(path,allow_pickle=False) as original:
                values = {key: original[key].copy() for key in original.files}
            values["distances"] = distances.copy()
            np.savez(path,**values)
            change_json(folder / "dataset.json",lambda value: value.update(provenance))

    def test_valid_aggregation_shared_selections_and_explicit_unavailability(self):
        report = self.audit()
        self.assertEqual(len(report["main"]),20)
        row = report["main"][0]
        self.assertAlmostEqual(row["recall_at_1"],7/12)
        self.assertAlmostEqual(row["success_at_1_1"],5/6)
        self.assertAlmostEqual(row["empty_rate"],1/12)
        self.assertEqual(row["median_ms"],5.)
        np.testing.assert_allclose(row["median_ms_seed_minmax"],[2.5,15.])
        self.assertEqual(row["index_MB"],20.)
        self.assertEqual(row["index_MB_seed_minmax"],[10.,60.])
        self.assertAlmostEqual(row["p95_ms"],7.7)
        self.assertEqual(row["seeds"][0]["query_count"],4)
        self.assertEqual(row["seeds"][0]["config_id"],report["main"][1]["seeds"][0]["config_id"])
        unavailable = report["main"][6]
        self.assertEqual(unavailable["display_method"],"store_all")
        self.assertEqual(unavailable["available_seed_count"],0)
        self.assertNotIn("index_MB",unavailable)
        self.assertEqual(unavailable["exclusion_reasons"],["not run: peak RAM estimate"])
        self.assertTrue(all(row["nonoptimal_fraction"] == .5 for row in report["frechet_reference_parents"]))
        self.assertIn("ground-truth-cache/common-n100000-k4.json",report["source_artifacts"])

    def test_publisher_audit_does_not_load_dtw_kernels(self):
        code = (
            "import sys, unittest; sys.path.insert(0, 'tests'); "
            "suite = unittest.defaultTestLoader.loadTestsFromName("
            "'test_publish_dtw.DTWPublisherTests."
            "test_valid_aggregation_shared_selections_and_explicit_unavailability'); "
            "result = unittest.TextTestRunner().run(suite); "
            "assert result.wasSuccessful(); assert 'fresh.dtw' not in sys.modules"
        )
        result = subprocess.run(
            [sys.executable,"-c",code],cwd=Path(__file__).resolve().parents[1],
            capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def test_conditional_ratio_tails_are_recomputed_and_seed_averaged(self):
        report = self.audit()
        row = report["main"][0]
        ratios = [[1+publisher.EPSILON/2,1.1,1,2],
                  [1+publisher.EPSILON/2,1.1,1],
                  [1+publisher.EPSILON/2,1.1,1,1]]
        self.assertAlmostEqual(row["mean_ratio"],np.mean([np.mean(value) for value in ratios]))
        self.assertAlmostEqual(row["p95_ratio"],np.mean([np.quantile(value,.95) for value in ratios]))
        self.assertEqual([seed["returned_ratio_count"] for seed in row["seeds"]],[4,3,4])

    def test_missing_seed_or_incomplete_completion_is_an_error(self):
        change_json(self.root / "cases.json",lambda rows: rows.pop())
        with self.assertRaisesRegex(ValueError,"Every configured case and seed"):
            self.audit()
        (self.root / "completion.json").unlink()
        with self.assertRaisesRegex(ValueError,"Missing required"):
            self.audit()

    def test_duplicate_seed_and_nonsequential_jobs_are_errors(self):
        path = self.root / "cases.json"
        original = read_json(path)
        altered = copy.deepcopy(original)
        altered[1] = copy.deepcopy(altered[0])
        write_json(path,altered)
        with self.assertRaisesRegex(ValueError,"Every configured case and seed"):
            self.audit()
        write_json(path,[original[1],original[0]]+original[2:])
        with self.assertRaisesRegex(ValueError,"sequential protocol"):
            self.audit()

    def test_missing_duplicate_and_out_of_split_query_are_errors(self):
        path = self.first / "test-queries.csv"
        original = read_csv(path)
        for mode in ("missing","duplicate","identity"):
            rows = copy.deepcopy(original)
            if mode == "missing":
                rows.pop()
            elif mode == "duplicate":
                rows.append(copy.deepcopy(rows[0]))
            else:
                rows[0]["query"] = "1000"
            write_csv(path,rows)
            with self.subTest(mode=mode),self.assertRaisesRegex(ValueError,"query|Query"):
                self.audit()

    def test_selected_missing_result_and_missing_method_are_errors(self):
        path = self.first / "main-results.json"
        original = read_json(path)
        missing = copy.deepcopy(original)
        missing[0].pop("config_id")
        write_json(path,missing)
        with self.assertRaisesRegex(ValueError,"selected main result"):
            self.audit()
        write_json(path,original[:-1])
        with self.assertRaisesRegex(ValueError,"Every display method"):
            self.audit()

    def test_missing_selected_config_log_is_an_error(self):
        self.change_test(lambda rows: rows.__setitem__(slice(None),[row for row in rows if row["config_id"] != "probe"]))
        with self.assertRaisesRegex(ValueError,"Every query"):
            self.audit()

    def test_test_optimum_tampering_is_an_error(self):
        self.change_test(lambda rows: rows[0].__setitem__("nn_distance","100"))
        with self.assertRaisesRegex(ValueError,"exact cached optimum"):
            self.audit()

    def test_truth_arrays_must_equal_cache_even_if_optimum_is_plausible(self):
        path = self.first / "ground-truth.npz"
        with np.load(path,allow_pickle=False) as original:
            values = {name: original[name].copy() for name in original.files}
        values["distances"][2] = .5
        np.savez(path,**values)
        with self.assertRaisesRegex(ValueError,"cached exact truth"):
            self.audit()

    def test_dtw_frechet_consistency_checks_tuning_test_and_warmup_queries(self):
        with np.load(self.first / "ground-truth.npz",allow_pickle=False) as truth, \
                np.load(self.first / "frechet-reference.npz",allow_pickle=False) as reference:
            dtw = truth["distances"].copy()
            frechet = reference["distances"].copy()
        for query in (0,2,6):
            changed = frechet.copy()
            changed[query] = dtw[query] + 2*publisher.EPSILON
            self.change_frechet_optima("common-n100000-k4",changed)
            with self.subTest(query=query),self.assertRaisesRegex(
                    ValueError,f"consistency check failed for common-n100000-k4-seed1701, query {query}"):
                self.audit()

    def test_dtw_frechet_consistency_accepts_ties_and_tolerance_boundary(self):
        with np.load(self.first / "ground-truth.npz",allow_pickle=False) as truth, \
                np.load(self.first / "frechet-reference.npz",allow_pickle=False) as reference:
            dtw = truth["distances"].copy()
            frechet = reference["distances"].copy()
        for difference in (0.,publisher.EPSILON/2,publisher.EPSILON):
            changed = frechet.copy()
            for query in (0,2,6):
                changed[query] = dtw[query] + difference
            self.change_frechet_optima("common-n100000-k4",changed)
            with self.subTest(difference=difference):
                report = self.audit()
                self.assertIn("consistency only",report["definitions"]["dtw_frechet_consistency"])
                self.assertEqual(report["main"][0]["recall_at_1"],7/12)

    def test_nba_dtw_frechet_consistency_uses_hash_checked_split_truth(self):
        folder = self.root / "nba-n23805-k5-seed2301"
        with np.load(folder / "ground-truth.npz",allow_pickle=False) as truth, \
                np.load(folder / "frechet-reference.npz",allow_pickle=False) as reference:
            changed = reference["distances"].copy()
            changed[2] = truth["distances"][2] + 2*publisher.EPSILON
        self.change_frechet_optima("nba-n23805-k5",changed)
        with self.assertRaisesRegex(ValueError,"consistency check failed for nba-n23805-k5-seed2301, query 2"):
            self.audit()

    def test_every_seed_truth_matches_cache_exactly_not_with_tie_tolerance(self):
        for seed in (1701,1702,1703):
            path = self.root / f"common-n100000-k4-seed{seed}" / "ground-truth.npz"
            with np.load(path,allow_pickle=False) as original:
                saved = {name: original[name].copy() for name in original.files}
            for field in ("ids","distances","frechet_parent_dtw"):
                changed = {name: values.copy() for name,values in saved.items()}
                if field == "ids":
                    changed[field][0] += 1
                else:
                    changed[field][0] = np.nextafter(changed[field][0],np.inf)
                    self.assertLess(changed[field][0]-saved[field][0],publisher.EPSILON)
                np.savez(path,**changed)
                with self.subTest(seed=seed,field=field),self.assertRaisesRegex(ValueError,"cached exact truth"):
                    self.audit()
                np.savez(path,**saved)

    def test_cache_initial_parent_hash_uses_frechet_not_dtw_optimal_ids(self):
        cache = self.root / "ground-truth-cache" / "common-n100000-k4.json"
        with np.load(self.first / "frechet-reference.npz",allow_pickle=False) as reference, \
                np.load(self.first / "ground-truth.npz",allow_pickle=False) as truth:
            self.assertEqual(read_json(cache)["initial_parents_sha256"],arrays_hash(reference["ids"]))
            self.assertNotEqual(arrays_hash(reference["ids"]),arrays_hash(truth["ids"]))
            wrong_hash = arrays_hash(truth["ids"])
        self.audit()
        change_json(cache,lambda record: record.update(initial_parents_sha256=wrong_hash))
        with self.assertRaisesRegex(ValueError,"cache provenance"):
            self.audit()
        change_json(cache,lambda record: record.pop("initial_parents_sha256"))
        with self.assertRaisesRegex(ValueError,"cache provenance"):
            self.audit()

    def test_cache_array_checksum_and_all_provenance_fields_are_checked(self):
        path = self.root / "ground-truth-cache" / "common-n100000-k4.json"
        original = read_json(path)
        for field in ("dataset_sha256","queries_sha256","initial_parents_sha256",
                      "dtw_kernel_sha256","array_sha256","full_collection_size"):
            altered = dict(original)
            altered[field] = 0 if field == "full_collection_size" else "f"*64
            write_json(path,altered)
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,"cache provenance"):
                self.audit()
        write_json(path,original)
        cache = path.with_suffix(".npz")
        with cache.open("ab") as handle:
            handle.write(b"changed")
        with self.assertRaisesRegex(ValueError,"cache provenance"):
            self.audit()

    def test_test_provenance_fields_are_all_checked(self):
        path = self.first / "test-queries.csv"
        original = read_csv(path)
        for field, bad in (("commit","tampered"),("config_sha256","f"*64),
                           ("dataset_sha256","f"*64),("queries_sha256","f"*64),
                           ("index_seed","999"),("metric","Frechet"),("phase","tune"),
                           ("config_id","unselected")):
            rows = copy.deepcopy(original)
            rows[0][field] = bad
            write_csv(path,rows)
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,"provenance"):
                self.audit()

    def test_reported_quality_ratio_and_below_optimum_are_errors(self):
        path = self.first / "test-queries.csv"
        original = read_csv(path)
        for field, bad in (("exact_nn","False"),("ratio","20"),("distance",".5")):
            rows = copy.deepcopy(original)
            exact_row = next(row for row in rows if row["query"] == "0")
            exact_row[field] = bad
            write_csv(path,rows)
            with self.subTest(field=field),self.assertRaises(ValueError):
                self.audit()

    def test_timeouts_and_deadline_overruns_are_errors(self):
        path = self.first / "test-queries.csv"
        original = read_csv(path)
        rows = copy.deepcopy(original)
        rows[0]["timeout"] = "True"
        write_csv(path,rows)
        with self.assertRaisesRegex(ValueError,"cannot contain timeouts"):
            self.audit()
        rows = copy.deepcopy(original)
        rows[0]["seconds"] = "61"
        write_csv(path,rows)
        with self.assertRaisesRegex(ValueError,"deadline"):
            self.audit()

    def test_unqualified_config_cannot_be_frozen_or_relabelled_qualified(self):
        path = self.first / "tuning-configurations.json"
        change_json(path,lambda rows: rows[0].update(status="tuning recall below target",tuning_recall=0.))
        with self.assertRaisesRegex(ValueError,"tuning quality"):
            self.audit()

    def test_actual_unqualified_selection_is_rejected_and_explicit_exclusion_is_valid(self):
        path = self.first / "tuning-queries.csv"
        rows = read_csv(path)
        for row in rows:
            if row["config_id"] == "intersection":
                row.update(distance=str(float(row["distance"])*2),ratio="2.0",exact_nn="False")
        write_csv(path,rows)
        change_json(self.first / "tuning-configurations.json",
                    lambda trials: trials[3].update(status="tuning recall below target",tuning_recall=0.))
        selections = read_json(self.first / "frozen-main-selections.json")
        selections["per_curve-intersection"] = read_json(self.first / "tuning-configurations.json")[3]
        write_json(self.first / "frozen-main-selections.json",selections)
        with self.assertRaisesRegex(ValueError,"chosen configurations"):
            self.audit()
        selections.pop("per_curve-intersection")
        write_json(self.first / "frozen-main-selections.json",selections)
        test_path = self.first / "test-queries.csv"
        write_csv(test_path,[row for row in read_csv(test_path) if row["config_id"] != "intersection"])
        main = read_json(self.first / "main-results.json")
        main[3] = dict(display_method="per_curve-intersection",index_seed=1701,
                       test_status="no feasible DTW tuning configuration",
                       exclusion_reasons=["tuning recall below target"])
        write_json(self.first / "main-results.json",main)
        report = self.audit()
        group = report["main"][3]
        self.assertEqual(group["available_seed_count"],2)
        self.assertNotIn("recall_at_1",group)
        self.assertNotIn("median_ms",group)

    def test_tuning_missing_query_false_recall_and_false_timing_are_errors(self):
        path = self.first / "tuning-queries.csv"
        original = read_csv(path)
        write_csv(path,original[:-1])
        with self.assertRaisesRegex(ValueError,"Every query"):
            self.audit()
        rows = copy.deepcopy(original)
        rows[0].update(distance="2",ratio="2",exact_nn="False")
        write_csv(path,rows)
        with self.assertRaisesRegex(ValueError,"tuning quality"):
            self.audit()
        write_csv(path,original)
        change_json(self.first / "tuning-configurations.json",lambda trials: trials[0].update(tuning_median_seconds=.2))
        with self.assertRaisesRegex(ValueError,"tuning latency"):
            self.audit()

    def test_false_tuning_timeout_cannot_hide_a_qualified_configuration(self):
        change_json(self.first / "tuning-configurations.json",
                    lambda rows: rows[0].update(tuning_complete=False,status="tuning timeout",
                                               tuning_median_seconds=None))
        with self.assertRaisesRegex(ValueError,"first actual timeout"):
            self.audit()

    def test_actual_tuning_timeout_is_an_explicit_exclusion_not_a_test_result(self):
        path = self.first / "tuning-queries.csv"
        rows = read_csv(path)
        for row in rows:
            if row["config_id"] == "intersection" and row["query"] == "1":
                row.update(timeout="True",found="False",id="-1",distance="",ratio="",exact_nn="False")
        write_csv(path,rows)
        change_json(self.first / "tuning-configurations.json",
                    lambda rows: rows[3].update(tuning_complete=False,status="tuning timeout",
                                               tuning_recall=.5,tuning_median_seconds=None))
        change_json(self.first / "frozen-main-selections.json",
                    lambda choices: choices.pop("per_curve-intersection"))
        path = self.first / "test-queries.csv"
        write_csv(path,[row for row in read_csv(path) if row["config_id"] != "intersection"])
        main = read_json(self.first / "main-results.json")
        main[3] = dict(display_method="per_curve-intersection",index_seed=1701,
                       test_status="no feasible DTW tuning configuration",
                       exclusion_reasons=["tuning timeout"])
        write_json(self.first / "main-results.json",main)
        group = self.audit()["main"][3]
        self.assertEqual(group["available_seed_count"],2)
        self.assertIn("tuning timeout",group["exclusion_reasons"])

    def test_frozen_choices_must_match_the_actual_selection_rule(self):
        trials = read_json(self.first / "tuning-configurations.json")
        path = self.first / "frozen-main-selections.json"
        change_json(path,lambda choices: choices.__setitem__("blocking",trials[1]))
        with self.assertRaisesRegex(ValueError,"chosen configurations"):
            self.audit()

    def test_baseline_choice_must_be_fastest_even_if_old_choice_still_qualifies(self):
        trials = read_json(self.first / "tuning-configurations.json")
        faster = dict(trials[0],config_id="faster-probe",tuning_median_seconds=.00005)
        trials.append(faster)
        write_json(self.first / "tuning-configurations.json",trials)
        rows = read_csv(self.first / "tuning-queries.csv")
        added = [dict(row,config_id="faster-probe",seconds=".00005")
                 for row in rows if row["config_id"] == "probe"]
        write_csv(self.first / "tuning-queries.csv",rows+added)
        choices = read_json(self.first / "frozen-main-selections.json")
        choices["blocking"] = faster
        write_json(self.first / "frozen-main-selections.json",choices)
        with self.assertRaisesRegex(ValueError,"chosen configurations"):
            self.audit()

    def test_unavailable_method_must_retain_actual_exclusions(self):
        path = self.root / "common-n100000-k8-seed1701" / "main-results.json"
        change_json(path,lambda rows: rows[2].update(exclusion_reasons=["unknown"]))
        with self.assertRaisesRegex(ValueError,"explicit actual tuning exclusions"):
            self.audit()

    def test_main_metrics_cannot_claim_false_quality(self):
        change_json(self.first / "main-results.json",lambda rows: rows[0].update(recall_at_1=1))
        with self.assertRaisesRegex(ValueError,"quality or timing"):
            self.audit()

    def test_resource_limits_are_checked(self):
        change_json(self.first / "resources.json",lambda value: value.update(peak_process_bytes=48000000001))
        with self.assertRaisesRegex(ValueError,"resources"):
            self.audit()

    def test_frechet_id_differences_do_not_count_as_nonoptimal_when_tied(self):
        report = self.audit()
        for row in report["frechet_reference_parents"]:
            self.assertEqual(row["nonoptimal_fraction"],.5)
            self.assertEqual(row["nonoptimal_fraction_seed_minmax"],[.5,.5])
        self.assertEqual(report["main"][0]["seeds"][0]["recall_at_1"],.5)
        change_json(self.first / "dataset.json",lambda value: value.update(frechet_parent_not_dtw_optimal=1))
        with self.assertRaisesRegex(ValueError,"Frechet-parent quality"):
            self.audit()

    def test_query_arrays_and_cross_seed_data_hashes_cannot_change(self):
        path = self.first / "ground-truth.npz"
        with np.load(path,allow_pickle=False) as original:
            values = {name: original[name].copy() for name in original.files}
        values["queries"][2,0,0,0] += 1
        np.savez(path,**values)
        with self.assertRaisesRegex(ValueError,"query identity or data hash"):
            self.audit()

    def test_original_synthetic_and_nba_provenance_are_checked(self):
        path = self.base / "synthetic" / "conjunctive-n100000-k4-seed1701" / "ground-truth.npz"
        with path.open("ab") as handle:
            handle.write(b"changed")
        with self.assertRaisesRegex(ValueError,"Frechet reference checksum"):
            self.audit()

    def test_nba_database_checksum_is_checked(self):
        path = self.base / "nba" / "database.npz"
        with path.open("ab") as handle:
            handle.write(b"changed")
        with self.assertRaisesRegex(ValueError,"NBA data artifact checksum"):
            self.audit()

    def test_explicit_nba_relocation_keeps_frozen_config_and_all_checksum_checks(self):
        manifest = read_json(self.root / "manifest.json")
        relocated = self.base / "relocated-nba"
        (self.base / "nba").rename(relocated)
        with self.assertRaisesRegex(ValueError,"Missing required source artifact"):
            self.audit()
        report = publisher.audit(self.root,nba_data=relocated)
        self.assertEqual(report["manifest"],manifest)
        self.assertEqual(report["input_relocations"],
                         {"nba_data": dict(declared=str(self.base / "nba"),audited=str(relocated))})
        self.assertIn(str(relocated / "database.npz"),report["source_artifacts"])
        with (relocated / "database.npz").open("ab") as handle:
            handle.write(b"changed")
        with self.assertRaisesRegex(ValueError,"NBA data artifact checksum"):
            publisher.audit(self.root,nba_data=relocated)

    def test_job_frechet_reference_must_equal_original_cached_control(self):
        path = self.first / "frechet-reference.npz"
        with np.load(path,allow_pickle=False) as original:
            values = {name: original[name].copy() for name in original.files}
        values["ids"][2] += 1
        np.savez(path,**values)
        with self.assertRaisesRegex(ValueError,"frozen Frechet ids"):
            self.audit()

    def test_dataset_hash_on_a_later_seed_must_match_frozen_data_control(self):
        path = self.root / "common-n100000-k4-seed1702" / "dataset.json"
        change_json(path,lambda value: value.update(dataset_sha256="f"*64))
        with self.assertRaisesRegex(ValueError,"data hashes"):
            self.audit()

    def test_nba_reference_checksum_is_checked(self):
        path = self.base / "nba" / "test-ground-truth.npz"
        with path.open("ab") as handle:
            handle.write(b"changed")
        with self.assertRaisesRegex(ValueError,"NBA Frechet reference checksum"):
            self.audit()

    def test_root_configuration_checksum_is_checked(self):
        change_json(self.root / "config.json",lambda value: value.update(target_recall=.5))
        with self.assertRaisesRegex(ValueError,"configuration checksum"):
            self.audit()

    def test_nba_and_synthetic_delta_grids_cannot_be_interchanged(self):
        config = read_json(self.root / "config.json")
        self.assertEqual(config["nba_deltas"],[8.,16.,32.,64.])
        self.assertEqual(config["deltas"],[64.,128.,256.])
        for folder,wrong_delta in ((self.first,8.),
                                   (self.root / "nba-n23805-k5-seed2301",128.)):
            path = folder / "tuning-configurations.json"
            original = read_json(path)
            changed = copy.deepcopy(original)
            changed[0]["delta"] = wrong_delta
            write_json(path,changed)
            with self.subTest(case=folder.name),self.assertRaisesRegex(ValueError,"frozen budget or grid"):
                self.audit()
            write_json(path,original)

    def test_partial_seed_availability_is_explicit_without_partial_averaging(self):
        report = self.audit()
        rows = copy.deepcopy(report["main"][0]["seeds"])
        rows[2] = {key: rows[2][key] for key in
                   ("case_id","kind","n","k","display_method","index_seed")}
        rows[2].update(test_status="no feasible DTW tuning configuration",exclusion_reasons=["tuning recall below target"])
        combined = publisher.combine_seeds(rows,[1701,1702,1703],.9)
        self.assertEqual(combined["available_seed_count"],2)
        self.assertNotIn("median_ms",combined)
        self.assertNotIn("recall_at_1",combined)
        self.assertEqual(len(combined["seeds"]),3)

    def test_print_safe_figure_has_two_panels_four_methods_and_store_all_gaps(self):
        report = self.audit()
        original = copy.deepcopy(report)

        def inspect(fig, path, **kwargs):
            self.assertEqual(len(fig.axes),2)
            self.assertEqual(fig.axes[0].get_ylabel(),"Time (ms)")
            self.assertEqual(fig.axes[1].get_ylabel(),"Index (MB)")
            self.assertEqual(fig.get_size_inches().tolist(),[3.35,3.15])
            fig.canvas.draw()
            top,bottom=(ax.get_position() for ax in fig.axes)
            self.assertGreater(top.y0,bottom.y1)
            self.assertAlmostEqual(top.x0,bottom.x0)
            boxes=[ax.get_tightbbox(fig.canvas.get_renderer()) for ax in fig.axes]
            self.assertFalse(boxes[0].overlaps(boxes[1]))
            legend=fig.legends[0].get_window_extent(fig.canvas.get_renderer())
            self.assertGreaterEqual(legend.y0,boxes[0].y1)
            self.assertEqual(Path(path).name,"compact-dtw.pdf")
            for ax in fig.axes:
                self.assertEqual(len(ax.lines),4)
                self.assertEqual(list(ax.get_xticks()),list(range(5)))
                self.assertEqual(ax.get_yscale(),"log")
                store = next(line for line in ax.lines if line.get_label() == "Store-all")
                self.assertTrue(np.isnan(store.get_ydata()[1]))
                self.assertTrue(np.isnan(store.get_ydata()[3]))
                self.assertEqual(store.get_linestyle(),"-.")
                self.assertEqual(store.get_marker(),"s")
                self.assertEqual(store.get_linewidth(),.7)
        with patch("fresh.nba_paper_examples.configure_paper_font"), \
                patch("fresh.nba_paper_examples.save_paper_pdf",side_effect=inspect):
            publisher.draw_figure(report,self.base / "compact-dtw.pdf")
        self.assertEqual(report,original)

    def test_abbreviated_table_headers_keep_seed_aggregation_in_prose_without_retuning(self):
        report = self.audit()
        original = copy.deepcopy(report)
        nba = next(row for row in report["main"]
                   if row["case_id"] == "nba-n23805-k5" and row["display_method"] == "blocking")
        self.assertLess(nba["recall_at_1"],report["config"]["tuning_acceptance_recall"])
        self.assertEqual(nba["test_status"],"complete")
        for table in (publisher.compact_table(report),publisher.appendix_table(report)):
            self.assertIn(r"\shortstack{Recall\\(\%)}",table)
            self.assertIn(r"\shortstack{$1.1\times$\\(\%)}",table)
            self.assertIn(r"\shortstack{Time\\(ms)}",table)
            self.assertIn(r"\shortstack{Index\\(MB)}",table)
            self.assertNotIn("Mean",table.splitlines()[2])
            self.assertNotIn("Median",table.splitlines()[2])
        compact = publisher.compact_table(report)
        self.assertIn(r"\shortstack{Ref. nonopt.\\(\%)}",compact)
        appendix = publisher.appendix_table(report)
        self.assertIn(r"\shortstack{Empty\\(\%)}",appendix)
        self.assertIn(r"\shortstack{p95\\(ms)}",appendix)
        self.assertIn(r"& r95 & r99\\",appendix)
        text = publisher.results_text(report)
        self.assertIn("Quality averages the three seed statistics",text)
        self.assertIn("three per-seed query medians",text)
        self.assertEqual(report,original)

    def test_appendix_is_only_tabular_with_all_rows_and_json_exclusions_preserved(self):
        report = self.audit()
        original = copy.deepcopy(report)
        appendix = publisher.appendix_table(report)
        self.assertTrue(appendix.startswith(r"\begin{tabular}"))
        self.assertTrue(appendix.endswith("\\end{tabular}\n"))
        self.assertEqual(appendix.count(r"\begin{tabular}"),1)
        self.assertEqual(appendix.count(r"\end{tabular}"),1)
        rows = appendix.split(r"\midrule",1)[1].split(r"\bottomrule",1)[0].strip().splitlines()
        self.assertEqual(len(rows),20)
        self.assertEqual(appendix.count(r"\multicolumn{8}{c}{unavailable}"),2)
        self.assertNotIn("see exclusions",appendix)
        self.assertNotIn("Quality averages",appendix)
        for group in report["main"]:
            self.assertNotIn(group["case_id"],appendix)
            if group["test_status"] != "complete":
                self.assertEqual(len(group["seeds"]),3)
                self.assertTrue(all(seed["exclusion_reasons"] == ["not run: peak RAM estimate"]
                                    for seed in group["seeds"]))
        self.assertEqual(report,original)

    def test_cli_exports_outputs_receipt_and_never_modifies_source(self):
        before = {path: publisher.digest(path) for path in self.root.rglob("*") if path.is_file()}
        output = self.base / "published"

        def write_pdf(report, destination):
            destination.write_bytes(b"%PDF-1.4\nfixture\n")

        with patch.object(publisher,"draw_figure",side_effect=write_pdf):
            publisher.main(["--source",str(self.root),"--output",str(output)])
        names = {"dtw-summary.json","dtw-publication-receipt.json","compact-dtw-table.tex",
                 "dtw-appendix-table.tex","dtw-results.tex","compact-dtw.pdf"}
        self.assertEqual({path.name for path in output.iterdir()},names)
        receipt = read_json(output / "dtw-publication-receipt.json")
        self.assertTrue(receipt["all_15_jobs_complete"])
        self.assertTrue(receipt["dtw_frechet_optimum_consistency_checked"])
        self.assertFalse(receipt["dtw_kernel_imported"])
        self.assertEqual(receipt["measurement_commit"],read_json(self.root / "manifest.json")["commit"])
        self.assertEqual(receipt["publisher_sha256"],publisher.digest(Path(publisher.__file__)))
        source_hashes = receipt["source_artifacts"]
        expected_source_hashes = {str(path.relative_to(self.root)):value for path,value in before.items()}
        actual_source_hashes = {name:value for name,value in source_hashes.items() if not Path(name).is_absolute()}
        self.assertEqual(actual_source_hashes,expected_source_hashes)
        for name in ("config.json","manifest.json","cases.json","completion.json"):
            self.assertIn(name,source_hashes)
        for case in read_json(self.root / "config.json")["cases"]:
            for suffix in (".json",".npz"):
                self.assertIn(f"ground-truth-cache/{case['id']}{suffix}",source_hashes)
        for case in read_json(self.root / "cases.json"):
            for name in ("dataset.json","ground-truth.npz","frechet-reference.npz",
                         "frozen-main-selections.json","tuning-configurations.json","tuning-queries.csv",
                         "test-queries.csv","main-results.json","resources.json"):
                self.assertIn(f"{case['case']}/{name}",source_hashes)
        for name, value in receipt["outputs"].items():
            self.assertEqual(value,publisher.digest(output / name))
        self.assertEqual(before,{path: publisher.digest(path) for path in self.root.rglob("*") if path.is_file()})
        table = (output / "compact-dtw-table.tex").read_text()
        self.assertIn("58.3 & 83.3 & 5.00 & 20.00 & 50.0",table)
        appendix = (output / "dtw-appendix-table.tex").read_text()
        self.assertEqual(appendix.count(" & Store-all & "),5)
        self.assertIn("unavailable",appendix)
        self.assertNotIn("not run: peak RAM estimate",appendix)
        self.assertNotIn("Rarest",appendix)
        self.assertIn("no Fr", (output / "dtw-results.tex").read_text())

    def test_invalid_audit_creates_no_publication_outputs(self):
        self.change_test(lambda rows: rows.pop())
        output = self.base / "not-created"
        with self.assertRaises(ValueError):
            publisher.publish(self.root,output)
        self.assertFalse(output.exists())
        with self.assertRaisesRegex(ValueError,"read-only"):
            publisher.publish(self.root,self.root)

    def test_output_inside_results_subfolder_preserves_source_and_excludes_own_exports(self):
        before = {str(path.relative_to(self.root)):publisher.digest(path)
                  for path in self.root.rglob("*") if path.is_file()}
        output = self.root / "paper"

        def write_pdf(report, destination):
            destination.write_bytes(b"%PDF-1.4\nfixture\n")

        with patch.object(publisher,"draw_figure",side_effect=write_pdf):
            for _ in range(2):
                publisher.main(["--source",str(self.root),"--output",str(output)])
                receipt = read_json(output / "dtw-publication-receipt.json")
                actual = {name:value for name,value in receipt["source_artifacts"].items()
                          if not Path(name).is_absolute()}
                self.assertEqual(actual,before)
        self.assertEqual(before,{name:publisher.digest(self.root / name) for name in before})
        self.assertEqual(len(list(output.iterdir())),6)

    def test_audit_receipt_detects_source_changing_during_read(self):
        sources = publisher.Sources(self.root)
        path = self.root / "completion.json"
        sources.json(path)
        change_json(path,lambda value: value.update(elapsed_seconds=101))
        with self.assertRaisesRegex(ValueError,"changed during audit"):
            sources.receipt()


class DTWQueryMetricTests(unittest.TestCase):
    def test_nn_distance_identity_is_exact_even_for_float32_cached_values(self):
        config = dict(c_success_factors=[1.,1.1],query_deadline_seconds=60)
        optimum = np.asarray([1.2],dtype=np.float32)
        row = query_row(0,float(optimum[0]),float(optimum[0]),.001)
        row = {key: str(value) for key,value in row.items()}
        row["nn_distance"] = "1.2"
        with self.assertRaisesRegex(ValueError,"exact cached optimum"):
            publisher.query_metrics([row],optimum,config,10)

    def test_zero_optimum_infinite_ratios_and_empty_results_are_json_safe(self):
        config = dict(c_success_factors=[1.,1.1],query_deadline_seconds=60)
        rows = [query_row(0,0.,0.,.001),query_row(1,0.,publisher.EPSILON/2,.002),
                query_row(2,1.,None,.003)]
        string_rows = [{key: "" if value is None else str(value) for key,value in row.items()} for row in rows]
        metrics = publisher.query_metrics(string_rows,np.asarray([0.,0.,1.]),config,10)
        self.assertEqual(metrics["recall_at_1"],2/3)
        self.assertEqual(metrics["success_at_1_1"],2/3)
        self.assertEqual(metrics["empty_rate"],1/3)
        self.assertEqual(metrics["returned_ratio_count"],2)
        self.assertEqual(metrics["unbounded_ratio_count"],1)
        self.assertEqual(metrics["p95_ratio"],"infinity")
        self.assertEqual(metrics["mean_ratio"],"infinity")
        json.dumps(metrics,allow_nan=False)

    def test_all_empty_results_have_undefined_not_zero_conditional_tails(self):
        config = dict(c_success_factors=[1.,1.1],query_deadline_seconds=60)
        row = query_row(0,1.,None,.001)
        row = {key: "" if value is None else str(value) for key,value in row.items()}
        metrics = publisher.query_metrics([row],np.ones(1),config,10)
        self.assertEqual(metrics["recall_at_1"],0)
        self.assertEqual(metrics["empty_rate"],1)
        self.assertIsNone(metrics["p95_ratio"])
        self.assertIsNone(metrics["mean_ratio"])

    def test_invalid_booleans_parents_times_and_counters_are_errors(self):
        config = dict(c_success_factors=[1.,1.1],query_deadline_seconds=60)
        good = {key: str(value) for key,value in query_row(0,1.,1.,.001).items()}
        for field,bad in (("found","false"),("id","10"),("seconds","nan"),
                          ("candidates","-1"),("verified","11"),("postings","1.5"),("seconds","0")):
            row = dict(good)
            row[field] = bad
            with self.subTest(field=field),self.assertRaises(ValueError):
                publisher.query_metrics([row],np.ones(1),config,10)


if __name__ == "__main__":
    unittest.main()
