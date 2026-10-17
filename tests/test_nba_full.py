import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import urllib.error
import uuid

import numpy as np

from fresh import nba_full as full
from fresh.nba_full_figures import illustrative_queries


PROJECT = Path(__file__).resolve().parents[1]


def fixture():
    config = json.loads((PROJECT / "configs/nba-full-20261012.json").read_text())
    players = [dict(playerid=i, firstname=f"Player{i}", lastname="Example", jersey=str(i))
               for i in range(1, 6)]
    event = dict(eventId="999", home=dict(teamid=2, name="Home", abbreviation="H", players=[]),
                 visitor=dict(teamid=1, name="Visitor", abbreviation="V", players=players), moments=[])
    for frame in range(176):
        t = max(0, frame - 100) / 75
        ball = [13 + (5.25 - 13) * t, 12 + 13 * t, 7 + 6 * np.sin(np.pi * t)]
        objects = [[-1, -1, *ball]]
        objects.extend([1, i, 10 + i + frame * .02, 5 + i * 7, 0] for i in range(1, 6))
        event["moments"].append([1, 1_000_000 + 40 * frame, 100 - frame * .04, 10, None, objects])
    shot = dict(EVENTNUM="25", PERIOD="1", PCTIMESTRING="1:36", PLAYER1_TEAM_ID="1.0",
                PLAYER1_ID="1", VISITORDESCRIPTION="Example shot", EVENTMSGTYPE="1")
    fixed = dict(GAME_ID="1", GAME_EVENT_ID="25", SHOT_TIME="96", QUARTER="1",
                 SHOT_DISTANCE=str(np.linalg.norm(np.array([13, 12]) - np.array([5.25, 25]))),
                 ACTION_TYPE="Jump Shot")
    return config, event, shot, fixed


class FullNBATests(unittest.TestCase):
    def setUp(self):
        # Project-owned scratch only: neither /tmp nor the existing pilot is changed.
        self.root = PROJECT / (".nba-full-test-" + uuid.uuid4().hex)
        self.root.mkdir()
        for name in ("raw", "events", "shots", "games", "extracted", "source"):
            (self.root / name).mkdir()
        self.config, self.event, self.shot, self.fixed = fixture()
        self.config.update(network_attempts=2, minimum_free_disk_bytes=0)

    def tearDown(self):
        shutil.rmtree(self.root)

    def csv(self, path, rows):
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def inputs(self, events=None):
        raw = self.root / "extracted" / "0000000001.json"
        raw.write_text(json.dumps(dict(gameid="0000000001", events=events or [self.event])))
        events = self.root / "events" / "0000000001.csv"
        fixed = self.root / "shots" / "shots_fixed.csv"
        self.csv(events, [self.shot])
        self.csv(fixed, [self.fixed])
        return raw, events, fixed

    def test_split_frozen_by_game_hash_no_outcome_dependence(self):
        assignments = [full.game_split(str(i).zfill(10), self.config) for i in range(2000)]
        self.assertEqual(assignments, [full.game_split(str(i).zfill(10), self.config) for i in range(2000)])
        counts = {name: assignments.count(name) for name in ("database", "tune", "test")}
        self.assertTrue(1500 < counts["database"] < 1700)
        self.assertTrue(150 < counts["test"] < 250)
        with self.assertRaises(ValueError):
            full.game_split("../pilot", self.config)

    def test_balanced_selection_deterministic_across_input_order(self):
        metadata = [dict(game_id=game, event_id=e) for game, count in
                    (("a", 10), ("b", 4), ("c", 2)) for e in range(count)]
        indices = full.balanced_selection(metadata, 6, "frozen")
        games = [metadata[i]["game_id"] for i in indices]
        self.assertEqual({g: games.count(g) for g in set(games)}, {"a": 2, "b": 2, "c": 2})
        reversed_metadata = list(reversed(metadata))
        actual = [metadata[i] for i in indices]
        reverse = [reversed_metadata[i] for i in full.balanced_selection(reversed_metadata, 6, "frozen")]
        self.assertEqual(actual, reverse)

    def test_downloader_checks_blob_cache_and_logs_missing_http(self):
        data = b"public fixture"
        expected = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        path = self.root / "raw" / "file"
        with patch("urllib.request.urlopen", return_value=io.BytesIO(data)) as request:
            result = full.download("https://public.invalid/file", path, self.config, self.root, expected, 100)
            self.assertFalse(result["cached"])
            result = full.download("https://public.invalid/file", path, self.config, self.root, expected, 100)
            self.assertTrue(result["cached"])
            self.assertEqual(request.call_count, 1)
        path.write_bytes(b"corrupt cache")
        with self.assertRaisesRegex(RuntimeError, "blob mismatch"):
            full.download("https://public.invalid/file", path, self.config, self.root, expected, 100)
        missing = urllib.error.HTTPError("https://public.invalid/missing", 404, "Not Found", {}, None)
        with patch("urllib.request.urlopen", side_effect=missing):
            with self.assertRaises(full.SourceError) as failure:
                full.download("https://public.invalid/missing", self.root / "raw" / "missing",
                              self.config, self.root, maximum=100)
        self.assertEqual(failure.exception.http_code, 404)
        log = (self.root / "progress.jsonl").read_text()
        self.assertIn('"http_code": 404', log)

    def test_network_retry_and_size_guard(self):
        failure = urllib.error.URLError("network unavailable")
        with patch("urllib.request.urlopen", side_effect=[failure, io.BytesIO(b"good")]), patch("time.sleep"):
            result = full.download("https://public.invalid/file", self.root / "raw" / "file",
                                   self.config, self.root, maximum=100)
        self.assertEqual(result["bytes"], 4)
        with patch("urllib.request.urlopen", return_value=io.BytesIO(b"too large")):
            with self.assertRaises(full.SourceError) as error:
                full.download("https://public.invalid/file", self.root / "raw" / "large",
                              self.config, self.root, maximum=2)
        self.assertEqual(error.exception.status, "oversized_source")
        self.assertFalse((self.root / "raw" / "large").exists())

    def test_unexpected_permission_error_is_not_treated_as_missing_source(self):
        with patch("urllib.request.urlopen", side_effect=PermissionError("permission")):
            with self.assertRaises(PermissionError):
                full.download("https://public.invalid/file", self.root / "raw" / "file",
                              self.config, self.root, maximum=100)

    def test_streaming_reuses_physical_filters_and_stitches_event_provenance(self):
        raw, events, fixed = self.inputs([self.event, self.event])
        clips, audit = full.validated_clips(raw, events, fixed, self.config, "0000000001", "example.7z")
        self.assertEqual(len(clips), 1)
        self.assertEqual(clips[0][0].shape, (5, 32, 2))
        self.assertEqual(clips[0][2]["tracking_event_ids"], [999])
        self.assertEqual(clips[0][2]["tracking_event_array_indices"], [0, 1])
        self.assertEqual(audit["unique_frames"], 176)
        self.assertEqual(audit["duplicate_candidate_frames"], 176)
        self.assertEqual(audit["conflicting_timestamps"], 0)
        self.fixed["SHOT_DISTANCE"] = "100"
        self.csv(fixed, [self.fixed])
        from fresh.nba import corrected_shots
        corrected_shots.cache_clear()
        clips, audit = full.validated_clips(raw, events, fixed, self.config, "0000000001", "example.7z")
        self.assertEqual(clips, [])
        self.assertEqual(audit["excluded"]["corrected shot anchor disagrees with nominal shot distance"], 1)

    def test_conflicting_outside_window_copy_excluded_and_corruption_logged(self):
        other = json.loads(json.dumps(self.event))
        other["moments"] = [other["moments"][50]]
        other["moments"][0][2] = 700
        raw, events, fixed = self.inputs([self.event, other])
        clips, audit = full.validated_clips(raw, events, fixed, self.config, "0000000001", "example.7z")
        self.assertEqual(audit["conflicting_timestamps"], 1)
        self.assertEqual(len(clips), 1)
        raw.write_text('{"invalid":')
        with self.assertRaises(full.SourceError) as error:
            full.parse_source(raw, events, fixed, self.config, "0000000001", "example.7z")
        self.assertEqual(error.exception.status, "corrupt_tracking")

    def test_archive_member_rejects_path_traversal_multiple_members_and_size(self):
        def result(text):
            return subprocess.CompletedProcess([], 0, text, "")
        header = "Path = archive.7z\n----------\n"
        for body in ("Path = ../0000000001.json\nSize = 1\n",
                     "Path = 0000000001.json\nSize = 9999999999\n",
                     "Path = 0000000001.json\nSize = 1\n\nPath = 0000000002.json\nSize = 1\n"):
            with patch("subprocess.run", return_value=result(header + body)):
                with self.assertRaises(full.SourceError):
                    full.archive_member(self.root / "raw" / "archive.7z", self.config)
        with patch("subprocess.run", return_value=result(header)):
            with self.assertRaises(full.SourceError) as error:
                full.archive_member(self.root / "raw" / "archive.7z", self.config)
        self.assertEqual(error.exception.status, "empty_archive")

    def test_real_extraction_owns_only_new_json_and_audits_resume(self):
        raw, events, fixed = self.inputs()
        archive = self.root / "raw" / "01.01.2016.AAA.at.BBB.7z"
        subprocess.run(["7z", "a", "-bd", "-y", str(archive), str(raw)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
        game, size = full.archive_member(archive, self.config)
        original_hash = full.digest(raw)
        full.remove_resolved_json(raw, self.root)
        extracted = full.extract_owned(archive, game, size, self.root, self.config)
        self.assertEqual(full.digest(extracted), original_hash)
        self.assertEqual(full.extract_owned(archive, game, size, self.root, self.config), extracted)
        with self.assertRaises(RuntimeError):
            full.remove_resolved_json(events, self.root)
        self.assertTrue(events.exists())
        full.remove_resolved_json(extracted, self.root)
        self.assertFalse(extracted.exists())
        self.assertTrue(archive.exists())

    def test_preparation_duplicate_game_and_resume_artifact_audit(self):
        raw, events, fixed = self.inputs()
        archive_name = "01.01.2016.AAA.at.BBB.7z"
        second_name = "01.02.2016.AAA.at.BBB.7z"
        archive = self.root / "raw" / archive_name
        subprocess.run(["7z", "a", "-bd", "-y", str(archive), str(raw)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
        shutil.copyfile(archive, self.root / "raw" / second_name)
        sources = dict(archives=[dict(path="data/" + name, url="https://public.invalid/" + name,
                                     github_blob_sha=full.blob_sha(archive), file_sha256=None)
                                 for name in (archive_name, second_name)],
                       corrected_shots=dict(url="https://public.invalid/shots", github_blob_sha=full.blob_sha(fixed)),
                       events={"0000000001": dict(url="https://public.invalid/events", github_blob_sha=full.blob_sha(events))})
        resolved = full.prepare_games(self.config, self.root, sources)
        self.assertEqual(resolved[archive_name]["status"], "usable")
        self.assertEqual(resolved[second_name]["status"], "duplicate_game_id")
        self.assertFalse(raw.exists())
        self.assertEqual(full.prepare_games(self.config, self.root, sources), resolved)
        path = self.root / resolved[archive_name]["curves_path"]
        path.write_bytes(b"corruption")
        with self.assertRaisesRegex(RuntimeError, "artifact failed audit"):
            full.prepare_games(self.config, self.root, sources)

    def test_storage_failure_stops_not_silently_resolves(self):
        with patch("shutil.disk_usage", return_value=type("Disk", (), {"free": 0})()):
            config = dict(self.config, minimum_free_disk_bytes=1)
            with self.assertRaises(RuntimeError):
                full.disk_guard(self.root, config)

    def test_metrics_include_empty_and_absolute_feet(self):
        rows = [dict(id=0, distance=2.0), dict(id=-1, distance=np.inf), dict(id=1, distance=4.0)]
        summary = full.quality_summary(rows, [2.0, 3.0, 2.0], [1.1, 2.0])
        self.assertAlmostEqual(summary["recall_at_1"], 1 / 3)
        self.assertAlmostEqual(summary["empty_rate"], 1 / 3)
        self.assertAlmostEqual(summary["c_success"]["2.0"], 2 / 3)
        self.assertEqual(summary["mean_returned_feet"], 3.0)
        self.assertEqual(summary["mean_excess_feet"], 1.0)
        self.assertEqual(summary["mean_ratio"], 1.5)
        self.assertEqual(summary["missing_ratio_count"], 1)
        self.assertIsNone(full.safe_row(dict(distance=np.inf))["distance"])

    def test_resource_profile_and_overrides_never_modify_preparation(self):
        original = json.loads(json.dumps(self.config))
        effective = full.benchmark_config(self.config)
        self.assertEqual(effective["index_budget_bytes"], 24_000_000_000)
        self.assertEqual(effective["peak_ram_cap_bytes"], 48_000_000_000)
        self.assertEqual(self.config, original)
        effective = full.benchmark_config(self.config, index_budget_bytes=123,
                                          peak_ram_cap_bytes=456, build_deadline_seconds=7)
        self.assertEqual(effective["index_budget_bytes"], 123)
        self.assertEqual(effective["peak_ram_cap_bytes"], 456)
        self.assertEqual(effective["build_deadline_seconds"], 7)
        self.assertEqual(self.config, original)
        with self.assertRaises(ValueError):
            full.benchmark_config(self.config, build_deadline_seconds=0)

    def test_only_blocking_uses_weighted_selection_baselines_choose_fastest(self):
        for method in full.METHOD_LABELS:
            rows = [dict(method=method, status="qualified", config_id="fast",
                         index_bytes=200_000_000, tuning_median_seconds=.001),
                    dict(method=method, status="qualified", config_id="small",
                         index_bytes=1_000_000, tuning_median_seconds=.002),
                    dict(method=method, status="overbudget: query deadline", config_id="invalid",
                         index_bytes=1, tuning_median_seconds=.00001)]
            selected = full.choose_main(rows, method, self.config)
            self.assertEqual(selected["config_id"], "small" if method == "blocking" else "fast")

    def test_any_timeout_or_over_deadline_suppresses_whole_config_summaries(self):
        valid = dict(id=0, distance=1., seconds=.001, timeout=False)
        for invalid in (dict(valid, timeout=True), dict(valid, seconds=60.001)):
            rows = [valid] * 99 + [invalid]
            result = full.query_summary(rows, np.ones(100), self.config)
            self.assertTrue(result["status"].startswith("overbudget:"))
            self.assertIsNone(result["quality"])
            self.assertIsNone(result["median_top1_seconds"])
            self.assertIsNone(result["p90_top1_seconds"])
            self.assertIsNone(result["p95_top1_seconds"])
            self.assertEqual(result["deadline_failed_queries"], [99])
        result = full.query_summary([valid], [1., 1.], self.config)
        self.assertEqual(result["not_run_queries"], [1])
        self.assertIsNone(result["quality"])
        result = full.query_summary([valid, dict(valid, seconds=.003)], [1., 1.], self.config)
        self.assertEqual(result["status"], "complete")
        self.assertAlmostEqual(result["p95_top1_seconds"], .0029)

    def test_ratios_distinguish_empty_unbounded_and_zero_optimal(self):
        missing = full.query_record(dict(id=-1, distance=np.inf, seconds=.001, timeout=False), 0, 0.)
        exact_zero = full.query_record(dict(id=0, distance=0., seconds=.001, timeout=False), 1, 0.)
        unbounded = full.query_record(dict(id=0, distance=1., seconds=.001, timeout=False), 2, 0.)
        self.assertEqual(missing["ratio_status"], "missing")
        self.assertIsNone(missing["ratio"])
        self.assertEqual(exact_zero["ratio"], 1.)
        self.assertEqual(unbounded["ratio_status"], "unbounded")
        self.assertEqual(unbounded["ratio"], "infinity")
        summary = full.quality_summary([dict(id=-1, distance=np.inf), dict(id=0, distance=1.)],
                                       [0., 0.], self.config["c_factors"])
        self.assertEqual(summary["missing_ratio_count"], 1)
        self.assertEqual(summary["unbounded_ratio_count"], 1)
        self.assertEqual(summary["mean_ratio"], "infinity")
        self.assertEqual(summary["p90_ratio"], "infinity")

    def test_measured_build_deadline_actual_storage_and_all_method_preflight(self):
        import resource
        config = full.benchmark_config(self.config)
        fake = SimpleNamespace(index_bytes=100, scratch_bytes=20, entries=1, raw_entries=2,
                               build_seconds=999.)
        index, metrics = full.measured_build(lambda: fake, config)
        self.assertIs(index, fake)
        self.assertGreater(metrics["build_seconds"], 0.)
        self.assertLess(metrics["build_seconds"], fake.build_seconds)
        self.assertGreater(metrics["build_sampled_peak_rss_bytes"], 0)
        with self.assertRaises(full.BenchmarkLimit) as failure:
            full.measured_build(lambda: fake, dict(config, index_budget_bytes=99))
        self.assertIn("actual index storage", failure.exception.status)
        original_limit = resource.getrlimit(resource.RLIMIT_AS)
        with self.assertRaises(full.BenchmarkLimit) as failure:
            full.measured_build(lambda: time.sleep(.03), dict(config, build_deadline_seconds=.005))
        self.assertIn("build deadline", failure.exception.status)
        self.assertGreater(failure.exception.metrics["build_seconds"], 0)
        self.assertEqual(resource.getrlimit(resource.RLIMIT_AS), original_limit)
        for method, stored in (("blocking", 12), ("probe_all", 1), ("store_all", 120),
                               ("one_block", 5), ("rarest", 5)):
            row = dict(method=method, stored=stored, L=2)
            with patch.object(full, "current_rss", return_value=100):
                with self.assertRaises(full.BenchmarkLimit) as failure:
                    full.build_preflight(row, np.zeros((3, 5, 32, 2)), dict(config, peak_ram_cap_bytes=100))
            self.assertIn("preallocation", failure.exception.status)

    def test_full_database_assembly_and_exact_ground_truth_resume(self):
        resolved = {}
        for split, count in (("database", 2), ("tune", 3), ("test", 1)):
            game = next(str(i).zfill(10) for i in range(1, 1000)
                        if full.game_split(str(i).zfill(10), self.config) == split)
            curves = np.zeros((count, 5, 32, 2))
            for i in range(5):
                curves[:, i, :, 0] = i * 10
            if split != "database":
                curves += .25
            prefix = self.root / "games" / game
            full.atomic_npz(prefix.with_suffix(".npz"), curves=curves, balls=np.zeros((count, 32, 2)))
            full.atomic_json(prefix.with_suffix(".json"),
                             [dict(game_id=game, event_id=e) for e in range(count)])
            resolved[game] = dict(archive=game + ".7z", game_id=game, status="usable", split=split,
                                 usable_clips=count, curves_path=f"games/{game}.npz",
                                 metadata_path=f"games/{game}.json")
        full.atomic_json(self.root / "source-manifest.json", {})
        full.atomic_json(self.root / "archive-status.json", resolved)
        config = dict(self.config, tuning_query_cap=2, test_query_cap=1, ground_truth_batch_size=1)
        info = full.assemble(config, self.root, resolved)
        self.assertEqual(info["counts"]["database"]["selected_clips"], 2)
        self.assertEqual(info["counts"]["tune"]["all_valid_clips"], 3)
        self.assertEqual(info["counts"]["tune"]["selected_clips"], 2)
        (self.root / "ground-truth").mkdir()
        full.exact_preparation(config, self.root)
        with np.load(self.root / "test-ground-truth.npz", allow_pickle=False) as arrays:
            np.testing.assert_allclose(arrays["distances"], [np.sqrt(.125)])
        full.exact_preparation(config, self.root)
        path = next((self.root / "ground-truth").glob("tune-*.npz"))
        path.write_bytes(b"tampered")
        with self.assertRaisesRegex(RuntimeError, "cache audit failed"):
            full.exact_preparation(config, self.root)

    def test_illustrative_policy_retains_empty_difficult_query(self):
        rows = [dict(id=0, distance=2., optimal_feet=2.),
                dict(id=1, distance=4., optimal_feet=2.),
                dict(id=-1, distance=None, optimal_feet=3.)]
        metadata = [dict(game_id=str(i).zfill(10), event_id=i) for i in range(3)]
        chosen = illustrative_queries(rows, metadata, self.config)
        self.assertEqual(chosen["median_quality"], 1)
        self.assertEqual(chosen["difficult"], 2)
        self.assertEqual(chosen["selected_best_optimal_distance"], 0)
        changed = [dict(r, id=-1) for r in rows]
        self.assertEqual(illustrative_queries(changed, metadata, self.config)["primary"], chosen["primary"])

    def test_local_timed_evaluation_is_guarded_before_any_files_or_work(self):
        with self.assertRaisesRegex(RuntimeError, "requires"):
            full.evaluate("nonexistent", "nonexistent", False, 0)
        with patch("socket.gethostname", return_value="local8cores16GB"):
            with self.assertRaisesRegex(RuntimeError, "hostname"):
                full.evaluate("nonexistent", "nonexistent", True, 0, "verified-newLab-host")

    def test_prepared_dataset_is_portable_and_read_only_but_hash_audited(self):
        project = self.root / "portable-project"
        (project / "csexp/configs").mkdir(parents=True)
        data = project / "csexp/data/nba-full-20261012"
        data.mkdir(parents=True)
        config_path = project / "csexp/configs/nba-full-20261012.json"
        full.atomic_json(config_path, self.config)
        full.atomic_json(data / "frozen-config.json", self.config)
        full.atomic_json(data / "ownership.json", dict(root="/original/extraction/host/path"))
        full.atomic_json(data / "source-manifest.json", {})
        full.atomic_json(data / "archive-status.json", {})
        full.atomic_npz(data / "database.npz", curves=np.zeros((1, 5, 32, 2)))
        full.atomic_npz(data / "test-ground-truth.npz", ids=np.array([0]), distances=np.array([1.]))
        info = dict(artifacts={"database.npz": full.digest(data / "database.npz")},
                    source_manifest_sha256=full.digest(data / "source-manifest.json"),
                    archive_audit_sha256=full.digest(data / "archive-status.json"))
        full.atomic_json(data / "dataset.json", info)
        full.atomic_json(data / "preparation-complete.json",
                        dict(dataset_sha256=full.digest(data / "dataset.json"),
                             ground_truth={"test": full.digest(data / "test-ground-truth.npz")}))
        loaded, actual = full.prepared_data(config_path)
        self.assertEqual(loaded, self.config)
        self.assertEqual(actual, data)
        self.assertEqual(json.loads((data / "ownership.json").read_text())["root"], "/original/extraction/host/path")
        (data / "database.npz").write_bytes(b"tampered")
        with self.assertRaisesRegex(RuntimeError, "artifact hash mismatch"):
            full.prepared_data(config_path)

    def test_all_four_main_candidate_rerankers_and_rarest_on_tiny_synthetic_fixture(self):
        from fresh.poc import Joint, PerCurve
        raw, events, fixed = self.inputs()
        clips, _ = full.validated_clips(raw, events, fixed, self.config, "0000000001", "example.7z")
        Q = clips[0][0] + .1
        X = np.stack([clips[0][0] + offset for offset in (0, 2, 4)])
        specs = [("blocking", [3, 2], False), ("probe_all", [1] * 5, False),
                 ("store_all", [1] * 5, True), ("one_block", None, False),
                 ("rarest", None, False)]
        for method, blocks, swapped in specs:
            with self.subTest(method=method):
                if blocks is None:
                    index = PerCurve(X, 64., 2, 2301)
                    answer = index.query(X, Q, self.config, method == "rarest")
                else:
                    index = Joint(X, 64., 2, 2301, blocks, swapped)
                    answer = index.query(X, Q, self.config)
                self.assertFalse(answer["timeout"])
                self.assertEqual(answer["id"], 0)
                self.assertAlmostEqual(answer["distance"], np.sqrt(.02))
                self.assertGreater(index.index_bytes, 0)
                self.assertGreater(index.scratch_bytes, 0)
                self.assertGreater(answer["candidates"], 0)
                self.assertGreater(answer["postings"], 0)
                ranked = full.reranked_candidates(index, X, Q, answer["candidates"])
                self.assertEqual(ranked[0]["database_index"], answer["id"])
                self.assertEqual(ranked[0]["distance_feet"], answer["distance"])

    def test_original_court_figure_renders_true_player_names_clocks_and_matching(self):
        from fresh.nba_full_figures import render_example
        from fresh.tuple_quality import exact_distance
        raw, events, fixed = self.inputs()
        clips, _ = full.validated_clips(raw, events, fixed, self.config, "0000000001", "example.7z")
        curves, ball, info = clips[0]
        other = dict(info, game_id="0000000002", event_id=26)
        X = np.array([curves + .2])
        distance = float(exact_distance(curves, X[0]))
        matches = [dict(database_index=0, distance_feet=distance,
                        candidate_rank=1, game_id=other["game_id"])]
        render_example(self.root / "example", curves, ball, info, matches, X,
                       np.array([ball + .2]), [other], "primary", distance)
        self.assertTrue((self.root / "example.png").exists())
        self.assertTrue((self.root / "example.pdf").exists())
        self.assertEqual(len(matches[0]["query_to_result_player_ids"]), 5)

    def controller_fixture(self):
        raw, events, fixed = self.inputs()
        clips, _ = full.validated_clips(raw, events, fixed, self.config, "0000000001", "example.7z")
        P = clips[0][0]
        config = dict(self.config, index_seeds=[2301], deltas_feet=[64.], Lmax=2,
                      blocking_families=[[3, 2]], warmup_queries=1)
        full.atomic_json(self.root / "frozen-config.json", config)
        full.atomic_json(self.root / "dataset.json", {})
        full.atomic_json(self.root / "source-manifest.json", {})
        full.atomic_json(self.root / "preparation-complete.json", {})
        full.atomic_npz(self.root / "database.npz", curves=np.stack([P, P + 2, P + 4]))
        full.atomic_npz(self.root / "tune.npz", curves=np.stack([P + .1, P + .2]))
        full.atomic_npz(self.root / "test.npz", curves=np.stack([P + .15]))
        full.atomic_npz(self.root / "tune-ground-truth.npz", ids=np.array([0, 0]),
                        distances=np.sqrt([.02, .08]))
        full.atomic_npz(self.root / "test-ground-truth.npz", ids=np.array([0]),
                        distances=np.sqrt([.045]))
        full.atomic_json(self.root / "database-metadata.json",
                        [dict(game_id=str(i).zfill(10), event_id=1) for i in range(1, 4)])
        full.atomic_json(self.root / "test-metadata.json", [dict(game_id="0000000004", event_id=1)])
        return config

    def test_controller_freezes_and_resumes_all_methods_on_synthetic_unit_fixture(self):
        # This is a three-record synthetic unit test, never a timed NBA run.
        import os
        from curveset import bench
        config = self.controller_fixture()
        output = self.root / "unit-evaluation"
        old_core = bench.PIN_CORE
        try:
            with patch.object(full, "prepared_data", return_value=(config, self.root)), \
                    patch("socket.gethostname", return_value="newLab-synthetic-unit-fixture"):
                full.evaluate("unused", output, True, min(os.sched_getaffinity(0)))
                before = full.digest(output / "seed-2301/main-results.json")
                full.evaluate("unused", output, True, min(os.sched_getaffinity(0)))
                self.assertEqual(before, full.digest(output / "seed-2301/main-results.json"))
        finally:
            bench.PIN_CORE = old_core
        results = json.loads((output / "seed-2301/main-results.json").read_text())
        self.assertEqual(set(results), set(full.METHOD_LABELS.values()))
        for result in results.values():
            self.assertEqual(result["quality"]["recall_at_1"], 1.0)
            self.assertEqual(len(result["queries"]), 1)
            self.assertIsNotNone(result["p95_top1_seconds"])
            self.assertGreater(result["final_build_metrics"]["build_seconds"], 0)
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(manifest["effective_benchmark_config_sha256"],
                         full.digest(output / "effective-benchmark-config.json"))
        self.assertEqual(manifest["benchmark_resources"]["index_budget_bytes"], 24_000_000_000)
        self.assertEqual(manifest["benchmark_resources"]["peak_ram_cap_bytes"], 48_000_000_000)
        self.assertEqual(manifest["method_labels"]["one_block"], "per_curve-intersection")
        self.assertTrue(manifest["code_git_revision"])
        self.assertIn("curveset/core.py", manifest["helpers"])
        self.assertTrue((output / "complete.json").exists())

    def test_controller_budget_gates_all_constructors_before_allocation(self):
        import os
        from curveset import bench
        config = self.controller_fixture()
        output = self.root / "unit-preallocation-gate"
        old_core = bench.PIN_CORE
        try:
            with patch.object(full, "prepared_data", return_value=(config, self.root)), \
                    patch("socket.gethostname", return_value="newLab-synthetic-unit-fixture"), \
                    patch("fresh.poc.Joint", side_effect=AssertionError("Joint must not allocate")), \
                    patch("fresh.poc.PerCurve", side_effect=AssertionError("PerCurve must not allocate")):
                full.evaluate("unused", output, True, min(os.sched_getaffinity(0)), peak_ram_cap_bytes=1)
        finally:
            bench.PIN_CORE = old_core
        rows = json.loads((output / "seed-2301/tuning-configurations.json").read_text())
        self.assertTrue(all("preallocation" in row["status"] for row in rows))
        self.assertEqual({row["method"] for row in rows}, set(full.METHOD_LABELS))

    def test_controller_test_deadline_failure_keeps_raw_rows_but_no_claims(self):
        import os
        from curveset import bench
        from fresh.poc import Joint, PerCurve
        config = self.controller_fixture()
        output = self.root / "unit-test-deadline"
        old_core = bench.PIN_CORE
        def query(original):
            def call(index, X, Q, cfg, *args, **kwargs):
                if np.isclose(Q[0, 0, 0] - X[0, 0, 0, 0], .15):
                    return dict(id=0, distance=np.sqrt(.045), seconds=60.001, timeout=False,
                                candidates=1, verified=1, probes=1, postings=1)
                return original(index, X, Q, cfg, *args, **kwargs)
            return call
        try:
            with patch.object(full, "prepared_data", return_value=(config, self.root)), \
                    patch("socket.gethostname", return_value="newLab-synthetic-unit-fixture"), \
                    patch.object(Joint, "query", new=query(Joint.query)), \
                    patch.object(PerCurve, "query", new=query(PerCurve.query)):
                full.evaluate("unused", output, True, min(os.sched_getaffinity(0)))
        finally:
            bench.PIN_CORE = old_core
        results = json.loads((output / "seed-2301/main-results.json").read_text())
        for result in results.values():
            self.assertTrue(result["status"].startswith("overbudget:"))
            self.assertIsNone(result["quality"])
            self.assertIsNone(result["median_top1_seconds"])
            self.assertIsNone(result["p95_top1_seconds"])
            self.assertEqual(result["queries"][0]["seconds"], 60.001)
            self.assertEqual(result["deadline_failed_queries"], [0])

    def test_frozen_source_manifest_has_all_urls_pinned_shas_and_split_policy(self):
        commit = dict(sha="a" * 40, commit=dict(tree=dict(sha="b" * 40)))
        tree = dict(truncated=False, tree=[
            dict(type="blob", path="data/01.01.2016.AAA.at.BBB.7z", sha="c" * 40, size=123),
            dict(type="blob", path="data/events/0000000001.csv", sha="d" * 40, size=456),
            dict(type="blob", path="data/shots/shots_fixed.csv", sha="e" * 40, size=789)])
        full.atomic_json(self.root / "frozen-config.json", self.config)
        with patch.object(full, "fetch_json", side_effect=[commit, tree]) as request:
            manifest = full.freeze_sources(self.config, self.root)
            self.assertEqual(len(manifest["archives"]), 1)
            self.assertIn("a" * 40, manifest["archives"][0]["url"])
            self.assertIsNone(manifest["archives"][0]["file_sha256"])
            self.assertEqual(manifest["split_rule"]["salt"], self.config["split_salt"])
            self.assertEqual(full.freeze_sources(self.config, self.root), manifest)
            self.assertEqual(request.call_count, 2)

    def test_small_public_sample_equals_existing_validated_pilot(self):
        pilot = PROJECT.parent / "data/nba-pilot"
        name = "10.31.2015.GSW.at.NOP.7z"
        game = "0021500035"
        required = [pilot / "raw" / name, pilot / "raw" / (game + ".json"),
                    pilot / "events" / (game + ".csv"), pilot / "shots/shots_fixed.csv"]
        if not all(path.exists() for path in required):
            self.skipTest("Read-only existing public pilot sample is not locally available")
        before = {path: (path.stat().st_size, path.stat().st_mtime_ns) for path in required}
        clips, audit = full.validated_clips(required[1], required[2], required[3], self.config, game, name)
        from fresh.nba import game_clips
        pilot_config = json.loads((PROJECT / "configs/nba-qualitative-20261012.json").read_text())
        expected, old_audit = game_clips(pilot_config, name, pilot)
        self.assertGreater(len(clips), 0)
        self.assertEqual(audit["usable_clips"], old_audit["usable_clips"])
        self.assertEqual(audit["excluded"], old_audit["excluded"])
        expected_by_event = {c[2]["event_id"]: c for c in expected}
        for curve, ball, info in clips:
            old = expected_by_event[info["event_id"]]
            np.testing.assert_array_equal(curve, old[0])
            np.testing.assert_array_equal(ball, old[1])
            self.assertEqual(info, old[2])
        self.assertEqual(before, {path: (path.stat().st_size, path.stat().st_mtime_ns) for path in required})


if __name__ == "__main__":
    unittest.main()
