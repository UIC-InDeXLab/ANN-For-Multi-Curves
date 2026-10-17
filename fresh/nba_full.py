"""Frozen public SportVU corpus preparation and separately authorized NBA evaluation.

Preparation (two low-priority threads; never times an index):
  nice -n 15 env NUMBA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
    ../.venv/bin/python -m fresh.nba_full prepare --config configs/nba-full-20261012.json

Resume uses the same command. Only owned, successfully resolved extracted JSON is
removed. Compressed archives, source hashes, per-game curves, exclusions, split,
and exact ground truth remain. The existing nba-pilot directory is never written.

The benchmark subcommand is intentionally separate and requires explicit newLab
authorization after the synthetic/selectivity queue. All selection uses tuning
games only. Its newlab profile uses 24GB index / 48GB peak RAM independently of
the frozen preparation configuration. Baselines select fastest qualified tuning
median; only blocking uses weighted cost. Any deadline failure excludes the
whole configuration from aggregate latency and quality claims; raw rows remain.
"""
import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import csv
import fcntl
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import resource
import shutil
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request

import ijson
import numpy as np

from fresh.nba import clock_value, corrected_shots, extract_clip, shot_rows


SCHEMA = 1
ARCHIVE = re.compile(r"\d{2}\.\d{2}\.\d{4}\.[A-Z]{3}\.at\.[A-Z]{3}\.7z")
GAME = re.compile(r"\d{10}")
RESOLVED = {"usable", "zero_usable", "duplicate_game_id", "empty_archive",
            "corrupt_archive", "corrupt_tracking", "unavailable_archive",
            "unavailable_events"}


class SourceError(Exception):
    """Expected, logged source failure, not an implementation exception."""
    def __init__(self, status, detail, http_code=None):
        super().__init__(detail)
        self.status, self.http_code = status, http_code


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def blob_sha(path):
    h = hashlib.sha1(f"blob {Path(path).stat().st_size}\0".encode())
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    part = path.with_name(path.name + ".part")
    with part.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    part.replace(path)


def atomic_npz(path, **arrays):
    path = Path(path)
    part = path.with_name(path.name + ".part")
    with part.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    part.replace(path)


def log(root, kind, **fields):
    row = dict(time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               kind=kind, **fields)
    with (root / "progress.jsonl").open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
    print(json.dumps(row, sort_keys=True), flush=True)


@contextmanager
def lock(root):
    with (root / ".preparation.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another preparation/evaluation owns {root}") from exc
        yield


def game_split(game_id, config):
    if not GAME.fullmatch(game_id):
        raise ValueError(f"Invalid game ID {game_id!r}")
    percentages = config["split_percentages"]
    if len(percentages) != 3 or sum(percentages) != 100:
        raise ValueError("Expected three split percentages summing to 100")
    # Integer arithmetic avoids platform-dependent floating-point split boundaries.
    number = int(hashlib.sha256(f"{config['split_salt']}:{game_id}".encode()).hexdigest(), 16)
    bucket = number * 100 // (1 << 256)
    return ("database" if bucket < percentages[0] else
            "tune" if bucket < sum(percentages[:2]) else "test")


def balanced_selection(metadata, cap, salt):
    """Round robin distinct games; selection is independent of retrieval outcomes."""
    groups = defaultdict(list)
    for i, row in enumerate(metadata):
        groups[row["game_id"]].append(i)
    key = lambda text: hashlib.sha256(f"{salt}:{text}".encode()).hexdigest()
    games = sorted(groups, key=key)
    for game in games:
        groups[game].sort(key=lambda i: key(f"{game}:{metadata[i]['event_id']}"))
    selected, depth = [], 0
    while len(selected) < cap:
        layer = [groups[game][depth] for game in games if len(groups[game]) > depth]
        if not layer:
            break
        selected.extend(layer[:cap - len(selected)])
        depth += 1
    return selected


def disk_guard(root, config, additional=0):
    free = shutil.disk_usage(root).free
    needed = config["minimum_free_disk_bytes"] + additional
    if free < needed:
        raise RuntimeError(f"Insufficient free disk: {free} bytes, need {needed}; no files deleted")


def download(url, path, config, root, expected_blob=None, maximum=None):
    """Sequential bounded download, audited caches, explicit HTTP/network retries."""
    path = Path(path)
    if path.exists():
        if not path.is_file() or (path.stat().st_size == 0 and expected_blob is None):
            raise RuntimeError(f"Invalid cached download: {path}")
        if expected_blob and blob_sha(path) != expected_blob:
            raise RuntimeError(f"Cached GitHub blob mismatch: {path}; refusing silent reuse")
        return dict(sha256=digest(path), bytes=path.stat().st_size, cached=True)
    part = path.with_name(path.name + ".part")
    for attempt in range(1, config["network_attempts"] + 1):
        disk_guard(root, config, maximum or (1 << 20))
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "ANN-public-NBA-research/1"})
            with urllib.request.urlopen(request, timeout=config["network_timeout_seconds"]) as response:
                with part.open("wb") as handle:
                    size = 0
                    while chunk := response.read(1 << 20):
                        size += len(chunk)
                        if maximum is not None and size > maximum:
                            raise SourceError("oversized_source", f"{url} exceeds {maximum} bytes")
                        handle.write(chunk)
                    handle.flush()
                    os.fsync(handle.fileno())
            if size == 0 and expected_blob is None:
                raise SourceError("empty_source", f"Zero-byte source: {url}")
            if expected_blob and blob_sha(part) != expected_blob:
                raise SourceError("checksum_mismatch", f"GitHub blob checksum mismatch: {url}")
            part.replace(path)
            return dict(sha256=digest(path), bytes=size, cached=False)
        except urllib.error.HTTPError as exc:
            retry = exc.code in (408, 429) or 500 <= exc.code <= 599
            log(root, "download_http_error", url=url, attempt=attempt, http_code=exc.code, retry=retry)
            exc.close()
            if not retry or attempt == config["network_attempts"]:
                raise SourceError("http_unavailable", f"HTTP {exc.code}: {url}", exc.code) from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException, OSError) as exc:
            # Disk and permission failures are implementation/environment failures.
            if isinstance(exc, OSError) and not isinstance(exc, (urllib.error.URLError, TimeoutError, ConnectionError)):
                raise
            log(root, "download_network_error", url=url, attempt=attempt,
                exception=type(exc).__name__, detail=str(exc))
            if attempt == config["network_attempts"]:
                raise SourceError("network_exhausted", f"{type(exc).__name__}: {url}: {exc}") from exc
        time.sleep(min(30, 2 ** attempt))
    raise AssertionError("Unreachable retry exit")


def fetch_json(url, path, config, root):
    download(url, path, config, root, maximum=32_000_000)
    return json.loads(path.read_text())


def initialize(config_path):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    root = (config_path.parent.parent / config["data_directory"]).resolve()
    if root.name != "nba-full-20261012":
        raise ValueError("This preparation must use the dedicated nba-full-20261012 directory")
    root.mkdir(parents=True, exist_ok=True)
    marker = root / "ownership.json"
    identity = dict(schema=SCHEMA, root=str(root), config_sha256=digest(config_path))
    if marker.exists():
        if json.loads(marker.read_text()) != identity:
            raise RuntimeError("Owned workdir config changed: use original frozen configuration")
    else:
        if any(root.iterdir()):
            raise RuntimeError("Refusing to claim a nonempty workdir")
        atomic_json(marker, identity)
        atomic_json(root / "frozen-config.json", config)
    for name in ("source", "raw", "events", "shots", "extracted", "games", "ground-truth"):
        (root / name).mkdir(exist_ok=True)
    return config, root


def freeze_sources(config, root):
    path = root / "source-manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text())
        if manifest["config_sha256"] != digest(root / "frozen-config.json"):
            raise RuntimeError("Frozen source/config digest mismatch")
        implementation = json.loads((root / "implementation.json").read_text())
        current = {str(Path(__file__).name): digest(__file__),
                   "nba.py": digest(Path(__file__).with_name("nba.py"))}
        if implementation["files"] != current:
            raise RuntimeError("Preparation implementation changed; refusing mixed-filter resumed caches")
        return manifest
    commit = fetch_json(config["source_api"] + "/commits/" + config["source_ref"],
                        root / "source" / "commit.json", config, root)
    sha = commit["sha"]
    tree = fetch_json(config["source_api"] + f"/git/trees/{sha}?recursive=1",
                      root / "source" / "tree.json", config, root)
    if tree.get("truncated"):
        raise RuntimeError("GitHub tree is truncated; cannot claim full accessible manifest")
    entries = {r["path"]: r for r in tree["tree"] if r["type"] == "blob"}
    base = f"https://raw.githubusercontent.com/sealneaward/nba-movement-data/{sha}/"
    def record(name):
        row = entries[name]
        return dict(path=name, url=base + name, github_blob_sha=row["sha"],
                    bytes_declared=row["size"], file_sha256=None)
    archives = [record(name) for name in sorted(entries)
                if name.startswith("data/") and name.endswith(".7z")]
    if not archives:
        raise RuntimeError("Source tree contains no raw archives")
    for row in archives:
        if not ARCHIVE.fullmatch(Path(row["path"]).name):
            raise RuntimeError(f"Unrecognized archive path: {row['path']}")
    shots = record("data/shots/shots_fixed.csv")
    events = {Path(name).stem: record(name) for name in sorted(entries)
              if re.fullmatch(r"data/events/\d{10}\.csv", name)}
    manifest = dict(schema=SCHEMA, source_repository=config["source_repository"],
                    source_commit=sha, source_tree_sha=commit["commit"]["tree"]["sha"],
                    config_sha256=digest(root / "frozen-config.json"),
                    split_rule=dict(salt=config["split_salt"], percentages=config["split_percentages"]),
                    illustrative_policy=config["illustrative_policy"],
                    corpus_definition=config["corpus_definition"],
                    archives=archives, events=events, corrected_shots=shots)
    atomic_json(path, manifest)
    atomic_json(root / "implementation.json",
                dict(schema=SCHEMA, files={str(Path(__file__).name): digest(__file__),
                                          "nba.py": digest(Path(__file__).with_name("nba.py"))}))
    log(root, "source_frozen", source_commit=sha, archives=len(archives), events=len(events))
    return manifest


def archive_member(archive, config):
    command = subprocess.run(["7z", "l", "-slt", str(archive)], capture_output=True, text=True)
    if command.returncode:
        raise SourceError("corrupt_archive", command.stderr[-2000:] or command.stdout[-2000:])
    blocks = command.stdout.split("----------", 1)
    if len(blocks) != 2:
        raise SourceError("corrupt_archive", "No 7z member listing")
    members = []
    for block in blocks[1].strip().split("\n\n"):
        fields = dict(line.split(" = ", 1) for line in block.splitlines() if " = " in line)
        if "Path" in fields:
            members.append(fields)
    if not members:
        raise SourceError("empty_archive", "Archive contains no members")
    if len(members) != 1 or not re.fullmatch(r"\d{10}\.json", members[0]["Path"]):
        raise SourceError("corrupt_archive", f"Unsafe/unexpected archive members: {members}")
    row = members[0]
    if row.get("Folder") == "+" or row.get("Attributes", "").startswith("D") or "L" in row.get("Attributes", ""):
        raise SourceError("corrupt_archive", "Archive member is a directory")
    if row.get("Encrypted") == "+":
        raise SourceError("corrupt_archive", "Encrypted archive is unavailable public input")
    size = int(row["Size"])
    if size == 0:
        raise SourceError("empty_archive", "Archive JSON has zero bytes")
    if size > config["maximum_extracted_bytes"]:
        raise SourceError("corrupt_archive", f"JSON exceeds declared extraction guard: {size}")
    return Path(row["Path"]).stem, size


def extract_owned(archive, game_id, size, root, config):
    """Stream a single prevalidated member to an owned path; never trust stored paths."""
    target = root / "extracted" / f"{game_id}.json"
    receipt_path = target.with_suffix(".extraction.json")
    identity = dict(game_id=game_id, bytes=size, archive_sha256=digest(archive))
    if target.exists():
        if target.stat().st_size != size:
            raise RuntimeError(f"Interrupted extraction requires inspection: {target}")
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if {k: receipt[k] for k in identity} != identity or receipt["json_sha256"] != digest(target):
                raise RuntimeError(f"Extracted JSON resume audit failed: {target}")
            return target
        log(root, "unreceipted_extraction_rebuilding", game_id=game_id)
    disk_guard(root, config, size)
    part = target.with_name(target.name + ".part")
    error_path = root / "extracted" / f"{game_id}.7z-stderr"
    with part.open("wb") as handle, error_path.open("wb") as errors:
        process = subprocess.Popen(["7z", "x", "-so", str(archive), f"{game_id}.json"],
                                   stdout=subprocess.PIPE, stderr=errors)
        try:
            total = 0
            while chunk := process.stdout.read(1 << 20):
                total += len(chunk)
                if total > min(size, config["maximum_extracted_bytes"]):
                    raise SourceError("corrupt_archive", "Extraction exceeds validated byte guard")
                handle.write(chunk)
            code = process.wait()
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait()
            process.stdout.close()
        handle.flush()
        os.fsync(handle.fileno())
    if code or part.stat().st_size != size:
        raise SourceError("corrupt_archive", error_path.read_text(errors="replace")[-2000:]
                          or "Extracted size disagrees with member listing")
    error_path.unlink()
    atomic_json(receipt_path, dict(identity, json_sha256=digest(part)))
    part.replace(target)
    return target


def remove_resolved_json(raw, root):
    raw = Path(raw)
    if raw.parent.resolve() != (root / "extracted").resolve() or not re.fullmatch(r"\d{10}\.json", raw.name):
        raise RuntimeError(f"Refusing deletion outside owned extracted JSON: {raw}")
    raw.unlink()


def validated_clips(raw, events_path, fixed_path, config, game_id, name):
    """Reuse declared clip validation; retain only potentially relevant shot windows.

    Streaming selected clock windows avoids materializing an entire duplicated game.
    Frames are deduplicated across raw tracking-event blocks; conflicting timestamps
    are excluded, just as in the pilot, without trusting tracking event IDs.
    """
    excluded = Counter()
    shots = shot_rows(events_path)
    fixed = corrected_shots(fixed_path)
    accepted, seconds = {}, set()
    for event_id, original in shots.items():
        period = int(float(original["PERIOD"]))
        official = clock_value(original["PCTIMESTRING"])
        correction = fixed.get((game_id, event_id))
        if correction is None:
            excluded["no corrected shot anchor"] += 1
            continue
        if correction["quarter"] != period or abs(correction["clock"] - official) > config["maximum_shot_anchor_shift_seconds"]:
            excluded["inconsistent corrected shot anchor"] += 1
            continue
        anchor = correction["clock"]
        accepted[event_id] = dict(original, _tracking_shot_clock=anchor,
                                  _nominal_distance_feet=correction["nominal_distance_feet"])
        seconds.update((period, second) for second in
                       range(int(anchor) - 4, int(anchor + config["clip_duration_seconds"]) + 3))
    with raw.open("rb") as handle:
        actual_id = next(ijson.items(handle, "gameid", use_float=True), None)
    if str(actual_id).zfill(10) != game_id:
        raise SourceError("corrupt_tracking", f"JSON game ID {actual_id!r} disagrees with {game_id}")
    frames, origins, conflicts, template = {}, {}, set(), None
    signatures = {}
    duplicates, total, outside = 0, 0, 0
    with raw.open("rb") as handle:
        for index, event in enumerate(ijson.items(handle, "events.item", use_float=True)):
            if template is None:
                template = dict(home=event["home"], visitor=event["visitor"])
            for moment in event["moments"]:
                total += 1
                if len(moment) < 6 or moment[1] is None or moment[2] is None:
                    continue
                period, timestamp, clock = int(moment[0]), int(moment[1]), float(moment[2])
                key = (period, timestamp)
                if key in conflicts:
                    continue
                # Remember compact signatures even outside selected clock windows:
                # a conflicting second copy can have a completely different clock.
                signature = hashlib.sha256(repr((clock, tuple(sorted(tuple(float(v) for v in row)
                                                       for row in moment[5])))).encode()).digest()
                if key in signatures:
                    duplicates += 1
                    if signature != signatures[key]:
                        conflicts.add(key)
                        frames.pop(key, None)
                        origins.pop(key, None)
                        continue
                else:
                    signatures[key] = signature
                if not np.isfinite(clock) or (period, int(clock)) not in seconds:
                    outside += 1
                    continue
                if key in frames:
                    origins[key].add((str(event["eventId"]), index))
                else:
                    frames[key] = moment
                    origins[key] = {(str(event["eventId"]), index)}
    if template is None:
        raise SourceError("corrupt_tracking", "No tracking events")
    buckets = defaultdict(list)
    for key, moment in sorted(frames.items()):
        buckets[(key[0], int(float(moment[2])))].append(moment)
    clips, windows = [], set()
    for event_id, shot in accepted.items():
        period, anchor = int(float(shot["PERIOD"])), shot["_tracking_shot_clock"]
        moments = [moment for second in range(int(anchor) - 4,
                   int(anchor + config["clip_duration_seconds"]) + 3)
                   for moment in buckets[(period, second)]]
        event = dict(template, eventId=event_id, moments=sorted(moments, key=lambda m: m[1]))
        result, reason = extract_clip(event, shot, config, game_id, name, None)
        if result is None:
            excluded[reason] += 1
            continue
        metadata = result[2]
        key = (metadata["quarter"], metadata["offense_team_id"],
               round(metadata["timestamp_start"], 3), round(metadata["timestamp_end"], 3))
        if key in windows:
            excluded["duplicate shot window"] += 1
            continue
        windows.add(key)
        contributing = set()
        for moment in moments:
            if metadata["timestamp_start"] <= moment[1] / 1000 <= metadata["timestamp_end"]:
                contributing.update(origins[(period, int(moment[1]))])
        metadata.update(tracking_event_ids=sorted({int(e) for e, i in contributing}),
                        tracking_event_array_indices=sorted({i for e, i in contributing}),
                        alignment_method="deduplicated game timeline by quarter/clock; not tracking-event-ID equality")
        clips.append(result)
    clips.sort(key=lambda c: (c[2]["quarter"], c[2]["event_id"]))
    return clips, dict(shot_rows=len(shots), usable_clips=len(clips), excluded=dict(excluded),
                      total_raw_moments=total, outside_candidate_windows=outside,
                      unique_frames=len(signatures) - len(conflicts),
                      unique_candidate_frames=len(frames), duplicate_candidate_frames=duplicates,
                      conflicting_timestamps=len(conflicts))


def parse_source(raw, events_path, fixed_path, config, game_id, name):
    # These exceptions are confined to the untrusted input parser. Errors elsewhere
    # (including cache, storage, and index implementations) must stop the process.
    try:
        return validated_clips(raw, events_path, fixed_path, config, game_id, name)
    except (ijson.JSONError, UnicodeError, csv.Error, KeyError, TypeError, ValueError, IndexError) as exc:
        raise SourceError("corrupt_tracking", f"{type(exc).__name__}: {exc}") from exc


def audit_game(root, row):
    if row.get("file_sha256"):
        path = root / "raw" / row["archive"]
        if not path.is_file() or digest(path) != row["file_sha256"]:
            raise RuntimeError(f"Resolved raw archive failed audit: {path}")
    if "events_provenance" in row:
        path = root / "events" / f"{row['game_id']}.csv"
        if not path.is_file() or digest(path) != row["events_provenance"]["file_sha256"]:
            raise RuntimeError(f"Resolved events source failed audit: {path}")
    for name, expected in row.get("artifacts", {}).items():
        path = root / name
        if not path.is_file() or digest(path) != expected:
            raise RuntimeError(f"Resolved cache artifact failed audit: {path}")
    if row.get("status") == "usable":
        with np.load(root / row["curves_path"], allow_pickle=False) as arrays:
            n = row["usable_clips"]
            if arrays["curves"].shape != (n, 5, 32, 2) or not np.isfinite(arrays["curves"]).all():
                raise RuntimeError(f"Invalid resolved curve cache: {row['archive']}")
            if arrays["balls"].shape != (n, 32, 2) or not np.isfinite(arrays["balls"]).all():
                raise RuntimeError(f"Invalid resolved ball cache: {row['archive']}")
        metadata = json.loads((root / row["metadata_path"]).read_text())
        if len(metadata) != n or any(m["game_id"] != row["game_id"] for m in metadata):
            raise RuntimeError(f"Invalid resolved metadata cache: {row['archive']}")


def provenance(root, sources, resolved):
    atomic_json(root / "manifest.json",
                dict(sources, archives=[dict(source, **resolved.get(Path(source["path"]).name, {}))
                                       for source in sources["archives"]]))


def prepare_games(config, root, sources):
    statuses_path = root / "archive-status.json"
    resolved = json.loads(statuses_path.read_text()) if statuses_path.exists() else {}
    valid_names = {Path(row["path"]).name for row in sources["archives"]}
    if set(resolved) - valid_names:
        raise RuntimeError("Resolved archive not in frozen source manifest")
    seen = {}
    for name in sorted(resolved):
        row = resolved[name]
        audit_game(root, row)
        if row["status"] in ("usable", "zero_usable"):
            seen[row["game_id"]] = name
    shots = sources["corrected_shots"]
    fixed = root / "shots" / "shots_fixed.csv"
    shot_download = download(shots["url"], fixed, config, root, shots["github_blob_sha"], 32_000_000)
    atomic_json(root / "shots" / "provenance.json", dict(shots, file_sha256=shot_download["sha256"]))
    for position, source in enumerate(sources["archives"]):
        name = Path(source["path"]).name
        if name in resolved:
            log(root, "resume_audited", archive=name, position=position + 1,
                total=len(sources["archives"]), status=resolved[name]["status"])
            stale = root / "extracted" / f"{resolved[name].get('game_id', 'none')}.json"
            if stale.exists() and resolved[name]["status"] in RESOLVED:
                remove_resolved_json(stale, root)
            continue
        row = dict(archive=name, status="pending", file_sha256=None)
        raw = None
        log(root, "archive_start", archive=name, position=position + 1, total=len(sources["archives"]))
        try:
            archive = root / "raw" / name
            try:
                result = download(source["url"], archive, config, root,
                                  source["github_blob_sha"], config["maximum_archive_bytes"])
            except SourceError as exc:
                status = ("corrupt_archive" if exc.status in ("checksum_mismatch", "oversized_source") else
                          "empty_archive" if exc.status == "empty_source" else "unavailable_archive")
                raise SourceError(status, str(exc), exc.http_code) from exc
            row.update(file_sha256=result["sha256"], downloaded_bytes=result["bytes"])
            if result["bytes"] == 0:
                raise SourceError("empty_archive", "Frozen public archive has zero bytes")
            game_id, size = archive_member(archive, config)
            row.update(game_id=game_id, split=game_split(game_id, config), extracted_bytes=size)
            if game_id in seen:
                row.update(status="duplicate_game_id", duplicate_of=seen[game_id], usable_clips=0)
            else:
                events_source = sources["events"].get(game_id)
                if events_source is None:
                    raise SourceError("unavailable_events", "No play-by-play source in frozen tree")
                events = root / "events" / f"{game_id}.csv"
                try:
                    event_result = download(events_source["url"], events, config, root,
                                            events_source["github_blob_sha"], 8_000_000)
                except SourceError as exc:
                    raise SourceError("unavailable_events", str(exc), exc.http_code) from exc
                row["events_provenance"] = dict(events_source, file_sha256=event_result["sha256"])
                raw = extract_owned(archive, game_id, size, root, config)
                row["tracking_json_sha256"] = digest(raw)
                clips, audit = parse_source(raw, events, fixed, config, game_id, name)
                row.update(audit, status="usable" if clips else "zero_usable")
                prefix = root / "games" / game_id
                curves = np.stack([clip[0] for clip in clips]) if clips else np.empty((0, 5, 32, 2))
                balls = np.stack([clip[1] for clip in clips]) if clips else np.empty((0, 32, 2))
                atomic_npz(prefix.with_suffix(".npz"), curves=curves, balls=balls)
                atomic_json(prefix.with_suffix(".json"), [clip[2] for clip in clips])
                paths = [prefix.with_suffix(".npz"), prefix.with_suffix(".json")]
                row.update(curves_path=str(paths[0].relative_to(root)),
                           metadata_path=str(paths[1].relative_to(root)),
                           artifacts={str(path.relative_to(root)): digest(path) for path in paths})
                seen[game_id] = name
        except SourceError as exc:
            row.update(status=exc.status, failure_detail=str(exc), http_code=exc.http_code, usable_clips=0)
            log(root, "source_excluded", archive=name, **{k: v for k, v in row.items() if k != "archive"})
        resolved[name] = row
        atomic_json(statuses_path, resolved)
        provenance(root, sources, resolved)
        # Persist resolved state and validated curves before removing only this JSON.
        if raw is not None and raw.exists():
            remove_resolved_json(raw, root)
        log(root, "archive_resolved", archive=name, position=position + 1,
            status=row["status"], usable_clips=row.get("usable_clips", 0),
            resolved=len(resolved), total=len(sources["archives"]),
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    return resolved


def assemble(config, root, resolved):
    completion = root / "dataset.json"
    if completion.exists():
        info = json.loads(completion.read_text())
        for name, expected in info["artifacts"].items():
            if digest(root / name) != expected:
                raise RuntimeError(f"Frozen dataset failed audit: {name}")
        return info
    pools, curves, balls = defaultdict(list), defaultdict(list), defaultdict(list)
    for row in sorted(resolved.values(), key=lambda r: r["archive"]):
        if row["status"] != "usable":
            continue
        split = row["split"]
        metadata = json.loads((root / row["metadata_path"]).read_text())
        with np.load(root / row["curves_path"], allow_pickle=False) as arrays:
            pools[split].extend(metadata)
            curves[split].extend(arrays["curves"])
            balls[split].extend(arrays["balls"])
    artifacts, counts, game_sets = {}, {}, {}
    for split in ("database", "tune", "test"):
        if not pools[split]:
            raise RuntimeError(f"No usable {split} clips; refusing an incomplete dataset")
        indices = (list(range(len(pools[split]))) if split == "database" else
                   balanced_selection(pools[split], config[f"{'tuning' if split == 'tune' else 'test'}_query_cap"],
                                      config["split_salt"] + ":" + split))
        selected = [pools[split][i] for i in indices]
        atomic_npz(root / f"{split}.npz", curves=np.asarray(curves[split])[indices],
                   balls=np.asarray(balls[split])[indices])
        atomic_json(root / f"{split}-metadata.json", selected)
        for name in (f"{split}.npz", f"{split}-metadata.json"):
            artifacts[name] = digest(root / name)
        counts[split] = dict(all_valid_clips=len(pools[split]), selected_clips=len(indices),
                            games=len({m["game_id"] for m in selected}))
        game_sets[split] = {m["game_id"] for m in selected}
    if any(game_sets[a] & game_sets[b] for a, b in
           (("database", "tune"), ("database", "test"), ("tune", "test"))):
        raise AssertionError("Game-level split leak")
    info = dict(schema=SCHEMA, artifacts=artifacts, counts=counts,
                status_counts=dict(Counter(r["status"] for r in resolved.values())),
                total_archives=len(resolved), corpus_definition=config["corpus_definition"],
                source_manifest_sha256=digest(root / "source-manifest.json"),
                archive_audit_sha256=digest(root / "archive-status.json"),
                illustrative_policy=config["illustrative_policy"])
    atomic_json(completion, info)
    log(root, "dataset_assembled", **counts)
    return info


def exact_preparation(config, root):
    """Resumable exact all-database nearest neighbors, not a timed index comparison."""
    from fresh.regimes import ground_truth
    with np.load(root / "database.npz", allow_pickle=False) as arrays:
        X = arrays["curves"]
    for split in ("tune", "test"):
        with np.load(root / f"{split}.npz", allow_pickle=False) as arrays:
            Q = arrays["curves"]
        records, all_ids, all_distances = [], [], []
        width = config["ground_truth_batch_size"]
        for start in range(0, len(Q), width):
            end = min(len(Q), start + width)
            path = root / "ground-truth" / f"{split}-{start:06d}-{end:06d}.npz"
            audit = path.with_suffix(".json")
            identity = dict(dataset_sha256=digest(root / "dataset.json"),
                            split=split, start=start, end=end, metric="exact unordered bottleneck discrete Frechet in feet")
            if path.exists() and audit.exists():
                old = json.loads(audit.read_text())
                if {k: old[k] for k in identity} != identity or old["sha256"] != digest(path):
                    raise RuntimeError(f"Ground-truth cache audit failed: {path}")
                with np.load(path, allow_pickle=False) as arrays:
                    ids, distances = arrays["ids"], arrays["distances"]
            else:
                log(root, "exact_ground_truth_start", split=split, start=start, end=end, total=len(Q))
                ids, distances, _ = ground_truth(X, Q[start:end], 0.0)
                if np.any(ids < 0) or not np.isfinite(distances).all():
                    raise AssertionError("Exact nearest-neighbor search returned invalid distances")
                atomic_npz(path, ids=ids, distances=distances)
                atomic_json(audit, dict(identity, sha256=digest(path)))
            if ids.shape != (end - start,) or distances.shape != (end - start,) or np.any(ids < 0) or np.any(ids >= len(X)) or not np.isfinite(distances).all():
                raise RuntimeError(f"Invalid ground-truth batch: {path}")
            records.append(dict(path=str(path.relative_to(root)), sha256=digest(path)))
            all_ids.extend(ids)
            all_distances.extend(distances)
            log(root, "exact_ground_truth_resolved", split=split, completed=end, total=len(Q))
        atomic_npz(root / f"{split}-ground-truth.npz", ids=np.asarray(all_ids, dtype=np.int64),
                   distances=np.asarray(all_distances))
        atomic_json(root / f"{split}-ground-truth.json",
                    dict(dataset_sha256=digest(root / "dataset.json"), batches=records,
                         sha256=digest(root / f"{split}-ground-truth.npz")))


def prepare(config_path, clips_only=False):
    config, root = initialize(config_path)
    from numba import set_num_threads
    set_num_threads(config["preparation_threads"])
    with lock(root):
        try:
            sources = freeze_sources(config, root)
            resolved = prepare_games(config, root, sources)
            info = assemble(config, root, resolved)
            if not clips_only:
                exact_preparation(config, root)
                atomic_json(root / "preparation-complete.json",
                            dict(dataset_sha256=digest(root / "dataset.json"),
                                 source_commit=sources["source_commit"], counts=info["counts"],
                                 ground_truth={s: digest(root / f"{s}-ground-truth.npz") for s in ("tune", "test")}))
            log(root, "preparation_complete" if not clips_only else "clips_complete", counts=info["counts"])
        except Exception as exc:
            log(root, "preparation_stopped", exception=type(exc).__name__, detail=str(exc))
            raise


def quality_summary(rows, optimum, factors):
    from fresh.regimes import approximation_summary
    n = len(rows)
    if n == 0 or n != len(optimum):
        raise ValueError("Quality metrics require all query outcomes and exact optima")
    returned, excess, ratios = [], [], []
    exact, found = 0, 0
    successes = {str(c): 0 for c in factors}
    for row, best in zip(rows, optimum):
        if row["id"] < 0:
            continue
        value = row["distance"]
        if not np.isfinite(value) or value < best - 1e-7:
            raise AssertionError("Returned distance inconsistent with exact ground truth")
        found += 1
        exact += int(value <= best + 1e-8)
        for c in factors:
            successes[str(c)] += int(value <= c * best + 1e-8)
        returned.append(value)
        excess.append(max(0.0, value - best))
        ratios.append(value / best if best else (1.0 if value == 0 else np.inf))
    return dict(query_count=n, recall_at_1=exact / n, empty_rate=1 - found / n,
                coverage=found / n, c_success={c: v / n for c, v in successes.items()},
                missing_ratio_count=n - found,
                ratio_scope="returned answers only; missing answers counted separately in empty_rate and missing_ratio_count",
                mean_optimal_feet=float(np.mean(optimum)), p90_optimal_feet=float(np.quantile(optimum, .9)),
                mean_returned_feet=float(np.mean(returned)) if returned else None,
                p90_returned_feet=float(np.quantile(returned, .9)) if returned else None,
                mean_excess_feet=float(np.mean(excess)) if excess else None,
                p90_excess_feet=float(np.quantile(excess, .9)) if excess else None,
                **approximation_summary(ratios))


def safe_row(row):
    return {k: (None if isinstance(v, (float, np.floating)) and not np.isfinite(v)
                else v.item() if isinstance(v, np.generic) else v) for k, v in row.items()}


METHOD_LABELS = {"blocking": "blocking", "probe_all": "probe_all", "store_all": "store_all",
                 "one_block": "per_curve-intersection", "rarest": "per_curve-rarest"}


class BenchmarkLimit(Exception):
    """Expected benchmark resource/deadline exclusion; never a parser failure."""
    def __init__(self, status, metrics=None):
        super().__init__(status)
        self.status, self.metrics = status, metrics or {}


def benchmark_config(prepared, profile="newlab", index_budget_bytes=None,
                     peak_ram_cap_bytes=None, build_deadline_seconds=None):
    if profile not in ("newlab", "preparation"):
        raise ValueError("Unknown benchmark resource profile")
    config = dict(prepared)
    defaults = ((24_000_000_000, 48_000_000_000) if profile == "newlab" else
                (prepared["index_budget_bytes"], prepared["peak_ram_cap_bytes"]))
    config.update(index_budget_bytes=defaults[0] if index_budget_bytes is None else index_budget_bytes,
                  peak_ram_cap_bytes=defaults[1] if peak_ram_cap_bytes is None else peak_ram_cap_bytes,
                  build_deadline_seconds=1800.0 if build_deadline_seconds is None else build_deadline_seconds)
    for field in ("index_budget_bytes", "peak_ram_cap_bytes", "build_deadline_seconds"):
        if not np.isfinite(config[field]) or config[field] <= 0:
            raise ValueError(f"Benchmark {field} must be positive and finite")
    config["benchmark_resource_profile"] = profile
    return config


def current_rss():
    with Path("/proc/self/status").open() as handle:
        for line in handle:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("Linux process RSS is unavailable")


def build_preflight(row, X, config):
    """Conservative uncompressed construction bound, including PerCurve scratch."""
    from curveset.index import index_bytes
    n, a, L = len(X), row["stored"], row["L"]
    entries = n * a
    if n >= 2 ** 31 or entries >= 2 ** 32:
        raise BenchmarkLimit("overbudget: integer capacity")
    per_curve = row["method"] in ("one_block", "rarest")
    scratch = n * (22 if per_curve else 12)
    upper_storage = index_bytes(entries, L) + L * 5 * 2 * 8 + 2 * 120 * 5
    temporary = entries * 64 + n * (5 if per_curve else 25) * 8
    resident = current_rss()
    estimate = resident + upper_storage + temporary + scratch
    metrics = dict(estimated_peak_ram_bytes=estimate, uncompressed_index_upper_bytes=upper_storage,
                   estimated_scratch_bytes=scratch, construction_temporary_upper_bytes=temporary,
                   prebuild_resident_rss_bytes=resident)
    if estimate > config["peak_ram_cap_bytes"]:
        raise BenchmarkLimit("overbudget: preallocation peak RAM estimate", metrics)
    return metrics


def measured_build(factory, config):
    """Measure every constructor and enforce wall/virtual-memory allocation caps.

    SIGALRM may be delivered after a native kernel returns; measured elapsed time
    is checked as well. RLIMIT_AS bounds allocations conservatively, while RSS is
    sampled separately and the process lifetime high-water mark is also retained.
    The monitoring thread is stopped before any timed query.
    """
    stop = threading.Event()
    samples = [current_rss()]
    def sample():
        while not stop.wait(.05):
            samples.append(current_rss())
    timer = threading.Thread(target=sample, name="nba-build-rss", daemon=True)
    old_limit = resource.getrlimit(resource.RLIMIT_AS)
    old_handler = signal.getsignal(signal.SIGALRM)
    old_timer = signal.getitimer(signal.ITIMER_REAL)
    cap = config["peak_ram_cap_bytes"]
    soft = cap if old_limit[1] == resource.RLIM_INFINITY else min(cap, old_limit[1])
    def deadline(signum, frame):
        raise BenchmarkLimit("overbudget: build deadline")
    started = time.perf_counter()
    failure, index = None, None
    timer.start()
    try:
        resource.setrlimit(resource.RLIMIT_AS, (soft, old_limit[1]))
        signal.signal(signal.SIGALRM, deadline)
        signal.setitimer(signal.ITIMER_REAL, config["build_deadline_seconds"])
        index = factory()
    except BenchmarkLimit as exc:
        failure = exc
    except MemoryError:
        failure = BenchmarkLimit("overbudget: build allocation RAM cap")
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
        resource.setrlimit(resource.RLIMIT_AS, old_limit)
        stop.set()
        timer.join()
        signal.setitimer(signal.ITIMER_REAL, *old_timer)
    elapsed = time.perf_counter() - started
    samples.append(current_rss())
    metrics = dict(build_seconds=elapsed, build_sampled_peak_rss_bytes=max(samples),
                   build_rss_sampling_seconds=.05, build_virtual_memory_cap_bytes=soft,
                   process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    if index is not None:
        metrics.update(index_bytes=index.index_bytes, scratch_bytes=index.scratch_bytes,
                       entries=index.entries, raw_entries=index.raw_entries,
                       constructor_reported_build_seconds=index.build_seconds)
    if failure or elapsed > config["build_deadline_seconds"] or max(samples) > cap:
        index = None
        raise BenchmarkLimit(failure.status if failure else "overbudget: measured build time/RAM", metrics)
    if index.index_bytes > config["index_budget_bytes"]:
        index = None
        raise BenchmarkLimit("overbudget: actual index storage", metrics)
    return index, metrics


def exceeds_deadline(row, seconds):
    return bool(row.get("timeout") or not np.isfinite(row["seconds"]) or row["seconds"] > seconds)


def query_record(row, query, optimal):
    result = dict(query=query, optimal_feet=float(optimal), **safe_row(row))
    if row["id"] < 0:
        result.update(ratio=None, ratio_status="missing", excess_feet=None)
    else:
        value = row["distance"]
        ratio = value / optimal if optimal > 0 else (1.0 if value == 0 else np.inf)
        result.update(ratio="infinity" if np.isinf(ratio) else float(ratio),
                      ratio_status="unbounded" if np.isinf(ratio) else "finite",
                      excess_feet=float(max(0.0, value - optimal)))
    return result


def query_summary(rows, optimum, config, requested_count=None, warmup_failed=False):
    failed = [i for i, row in enumerate(rows) if exceeds_deadline(row, config["query_deadline_seconds"])]
    requested = len(optimum) if requested_count is None else requested_count
    valid = not warmup_failed and not failed and len(rows) == requested
    result = dict(status="complete" if valid else "overbudget: query deadline",
                  query_count=len(rows), requested_queries=requested, deadline_failed_queries=failed,
                  warmup_deadline_failed=warmup_failed,
                  not_run_queries=list(range(len(rows), requested)),
                  quality=None, median_top1_seconds=None, p90_top1_seconds=None, p95_top1_seconds=None)
    if valid:
        seconds = [row["seconds"] for row in rows]
        result.update(quality=quality_summary(rows, optimum, config["c_factors"]),
                      median_top1_seconds=float(np.median(seconds)),
                      p90_top1_seconds=float(np.quantile(seconds, .9)),
                      p95_top1_seconds=float(np.quantile(seconds, .95)))
    return result


def choose_main(rows, method, config):
    from fresh.poc import weighted_cost
    eligible = [row for row in rows if row["method"] == method and row["status"] == "qualified"]
    if not eligible:
        return None
    if method == "blocking":
        key = lambda row: (weighted_cost(row, config["blocking_main_latency_weight"], config),
                           row["index_bytes"], row["config_id"])
    else:
        key = lambda row: (row["tuning_median_seconds"], row["index_bytes"], row["config_id"])
    return min(eligible, key=key)


def reranked_candidates(index, X, Q, count):
    """Untimed diagnostics on the candidate IDs left by the just-completed top1 call."""
    from fresh.tuple_quality import exact_distance
    if hasattr(index, "base"):
        candidates = index.base.cand
    else:
        candidates = index.candidates
    return sorted((dict(database_index=int(p), distance_feet=float(exact_distance(X[p], Q)))
                   for p in candidates[:count]), key=lambda r: (r["distance_feet"], r["database_index"]))


def prepared_data(config_path):
    """Read-only, hash-audited dataset loading, portable to the newLab workspace.

    The original extraction owner path remains recorded, but evaluation may read
    a copied corpus from another absolute path. It never assumes deletion rights.
    """
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    root = (config_path.parent.parent / config["data_directory"]).resolve()
    if root.name != "nba-full-20261012":
        raise RuntimeError("Evaluation requires the dedicated full-corpus data directory")
    if json.loads((root / "frozen-config.json").read_text()) != config:
        raise RuntimeError("Evaluation configuration differs from frozen preparation")
    complete = json.loads((root / "preparation-complete.json").read_text())
    if complete["dataset_sha256"] != digest(root / "dataset.json"):
        raise RuntimeError("Prepared dataset hash mismatch")
    info = json.loads((root / "dataset.json").read_text())
    for name, expected in info["artifacts"].items():
        if digest(root / name) != expected:
            raise RuntimeError(f"Prepared dataset artifact hash mismatch: {name}")
    for name, expected in (("source-manifest.json", info["source_manifest_sha256"]),
                           ("archive-status.json", info["archive_audit_sha256"])):
        if digest(root / name) != expected:
            raise RuntimeError(f"Prepared provenance artifact hash mismatch: {name}")
    for split, expected in complete["ground_truth"].items():
        if digest(root / f"{split}-ground-truth.npz") != expected:
            raise RuntimeError(f"Exact ground-truth artifact hash mismatch: {split}")
    return config, root


def evaluate(config_path, output, authorize, core_number, newlab_hostname=None,
             resource_profile="newlab", index_budget_bytes=None, peak_ram_cap_bytes=None,
             build_deadline_seconds=None):
    """No entry from prepare; parent must explicitly dequeue on newLab."""
    if not authorize:
        raise RuntimeError("Timed evaluation requires --confirm-newlab-after-synthetic")
    hostname = socket.gethostname()
    if newlab_hostname is not None and hostname != newlab_hostname:
        raise RuntimeError("The current hostname does not match the explicitly requested benchmark host")
    from numba import set_num_threads
    from fresh.poc import Joint, PerCurve, weighted_cost
    from fresh.regimes import family_first_hits
    from curveset.tune import first_hit_per_curve, tables_for_recall
    from curveset import bench
    prepared, root = prepared_data(config_path)
    config = benchmark_config(prepared, resource_profile, index_budget_bytes,
                              peak_ram_cap_bytes, build_deadline_seconds)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    bench.PIN_CORE = core_number
    set_num_threads(1)
    code_root = Path(__file__).resolve().parents[1]
    git = subprocess.run(["git", "--no-pager", "rev-parse", "HEAD"], cwd=code_root,
                         capture_output=True, text=True)
    if git.returncode:
        raise RuntimeError("Benchmark requires a staged Git snapshot with a recorded code revision")
    raw_manifest = json.loads((root / "source-manifest.json").read_text())
    helper_paths = [Path(__file__).with_name(name) for name in
                    ("nba.py", "poc.py", "regimes.py", "tuple_quality.py", "e1.py")]
    helper_paths += sorted((code_root / "curveset").glob("*.py"))
    identity = dict(config_sha256=digest(root / "frozen-config.json"),
                    effective_benchmark_config_sha256=hashlib.sha256(
                        (json.dumps(config, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()).hexdigest(),
                    benchmark_resources={name: config[name] for name in
                        ("benchmark_resource_profile", "index_budget_bytes", "peak_ram_cap_bytes",
                         "build_deadline_seconds", "query_deadline_seconds")},
                    code_git_revision=git.stdout.strip(),
                    raw_source_repository=raw_manifest.get("source_repository"),
                    raw_source_git_commit=raw_manifest.get("source_commit"),
                    method_labels=METHOD_LABELS,
                    selection_policy={"blocking": "weighted tuning cost",
                                      "other_methods": "fastest qualified tuning median"},
                    dataset_sha256=digest(root / "dataset.json"),
                    ground_truth={s: digest(root / f"{s}-ground-truth.npz") for s in ("tune", "test")},
                    source_manifest_sha256=digest(root / "source-manifest.json"),
                    implementation_sha256=digest(__file__), nba_sha256=digest(Path(__file__).with_name("nba.py")),
                    helpers={str(path.relative_to(code_root)): digest(path) for path in helper_paths},
                    hostname=socket.gethostname(), pin_core=core_number, timing_threads=1,
                    timing_scope="top1 only; candidate topK diagnostic reranking is outside all query timers")
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != identity:
        raise RuntimeError("Evaluation resume provenance/config/hardware changed")
    atomic_json(manifest_path, identity)
    atomic_json(output / "effective-benchmark-config.json", config)
    with np.load(root / "database.npz", allow_pickle=False) as arrays:
        X = arrays["curves"]
    with np.load(root / "tune.npz", allow_pickle=False) as arrays:
        Qt = arrays["curves"]
    with np.load(root / "test.npz", allow_pickle=False) as arrays:
        Qtest = arrays["curves"]
    with np.load(root / "tune-ground-truth.npz", allow_pickle=False) as arrays:
        NN, optimal_tune = X[arrays["ids"]], arrays["distances"]
    with np.load(root / "test-ground-truth.npz", allow_pickle=False) as arrays:
        optimal_test = arrays["distances"]
    database_metadata = json.loads((root / "database-metadata.json").read_text())
    test_metadata = json.loads((root / "test-metadata.json").read_text())
    qwarm = Qt[:config["warmup_queries"]]
    for seed in config["index_seeds"]:
        seed_root = output / f"seed-{seed}"
        seed_root.mkdir(exist_ok=True)
        complete = seed_root / "complete.json"
        if complete.exists():
            for name, expected in json.loads(complete.read_text())["artifacts"].items():
                if digest(seed_root / name) != expected:
                    raise RuntimeError(f"Completed seed artifact failed audit: {name}")
            continue
        tuning_path = seed_root / "tuning-configurations.json"
        tuning = json.loads(tuning_path.read_text()) if tuning_path.exists() else []
        done = {r["config_id"] for r in tuning}

        def build(row):
            phase_metrics = build_preflight(row, X, config)
            log(output, "build_start", seed=seed, config_id=row["config_id"], **phase_metrics)
            if row["method"] in ("one_block", "rarest"):
                idx, metrics = measured_build(lambda: PerCurve(X, row["delta"], row["L"], seed), config)
                call = lambda Q: idx.query(X, Q, config, row["method"] == "rarest")
            else:
                idx, metrics = measured_build(lambda: Joint(X, row["delta"], row["L"], seed,
                                                           row["blocks"], row["swapped"]), config)
                call = lambda Q: idx.query(X, Q, config)
            metrics.update(phase_metrics)
            log(output, "build_complete", seed=seed, config_id=row["config_id"], **metrics)
            return idx, call, metrics

        def run_queries(call, queries, cid, phase):
            rows, warmups = [], []
            log(output, "query_phase_start", seed=seed, config_id=cid, phase=phase, requested=len(queries))
            with bench.pinned():
                for Q in qwarm:
                    answer = call(Q)
                    warmups.append(answer)
                    if exceeds_deadline(answer, config["query_deadline_seconds"]):
                        break
                warmup_failed = any(exceeds_deadline(r, config["query_deadline_seconds"]) for r in warmups)
                if not warmup_failed:
                    for q, Q in enumerate(queries):
                        answer = call(Q)
                        rows.append(answer)
                        if (q + 1) % 100 == 0:
                            log(output, "query_progress", seed=seed, config_id=cid, phase=phase,
                                completed=q + 1, requested=len(queries))
                        if exceeds_deadline(answer, config["query_deadline_seconds"]):
                            break
            atomic_json(seed_root / (cid + f"-{phase}-warmups.json"), [safe_row(r) for r in warmups])
            log(output, "query_phase_finished", seed=seed, config_id=cid, phase=phase,
                completed=len(rows), requested=len(queries), warmup_deadline_failed=warmup_failed,
                deadline_failed=any(exceeds_deadline(r, config["query_deadline_seconds"]) for r in rows))
            return rows, warmup_failed

        specs = [("probe_all", [1] * 5, False), ("store_all", [1] * 5, True)]
        specs += [("blocking", sizes, swap) for sizes in config["blocking_families"] for swap in (False, True)]
        specs += [("one_block", None, False), ("rarest", None, False)]
        for delta in config["deltas_feet"]:
            for method, sizes, swapped in specs:
                cid = f"{method}-d{delta:g}-b{''.join(map(str, sizes or []))}-swap{int(swapped)}"
                if cid in done:
                    continue
                if method == "rarest" and any(r["config_id"].startswith(cid + "-L")
                                              and r["status"] == "qualified" for r in tuning):
                    continue
                log(output, "benchmark_tuning_start", seed=seed, config_id=cid, method_label=METHOD_LABELS[method])
                if sizes is None:
                    shifts = np.random.default_rng(seed).uniform(0, delta, (config["Lmax"], 2))
                    hits = first_hit_per_curve(Qt, NN, shifts, delta)
                else:
                    shifts = np.random.default_rng(seed).uniform(0, delta, (config["Lmax"], 5, 2))
                    hits = family_first_hits(Qt, NN, shifts, delta, np.asarray(sizes), swapped)
                # Rarest is a necessary-condition candidate generator and may qualify
                # earlier than intersection: test each prefix on tuning only.
                minimum = tables_for_recall(hits, config["tuning_acceptance_recall"], config["Lmax"])
                from math import factorial, prod
                a = prod(factorial(s) for s in sizes) if sizes else 5
                if sizes and swapped:
                    a = factorial(5) // a
                base = dict(config_id=cid, method=method, delta=delta, blocks=sizes,
                            swapped=swapped, index_seed=seed, stored=a,
                            probed=factorial(5) // a if sizes else 5,
                            L=minimum, tuning_acceptance_recall=config["tuning_acceptance_recall"])
                if minimum is None and method != "rarest":
                    tuning.append(dict(base, status="no tuned L in grid"))
                    atomic_json(tuning_path, tuning)
                    continue
                candidates_L = range(1, (minimum or config["Lmax"]) + 1) if method == "rarest" else [minimum]
                for L in candidates_L:
                    row = dict(base, L=L, config_id=cid + (f"-L{L}" if method == "rarest" else ""))
                    if row["config_id"] in done:
                        previous = next(r for r in tuning if r["config_id"] == row["config_id"])
                        if previous["status"] == "qualified" or previous["status"].startswith("overbudget:"):
                            break
                        continue
                    row["display_method"] = METHOD_LABELS[method]
                    try:
                        index, call, metrics = build(row)
                    except BenchmarkLimit as exc:
                        row.update(status=exc.status, **exc.metrics)
                        log(output, "configuration_overbudget", seed=seed, config_id=row["config_id"],
                            phase="tuning-build", status=exc.status, **exc.metrics)
                        tuning.append(row)
                        break
                    row.update(metrics)
                    rows, warmup_failed = run_queries(call, Qt, row["config_id"], "tuning")
                    summary = query_summary(rows, optimal_tune, config, warmup_failed=warmup_failed)
                    stats = summary["quality"]
                    row.update(status=(summary["status"] if stats is None else
                                       "qualified" if stats["recall_at_1"] >= config["tuning_acceptance_recall"] else
                                       "tuning recall below target"),
                               tuning_recall=stats["recall_at_1"] if stats else None,
                               tuning_median_seconds=summary["median_top1_seconds"],
                               tuning_p90_seconds=summary["p90_top1_seconds"],
                               tuning_p95_seconds=summary["p95_top1_seconds"],
                               tuning_quality=stats, tuning_budget_summary=summary)
                    atomic_json(seed_root / (row["config_id"] + "-tuning-queries.json"),
                                [query_record(r, q, optimal_tune[q]) for q, r in enumerate(rows)])
                    tuning.append(row)
                    del index, call
                    atomic_json(tuning_path, tuning)
                    if method == "rarest" and (row["status"] == "qualified" or row["status"].startswith("overbudget:")):
                        break
                atomic_json(tuning_path, tuning)
        qualified = [r for r in tuning if r["status"] == "qualified"]
        selections = {}
        for method in config["main_methods"] + config["supplementary_methods"]:
            selected = choose_main(tuning, method, config)
            if selected is not None:
                selections[method] = selected
        ablation = {}
        for weight in config["ablation_latency_weights"]:
            rows = [r for r in qualified if r["method"] == "blocking"]
            if rows:
                ablation[str(weight)] = min(rows, key=lambda r: (weighted_cost(r, weight, config), r["index_bytes"], r["config_id"]))
        frozen = dict(main={METHOD_LABELS[m]: row for m, row in selections.items()},
                      weight_ablation=ablation, selection_scope="tuning games only",
                      blocking_cost="index_bytes / 100MB + lambda * median_top1_seconds / 1ms",
                      baseline_selection="fastest qualified tuning median; storage used only as a tie-breaker")
        selection_path = seed_root / "frozen-selections.json"
        if selection_path.exists() and json.loads(selection_path.read_text()) != frozen:
            raise RuntimeError("Frozen tuning selection changed; no test-based retuning allowed")
        atomic_json(selection_path, frozen)
        evaluations = {}
        requested = {r["config_id"]: r for r in [*selections.values(), *ablation.values()]}
        for cid, row in requested.items():
            result_path = seed_root / (cid + "-test.json")
            if result_path.exists():
                evaluations[cid] = json.loads(result_path.read_text())
                continue
            try:
                index, call, build_metrics = build(row)
            except BenchmarkLimit as exc:
                result = dict(configuration=row, status=exc.status, quality=None,
                              median_top1_seconds=None, p90_top1_seconds=None, p95_top1_seconds=None,
                              queries=[], requested_queries=len(Qtest), not_run_queries=list(range(len(Qtest))),
                              final_build_metrics=exc.metrics)
                atomic_json(result_path, result)
                evaluations[cid] = result
                log(output, "configuration_overbudget", seed=seed, config_id=cid,
                    phase="final-build", status=exc.status, **exc.metrics)
                continue
            if index.index_bytes != row["index_bytes"]:
                raise AssertionError("Storage changed between tuning and test")
            timed, warmup_failed = run_queries(call, Qtest, cid, "test")
            summary = query_summary(timed, optimal_test, config, warmup_failed=warmup_failed)
            diagnostics = []
            # Finish every timed top1 call before additional candidate reranking.
            # A second untimed call reconstructs its deterministic candidate list.
            for q, Q in enumerate(Qtest if summary["status"] == "complete" else []):
                answer = call(Q)
                ranked = reranked_candidates(index, X, Q, answer["candidates"]) if not answer["timeout"] else []
                distinct, seen_games = [], set()
                for rank, match in enumerate(ranked, 1):
                    game = database_metadata[match["database_index"]]["game_id"]
                    if game not in seen_games:
                        distinct.append(dict(match, candidate_rank=rank, game_id=game))
                        seen_games.add(game)
                    if len(distinct) == 2:
                        break
                diagnostics.append(dict(query=q, query_game_id=test_metadata[q]["game_id"],
                                        candidate_topk=ranked[:config["topk_diagnostic_count"]],
                                        distinct_game_results=distinct,
                                        timed_top1_id=timed[q]["id"],
                                        diagnostic_top1_id=answer["id"],
                                        scope="untimed reranking of retrieved candidates, not global topK ground truth"))
            result = dict(configuration=row, **summary, final_build_metrics=build_metrics,
                          process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                          queries=[query_record(r, q, optimal_test[q]) for q, r in enumerate(timed)])
            atomic_json(seed_root / (cid + "-candidate-diagnostics.json"), diagnostics)
            atomic_json(result_path, result)
            evaluations[cid] = result
            del index, call
        main = {METHOD_LABELS[m]: evaluations[r["config_id"]] for m, r in selections.items()}
        for method in config["main_methods"] + config["supplementary_methods"]:
            main.setdefault(METHOD_LABELS[method], dict(status="no feasible tuning-qualified configuration",
                                                       quality=None, median_top1_seconds=None,
                                                       p90_top1_seconds=None, p95_top1_seconds=None))
        atomic_json(seed_root / "main-results.json", main)
        atomic_json(seed_root / "weight-ablation-results.json",
                    {w: evaluations[r["config_id"]] for w, r in ablation.items()})
        atomic_json(complete, dict(index_seed=seed, artifacts={p.name: digest(p)
                    for p in seed_root.iterdir() if p.is_file() and p != complete}))
    atomic_json(output / "complete.json", dict(index_seeds=config["index_seeds"], manifest_sha256=digest(manifest_path)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--config", default="configs/nba-full-20261012.json")
    p.add_argument("--clips-only", action="store_true", help="Explicit diagnostic mode; do not mark full preparation complete")
    p = sub.add_parser("benchmark")
    p.add_argument("--config", default="configs/nba-full-20261012.json")
    p.add_argument("--output", required=True)
    p.add_argument("--confirm-benchmark", "--confirm-newlab-after-synthetic",
                   dest="confirm_newlab_after_synthetic", action="store_true",
                   help="Confirm that this machine is available for sequential timed benchmarks")
    p.add_argument("--expected-hostname", "--newlab-hostname", dest="newlab_hostname",
                   help="Optional hostname guard")
    p.add_argument("--resource-profile", choices=("newlab", "preparation"), default="newlab",
                   help="newlab defaults to 24GB index and 48GB peak RAM, without altering frozen preparation")
    p.add_argument("--index-budget-bytes", type=int)
    p.add_argument("--peak-ram-cap-bytes", type=int)
    p.add_argument("--build-deadline-seconds", type=float)
    p.add_argument("--pin-core", type=int, default=3)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.config, args.clips_only)
    else:
        evaluate(args.config, args.output, args.confirm_newlab_after_synthetic, args.pin_core, args.newlab_hostname,
                 args.resource_profile, args.index_budget_bytes, args.peak_ram_cap_bytes, args.build_deadline_seconds)


if __name__ == "__main__":
    main()
