"""Ten-player NBA plays: the five offensive trajectories of the selected clips plus the five defenders.

The clips, windows, sampling, and orientation of the five-player corpus are reconstructed from its metadata, and
the offensive curves are checked against the stored arrays before the defensive curves are added.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import ijson
import numpy as np


def dedup_frames(raw):
    """Game timeline keyed by (period, timestamp), as in fresh.nba.game_clips."""
    frames, conflicts = {}, set()
    with raw.open("rb") as handle:
        for event in ijson.items(handle, "events.item", use_float=True):
            for moment in event["moments"]:
                if len(moment) < 6 or moment[1] is None or moment[2] is None:
                    continue
                key = (int(moment[0]), int(moment[1]))
                if key in conflicts:
                    continue
                signature = (float(moment[2]), tuple(sorted(tuple(row) for row in moment[5])))
                if key in frames:
                    old = frames[key]
                    if signature != (float(old[2]), tuple(sorted(tuple(row) for row in old[5]))):
                        conflicts.add(key)
                        del frames[key]
                else:
                    frames[key] = moment
    buckets = {}
    for key, moment in sorted(frames.items()):
        buckets.setdefault((key[0], int(float(moment[2]))), []).append(moment)
    return buckets


def clip_curves(buckets, meta, config):
    """Offense curves in the stored order and the defenders, or (None, reason)."""
    period, anchor = int(meta["quarter"]), float(meta["corrected_shot_clock_seconds"])
    team = int(meta["offense_team_id"])
    duration = config["clip_duration_seconds"]
    moments = []
    for second in range(int(anchor) - 4, int(anchor + duration) + 3):
        moments.extend(buckets.get((period, second), []))
    moments.sort(key=lambda m: m[1])
    frames = {}
    for moment in moments:
        if len(moment) < 6 or int(moment[0]) != period or moment[2] is None:
            continue
        if abs(float(moment[2]) - anchor) > duration + 2:
            continue
        offense = {int(r[1]): np.asarray(r[2:4], float) for r in moment[5] if int(r[0]) == team and int(r[1]) > 0}
        ball = [r for r in moment[5] if int(r[0]) == -1]
        if len(offense) != 5 or len(ball) != 1:
            continue
        defense = {int(r[1]): np.asarray(r[2:4], float) for r in moment[5] if int(r[0]) not in (team, -1) and int(r[1]) > 0}
        frames[float(moment[1]) / 1000] = (offense, defense)
    anchor_time = float(meta["timestamp_end"])
    start = anchor_time - duration
    times = np.array(sorted(t for t in frames if start - .04 <= t <= anchor_time + .0001))
    if len(times) < 2:
        return None, "no frames"
    sample = np.linspace(max(start, times[0]), anchor_time, config["m"])
    if abs(sample[0] - meta["timestamp_start"]) > 1e-6:
        return None, "window differs from the stored clip"
    ids = [int(p) for p in meta["player_ids"]]
    if any(set(frames[t][0]) != set(ids) for t in times):
        return None, "offense lineup differs from the stored clip"
    offense = np.stack([np.stack([np.interp(sample, times, [frames[t][0][p][d] for t in times]) for d in range(2)], axis=1)
                        for p in ids])
    defenders = sorted(frames[times[0]][1])
    if len(defenders) != 5 or any(sorted(frames[t][1]) != defenders for t in times):
        return None, "defense not five consistent players"
    defense = np.stack([np.stack([np.interp(sample, times, [frames[t][1][p][d] for t in times]) for d in range(2)], axis=1)
                        for p in defenders])
    if meta["orientation_rotated"]:
        offense, defense = np.array([94., 50.]) - offense, np.array([94., 50.]) - defense
    return (offense, defense), None


def process_archive(args):
    archive, clips, config, raw_dir = args
    work = Path(tempfile.mkdtemp(prefix="nba10-"))
    try:
        subprocess.run(["7z", "x", "-y", f"-o{work}", str(Path(raw_dir) / archive)], check=True, capture_output=True)
        raw = next(work.glob("*.json"))
        buckets = dedup_frames(raw)
        out = []
        for split, index, meta in clips:
            result, reason = clip_curves(buckets, meta, config)
            out.append((split, index, result, reason))
        return archive, out
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main(args):
    root = Path(args.data)
    config = json.loads((root / "frozen-config.json").read_text())
    stored, metadata, jobs = {}, {}, {}
    for split in ("database", "tune", "test"):
        name = "database" if split == "database" else split
        with np.load(root / f"{name}.npz", allow_pickle=False) as arrays:
            stored[split] = arrays["curves"].copy()
        metadata[split] = json.loads((root / f"{name}-metadata.json").read_text())
        for index, meta in enumerate(metadata[split]):
            jobs.setdefault(meta["archive"], []).append((split, index, meta))
    tasks = [(archive, clips, config, str(root / "raw")) for archive, clips in sorted(jobs.items())]
    if args.limit:
        tasks = tasks[:args.limit]
    results = {split: [None] * len(metadata[split]) for split in stored}
    reasons = {}
    mismatched = 0
    with ProcessPoolExecutor(args.workers) as pool:
        for done, (archive, out) in enumerate(pool.map(process_archive, tasks), 1):
            for split, index, result, reason in out:
                if result is None:
                    reasons[reason] = reasons.get(reason, 0) + 1
                    continue
                offense, defense = result
                if not np.allclose(offense, stored[split][index], atol=1e-6):
                    mismatched += 1
                    reasons["offense differs from the stored curves"] = reasons.get("offense differs from the stored curves", 0) + 1
                    continue
                results[split][index] = (offense, defense, metadata[split][index])
            print(f"{done}/{len(tasks)} {archive}", flush=True)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in ("database", "tune", "test"):
        kept = [i for i, r in enumerate(results[split]) if r is not None]
        curves = np.empty((len(kept), 10, config["m"], 2))
        for row, i in enumerate(kept):
            offense, defense, meta = results[split][i]
            seed = int(hashlib.sha256(f"{meta['game_id']}:{meta['event_id']}:both".encode()).hexdigest()[:12], 16) + config["dataset_seed"]
            curves[row] = np.concatenate([offense, defense])[np.random.default_rng(seed).permutation(10)]
        np.savez(output / f"{split}.npz", curves=curves, source_index=np.array(kept))
        counts[split] = dict(selected=len(metadata[split]), kept=len(kept))
    (output / "dataset.json").write_text(json.dumps(dict(counts=counts, dropped=reasons, k=10, m=config["m"],
                                                         source=str(root), orientation_note="shared rotation as in the five-player corpus"),
                                                    indent=1))
    print(json.dumps(dict(counts=counts, dropped=reasons), indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0)
    main(parser.parse_args())
