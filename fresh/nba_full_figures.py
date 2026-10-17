"""Transparent original court diagrams from untimed full-corpus candidate diagnostics.

Run after frozen test evaluation, e.g.
  python -m fresh.nba_full_figures --data data/nba-full-20261012 \
    --results RESULTS/seed-2301 --method blocking

Policies are frozen before retrieval in the full-corpus config. Primary is the
hash-first test query regardless of outcome. Median/worst quality include empty
answers; best-optimal-distance is explicitly selected, not a representative case.
Only actually retrieved candidates are shown. Distances are geometric, not claims
about tactical intent, and topK diagnostic work is outside all top1 timers.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fresh.nba import clock_text, court, optimal_mapping
from fresh.nba_full import atomic_json, digest


def illustrative_queries(rows, metadata, config):
    if len(rows) != len(metadata):
        raise ValueError("Illustration policy requires every test query outcome")
    salt = config["split_salt"]
    primary = min(range(len(rows)), key=lambda i: hashlib.sha256(
        f"{salt}:{metadata[i]['game_id']}:{metadata[i]['event_id']}".encode()).hexdigest())
    def quality(i):
        row = rows[i]
        if row["id"] < 0:
            return (1, float("inf"), i)
        best, value = row["optimal_feet"], row["distance"]
        ratio = value / best if best else (1 if value == 0 else float("inf"))
        return (0, ratio, i)
    ordered = sorted(range(len(rows)), key=quality)
    return dict(primary=primary,
                median_quality=ordered[(len(ordered) - 1) // 2],
                difficult=ordered[-1],
                selected_best_optimal_distance=min(range(len(rows)), key=lambda i: (rows[i]["optimal_feet"], i)))


def render_example(destination, Q, query_ball, query_meta, matches, X, balls, metadata, label, optimum):
    colors = ["#007f5f", "#277da1", "#b5179e", "#e76f51", "#7a6c15"]
    panels = [(Q, query_ball, query_meta, "Query", None)]
    for match in matches:
        p = match["database_index"]
        order, pairwise, distance = optimal_mapping(Q, X[p])
        if abs(distance - match["distance_feet"]) > 1e-7:
            raise AssertionError("Illustration distance disagrees with exact recovered matching")
        info = dict(metadata[p])
        info["player_names"] = [info["player_names"][i] for i in order]
        info["jerseys"] = [info["jerseys"][i] for i in order]
        match.update(query_to_result_player_ids=[metadata[p]["player_ids"][i] for i in order],
                     pairwise_frechet_feet=pairwise.tolist())
        panels.append((X[p, order], balls[p], info, f"Retrieved candidate rank {match['candidate_rank']}", match))
    points = np.concatenate([np.concatenate([curves.reshape(-1, 2), ball]) for curves, ball, *rest in panels])
    limits = ([min(-2., points[:, 0].min() - 2), max(49., points[:, 0].max() + 2)],
              [min(-2., points[:, 1].min() - 2), max(52., points[:, 1].max() + 2)])
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 9), squeeze=False)
    for ax, (curves, ball, info, heading, match) in zip(axes[0], panels):
        court(ax, limits)
        for i, curve in enumerate(curves):
            ax.plot(curve[:, 0], curve[:, 1], color=colors[i], linewidth=2)
            ax.scatter(*curve[0], color=colors[i], marker="o", s=45)
            ax.scatter(*curve[-1], color=colors[i], marker="X", s=55)
            ax.text(.02, -.15 - .045 * i, f"{i + 1}: {info['player_names'][i]} #{info['jerseys'][i]}",
                    color=colors[i], fontsize=9, transform=ax.transAxes)
        ax.plot(ball[:, 0], ball[:, 1], "--", color="#ff9f1c", label="Ball (context only)")
        value = "" if match is None else f"\nGeometric distance: {match['distance_feet']:.2f} ft"
        ax.set_title(f"{heading}: {info['matchup']}\n{info['date']}; game {info['game_id']}; event {info['event_id']}\n"
                     f"Q{info['quarter']} tracking clock {clock_text(info['clock_start'])}–{clock_text(info['clock_end'])}\n"
                     f"Official event clock {info['official_event_clock']}" + value, fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle(f"{label.replace('_', ' ')}; exact optimum {optimum:.2f} ft\n"
                 "Five-player geometric similarity, not tactical semantics; ball excluded from metric.\n"
                 "Circle=start, X=end; colors follow exact recovered player correspondence.", fontsize=10)
    fig.tight_layout(rect=(0, .14, 1, .9))
    fig.savefig(destination.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(destination.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--method", default="blocking")
    args = parser.parse_args()
    data, results = args.data.resolve(), args.results.resolve()
    config = json.loads((data / "frozen-config.json").read_text())
    manifest = json.loads((results.parent / "manifest.json").read_text())
    if manifest["dataset_sha256"] != digest(data / "dataset.json"):
        raise RuntimeError("Figure data differs from frozen evaluation")
    selections = json.loads((results / "frozen-selections.json").read_text())
    selection = selections["main"].get(args.method)
    if selection is None:
        raise RuntimeError(f"No tuning-qualified {args.method}; do not silently substitute a favorable method")
    cid = selection["config_id"]
    evaluation = json.loads((results / (cid + "-test.json")).read_text())
    if evaluation.get("status") != "complete":
        raise RuntimeError("Overbudget or incomplete configurations have no representative aggregate illustrations")
    diagnostics = json.loads((results / (cid + "-candidate-diagnostics.json")).read_text())
    metadata = {split: json.loads((data / f"{split}-metadata.json").read_text()) for split in ("database", "test")}
    arrays = {}
    for split in metadata:
        with np.load(data / f"{split}.npz", allow_pickle=False) as handle:
            arrays[split] = (handle["curves"], handle["balls"])
    destination = results / "illustrations" / args.method
    destination.mkdir(parents=True, exist_ok=True)
    chosen = illustrative_queries(evaluation["queries"], metadata["test"], config)
    illustrated = []
    for label, q in chosen.items():
        row = evaluation["queries"][q]
        diagnostic = diagnostics[q]
        if row["id"] < 0:
            matches = []
        else:
            matches = [dict(match) for match in diagnostic["distinct_game_results"]]
            if not matches or matches[0]["database_index"] != row["id"]:
                raise RuntimeError("Untimed diagnostic top1 differs from timed top1; refusing misleading illustration")
        if len({m["game_id"] for m in matches}) != len(matches):
            raise AssertionError("Illustration results must be from distinct database games")
        render_example(destination / label, arrays["test"][0][q], arrays["test"][1][q],
                       metadata["test"][q], matches, *arrays["database"], metadata["database"],
                       label + (" (empty retrieval)" if row["id"] < 0 else ""), row["optimal_feet"])
        illustrated.append(dict(policy=label, query=q, query_metadata=metadata["test"][q],
                                empty=row["id"] < 0, optimal_feet=row["optimal_feet"], results=matches))
    atomic_json(destination / "examples.json", dict(frozen_policy=config["illustrative_policy"],
                                                   method=args.method, examples=illustrated))


if __name__ == "__main__":
    main()
