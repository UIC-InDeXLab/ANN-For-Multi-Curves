"""Court diagram of a ten-player NBA retrieval: a held-out play and the play that blocking returns for it.

Both plays contain the five offensive and the five defensive trajectories. Colors and numbers show the recovered
correspondence; solid lines mark offensive and dashed lines defensive players, which the retrieval does not see.
Example selection (deterministic): among test queries whose returned play is the exact nearest neighbor, comes from a
different game and a different offensive team, keep those with an optimal distance at or below the median of that set,
and take the one that matches the most offensive players to offensive players, breaking ties by the longest mean trajectory.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fresh.nba import court, optimal_mapping
from fresh.nba_paper_examples import configure_paper_font

COLORS = ["#007f5f", "#277da1", "#b5179e", "#e76f51", "#7a6c15", "#d62828", "#6a4c93", "#2a9d8f", "#f4a261", "#555555"]


def offense_flags(ten, five):
    """True for the trajectories of a ten-player play that also occur in the stored five-player offense."""
    flags = np.zeros(len(ten), bool)
    for curve in five:
        distance = np.abs(ten - curve).reshape(len(ten), -1).max(axis=1)
        flags[int(np.argmin(distance))] = True
    return flags


def read_returned(results, config_id):
    rows = {}
    with (Path(results) / "test-queries.csv").open() as handle:
        for row in csv.DictReader(handle):
            if row["config_id"] == config_id and row["phase"] == "test":
                rows[int(row["query"])] = (int(row["id"]), float(row["distance"]), float(row["nn_distance"]))
    return rows


def choose(rows, X, Q, metadata_db, metadata_test, flags_db, flags_test):
    """Apply the selection rule of the module docstring to one seed's blocking answers."""
    exact = {q: r for q, r in rows.items() if r[0] >= 0 and abs(r[1] - r[2]) <= 1e-9}
    different = {q: r for q, r in exact.items()
                 if metadata_db[r[0]]["game_id"] != metadata_test[q]["game_id"]
                 and metadata_db[r[0]]["offense_team_id"] != metadata_test[q]["offense_team_id"]}
    if not different:
        raise ValueError("No exact answer comes from a different game and team")
    cutoff = float(np.median([r[1] for r in different.values()]))
    pool = [q for q, r in different.items() if r[1] <= cutoff]
    length = lambda q: float(np.linalg.norm(np.diff(Q[q], axis=1), axis=2).sum(axis=1).mean())

    def agreement(q):
        order, _, _ = optimal_mapping(Q[q], X[rows[q][0]])
        return int(sum(flags_test[q][i] and flags_db[rows[q][0]][order[i]] for i in range(len(order))))

    return max(pool, key=lambda q: (agreement(q), length(q), -q))


def draw(ax, curves, order, flags, title, subtitle):
    court(ax, ((0, 62), (-1, 51)))
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    for slot in range(len(curves)):
        curve = curves[order[slot]]
        offense = flags[order[slot]]
        color = COLORS[slot]
        ax.plot(curve[:, 0], curve[:, 1], color=color, linewidth=1.5 if offense else 1.0,
                linestyle="-" if offense else (0, (3, 1.6)), zorder=3)
        ax.plot(*curve[0], "o", color=color, markersize=3.6, zorder=4)
        ax.plot(*curve[-1], "x", color=color, markersize=3.6, markeredgewidth=1.0, zorder=4)
        ax.annotate(str(slot + 1), curve[0], xytext=(2.2, 2.2), textcoords="offset points", fontsize=6, color=color)
    ax.set_title(title, pad=2, fontsize=8)
    ax.text(0.5, -0.02, subtitle, transform=ax.transAxes, ha="center", va="top", fontsize=6.5)


def render(destination, Q, flags_q, P, flags_p, distance, query_info, result_info):
    configure_paper_font()
    # Number the slots by the query; the returned curves are permuted onto the query's slots.
    order_q = np.arange(len(Q))
    order_p, pairwise, optimum = optimal_mapping(Q, P)
    if abs(optimum - distance) > 1e-7:
        raise AssertionError("The recovered matching disagrees with the measured distance")
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.75))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.12, wspace=0.04)
    draw(axes[0], Q, order_q, flags_q, "Query",
         f"{query_info['offense_team_name']}, {query_info['matchup']}, {query_info['date']}")
    draw(axes[1], P, order_p, flags_p, f"Returned nearest neighbor ({distance:.2f} ft)",
         f"{result_info['offense_team_name']}, {result_info['matchup']}, {result_info['date']}")
    fig.savefig(destination, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    pairs = [(int(flags_q[i]), int(flags_p[order_p[i]])) for i in range(len(Q))]
    return dict(optimal_feet=optimum, pairwise_feet=[float(v) for v in pairwise],
                offense_matched_to_offense=sum(1 for a, b in pairs if a and b),
                offense_matched_to_defense=sum(1 for a, b in pairs if a and not b))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="../data/nba-both-20261016")
    parser.add_argument("--five", default="../data/nba-full-20261012")
    parser.add_argument("--results", required=True, help="seed folder with test-queries.csv and main-results.json")
    parser.add_argument("--destination", required=True)
    parser.add_argument("--query", type=int, help="override the selection rule")
    args = parser.parse_args()
    data, five = Path(args.data), Path(args.five)
    X = np.load(data / "database.npz")["curves"]
    Q = np.load(data / "test.npz")["curves"]
    metadata_db = json.loads((five / "database-metadata.json").read_text())
    metadata_test = json.loads((five / "test-metadata.json").read_text())
    main_rows = json.loads((Path(args.results) / "main-results.json").read_text())
    config_id = next(r["config_id"] for r in main_rows if r["display_method"] == "blocking")
    rows = read_returned(args.results, config_id)
    five_db = np.load(five / "database.npz")["curves"]
    five_test = np.load(five / "test.npz")["curves"]
    if args.query is not None:
        q = args.query
    else:
        needed = {r[0] for r in rows.values() if r[0] >= 0}
        flags_db = {p: offense_flags(X[p], five_db[p]) for p in needed}
        flags_test = {q: offense_flags(Q[q], five_test[q]) for q in rows}
        q = choose(rows, X, Q, metadata_db, metadata_test, flags_db, flags_test)
    p, distance, optimum = rows[q]
    info = render(args.destination, Q[q], offense_flags(Q[q], five_test[q]), X[p], offense_flags(X[p], five_db[p]),
                  distance, metadata_test[q], metadata_db[p])
    info.update(query=q, database_index=p, config_id=config_id, query_game=metadata_test[q]["game_id"],
                result_game=metadata_db[p]["game_id"], query_clock=[metadata_test[q]["clock_start"], metadata_test[q]["clock_end"]],
                result_clock=[metadata_db[p]["clock_start"], metadata_db[p]["clock_end"]])
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
