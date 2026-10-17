"""Additional preregistered moving-play illustration; no changes to indexing or test scores."""
import argparse
import json
from pathlib import Path

import numpy as np

from fresh.e1 import write_json
from fresh.nba import render
from fresh.nba_full import digest
from fresh.nba_full_figures import render_example


def choose(rows,diagnostics,queries,database,policy):
    if len(rows)!=len(diagnostics) or len(rows)!=len(queries):
        raise ValueError("Motion illustration requires every frozen query outcome")
    eligible=[]
    for i,row in enumerate(rows):
        if row["id"]<0:
            continue
        if queries[i]["mean_path_length_feet"]<policy["minimum_query_mean_path_length_feet"]:
            continue
        p=row["id"]
        if database[p]["mean_path_length_feet"]<policy["minimum_result_mean_path_length_feet"]:
            continue
        actual=diagnostics[i]["distinct_game_results"]
        if not actual or actual[0]["database_index"]!=p:
            raise ValueError("The illustrated top result differs from the timed retrieved tuple")
        if any(match["game_id"]==queries[i]["game_id"] for match in actual):
            raise ValueError("Illustration query game leaks into the retrieved database")
        eligible.append((row["distance"],i))
    if not eligible:
        raise ValueError("No eligible moving-play example; do not substitute a static or synthetic match")
    _,selected=min(eligible)
    return selected,len(eligible)


def main(args):
    policy=json.loads(args.policy.read_text())
    dataset=args.data.resolve();results=args.results.resolve()
    if not (results/"complete.json").exists():
        raise ValueError("Full NBA benchmark is not complete")
    manifest=json.loads((results/"manifest.json").read_text())
    if manifest["dataset_sha256"]!=digest(dataset/"dataset.json"):
        raise ValueError("Example arrays differ from the evaluated dataset")
    seed=results/f"seed-{policy['index_seed']}"
    selections=json.loads((seed/"frozen-selections.json").read_text())
    selected=selections["main"].get(policy["method"])
    if selected is None:
        raise ValueError("No tuning-qualified blocking index; do not choose another seed/method after testing")
    cid=selected["config_id"]
    records=json.loads((seed/(cid+"-test.json")).read_text())
    rows=records.get("queries",[])
    if records.get("status") not in (None,"complete") or not rows or any(r["timeout"] for r in rows):
        raise ValueError("Illustration method did not completely evaluate its query set")
    diagnostics=json.loads((seed/(cid+"-candidate-diagnostics.json")).read_text())
    qmeta=json.loads((dataset/"test-metadata.json").read_text())
    dmeta=json.loads((dataset/"database-metadata.json").read_text())
    q,count=choose(rows,diagnostics,qmeta,dmeta,policy)
    with np.load(dataset/"database.npz",allow_pickle=False) as arrays:
        X,balls=arrays["curves"],arrays["balls"]
    with np.load(dataset/"test.npz",allow_pickle=False) as arrays:
        Q,qballs=arrays["curves"],arrays["balls"]
    destination=seed/"illustrations"/policy["method"]
    destination.mkdir(parents=True,exist_ok=True)
    matches=[dict(r) for r in diagnostics[q]["distinct_game_results"]]
    render_example(destination/"selected_moving_play",Q[q],qballs[q],qmeta[q],matches,X,balls,dmeta,
                   "Selected moving-play example; not representative",rows[q]["optimal_feet"])
    if args.animation:
        animated=destination/"moving-animation"
        animated.mkdir(exist_ok=True)
        optimum=rows[q]["optimal_feet"]
        display=[dict(match,rank=match["candidate_rank"],
                      approximation_ratio=match["distance_feet"]/optimum if optimum else
                      1.0 if match["distance_feet"]==0 else "infinity") for match in matches]
        render(animated,(Q[q],qballs[q],qmeta[q]),display,X,balls,dmeta)
    write_json(destination/"motion-example.json",dict(
        policy=policy,policy_sha256=digest(args.policy),query=q,eligible_queries=count,
        query_metadata=qmeta[q],retrieved_results=matches,optimal_feet=rows[q]["optimal_feet"],
        retrieved_metadata=[dict(candidate_rank=match["candidate_rank"],
                                 database_index=match["database_index"],
                                 metadata=dmeta[match["database_index"]]) for match in matches],
        chosen_query_distance=rows[q]["distance"],changed_test_outcomes=False))
    print(f"Illustrated actual moving query {q} out of {count} eligible queries",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--results",type=Path,required=True)
    parser.add_argument("--policy",type=Path,default=Path("configs/nba-illustration-policy-20261013.json"))
    parser.add_argument("--animation",action="store_true",
                        help="Also render a GIF and start/middle/end panels for the same selected retrieval")
    main(parser.parse_args())
