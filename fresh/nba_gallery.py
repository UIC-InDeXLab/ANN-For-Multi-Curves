"""Illustrative NBA screening with a frozen index; retain every query outcome."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from fresh.e1 import write_json
from fresh.nba import game_clips,render
from fresh.poc import Joint
from fresh.regimes import approximation_summary
from fresh.tuple_quality import exact_distance


def quality_metrics(outcomes):
    if not outcomes:
        raise ValueError("No completed NBA queries to summarize")
    ratios=np.array([float(r["results"][0]["approximation_ratio"]) for r in outcomes if r["results"]])
    result=dict(query_count=len(outcomes),exact_nn_count=sum(r["found_exact_nearest"] for r in outcomes),
                recall_at_1=sum(r["found_exact_nearest"] for r in outcomes)/len(outcomes),
                empty_query_count=sum(not r["results"] for r in outcomes),
                coverage=sum(bool(r["results"]) for r in outcomes)/len(outcomes),
                c_success={str(c):sum(bool(r["results"]) and float(r["results"][0]["approximation_ratio"])<=c
                                     for r in outcomes)/len(outcomes) for c in (1.,1.05,1.1,1.25,1.5,2.)},
                **approximation_summary(ratios.tolist()))
    result["median_ratio"]=float(np.median(ratios)) if len(ratios) else None
    result["max_ratio"]=float(ratios.max()) if len(ratios) and np.isfinite(ratios.max()) else "infinity" if len(ratios) else None
    return result


def main(args):
    source=args.source.resolve();output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    config=json.loads((source/"config.json").read_text())
    previous=json.loads((source/"results.json").read_text())
    metadata=json.loads((source/"clip-metadata.json").read_text())
    with np.load(source/"clips.npz") as saved:
        X=saved["database"];balls=saved["balls"]
    queries,audit=game_clips(config,config["query_archive"],Path(config["data_directory"]).resolve())
    query_team=next(row["query"]["offense_team_id"] for row in previous["outcomes"]
                    if row["query"]["event_id"]==config["primary_display_event"])
    selected=previous["selected_index"]
    index=Joint(X,selected["delta"],selected["L"],config["index_seed"],selected["blocks"],False)
    write_json(output/"selection-policy.json",dict(
        source=str(source),frozen_index=selected,retuned=False,query_game=config["query_archive"],
        selection="Smallest returned distance among moving query/result pairs (both mean path length >=8ft), query offense team fixed to the primary query's team, whose top-ranked result and query are both labeled 3PT; illustration only",
        query_team_id=query_team,
        all_valid_queries_evaluated=len(queries),not_a_random_representative_example=True,
        labels_not_used_for_indexing_or_ranking=True,primary_predeclared_query_preserved=str(source/"nba-query-and-results.pdf")))
    outcomes=[];eligible=[];eligible_queries={}
    for q,(Q,ball,info) in enumerate(queries):
        response=index.query(X,Q,config)
        if response["timeout"]:
            raise RuntimeError("Gallery query timed out; no complete-gallery claim")
        candidates=index.base.cand[:response["candidates"]].copy()
        ranked=sorted((float(exact_distance(X[p],Q)),int(p)) for p in candidates)
        exact=np.array([exact_distance(P,Q) for P in X])
        optimum=float(exact.min())
        results=[dict(rank=i+1,database_index=p,distance_feet=value,
                      approximation_ratio=value/optimum if optimum>0 else 1 if value==0 else "infinity",
                      clip=metadata[p]) for i,(value,p) in enumerate(ranked)]
        result=dict(query=info,candidates=len(candidates),exact_optimal_distance_feet=optimum,
                    found_exact_nearest=bool(ranked and abs(ranked[0][0]-optimum)<=1e-8),
                    results=results[:5])
        outcomes.append(result)
        write_json(output/"all-query-results.json",outcomes)
        if results and info["offense_team_id"]==query_team and "3PT" in info["shot_description"] and "3PT" in results[0]["clip"]["shot_description"] and \
                info["mean_path_length_feet"]>=8 and results[0]["clip"]["mean_path_length_feet"]>=8:
            eligible.append((results[0]["distance_feet"],q,results))
            eligible_queries[q]=(Q,ball,info)
        if q%20==0:print(f"NBA gallery {q+1}/{len(queries)}",flush=True)
    if not eligible:
        raise RuntimeError("No moving three-point example matched the illustrative policy")
    _,q,ranked=min(eligible,key=lambda item:(item[0],item[1]))
    chosen=eligible_queries[q]
    display=[ranked[0]]
    other=next((r for r in ranked[1:] if r["clip"]["game_id"]!=ranked[0]["clip"]["game_id"]),None)
    if other is not None:display.append(other)
    elif len(ranked)>1:display.append(ranked[1])
    render(output,chosen,display,X,balls,metadata)
    report=dict(source_results=str(source),selected_query=chosen[2],displayed_results=display,
                index=selected,database_clips=len(X),query_count=len(queries),
                query_recall_at_1=sum(r["found_exact_nearest"] for r in outcomes)/len(outcomes),
                empty_query_count=sum(not r["results"] for r in outcomes),
                illustrative_eligible_query_count=len(eligible),
                chosen_by_policy=True,no_test_retuning=True,
                semantic_validation="No expert tactical judgments; this figure shows geometric matches and action-label context only")
    report["quality_metrics"]=quality_metrics(outcomes)
    write_json(output/"quality-metrics.json",report["quality_metrics"])
    write_json(output/"results.json",report)
    write_json(output/"completion.json",dict(completed=time.time()))
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--source",type=Path);parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--summarize-only",action="store_true")
    args=parser.parse_args()
    if args.summarize_only:
        rows=json.loads((args.output/"all-query-results.json").read_text())
        stats=quality_metrics(rows);write_json(args.output/"quality-metrics.json",stats)
        report=json.loads((args.output/"results.json").read_text());report["quality_metrics"]=stats
        write_json(args.output/"results.json",report)
    else:
        if args.source is None:parser.error("--source is required for a new NBA gallery run")
        main(args)
