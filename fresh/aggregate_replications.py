"""Aggregate complete seed replications without silently dropping failed seeds."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from fresh.e1 import write_json
from fresh.regimes import approximation_summary


def paired_interval(matrix):
    means=matrix.mean(axis=1)
    rng=np.random.default_rng(1704)
    trials=means[rng.integers(0,len(means),(2000,len(means)))].mean(axis=1)
    return [float(x) for x in np.quantile(trials,[.025,.975])]


def aggregate_group(group,seeds,root,config,ablation=False):
    completed=[r for r in group if r["test_status"]=="complete"]
    base=dict(case=group[0]["case_id"],case_id=group[0]["case_id"],kind=group[0]["kind"],
              n=group[0]["n"],k=group[0]["k"],display_method=group[0].get("display_method","blocking"),
              seed_statuses=[dict(seed=r["index_seed"],status=r["test_status"],
                                  reasons=r.get("exclusion_reasons")) for r in group])
    if ablation:
        base["latency_weight"]=group[0]["latency_weight"]
    if len(group)!=len(seeds) or set(r["index_seed"] for r in group)!=set(seeds) or len(completed)!=len(seeds):
        return dict(base,test_status="incomplete across required seeds; no aggregate performance summary")
    group=sorted(group,key=lambda r:r["index_seed"])
    for field in ("dataset_sha256","queries_sha256"):
        if len({r[field] for r in group})!=1:
            raise ValueError(f"Paired replication requires identical {field}")
    recall=[]
    ratios=[]
    for r in group:
        with (root/r["case"]/"test-queries.csv").open() as handle:
            rows=sorted([row for row in csv.DictReader(handle) if row["config_id"]==r["config_id"]],
                        key=lambda row:int(row["query"]))
        if len(rows)!=config["test_queries"]:
            raise ValueError("Incomplete replication query log")
        recall.append([float(row["exact_nn"]=="True") for row in rows])
        ratios.extend(float(row["ratio"]) for row in rows if row["ratio"])
    result=dict(base,test_status="complete",seed_count=len(seeds),
                test_queries=config["test_queries"]*len(seeds),queries_per_seed=config["test_queries"],
                recall_at_1=float(np.mean([r["recall_at_1"] for r in group])),
                recall_95_interval=paired_interval(np.asarray(recall).T),
                recall_at_1_seed_minmax=[min(r["recall_at_1"] for r in group),max(r["recall_at_1"] for r in group)],
                test_target_met=all(r["test_target_met"] for r in group),
                per_seed_parameters=[dict(seed=r["index_seed"],delta=r.get("delta"),L=r.get("L"),
                                          blocks=r.get("blocks"),swapped=r.get("swapped"),
                                          index_MB=r["index_MB"],median_ms=r["median_ms"],
                                          recall_at_1=r["recall_at_1"]) for r in group],
                c_success={c:float(np.mean([r["c_success"][c] for r in group])) for c in group[0]["c_success"]},
                blocks=group[0].get("blocks"),blocks_vary_by_seed=len({str(r.get("blocks")) for r in group})>1,
                raw_curve_bytes=group[0]["raw_curve_bytes"],mean_relevant_sets=group[0]["mean_relevant_sets"])
    for metric in ("median_ms","p95_ms","mean_ms","index_MB","index_bytes","coverage","empty_rate",
                   "mean_candidates","mean_verified","mean_postings","mean_probes","mean_nn_distance",
                   "mean_returned_distance"):
        values=[r[metric] for r in group if r.get(metric) is not None]
        if metric in ("median_ms","p95_ms","mean_ms","index_MB","index_bytes"):
            result[metric]=float(np.median(values)) if values else None
        elif metric=="mean_returned_distance":
            answered=[(r[metric],r["coverage"]*config["test_queries"]) for r in group if r.get(metric) is not None]
            result[metric]=float(np.average([v for v,_ in answered],weights=[n for _,n in answered])) if answered else None
        else:
            result[metric]=float(np.mean(values)) if values else None
        if values:
            result[metric+"_seed_minmax"]=[min(values),max(values)]
    result.update(approximation_summary(ratios))
    return result


def aggregate(root,plot=False):
    report=json.loads((root/"summary.json").read_text())
    config=report["config"]
    seeds=config["index_seeds"]
    main,ablation=[],[]
    for source,result,is_ablation in ((report["main"],main,False),(report["ablation"],ablation,True)):
        groups={}
        for row in source:
            key=(row["case_id"],row["latency_weight"] if is_ablation else row["display_method"])
            groups.setdefault(key,[]).append(row)
        for _,group in sorted(groups.items()):
            result.append(aggregate_group(group,seeds,root,config,is_ablation))
    output=dict(config=config,manifest=report["manifest"],main=main,ablation=ablation,
                notes=["Only all-required-seed groups are summarized; partial seed groups remain explicit.",
                       "Latency/storage use median per-seed summaries with seed min-max ranges.",
                       "Coverage/empty rates and work counts average all equal-sized seed query sets; returned-distance means weight the answered-query counts.",
                       "Recall interval bootstraps query IDs while keeping the same queries' seed outcomes together.",
                       "Three index seeds share a fixed synthetic dataset; dataset generalization is not tested.",
                       "Different tuning choices per seed are retained; no held-out retuning."])
    write_json(root/"replicated-summary.json",output)
    with (root/"replicated-main-table.csv").open("w",newline="") as handle:
        columns=["case","display_method","n","k","test_status","index_MB","median_ms","p95_ms",
                 "recall_at_1","mean_ratio","p90_ratio","empty_rate","test_target_met"]
        writer=csv.DictWriter(handle,fieldnames=columns,extrasaction="ignore")
        writer.writeheader()
        writer.writerows(main)
    if plot:
        from fresh.plot_poc import figures
        figures(output,root/"replicated-figures")
    print(f"Aggregated {len(main)} main case/method groups; "
          f"{sum(r['test_status']=='complete' for r in main)} complete across {len(seeds)} seeds")
    return output


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("results",type=Path)
    parser.add_argument("--plot",action="store_true")
    args=parser.parse_args()
    aggregate(args.results,args.plot)
