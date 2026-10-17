"""Aggregate the full game-split benchmark, distinguishing metric accuracy from semantics."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fresh.e1 import write_json
from fresh.aggregate_replications import paired_interval
from fresh.regimes import approximation_summary
from fresh.nba_full import digest
from fresh.plot_styles import method_style


LABELS={"blocking":"Blocking","per_curve-intersection":"Intersection",
        "probe_all":"Probe-all","store_all":"Store-all"}
HATCHES={"blocking":"/","per_curve-intersection":"\\","probe_all":"x","store_all":"."}


def presentation_rows(rows):
    """Select displayed methods without changing the frozen measurements."""
    return sorted((r for r in rows if normalized_name(r["method"]) in LABELS),
                  key=lambda r:list(LABELS).index(normalized_name(r["method"])))


def plot_names(rows):
    return [LABELS[normalized_name(r["method"])] for r in rows]


def method_bars(ax,x,values,rows):
    bars=ax.bar(x,values,color=[method_style(normalized_name(r["method"]))["color"] for r in rows],
                edgecolor="black",linewidth=.55)
    for bar,row in zip(bars,rows):
        bar.set_hatch(HATCHES[normalized_name(row["method"])])
    return bars


def normalized_name(name):
    return {"one_block":"per_curve-intersection","rarest":"per_curve-rarest"}.get(name,name)


def game_interval(matrix,games):
    if len(games)!=len(matrix):
        raise ValueError("Game labels do not match the paired query outcomes")
    labels=sorted(set(games))
    means=matrix.mean(axis=1)
    totals=np.array([means[np.array(games)==label].sum() for label in labels])
    counts=np.array([sum(game==label for game in games) for label in labels])
    rng=np.random.default_rng(2304)
    samples=rng.integers(0,len(labels),(2000,len(labels)))
    bootstrap=totals[samples].sum(axis=1)/counts[samples].sum(axis=1)
    return [float(x) for x in np.quantile(bootstrap,[.025,.975])]


def combine(entries,seeds,games=None):
    base=dict(method=normalized_name(entries[0]["method"]),
              seed_statuses=[dict(seed=r["seed"],status=r["status"]) for r in entries])
    if len(entries)!=len(seeds) or {e["seed"] for e in entries}!=set(seeds) or any(e["status"]!="complete" for e in entries):
        return dict(base,status="not completely evaluated across required seeds")
    entries=sorted(entries,key=lambda r:r["seed"])
    matrix=[];ratios=[];feet=[];excess=[];times=[]
    for entry in entries:
        rows=entry["queries"]
        optimum=np.array([row["optimal_feet"] for row in rows])
        matrix.append([float(row["id"]>=0 and row["distance"]<=best+1e-8) for row,best in zip(rows,optimum)])
        values=[]
        for row,best in zip(rows,optimum):
            values.append(row["seconds"])
            if row["id"]<0:
                continue
            value=row["distance"]
            feet.append(value);excess.append(max(0,value-best))
            ratios.append(value/best if best else 1 if value==0 else np.inf)
        times.append(values)
    if len({len(t) for t in times})!=1:
        raise ValueError("Different query counts across seed replications")
    reference=[(r["query"],r["optimal_feet"]) for r in entries[0]["queries"]]
    if any([(r["query"],r["optimal_feet"]) for r in entry["queries"]]!=reference for entry in entries[1:]):
        raise ValueError("NBA accuracy replication requires identical query identities and exact optima")
    recall=np.asarray(matrix).T
    result=dict(base,status="complete",queries_per_seed=recall.shape[0],index_seeds=seeds,
                exact_recall=float(recall.mean()),
                recall_query_interval=paired_interval(recall) if games is None else game_interval(recall,games),
                recall_interval_unit="query" if games is None else "held-out game",
                recall_seed_range=[float(recall.mean(axis=0).min()),float(recall.mean(axis=0).max())],
                median_ms=float(np.median([np.median(t)*1000 for t in times])),
                p95_ms=float(np.median([np.quantile(t,.95)*1000 for t in times])),
                median_seed_ms=[float(np.median(t)*1000) for t in times],
                index_MB=float(np.median([e["configuration"]["index_bytes"]/1e6 for e in entries])),
                index_seed_MB=[e["configuration"]["index_bytes"]/1e6 for e in entries],
                coverage=len(feet)/(recall.shape[0]*len(seeds)),
                mean_returned_feet=float(np.mean(feet)) if feet else None,
                mean_excess_feet=float(np.mean(excess)) if excess else None,
                p90_excess_feet=float(np.quantile(excess,.9)) if excess else None,
                configurations=[dict(seed=e["seed"],**e["configuration"]) for e in entries],
                **approximation_summary(ratios))
    first=entries[0]["queries"]
    result["mean_optimal_feet"]=float(np.mean([r["optimal_feet"] for r in first]))
    factors=[1,1.05,1.1,1.25,1.5,2]
    result["c_success"]={
        str(float(c)):sum(row["id"]>=0 and row["distance"]<=c*row["optimal_feet"]+1e-8
                         for e in entries for row in e["queries"])/(len(first)*len(seeds))
        for c in factors}
    return result


def comparison_figure(completed):
    completed=presentation_rows(completed)
    fig,axes=plt.subplots(1,3,figsize=(12,4.7))
    x=np.arange(len(completed))
    names=plot_names(completed)
    for ax,metric,title in ((axes[0],"median_ms","Median query latency (ms)"),
                            (axes[1],"index_MB","Index array storage (MB)"),
                            (axes[2],"exact_recall","Exact nearest-neighbor recall (%)")):
        values=np.array([r[metric]*(100 if metric=="exact_recall" else 1) for r in completed])
        method_bars(ax,x,values,completed)
        ax.set_xticks(x,names,fontsize=9);ax.set_ylabel(title,fontsize=12)
        ax.tick_params(axis="y",labelsize=11);ax.grid(axis="y",alpha=.25)
        if not completed:
            ax.text(.5,.5,"No complete qualifying method/seed group",ha="center",va="center",
                    transform=ax.transAxes)
        elif metric=="exact_recall":
            intervals=np.array([r["recall_query_interval"] for r in completed])*100
            ax.vlines(x,intervals[:,0],intervals[:,1],color="black",linewidth=.7)
            ax.hlines(intervals[:,0],x-.08,x+.08,color="black",linewidth=.7)
            ax.hlines(intervals[:,1],x-.08,x+.08,color="black",linewidth=.7)
            ax.set_ylim(0,100);ax.axhline(90,color="black",linestyle=":",linewidth=.7)
        else:
            field="median_seed_ms" if metric=="median_ms" else "index_seed_MB"
            ranges=np.array([[min(r[field]),max(r[field])] for r in completed])
            ax.errorbar(x,values,yerr=np.array([values-ranges[:,0],ranges[:,1]-values]),
                        fmt="none",ecolor="black",capsize=3,elinewidth=.7,capthick=.7)
            if metric=="index_MB":
                ax.set_yscale("log")
            else:
                ax.set_ylim(bottom=0)
    fig.tight_layout()
    return fig


def quality_figure(completed):
    completed=presentation_rows(completed)
    x=np.arange(len(completed));names=plot_names(completed)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    factors=[1,1.05,1.1,1.25,1.5,2]
    for row in completed:
        method=normalized_name(row["method"])
        axes[0].plot(range(len(factors)),[100*row["c_success"][str(float(c))] for c in factors],
                     label=LABELS[method],**method_style(method))
    axes[0].set_xticks(range(6),["1","1.05","1.1","1.25","1.5","2"])
    axes[0].set_xlabel("Allowed distance factor",fontsize=11)
    axes[0].set_ylabel("All queries answered within factor (%)",fontsize=11)
    method_bars(axes[1],x,[r["mean_excess_feet"] for r in completed],completed)
    axes[1].set_xticks(x,names,fontsize=9)
    axes[1].set_ylabel("Mean excess distance (feet), answered queries",fontsize=11)
    for ax in axes:
        ax.tick_params(axis="y",labelsize=10)
        ax.grid(axis="y",alpha=.25)
    if completed:
        axes[0].legend(fontsize=9)
    else:
        for ax in axes:
            ax.text(.5,.5,"No complete qualifying method/seed group",ha="center",va="center",
                    transform=ax.transAxes)
    fig.tight_layout()
    return fig


def render_frozen_figures(root,data,output):
    """Render only the two quantitative PDFs from an unchanged audited report."""
    if not all(path.exists() for path in (root/"complete.json",root/"analysis-complete.json",
                                         data/"preparation-complete.json")):
        raise ValueError("Full corpus preparation, timed NBA evaluation, or result analysis is incomplete")
    source=root/"replicated-summary.json"
    receipt=json.loads((root/"analysis-complete.json").read_text())
    if receipt["summary_sha256"]!=digest(source) or receipt["dataset_sha256"]!=digest(data/"dataset.json"):
        raise ValueError("NBA analysis receipt does not match the current summary and corpus")
    report=json.loads(source.read_text())
    completed=[r for r in report["main"] if r["status"]=="complete"]
    for name,plot in (("nba-full-comparison.pdf",comparison_figure),
                      ("nba-full-quality.pdf",quality_figure)):
        fig=plot(completed)
        try:
            fig.savefig(output/name,bbox_inches="tight")
        finally:
            plt.close(fig)


def summarize(root,data):
    marker=root/"complete.json"
    if not marker.exists():
        raise ValueError("Full NBA evaluation has not completed")
    config=json.loads((data/"frozen-config.json").read_text())
    dataset=json.loads((data/"dataset.json").read_text())
    manifest=json.loads((root/"manifest.json").read_text())
    games=[row["game_id"] for row in json.loads((data/"test-metadata.json").read_text())]
    if json.loads(marker.read_text())["manifest_sha256"]!=digest(root/"manifest.json"):
        raise ValueError("NBA completion marker does not match the evaluation manifest")
    if manifest["dataset_sha256"]!=digest(data/"dataset.json"):
        raise ValueError("NBA dataset differs from the evaluated corpus")
    groups=defaultdict(list);ablation=defaultdict(list)
    for seed in config["index_seeds"]:
        folder=root/f"seed-{seed}"
        if not (folder/"complete.json").exists():
            raise ValueError("A required NBA seed is incomplete")
        for name,expected in json.loads((folder/"complete.json").read_text())["artifacts"].items():
            if digest(folder/name)!=expected:
                raise ValueError(f"NBA seed artifact changed after completion: {folder/name}")
        for filename,target in (("main-results.json",groups),("weight-ablation-results.json",ablation)):
            rows=json.loads((folder/filename).read_text())
            for name,row in rows.items():
                queries=row.get("queries")
                expected_queries=dataset["counts"]["test"]["selected_clips"]
                complete_queries=bool(queries and len(queries)==expected_queries and
                                      len({r["query"] for r in queries})==expected_queries)
                if not complete_queries:
                    status=row.get("status","incomplete query list")
                    if status=="complete":
                        status="incomplete query list"
                elif any(r["timeout"] or r["seconds"] is None or not np.isfinite(r["seconds"]) or r["seconds"]>60
                         for r in queries):
                    status="deadline failure; no aggregate performance"
                else:
                    status=row.get("status") or "complete"
                target[name].append(dict(method=name if target is groups else "blocking",seed=seed,
                                         status=status,configuration=row.get("configuration"),queries=queries))
    result=dict(dataset=dataset,manifest=manifest,main=[combine(v,config["index_seeds"],games) for _,v in sorted(groups.items())],
                ablation=[dict(combine(v,config["index_seeds"],games),latency_weight=float(key)) for key,v in sorted(ablation.items())],
                notes=["Results are full validated-accessible-corpus measurements, not the whole NBA season.",
                       "Every query game is held out of database and tuning.",
                       "Index-array storage and total RAM are distinct; achieved quality accompanies latency.",
                       "Accuracy intervals resample held-out games, preserving their query and index-seed dependence.",
                       "Absolute distance and excess distance in feet do not certify tactical similarity.",
                       "Any missing or deadline-failed seed group has no headline aggregate performance."])
    write_json(root/"replicated-summary.json",result)
    completed=[r for r in result["main"] if r["status"]=="complete"]
    fig=comparison_figure(completed)
    fig.savefig(root/"nba-full-comparison.pdf",bbox_inches="tight")
    fig.savefig(root/"nba-full-comparison.png",dpi=180,bbox_inches="tight")
    plt.close(fig)
    fig=quality_figure(completed)
    fig.savefig(root/"nba-full-quality.pdf",bbox_inches="tight")
    fig.savefig(root/"nba-full-quality.png",dpi=180,bbox_inches="tight");plt.close(fig)
    print(json.dumps(result["main"],indent=2),flush=True)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("results",type=Path);parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--render-only",action="store_true",
                        help="Render only quantitative PDFs from the frozen summary; do not summarize or write PNGs")
    parser.add_argument("--output",type=Path,help="Existing PDF output directory (requires --render-only)")
    args=parser.parse_args()
    if args.render_only:
        render_frozen_figures(args.results,args.data,args.output or args.results)
    elif args.output:
        parser.error("--output requires --render-only")
    else:
        summarize(args.results,args.data)
