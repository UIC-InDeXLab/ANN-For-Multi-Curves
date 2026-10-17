"""Finalize first-hit quality using complete cells and exact all-tuple references."""
import argparse
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fresh.e1 import write_json
from fresh.regimes import approximation_summary
from fresh.tuple_quality import analyze_case,first_hit_rows
from fresh.nba_paper_examples import configure_paper_font, save_paper_pdf
from fresh.plot_styles import LINE_WIDTH


def plot_report(report,output):
    configure_paper_font()
    plt.rcParams.update({"font.size":7})
    fig,axes=plt.subplots(1,3,figsize=(7,2.4))
    colors={"random":"#277da1","aligned":"#007f5f","separated-star":"#b5179e"}
    markers={"random":"o","aligned":"s","separated-star":"D"}
    names={"random":"Random paths","aligned":"Translated copies","separated-star":"Alternating segments"}
    for workload in report["workloads"]:
        for d,style in ((2,"-"),(3,"--")):
            values=sorted((r for r in report["cases"] if r["workload"]==workload and r["d"]==d),
                          key=lambda r:r["k"])
            x=[r["k"] for r in values]
            for col,key in enumerate(("mean_ratio","p90_ratio","mean_uniform_random_over_optimum")):
                axes[col].plot(x,[r[key] for r in values],marker=markers[workload],
                               markersize=3,markeredgewidth=.55,linewidth=LINE_WIDTH,
                               linestyle=style,color=colors[workload],label=f"{names[workload]}, {d}-D")
    for ax,label in zip(axes,("Mean first-hit / optimum","90th percentile / optimum",
                              "Uniform-random / optimum")):
        ax.set_xlabel("Objects per multi-curve")
        ax.set_ylabel(label)
        ax.set_yscale("log")
        ax.grid(alpha=.2,linewidth=.4)
        ax.spines[["top","right"]].set_visible(False)
    fig.legend(*axes[0].get_legend_handles_labels(),loc="upper center",ncol=3,
               bbox_to_anchor=(.5,1.14),fontsize=6.5,handlelength=2.5)
    fig.tight_layout()
    save_paper_pdf(fig,output/"first-hit-quality.pdf")
    fig.savefig(output/"first-hit-quality.png",dpi=220,bbox_inches="tight")
    plt.close(fig)


def finish(parent,continuation,output):
    if not (continuation/"completion.json").exists():
        raise ValueError("The first-hit continuation is not complete")
    config=json.loads((parent/"config.json").read_text())
    output.mkdir(parents=True,exist_ok=False)
    cases=[]
    first_ratios=[]
    for d in (2,3):
        for k in range(2,11):
            for workload in ("random","aligned","separated-star"):
                name=f"{workload}-d{d}-k{k}"
                source=continuation if k==10 else parent
                source_config=json.loads((source/"config.json").read_text())
                rows,status=first_hit_rows(source/name,"cost-model",source_config)
                info=json.loads((source/name/"dataset.json").read_text())
                cache=None
                for directory in ("fresh-e1-20261012-random-quality-k2","fresh-e1-20261012-random-quality-k3to7"):
                    file=parent.parent/directory/f"{name}-quality.json"
                    if file.exists():
                        candidate=json.loads(file.read_text())
                        if (candidate["dataset_sha256"]==info["dataset_sha256"] and
                            candidate["experiment_commit"]==status["commit"] and
                            candidate["query_count"]==source_config["test_queries"]):
                            cache=candidate
                            break
                if cache is None:
                    cache=analyze_case(source,output,name,"cost-model")
                else:
                    write_json(output/f"{name}-quality.json",dict(cache,validated_cache_reused=True))
                ratios=[float(row["returned_distance"])/float(row["nn_distance"]) for row in rows
                        if row["found"]=="True" and float(row["nn_distance"])>0]
                denominators=[float(row["nn_distance"]) for row in rows]
                if any(value<=0 for value in denominators):
                    raise ValueError("Zero optimum needs explicit unbounded-ratio handling before publication")
                radii=[float(row["returned_distance"])/source_config["r"] for row in rows if row["found"]=="True"]
                row=dict(cache,**approximation_summary(ratios),theoretical_c=np.sqrt(d)*status["delta"]/source_config["r"],
                         mean_radius_factor=float(np.mean(radii)) if radii else None,
                         median_ratio=float(np.median(ratios)) if ratios else None,
                         p95_ratio=float(np.quantile(ratios,.95)) if ratios else None,
                         max_ratio=max(ratios) if ratios else None,
                         empty_rate=sum(r["found"]!="True" for r in rows)/len(rows),
                         query_sha256=info["queries_sha256"],source_results=str(source),
                         quality_only_local=(k==10))
                cases.append(row)
                first_ratios.extend(ratios)
                write_json(output/"case-quality.json",cases)
                print(f"FINAL QUALITY {name}",flush=True)

    def group_summary(group):
        return dict(cases=len(group),first_hit_evaluations=sum(r["first_hit_evaluations"] for r in group),
                    distinct_queries=sum(r["query_count"] for r in group),
                    mean_optimal_distance=float(np.mean([r["mean_optimal_distance"] for r in group])),
                    mean_first_hit_distance=float(np.average(
                        [r["mean_first_hit_distance"] for r in group],
                        weights=[r["returned_ratio_count"] for r in group])),
                    mean_uniform_random_tuple_distance=float(np.mean([r["mean_uniform_random_tuple_distance"] for r in group])),
                    mean_ratio=float(np.average([r["mean_ratio"] for r in group],weights=[r["returned_ratio_count"] for r in group])),
                    mean_radius_factor=float(np.average([r["mean_radius_factor"] for r in group],weights=[r["returned_ratio_count"] for r in group])),
                    mean_uniform_random_over_optimum=float(np.mean([r["mean_uniform_random_over_optimum"] for r in group])),
                    empty_rate=float(np.mean([r["empty_rate"] for r in group])))
    report=dict(created=time.time(),experiment="first-hit approximation quality",
                k=list(range(2,11)),dimensions=[2,3],workloads=["random","aligned","separated-star"],
                index_seeds=config["index_seeds"],queries_per_seed=200,n=11800,r=1,
                cases=cases,overall=dict(group_summary(cases),median_ratio=float(np.median(first_ratios)),
                                        p90_ratio=float(np.quantile(first_ratios,.9)),max_ratio=max(first_ratios)),
                by_dimension=[dict(d=d,**group_summary([r for r in cases if r["d"]==d]),
                                   theoretical_c=float(np.sqrt(d)*2*d*32/(1-.75))) for d in (2,3)],
                by_workload=[dict(workload=w,**group_summary([r for r in cases if r["workload"]==w]))
                             for w in ("random","aligned","separated-star")],
                sources=dict(parent=str(parent),continuation=str(continuation)),
                notes=["Parent E1 sweep was stopped; only complete first-hit method/seed groups are included.",
                       "Omitted k10 cells were completed locally solely for quality; local timings are not server comparisons.",
                       "All-tuple means are exact, with exhaustive minima/relevance counts checked against ground truth.",
                       "First-hit construction uses theory parameters and a fresh feasibility-guided family; it is not the empirical comparison's tuned reranking configuration."])
    write_json(output/"approximation-summary.json",report)
    plot_report(report,output)
    write_json(output/"completion.json",dict(completed=time.time(),case_count=len(cases)))
    print(json.dumps(report["by_dimension"],indent=2),flush=True)
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--parent",type=Path,required=True)
    parser.add_argument("--continuation",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    finish(args.parent,args.continuation,args.output)
