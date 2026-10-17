"""Create POC figures directly from complete held-out logs and frozen selections."""
import argparse
import csv
import hashlib
import json
from math import factorial
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fresh.analyze_regimes import wilson
from fresh.e1 import write_json
from fresh.regimes import approximation_summary
from fresh.plot_styles import COLORS, MARKERS, method_style
from fresh.nba_paper_examples import configure_paper_font, save_paper_pdf


MAIN = ["blocking","probe_all","store_all","per_curve-intersection","brute-force"]
LABELS = {"blocking":"Blocking","probe_all":"Probe all","store_all":"Store all",
          "per_curve-intersection":"Per-curve intersection",
          "brute-force":"Brute force"}
REGIMES = ["conjunctive","rare","repeated"]
TITLES = {"conjunctive":"Common paths, selective combinations",
          "rare":"Distinctive paths","repeated":"Repeated paths / multiplicities"}


def metrics(rows,expected,factors):
    if len(rows)!=expected or len({int(r["query"]) for r in rows})!=expected or any(r["timeout"]!="False" for r in rows):
        raise ValueError("Refusing to summarize an incomplete POC configuration")
    times=np.array([float(r["seconds"]) for r in rows])
    answered=[r for r in rows if r["found"]=="True"]
    ratios=[float(r["ratio"]) for r in answered if r["ratio"]]
    exact=sum(r["exact_nn"]=="True" for r in rows)
    c_success={str(c):sum(r["found"]=="True" and float(r["distance"])<=c*float(r["nn_distance"])+1e-8
                          for r in rows)/expected for c in factors}
    return dict(recall_at_1=exact/expected,exact_successes=exact,test_queries=expected,
                recall_95_interval=wilson(exact,expected),coverage=len(answered)/expected,
                empty_rate=1-len(answered)/expected,median_ms=float(np.median(times)*1000),
                p95_ms=float(np.quantile(times,.95)*1000),mean_ms=float(times.mean()*1000),
                serial_qps=expected/times.sum(),c_success=c_success,**approximation_summary(ratios),
                mean_nn_distance=float(np.mean([float(r["nn_distance"]) for r in rows])),
                mean_returned_distance=float(np.mean([float(r["distance"]) for r in answered])) if answered else None,
                mean_candidates=float(np.mean([int(r["candidates"]) for r in rows])),
                mean_verified=float(np.mean([int(r["verified"]) for r in rows])),
                mean_postings=float(np.mean([int(r["postings"]) for r in rows])),
                mean_probes=float(np.mean([int(r["probes"]) for r in rows])))


def summarize(root):
    config=json.loads((root/"config.json").read_text())
    cases=json.loads((root/"cases.json").read_text())
    table,ablation=[],[]
    for case in cases:
        if case["status"]!="complete":
            continue
        folder=root/case["case"]
        dataset=json.loads((folder/"dataset.json").read_text())
        with (folder/"test-queries.csv").open() as handle:
            rows=list(csv.DictReader(handle))
        main=json.loads((folder/"main-results.json").read_text())
        alternatives=json.loads((folder/"ablation-results.json").read_text())
        for source,output in ((main,table),(alternatives,ablation)):
            for item in source:
                seed=dataset.get("index_seed",config.get("index_seed"))
                if item.get("index_seed",seed)!=seed:
                    raise ValueError("Result index seed disagrees with the dataset job")
                item_fields={key:value for key,value in item.items() if key!="index_seed"}
                base=dict(case=case["case"],kind=dataset["kind"],n=dataset["n"],k=dataset["k"],
                          case_id=dataset.get("case_id",case["case"]),
                          index_seed=seed,
                          raw_curve_bytes=dataset["raw_curve_bytes"],planted_fraction=dataset["planted_fraction"],
                          dataset_sha256=dataset["dataset_sha256"],queries_sha256=dataset["queries_sha256"],
                          mean_relevant_sets=dataset["mean_test_relevant_sets"],**item_fields)
                if item["test_status"]=="complete":
                    query_rows=[r for r in rows if r["config_id"]==item["config_id"]]
                    base.update(metrics(query_rows,config["test_queries"],config["c_success_factors"]))
                    base["test_target_met"]=base["recall_at_1"]>=config["target_recall"]
                    base["index_MB"]=item["index_bytes"]/1e6
                output.append(base)
        missing=[r for r in main if r["test_status"]!="complete"]
        if missing:
            tuning=json.loads((folder/"tuning-configurations.json").read_text())
            for row in table:
                if row["case"]!=case["case"] or row["test_status"]=="complete":
                    continue
                if row["display_method"]=="store_all":
                    row["exclusion_reasons"]=sorted({r["status"] for r in tuning
                                                     if r.get("stored")==factorial(dataset["k"])})
    report=dict(config=config,manifest=json.loads((root/"manifest.json").read_text()),
                analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                cases=cases,main=table,ablation=ablation,
                notes=["One main blocking label; lambda=1 normalized by 100 MB and 1 ms is fixed before testing.",
                       "All index types use unique fingerprint-parent postings; raw array payload differs from total process RAM.",
                       f"{config['test_queries']} held-out queries per seed; {len(config.get('index_seeds',[config.get('index_seed')]))} index seed(s), synthetic evidence only.",
                       "Submillisecond timing is descriptive; repeated timing/seeds will be needed for small differences.",
                       "Planted-set fraction is reported per dataset; nested sizes preserve query and background prefixes.",
                       "No survivor-only summaries; excluded builds carry their precise resource reason."])
    write_json(root/"summary.json",report)
    columns=["case","display_method","n","k","test_status","index_MB","median_ms","p95_ms","recall_at_1",
             "mean_ratio","p90_ratio","empty_rate","mean_candidates","mean_postings","blocks","L","delta"]
    with (root/"main-table.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=columns,extrasaction="ignore")
        writer.writeheader()
        writer.writerows(table)
    return report


def save(fig,folder,name):
    fig.savefig(folder/f"{name}.png",dpi=180,bbox_inches="tight")
    save_paper_pdf(fig,folder/f"{name}.pdf")
    plt.close(fig)


def plot_value(value):
    if value is None:
        return np.nan
    return float("inf") if value=="infinity" else float(value)


def figures(report,folder):
    folder.mkdir(parents=True,exist_ok=True)
    configure_paper_font()
    plt.rcParams.update({"font.size":7,"axes.grid":True,"grid.alpha":.2,"grid.linewidth":.4,"axes.spines.top":False,
                         "axes.spines.right":False})
    rows=[r for r in report["main"] if r["test_status"]=="complete"]
    main_n=report["config"].get("main_plot_n",2000)
    main_k=sorted({r["k"] for r in rows if r["n"]==main_n})
    fig,axes=plt.subplots(4,3,figsize=(7,6.5),sharex="col")
    fields=[("median_ms","Median query latency (ms)",True),
            ("index_MB","Index array storage (MB)",True),
            ("recall_at_1","Exact NN success (%)",False),
            ("mean_ratio","Mean distance ratio among answers",False)]
    for col,kind in enumerate(REGIMES):
        axes[0,col].set_title(TITLES[kind])
        for method in MAIN:
            selected=sorted([r for r in rows if r["kind"]==kind and r["n"]==main_n and r["display_method"]==method],
                            key=lambda r:r["k"])
            if not selected:
                continue
            x=[r["k"] for r in selected]
            for row,(field,label,log) in enumerate(fields):
                if field=="index_MB" and method=="brute-force":
                    continue
                y=[plot_value(r[field])*(100 if field=="recall_at_1" else 1) for r in selected]
                axes[row,col].plot(x,y,**method_style(method),label=LABELS[method])
                if field+"_seed_minmax" in selected[0]:
                    low=[r[field+"_seed_minmax"][0]*(100 if field=="recall_at_1" else 1) for r in selected]
                    high=[r[field+"_seed_minmax"][1]*(100 if field=="recall_at_1" else 1) for r in selected]
                    axes[row,col].fill_between(x,low,high,color=COLORS[method],alpha=.07,linewidth=0)
                for xx,yy,r in zip(x,y,selected):
                    if not r["test_target_met"]:
                        axes[row,col].plot(xx,yy,marker=MARKERS[method],linestyle="none",
                                           markerfacecolor="white",markeredgecolor=COLORS[method],
                                           markersize=3.5,markeredgewidth=.55,zorder=5)
                if log:
                    axes[row,col].set_yscale("log")
                if field=="recall_at_1":
                    lower=[(r["recall_at_1"]-r["recall_95_interval"][0])*100 for r in selected]
                    upper=[(r["recall_95_interval"][1]-r["recall_at_1"])*100 for r in selected]
                    axes[row,col].errorbar(x,y,yerr=[lower,upper],color=COLORS[method],fmt="none",
                                          capsize=1.5,elinewidth=.5,capthick=.5)
                    axes[row,col].set_ylim(70,101)
                    axes[row,col].axhline(90,color="black",linestyle=":",linewidth=.8)
                elif field=="mean_ratio":
                    axes[row,col].ticklabel_format(axis="y",style="plain",useOffset=False)
                    axes[row,col].axhline(1,color="black",linestyle=":",linewidth=.8)
                axes[row,col].set_ylabel(label if col==0 else "")
        axes[3,col].set_xlabel("Objects per tuple, k")
        axes[3,col].set_xticks(main_k)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",ncol=5,bbox_to_anchor=(.5,1.015),
               fontsize=6.5,handlelength=2.5)
    seed_count=len(report["config"].get("index_seeds",[report["config"].get("index_seed")]))
    fig.suptitle(f"{main_n:,} tuples, {report['config']['test_queries']:,} test queries/seed, "
                 f"{seed_count} seed(s); open points fail the test target in at least one seed",y=1.045)
    fig.tight_layout()
    save(fig,folder,"01_main_comparison")

    fig,axes=plt.subplots(1,3,figsize=(7,2.2))
    for col,kind in enumerate(REGIMES):
        axes[col].set_title(TITLES[kind])
        for method in MAIN:
            selection=[r for r in rows if r["kind"]==kind and r["n"]==main_n and r["k"]==8 and r["display_method"]==method]
            if not selection:
                continue
            row=selection[0]
            factors=[1,1.01,1.05,1.1,1.5,2]
            axes[col].plot(np.arange(len(factors)),[100*row["c_success"][str(float(c))] for c in factors],
                           **method_style(method),label=LABELS[method])
        axes[col].set_xticks(range(6),labels=["1","1.01","1.05","1.1","1.5","2"])
        axes[col].set_ylim(75,101)
        axes[col].set_xlabel("Allowed factor c (discrete thresholds)")
        axes[col].set_ylabel("Queries answered within c × optimum (%)" if col==0 else "")
    fig.legend(*axes[0].get_legend_handles_labels(),loc="upper center",ncol=5,bbox_to_anchor=(.5,1.05),
               fontsize=6.5,handlelength=2.5)
    fig.tight_layout()
    save(fig,folder,"02_approximation_quality")

    fig,axes=plt.subplots(3,3,figsize=(7,4.6),gridspec_kw={"height_ratios":[1,1,.55]})
    for col,kind in enumerate(REGIMES):
        group=sorted([r for r in report["ablation"] if r["test_status"]=="complete"
                      and r["kind"]==kind and r["n"]==main_n and r["k"]==8],key=lambda r:r["latency_weight"])
        if not group:
            continue
        x=[r["latency_weight"] for r in group]
        for ax,field,label in ((axes[0,col],"median_ms","Median latency (ms)"),
                               (axes[1,col],"index_MB","Index array storage (MB)")):
            ax.plot(x,[r[field] for r in group],**method_style("blocking"))
            ax.axvline(report["config"]["blocking_main_latency_weight"],color="black",
                       linestyle=":",linewidth=.6,label="Main weight")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_ylabel(label if col==0 else "")
            for r in group:
                if not r["test_target_met"]:
                    ax.plot(r["latency_weight"],r[field],marker=MARKERS["blocking"],linestyle="none",
                            markerfacecolor="white",markeredgecolor=COLORS["blocking"],
                            markersize=3.5,markeredgewidth=.55,zorder=5)
        axes[0,col].set_title(TITLES[kind])
        axes[1,col].set_xlabel(r"Latency weight $\lambda$ (memory weight = 1)")
        axes[2,col].axis("off")
        cells=[[f"{r['latency_weight']:g}",
                "varies by seed" if r.get("blocks_vary_by_seed") else "+".join(map(str,r["blocks"])),
                f"{100*r['recall_at_1']:.1f}%"]
               for r in group]
        table=axes[2,col].table(cellText=cells,colLabels=[r"$\lambda$","Selected blocks","Test recall"],
                                 cellLoc="center",loc="center",colWidths=[.15,.6,.25])
        table.auto_set_font_size(False)
        table.set_fontsize(6.5)
        table.scale(1,1.1)
    fig.suptitle("Weight ablation: tuning selects each setting; held-out results do not choose the main weight")
    fig.tight_layout()
    save(fig,folder,"03_weight_ablation")

    fig,axes=plt.subplots(1,3,figsize=(7,2.3))
    for method in MAIN:
        selected=sorted([r for r in rows if r["kind"]=="conjunctive" and r["k"]==8 and r["display_method"]==method],
                        key=lambda r:r["n"])
        if not selected:
            continue
        for col,(field,label) in enumerate([("median_ms","Median query latency (ms)"),
                                           ("index_MB","Index array storage (MB)"),
                                           ("recall_at_1","Exact NN success (%)")]):
            if field=="index_MB" and method=="brute-force":
                continue
            axes[col].plot([r["n"] for r in selected],
                           [r[field]*(100 if col==2 else 1) for r in selected],
                           **method_style(method),label=LABELS[method])
            for r in selected:
                if not r["test_target_met"]:
                    value=r[field]*(100 if col==2 else 1)
                    axes[col].plot(r["n"],value,marker=MARKERS[method],linestyle="none",
                                   markerfacecolor="white",markeredgecolor=COLORS[method],
                                   markersize=3.5,markeredgewidth=.55,zorder=5)
            axes[col].set_xscale("log")
            if col<2:
                axes[col].set_yscale("log")
            axes[col].set_xlabel("Tuples n (common paths, k = 8)")
            axes[col].set_ylabel(label)
            sizes=sorted({r["n"] for r in rows if r["kind"]=="conjunctive" and r["k"]==8})
            axes[col].set_xticks(sizes,labels=[f"{n:,}" for n in sizes])
    axes[2].axhline(90,color="black",linestyle=":",linewidth=.8)
    excluded=[r for r in report["main"] if r["kind"]=="conjunctive" and r["k"]==8
              and r["display_method"]=="store_all" and r["test_status"]!="complete"]
    if excluded:
        excluded_sizes=", ".join(f"{r['n']:,}" for r in excluded)
        axes[1].text(.03,.03,f"Store all at {excluded_sizes}: not fully evaluated\n(see exclusion statuses)",
                     transform=axes[1].transAxes,fontsize=8,bbox=dict(facecolor="white",alpha=.9,edgecolor="none"))
    fig.legend(*axes[0].get_legend_handles_labels(),loc="upper center",ncol=5,bbox_to_anchor=(.5,1.05),
               fontsize=6.5,handlelength=2.5)
    fig.tight_layout()
    save(fig,folder,"04_scaling")

    fig,axes=plt.subplots(1,3,figsize=(7,2.3))
    for col,kind in enumerate(REGIMES):
        selection=[r for r in rows if r["kind"]==kind and r["n"]==main_n and r["k"]==8
                   and r["display_method"] in ("blocking","per_curve-intersection")]
        names=[LABELS[r["display_method"]].replace(" ","\n",1) for r in selection]
        x=np.arange(len(selection))
        axes[col].bar(x-.18,[r["mean_postings"] for r in selection],.36,label="Postings read",
                      hatch="//",edgecolor="black",linewidth=.4)
        axes[col].bar(x+.18,[r["mean_candidates"] for r in selection],.36,label="Candidate tuples",
                      hatch="..",edgecolor="black",linewidth=.4)
        axes[col].set_xticks(x,names,fontsize=8)
        axes[col].set_yscale("symlog",linthresh=1)
        axes[col].set_title(TITLES[kind])
        axes[col].set_ylabel("Mean count per query" if col==0 else "")
    fig.legend(*axes[0].get_legend_handles_labels(),loc="upper center",ncol=2,bbox_to_anchor=(.5,1.05))
    fig.tight_layout()
    save(fig,folder,"05_candidate_work")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("results",type=Path)
    args=parser.parse_args()
    report=summarize(args.results)
    figures(report,args.results/"figures")
    print(f"Generated {len(report['main'])} main rows and five figures in {args.results/'figures'}")
