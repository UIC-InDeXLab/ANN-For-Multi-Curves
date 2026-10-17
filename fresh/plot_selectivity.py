"""Measured crossover and accuracy, rather than assuming a monotone speed advantage."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fresh.e1 import write_json
from fresh.nba_full import digest
from fresh.nba_paper_examples import configure_paper_font, save_paper_pdf
from fresh.plot_styles import method_style


def report_and_plot(root):
    if not (root/"completion.json").exists():
        raise ValueError("Do not publish a partial selectivity sweep")
    report=json.loads((root/"replicated-summary.json").read_text())
    config=report["config"]
    frequency={}
    for case in config["cases"]:
        pool=int(round(1/(1-(1-case["frequency"])**(1/case["k"]))))
        frequency[case["id"]]=dict(target=case["frequency"],pool=pool,
                                   realized_theoretical=1-(1-1/pool)**case["k"])
    table=[]
    for case_id,entry in frequency.items():
        rows=[r for r in report["main"] if r["case_id"]==case_id]
        blocking=next((r for r in rows if r["display_method"]=="blocking"),None)
        pc=[r for r in rows if r["display_method"]=="per_curve-intersection"]
        if blocking is None or len(pc)!=1:
            raise ValueError("Missing required comparison in selectivity data")
        row=dict(case_id=case_id,**entry)
        if blocking["test_status"]!="complete" or any(r["test_status"]!="complete" for r in pc):
            row["status"]="not all required methods/seeds completed"
        else:
            best=pc[0]
            row.update(status="complete",
                       best_per_curve_variant=best["display_method"],
                       speedup_best_pc_over_blocking=best["median_ms"]/blocking["median_ms"],
                       blocking_latency_ms=blocking["median_ms"],best_pc_latency_ms=best["median_ms"],
                       blocking_index_MB=blocking["index_MB"],best_pc_index_MB=best["index_MB"],
                       blocking_recall=blocking["recall_at_1"],best_pc_recall=best["recall_at_1"],
                       blocking_mean_postings=blocking["mean_postings"],best_pc_mean_postings=best["mean_postings"],
                       blocking_mean_candidates=blocking["mean_candidates"],best_pc_mean_candidates=best["mean_candidates"],
                       blocking_meets_target_every_seed=blocking["test_target_met"],
                       pc_meets_target_every_seed=best["test_target_met"],
                       mean_relevant_sets=blocking["mean_relevant_sets"],
                       best_variant_is_posthoc_envelope=False)
        table.append(row)
    table.sort(key=lambda r:r["realized_theoretical"])
    brackets=[]
    valid=[r for r in table if r["status"]=="complete"]
    for a,b in zip(valid,valid[1:]):
        if (a["speedup_best_pc_over_blocking"]-1)*(b["speedup_best_pc_over_blocking"]-1)<0:
            brackets.append([a["realized_theoretical"],b["realized_theoretical"]])
    write_json(root/"intersection-summary.json",dict(
        comparisons=table,observed_crossover_brackets=brackets,
        source_sha256=digest(root/"replicated-summary.json"),
        note="Comparison uses only per-curve intersection. Frequency changes joint-answer counts too. Latency ratios are shown beside achieved accuracy and footprint, not asserted as equal-quality dominance."))
    configure_paper_font()
    plt.rcParams.update({"font.size":8})
    fig,axes=plt.subplots(2,2,figsize=(7,4.4))
    x=[100*r["realized_theoretical"] for r in valid]
    axes[0,0].plot(x,[r["speedup_best_pc_over_blocking"] for r in valid],**method_style("blocking"))
    axes[0,0].axhline(1,color="black",linestyle=":",linewidth=.6)
    axes[0,0].set_ylabel("Intersection / blocking time")
    for key,method,label in (("blocking","blocking","Blocking"),
                            ("best_pc","per_curve-intersection","Intersection")):
        axes[0,1].plot(x,[r[f"{key}_recall"]*100 for r in valid],**method_style(method),label=label)
        axes[1,0].plot(x,[r[f"{key}_index_MB"] for r in valid],**method_style(method),label=label)
    axes[0,1].axhline(90,color="black",linestyle=":",linewidth=.6)
    axes[0,1].set_ylabel("Achieved exact-NN recall (%)")
    axes[1,0].set_ylabel("Index array storage (MB)")
    axes[1,0].set_yscale("log")
    axes[1,1].plot(x,[r["mean_relevant_sets"] for r in valid],**method_style("brute-force"))
    axes[1,1].set_ylabel("Mean relevant multi-curves per query")
    axes[1,1].set_yscale("log")
    for ax in axes.flat:
        ax.set_xscale("log");ax.set_xlabel("Background path occurrence (%)");ax.grid(alpha=.25)
    fig.legend(*axes[0,1].get_legend_handles_labels(),loc="upper center",ncol=2,handlelength=2.5)
    fig.tight_layout(rect=(0,0,1,.95))
    fig.savefig(root/"selectivity-crossover.png",dpi=180,bbox_inches="tight")
    save_paper_pdf(fig,root/"selectivity-crossover.pdf")
    plt.close(fig)
    return table


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("results",type=Path)
    report_and_plot(parser.parse_args().results)
