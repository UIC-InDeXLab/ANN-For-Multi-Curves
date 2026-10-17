"""Audit and publish measured hard-budget object-count feasibility."""
import argparse
import csv
import json
from math import factorial
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fresh.e1 import arrays_hash, write_json
from fresh.feasibility import select_fastest
from fresh.nba_full import digest
from fresh.nba_paper_examples import configure_paper_font
from fresh.plot_paper_experiments import row_figure, save_row
from fresh.poc import jobs
from fresh.plot_styles import COLORS, MARKERS, method_style


METHODS=("blocking","store_all","probe_all","per_curve-intersection")
LABELS={"blocking":"Blocking","store_all":"Store-all","probe_all":"Probe-all",
        "per_curve-intersection":"Intersection"}


def query_metrics(rows,optimum,deadline,expected):
    if len(rows)!=expected or {int(row["query"]) for row in rows}!=set(range(expected)):
        raise ValueError("Every held-out query must appear once, including timeouts")
    success=[]
    expired=[]
    elapsed=[]
    for row in rows:
        q=int(row["query"])
        seconds=float(row["seconds"])
        if seconds<0 or not np.isfinite(seconds) or float(row["nn_distance"])!=optimum[q]:
            raise ValueError("Query time or exact ground truth is invalid")
        timeout=row["timeout"]=="True" or seconds>deadline
        found=int(row["id"])>=0 and row["found"]=="True"
        exact=found and not timeout and float(row["distance"])<=optimum[q]+1e-8
        if (row["exact_nn"]=="True")!=exact:
            raise ValueError("Reported recall differs from exact success within the deadline")
        if timeout and found:
            raise ValueError("A timed-out query cannot be counted as a returned answer")
        success.append(exact)
        expired.append(timeout)
        elapsed.append(seconds*1000)
    return dict(success_rate=float(np.mean(success)),timeout_rate=float(np.mean(expired)),
                median_ms=float(np.median(elapsed)),p95_ms=float(np.quantile(elapsed,.95)),
                max_ms=float(np.max(elapsed)),query_count=expected)


def combine_seeds(rows,seeds,target,budget):
    if len(rows)!=len(seeds) or {row["index_seed"] for row in rows}!=set(seeds):
        raise ValueError("All required seeds must be represented")
    measured=[row for row in rows if row["measured"]]
    feasible=(len(measured)==len(seeds) and
              all(row["success_rate"]>=target and row["index_bytes"]<=budget for row in measured))
    result=dict(k=rows[0]["k"],method=rows[0]["method"],feasible=feasible,
                all_seeds_measured=len(measured)==len(seeds),seeds=rows)
    if len(measured)==len(seeds):
        for metric in ("median_ms","p95_ms","index_MB","success_rate","timeout_rate"):
            values=[row[metric] for row in measured]
            result[metric]=float(np.mean(values)) if metric in ("success_rate","timeout_rate") else float(np.median(values))
            result[metric+"_minmax"]=[min(values),max(values)]
    return result


def audit(root, source_config=None, not_run_ks=()):
    """not_run_ks lists object counts that could not be run (for example because one untimed query already
    exceeds the compilation deadline); every other configured count and seed must complete."""
    if not (root/"completion.json").is_file() and not not_run_ks:
        raise ValueError("Feasibility benchmark is incomplete")
    config=json.loads((root/"config.json").read_text())
    manifest=json.loads((root/"manifest.json").read_text())
    source_config=Path(source_config) if source_config else Path(__file__).resolve().parents[1]/"configs"/"feasibility-20261014.json"
    if (json.loads(source_config.read_text())!=config or
            digest(source_config)!=manifest["config_sha256"]):
        raise ValueError("Frozen configuration checksum changed")
    expected=[row for row in jobs(config) if row["k"] not in not_run_ks]
    cases=[row for row in json.loads((root/"cases.json").read_text())
           if row["case"] in {job["id"] for job in expected}]
    if (len(cases)!=len(expected) or
            {row["case"] for row in cases}!={row["id"] for row in expected} or
            any(row["status"]!="complete" or row["returncode"]!=0 for row in cases)):
        raise ValueError("Every configured object count and index seed must complete")
    configured={row["id"]:row for row in expected}
    groups={}
    artifacts={}
    hashes={}
    for case in cases:
        folder=root/case["case"]
        dataset=json.loads((folder/"dataset.json").read_text())
        job=configured[case["case"]]
        if any(dataset[field]!=job[field] for field in ("n","k","index_seed")):
            raise ValueError("Dataset identity differs from the frozen case")
        hashes.setdefault(dataset["k"],set()).add((dataset["dataset_sha256"],dataset["queries_sha256"]))
        with np.load(folder/"ground-truth.npz",allow_pickle=False) as truth:
            if arrays_hash(truth["queries"])!=dataset["queries_sha256"]:
                raise ValueError("Ground-truth queries differ from dataset metadata")
            h=config["tuning_queries"]
            tuning_optimum=truth["distances"][:h]
            optimum=truth["distances"][h:h+config["test_queries"]]
        if not np.isfinite(optimum).all() or len(optimum)!=config["test_queries"]:
            raise ValueError("Ground truth must cover every held-out query")
        selections=json.loads((folder/"frozen-main-selections.json").read_text())
        results=json.loads((folder/"main-results.json").read_text())
        tuning=json.loads((folder/"tuning-configurations.json").read_text())
        presented=[row for row in results if row["display_method"] in METHODS]
        if len(presented)!=len(METHODS) or {row["display_method"] for row in presented}!=set(METHODS):
            raise ValueError("Every baseline must have a measured result or explicit exclusion")
        query_file=folder/"test-queries.csv"
        with query_file.open() as handle:
            logs=list(csv.DictReader(handle))
        for log in logs:
            if (log["commit"]!=manifest["commit"] or
                    log["config_sha256"]!=manifest["config_sha256"] or
                    log["dataset_sha256"]!=dataset["dataset_sha256"] or
                    int(log["index_seed"])!=dataset["index_seed"]):
                raise ValueError("A query log differs from frozen run provenance")
        qualified={row["config_id"]:row for row in tuning if row["status"]=="qualified"}
        with (folder/"tuning-queries.csv").open() as handle:
            tuning_logs=list(csv.DictReader(handle))
        for result in presented:
            method=result["display_method"]
            summary=dict(k=dataset["k"],method=method,index_seed=dataset["index_seed"],
                         raw_curve_bytes=dataset["raw_curve_bytes"],case=case["case"])
            if method in selections:
                selection=selections[method]
                if selection["config_id"] not in qualified or selection["index_bytes"]>config["index_budget_bytes"]:
                    raise ValueError("A selected configuration must qualify using tuning queries and actual storage")
                cfg_id=selection["config_id"]
                if selection!=qualified[cfg_id]:
                    raise ValueError("Frozen selection differs from the qualified tuning record")
                if method=="blocking":
                    options=[row for row in tuning if row["method"]=="blocking-family"]
                elif method in ("store_all","probe_all"):
                    count=factorial(dataset["k"]) if method=="store_all" else 1
                    options=[row for row in tuning if row["method"]=="blocking-family" and row["stored"]==count]
                else:
                    options=[row for row in tuning if row["method"]==method]
                if selection!=select_fastest(options):
                    raise ValueError("Frozen selection is not the fastest qualifying tuning configuration")
                tuning_rows=[row for row in tuning_logs if row["config_id"]==cfg_id]
                tuning_metrics=query_metrics(
                    [dict(row,nn_distance=tuning_optimum[int(row["query"])]) for row in tuning_rows],
                    tuning_optimum,config["query_deadline_seconds"],config["tuning_queries"])
                if tuning_metrics["success_rate"]<config["tuning_acceptance_recall"]:
                    raise ValueError("Tuning selection lacks a full qualifying query log")
                if not np.isclose(tuning_metrics["median_ms"]/1000,selection["tuning_median_seconds"],
                                  rtol=1e-12,atol=1e-12):
                    raise ValueError("Selection latency differs from its tuning query log")
                if result["config_id"]!=cfg_id or result["index_bytes"]!=selection["index_bytes"]:
                    raise ValueError("Test result does not match the frozen selection")
                query_rows=[row for row in logs if row["config_id"]==cfg_id]
                measured=query_metrics(query_rows,optimum,config["query_deadline_seconds"],config["test_queries"])
                if result["recall_at_1"]!=measured["success_rate"]:
                    raise ValueError("Reported test success differs from the complete query log")
                summary.update(measured=True,index_bytes=selection["index_bytes"],
                               index_MB=selection["index_bytes"]/1e6,
                               config_id=cfg_id,blocks=selection.get("blocks"),swapped=selection.get("swapped",False),
                               L=selection["L"],delta=selection["delta"],
                               scratch_bytes=selection["scratch_bytes"],**measured)
                summary["feasible"]=(summary["success_rate"]>=config["target_recall"])
            else:
                reasons=result.get("exclusion_reasons")
                if not reasons:
                    raise ValueError("An unselected method needs its explicit exclusion reasons")
                summary.update(measured=False,feasible=False,reasons=reasons)
            groups.setdefault((summary["k"],method),[]).append(summary)
        resources=json.loads((folder/"resources.json").read_text())
        if resources["peak_process_bytes"]>config["peak_ram_cap_bytes"] or resources.get("interruption"):
            raise ValueError("A benchmark job exceeded its process-resource limits")
        for name in ("dataset.json","ground-truth.npz","tuning-configurations.json",
                     "tuning-queries.csv","frozen-main-selections.json","main-results.json",
                     "test-queries.csv","resources.json"):
            artifacts[str((folder/name).relative_to(root))]=digest(folder/name)
    if any(len(values)!=1 for values in hashes.values()):
        raise ValueError("All seeds of an object count must share a fixed dataset and queries")
    combined=[combine_seeds(rows,config["index_seeds"],config["target_recall"],config["index_budget_bytes"])
              for key,rows in sorted(groups.items())]
    maxima={method:max((row["k"] for row in combined if row["method"]==method and row["feasible"]),default=None)
            for method in METHODS}
    result=dict(config=config,manifest=manifest,groups=combined,largest_feasible_tested_k=maxima,
                not_run_ks=sorted(not_run_ks),
                artifacts=artifacts,publisher_sha256=digest(Path(__file__)),
                notes=["Maximum is over the tested object counts and configuration shortlist only.",
                       "Every seed must meet both allocated-index and exact-NN-within-deadline requirements.",
                       "Raw curves and query scratch are excluded from the serving-index budget.",
                       "No configurations are substituted after held-out evaluation."])
    write_json(root/"feasibility-summary.json",result)
    return result


def figures(report,destination):
    configure_paper_font()
    plt.rcParams.update({"font.size":9,"axes.titlesize":9,"axes.labelsize":9})
    fig,axes=row_figure(4)
    try:
        config=report["config"]
        ks=sorted({case["k"] for case in config["cases"]})
        for i,method in enumerate(METHODS):
            groups=sorted((row for row in report["groups"] if row["method"]==method),key=lambda row:row["k"])
            for row in groups:
                axes[0].plot(row["k"],i,MARKERS[method] if row["feasible"] else "x",
                             color=COLORS[method] if row["feasible"] else "black",
                             markersize=5,markeredgewidth=1)
            measured=[row for row in groups if row["all_seeds_measured"]]
            for ax,metric,factor in ((axes[1],"index_MB",1),(axes[2],"median_ms",1),
                                    (axes[3],"success_rate",100)):
                ax.plot([row["k"] for row in measured],[row[metric]*factor for row in measured],
                        **method_style(method),label=LABELS[method])
                ax.fill_between([row["k"] for row in measured],
                                [row[metric+"_minmax"][0]*factor for row in measured],
                                [row[metric+"_minmax"][1]*factor for row in measured],
                                color=COLORS[method],alpha=.07,linewidth=0)
        axes[0].set_yticks(range(len(METHODS)),labels=[LABELS[method] for method in METHODS])
        axes[0].invert_yaxis()
        axes[0].set_title("Feasible in all seeds")
        axes[1].set_title("Allocated index")
        axes[1].set_ylabel("Index size (MB)")
        axes[1].set_yscale("log")
        axes[1].axhline(config["index_budget_bytes"]/1e6,color="black",linestyle=":",linewidth=.8)
        axes[2].set_title("Query time")
        axes[2].set_ylabel("Median time (ms)")
        axes[2].set_yscale("log")
        axes[2].axhline(config["query_deadline_seconds"]*1000,color="black",linestyle=":",linewidth=.8)
        axes[3].set_title("Exact NN by deadline")
        axes[3].set_ylabel("Success (%)")
        measured_rates=[row["success_rate_minmax"][0]*100 for row in report["groups"]
                        if row["all_seeds_measured"]]
        lower=max(0,min(85,int(min(measured_rates)//5)*5)) if measured_rates else 0
        axes[3].set_ylim(lower,100.5)
        axes[3].axhline(config["target_recall"]*100,color="black",linestyle=":",linewidth=.8)
        for ax in axes:
            ax.set_xlabel("Objects per set")
            ax.set_xticks(ks[::2])
        fig.legend(*axes[1].get_legend_handles_labels(),loc="upper center",ncol=4,
                   frameon=False,fontsize=8,columnspacing=.7,handlelength=2.5)
        save_row(fig,destination)
    finally:
        plt.close(fig)


def summary_table(report):
    lines=[r"\begin{tabular}{lr}",r"\toprule",r"Method & Largest feasible tested $k$\\",r"\midrule"]
    for method in METHODS:
        value=report["largest_feasible_tested_k"][method]
        lines.append(f"{LABELS[method]} & {value if value is not None else '--'}"+r"\\")
    lines.extend([r"\bottomrule",r"\end{tabular}"])
    return "\n".join(lines)+"\n"


def detail_tables(report):
    lines=[]
    ks=sorted({row["k"] for row in report["groups"]})
    for start in range(0,len(ks),3):
        selected=ks[start:start+3]
        lines.extend([r"\begin{table*}[t]",r"\centering",
                      r"\caption{Measured hard-budget feasibility at one million common-path multi-curves. Every seed must satisfy the 1 GB allocated-index budget and at least 90\% exact-NN success within 10 ms. Time and memory are medians of per-seed summaries; success and timeout rates average all seed queries. Aggregates require all seeds to be measured; exclusions are retained.}",
                      r"\small",r"\begin{tabular}{rlrrrrl}",r"\toprule",
                      r"$k$ & Method & MB & median ms & on-time NN (\%) & timeout (\%) & All seeds feasible?\\",
                      r"\midrule"])
        for k in selected:
            for method in METHODS:
                row=next(row for row in report["groups"] if row["k"]==k and row["method"]==method)
                prefix=f"{k} & {LABELS[method]}"
                if row["all_seeds_measured"]:
                    lines.append(prefix+f" & {row['index_MB']:.2f} & {row['median_ms']:.2f} & "
                                 f"{100*row['success_rate']:.1f} & {100*row['timeout_rate']:.1f} & "
                                 f"{'yes' if row['feasible'] else 'no'}"+r"\\")
                else:
                    lines.append(prefix+r" & \multicolumn{4}{c}{no qualifying configuration in at least one seed} & no\\")
        lines.extend([r"\bottomrule",r"\end{tabular}",r"\end{table*}"])
    exclusions=[]
    for row in report["groups"]:
        unmeasured=[seed for seed in row["seeds"] if not seed["measured"]]
        if unmeasured:
            reasons=sorted({reason for seed in unmeasured for reason in seed["reasons"]})
            exclusions.append(f"{LABELS[row['method']]} at $k={row['k']}$: "+", ".join(reasons)+".")
    if exclusions:
        lines.extend([r"\paragraph{Configurations without a qualifying tuning result.}",
                      "These reasons cover the declared configuration search; they do not prove that all possible implementations are infeasible.",
                      " ".join(exclusions)])
    lines.append(r"\paragraph{Selected configurations.}")
    descriptions=[]
    for row in report["groups"]:
        measured=[seed for seed in row["seeds"] if seed["measured"]]
        if not measured:
            continue
        params=[]
        for seed in sorted(measured,key=lambda item:item["index_seed"]):
            block=("+".join(str(size) for size in seed["blocks"]) if seed["blocks"] else "per-curve")
            if seed.get("swapped"):
                block+=" (reversed storage/probing)"
            params.append(f"{seed['index_seed']}: {block}, $\\delta={seed['delta']:g}$, $L={seed['L']}$, "
                          f"{seed['index_MB']:.2f}\\,MB, {seed['median_ms']:.2f}\\,ms, "
                          f"{100*seed['success_rate']:.1f}\\% on-time NN success")
        descriptions.append(f"{LABELS[row['method']]} at $k={row['k']}$ ("+"; ".join(params)+").")
    lines.append(" ".join(descriptions))
    return "\n".join(lines)+"\n"


def publish(root,paper):
    report=audit(root)
    output=paper/"experiments"
    figures(report,output/"figures"/"compact-feasibility")
    summary=output/"compact-feasibility-table.tex"
    details=output/"feasibility-results.tex"
    summary.write_text("% Generated from audited hard-budget logs by fresh.publish_feasibility.\n"+summary_table(report))
    details.write_text("% Generated from audited hard-budget logs by fresh.publish_feasibility.\n"+detail_tables(report))
    paths=[summary,details,output/"figures"/"compact-feasibility.pdf"]
    write_json(output/"feasibility-provenance.json",
               dict(source=str(root/"feasibility-summary.json"),
                    source_sha256=digest(root/"feasibility-summary.json"),
                    publisher_sha256=digest(Path(__file__)),
                    outputs={str(path.relative_to(paper)):digest(path) for path in paths}))
    print("Largest feasible tested object counts:",report["largest_feasible_tested_k"])


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",type=Path,required=True)
    parser.add_argument("--paper",type=Path,required=True)
    args=parser.parse_args()
    publish(args.results.resolve(),args.paper.resolve())
