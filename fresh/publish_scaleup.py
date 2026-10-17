"""Publish larger synthetic comparisons from complete, paired benchmark logs."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fresh.aggregate_replications import aggregate
from fresh.e1 import arrays_hash, write_json
from fresh.nba_full import digest
from fresh.nba_paper_examples import configure_paper_font
from fresh.plot_paper_experiments import row_figure, save_row
from fresh.plot_poc import COLORS, summarize
from fresh.poc import jobs
from fresh.plot_styles import MARKERS, method_style


METHODS=("blocking","probe_all","store_all","per_curve-intersection","brute-force")
DISPLAY_NAMES={"blocking":"Blocking","probe_all":"Probe-all","store_all":"Store-all",
               "per_curve-intersection":"Intersection",
               "brute-force":"Exact scan"}
PROTOCOL_EXCEPTIONS={"experiment","stage","main_plot_n","cases"}


def checked_reports(original_root, extension_root):
    for root in (original_root,extension_root):
        if not (root/"completion.json").is_file():
            raise ValueError(f"Benchmark is not complete: {root}")
        config=json.loads((root/"config.json").read_text())
        expected={row["id"] for row in jobs(config)}
        cases=json.loads((root/"cases.json").read_text())
        if (len(cases)!=len(expected) or {row["case"] for row in cases}!=expected
                or any(row["status"]!="complete" or row["returncode"]!=0 for row in cases)):
            raise ValueError("Every configured case and index seed must complete")
    manifest=json.loads((extension_root/"manifest.json").read_text())
    config=json.loads((extension_root/"config.json").read_text())
    artifact_hashes={}
    for case in json.loads((extension_root/"cases.json").read_text()):
        folder=extension_root/case["case"]
        dataset=json.loads((folder/"dataset.json").read_text())
        with np.load(folder/"ground-truth.npz",allow_pickle=False) as truth:
            optimum=truth["distances"][config["tuning_queries"]:
                                       config["tuning_queries"]+config["test_queries"]]
            if arrays_hash(truth["queries"])!=dataset["queries_sha256"]:
                raise ValueError("Ground-truth queries differ from the declared dataset")
        if len(optimum)!=config["test_queries"] or not np.isfinite(optimum).all():
            raise ValueError("A scale-up job has invalid exact ground truth")
        with (folder/"test-queries.csv").open() as handle:
            for row in csv.DictReader(handle):
                q=int(row["query"])
                if (row["commit"]!=manifest["commit"] or
                        row["config_sha256"]!=manifest["config_sha256"] or
                        row["dataset_sha256"]!=dataset["dataset_sha256"] or
                        int(row["index_seed"])!=case["index_seed"] or
                        not 0<=q<len(optimum) or float(row["nn_distance"])!=optimum[q]):
                    raise ValueError("A query log differs from its frozen protocol or exact ground truth")
                found=int(row["id"])>=0
                if (row["timeout"]!="False" or (row["found"]=="True")!=found or
                        found and int(row["id"])>=dataset["n"]):
                    raise ValueError("A query log contains an invalid answer or timeout")
                exact=found and float(row["distance"])<=optimum[q]+1e-8
                if (row["exact_nn"]=="True")!=exact:
                    raise ValueError("Recorded recall disagrees with exact ground truth")
        for name in ("dataset.json","ground-truth.npz","frozen-main-selections.json",
                     "tuning-configurations.json","main-results.json","test-queries.csv","resources.json"):
            artifact_hashes[str((folder/name).relative_to(extension_root))]=digest(folder/name)
    original=json.loads((original_root/"replicated-summary.json").read_text())
    extension_config=json.loads((extension_root/"config.json").read_text())
    before=original["config"]
    if set(before)!=set(extension_config):
        raise ValueError("The scale-up protocol has different configuration fields")
    if any(before[key]!=extension_config[key] for key in before if key not in PROTOCOL_EXCEPTIONS):
        raise ValueError("Scale-up must preserve the sampling, tuning, and resource protocol")
    for field in ("python","numpy","numba","cpu_model","pin_core","build_threads"):
        if manifest[field]!=original["manifest"][field]:
            raise ValueError(f"Benchmark environments differ: {field}")
    summarize(extension_root)
    extension=aggregate(extension_root)
    hashes={}
    for root in (original_root,extension_root):
        for row in json.loads((root/"summary.json").read_text())["main"]:
            key=(row["kind"],row["k"])
            hashes.setdefault(key,set()).add(row["queries_sha256"])
    if any(len(values)!=1 for values in hashes.values()):
        raise ValueError("Nested collection sizes must use identical query sets")
    expected_cases={(row["kind"],row["n"],row["k"]) for row in extension_config["cases"]}
    for kind,n,k in expected_cases:
        rows=[row for row in extension["main"] if row["display_method"] in METHODS
              and (row["kind"],row["n"],row["k"])==(kind,n,k)]
        if len(rows)!=len(METHODS) or {row["display_method"] for row in rows}!=set(METHODS):
            raise ValueError("A scale-up case is missing a baseline")
        for row in rows:
            if row["display_method"]!="store_all" and row["test_status"]!="complete":
                raise ValueError("Do not publish incomplete indexed or exact-scan comparisons")
            if row["display_method"]=="brute-force" and row["recall_at_1"]!=1:
                raise ValueError("Exact-scan validation must recover every true nearest neighbor")
        alternatives=[row for row in extension["ablation"]
                      if (row["kind"],row["n"],row["k"])==(kind,n,k)]
        if (len(alternatives)!=len(extension_config["ablation_latency_weights"]) or
                {row["latency_weight"] for row in alternatives}!=
                set(extension_config["ablation_latency_weights"]) or
                any(row["test_status"]!="complete" for row in alternatives)):
            raise ValueError("The scale-up weight ablation is incomplete")
    write_json(extension_root/"publication-audit.json",
               dict(artifacts=artifact_hashes,manifest_sha256=digest(extension_root/"manifest.json"),
                    exact_ground_truth_checked=True,all_seeds_complete=True))
    return original,extension


def scale_rows(original,extension):
    rows=[row for report in (original,extension) for row in report["main"]
          if row["kind"]=="conjunctive" and row["k"]==8 and row["display_method"] in METHODS]
    for n in (100000,1000000,2000000,5000000):
        group=[row for row in rows if row["n"]==n]
        if len(group)!=len(METHODS) or {row["display_method"] for row in group}!=set(METHODS):
            raise ValueError("Scaling figure requires every declared size and baseline")
    return rows


def draw_scaling(rows,destination):
    configure_paper_font()
    plt.rcParams.update({"font.size":9,"axes.titlesize":9,"axes.labelsize":9})
    fig,axes=row_figure(3)
    try:
        for method in METHODS:
            values=sorted((row for row in rows if row["display_method"]==method
                           and row["test_status"]=="complete"),key=lambda row:row["n"])
            if not values:
                continue
            color=COLORS.get(method,"#9c6644")
            x=[row["n"]/1e6 for row in values]
            for ax,metric,factor in ((axes[0],"median_ms",1),(axes[1],"index_MB",1),
                                    (axes[2],"recall_at_1",100)):
                if metric=="index_MB" and method=="brute-force":
                    continue
                y=[factor*row[metric] for row in values]
                ax.plot(x,y,**method_style(method),label=DISPLAY_NAMES[method])
                limits=[row[metric+"_seed_minmax"] for row in values]
                ax.fill_between(x,[factor*low for low,high in limits],
                                [factor*high for low,high in limits],color=color,alpha=.07,linewidth=0)
                if metric=="median_ms":
                    for point,row in zip(x,values):
                        if not row["test_target_met"]:
                            ax.plot(point,row[metric],marker=MARKERS[method],linestyle="none",
                                    markersize=3.5,markeredgewidth=.55,
                                    markerfacecolor="white",markeredgecolor=color)
        for ax,title,label in zip(axes,("Query time","Index size","Exact-NN recall"),
                                  ("Time (ms)","Index size (MB)","Recall (%)")):
            ax.set_title(title)
            ax.set_xlabel("Multi-curves (millions)")
            ax.set_xscale("log")
            ax.set_xticks([.1,1,2,5],labels=["0.1","1","2","5"])
            ax.minorticks_off()
            ax.set_ylabel(label)
        axes[0].set_yscale("log")
        axes[1].set_yscale("log")
        recalls=[min(row["recall_at_1"],row["recall_at_1_seed_minmax"][0])*100
                 for row in rows if row["test_status"]=="complete"]
        axes[2].set_ylim(max(0,min(90,int(min(recalls)//5)*5)),100.5)
        axes[2].axhline(90,color="black",linestyle=":",linewidth=.7)
        fig.legend(*axes[0].get_legend_handles_labels(),loc="upper center",ncol=5,
                   frameon=False,fontsize=8,columnspacing=.8,handlelength=2.5)
        save_row(fig,destination)
    finally:
        plt.close(fig)


def appendix_table(extension):
    lines=[
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Larger synthetic collections ($k=8$, $m=32$). Time and index size are medians across three seeds; recall averages the 1,000 held-out queries per seed. Every method uses tuning-only selection and exact candidate verification. Raw curves are separate from index size.}",
        r"\label{tab:exp-scaleup}",
        r"\small",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Paths & Method & $n$ (millions) & MB & median ms & p95 ms & Recall (\%)\\",
        r"\midrule",
    ]
    ordered=sorted((row for row in extension["main"] if row["display_method"] in METHODS),
                   key=lambda row:(row["kind"],row["n"],METHODS.index(row["display_method"])))
    for row in ordered:
        prefix=f"{'Common' if row['kind']=='conjunctive' else 'Rare'} & {DISPLAY_NAMES[row['display_method']]} & {row['n']/1e6:g}"
        if row["test_status"]=="complete":
            lines.append(prefix+f" & {row['index_MB']:.2f} & {row['median_ms']:.2f} & "
                         f"{row['p95_ms']:.2f} & {100*row['recall_at_1']:.1f}"+r"\\")
        else:
            reasons={reason for seed in row["seed_statuses"] for reason in seed["reasons"]}
            if not reasons or not reasons<={"not run: integer capacity","not run: peak RAM estimate"}:
                raise ValueError("An excluded store-all run needs its explicit construction-resource reason")
            reason=("integer capacity" if reasons=={"not run: integer capacity"} else
                    "construction RAM estimate" if reasons=={"not run: peak RAM estimate"} else
                    "construction RAM or integer capacity")
            lines.append(prefix+r" & \multicolumn{4}{c}{not built: "+reason+r"}\\")
    lines.extend([r"\bottomrule",r"\end{tabular}",r"\end{table*}"])
    return "\n".join(lines)+"\n"


def ablation_table(extension):
    lines=[
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Storage--latency preferences at larger sizes ($k=8$). Each setting is selected on tuning queries only. Time and storage are medians across three seeds; recall averages all held-out queries. Different seeds may select different block sizes. The default is $\lambda=1$; the other settings are ablations, not additional main methods.}",
        r"\label{tab:exp-scaleup-weights}",
        r"\small",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Paths & $n$ (millions) & $\lambda$ & MB & median ms & Recall (\%)\\",
        r"\midrule",
    ]
    for row in sorted(extension["ablation"],key=lambda row:(row["kind"],row["n"],row["latency_weight"])):
        if row["test_status"]!="complete":
            raise ValueError("Do not publish a partial weight-ablation seed group")
        lines.append(f"{'Common' if row['kind']=='conjunctive' else 'Rare'} & {row['n']/1e6:g}"
                     f" & {row['latency_weight']:g} & {row['index_MB']:.2f} & "
                     f"{row['median_ms']:.2f} & {100*row['recall_at_1']:.1f}"+r"\\")
    lines.extend([r"\bottomrule",r"\end{tabular}",r"\end{table*}"])
    return "\n".join(lines)+"\n"


def publish(original_root,extension_root,paper):
    original,extension=checked_reports(original_root,extension_root)
    rows=scale_rows(original,extension)
    figures=paper/"experiments"/"figures"
    draw_scaling(rows,figures/"compact-scaleup")
    fragment=paper/"experiments"/"scaleup-results.tex"
    fragment.write_text("% Generated from complete scale-up reports by fresh.publish_scaleup.\n"
                        +appendix_table(extension)+ablation_table(extension))
    receipt=dict(publisher_sha256=digest(Path(__file__)),
                 original_source=dict(file=str(original_root/"replicated-summary.json"),
                                     sha256=digest(original_root/"replicated-summary.json")),
                 extension_source=dict(file=str(extension_root/"replicated-summary.json"),
                                      sha256=digest(extension_root/"replicated-summary.json")),
                 audit_source=dict(file=str(extension_root/"publication-audit.json"),
                                   sha256=digest(extension_root/"publication-audit.json")),
                 outputs={str(path.relative_to(paper)):digest(path) for path in
                          (fragment,figures/"compact-scaleup.pdf")},
                 same_sampling_and_tuning=True,changed_test_outcomes=False,retuned=False)
    write_json(paper/"experiments"/"scaleup-provenance.json",receipt)
    print("Published larger-size scaling figure and full appendix table from completed comparisons.")
    for row in extension["main"]:
        if row["display_method"] not in METHODS:
            continue
        print(row["kind"],row["n"],row["display_method"],row["test_status"],
              row.get("median_ms"),row.get("index_MB"),row.get("recall_at_1"))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original",type=Path,required=True)
    parser.add_argument("--extension",type=Path,required=True)
    parser.add_argument("--paper",type=Path,required=True)
    args=parser.parse_args()
    publish(args.original.resolve(),args.extension.resolve(),args.paper.resolve())
