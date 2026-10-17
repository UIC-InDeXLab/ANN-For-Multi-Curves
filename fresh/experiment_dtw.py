"""Empirical DTW retrieval using unchanged grid indexes and exact DTW verification."""
import argparse
import hashlib
import json
from math import factorial,prod
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numba
import numpy as np
from numba import njit,prange

from curveset.bench import pinned
from curveset.tune import first_hit_per_curve,tables_for_recall
from fresh.dtw import dtw_distance,dtw_ground_truth,dtw_verify
from fresh.e1 import append_row,arrays_hash,monitor_worker,terminate_worker,write_json
from fresh.nba_full import digest
from fresh.poc import Joint,PerCurve,choose_blocking,dataset,family_specs,observed_peak_gate,tuning_target
from fresh.regimes import family_first_hits,timed_queries,warm


METHODS=("blocking","probe_all","store_all","per_curve-intersection")


def study_jobs(config):
    jobs=[]
    for case in config["cases"]:
        seeds=config["nba_index_seeds"] if case["kind"]=="nba" else config["synthetic_index_seeds"]
        for seed in seeds:
            jobs.append(dict(case,case_id=case["id"],index_seed=seed,id=f"{case['id']}-seed{seed}"))
    return jobs


def load_dataset(config,case):
    if case["kind"]!="nba":
        X,Q,info=dataset(config,case)
        reference=Path(config["synthetic_reference"])/f"{case['kind']}-n{case['n']}-k{case['k']}-seed1701"
        original=json.loads((reference/"dataset.json").read_text())
        if (original["dataset_sha256"]!=info["dataset_sha256"] or
                original["queries_sha256"]!=info["queries_sha256"]):
            raise ValueError("DTW and Frechet controls must use identical synthetic arrays")
        with np.load(reference/"ground-truth.npz",allow_pickle=False) as source:
            if not np.array_equal(source["queries"],Q):
                raise ValueError("Synthetic reference query order changed")
            initial_ids=source["ids"].copy()
            frechet_optimum=source["distances"].copy()
        info.update(frechet_reference=str(reference),
                    frechet_reference_sha256=digest(reference/"ground-truth.npz"),
                    initial_parent_policy="Frechet NN is only a DTW upper-bound candidate")
    else:
        root=Path(config["nba_data"])
        if not (root/"preparation-complete.json").is_file() or digest(root/"dataset.json")!=config["nba_dataset_sha256"]:
            raise ValueError("DTW NBA data must match the audited complete corpus")
        corpus=json.loads((root/"dataset.json").read_text())
        for name in ("database.npz","tune.npz","test.npz"):
            if digest(root/name)!=corpus["artifacts"][name]:
                raise ValueError("A frozen NBA curve array changed")
        with np.load(root/"database.npz",allow_pickle=False) as source:
            X=source["curves"].copy()
        with np.load(root/"tune.npz",allow_pickle=False) as source:
            Qt=source["curves"].copy()
        with np.load(root/"test.npz",allow_pickle=False) as source:
            Qtest=source["curves"].copy()
        if (len(Qt)!=config["tuning_queries"] or len(Qtest)!=config["test_queries"] or
                X.shape!=(case["n"],case["k"],config["m"],config["d"])):
            raise ValueError("NBA shapes or query splits differ from the declared study")
        Q=np.concatenate((Qt,Qtest,X[:config["warmup_queries"]]))
        ids=[];distances=[]
        for split in ("tune","test"):
            path=root/f"{split}-ground-truth.npz"
            if digest(path)!=config["nba_reference_artifacts"][path.name]:
                raise ValueError("The frozen NBA Frechet reference parents changed")
            with np.load(path,allow_pickle=False) as source:
                ids.extend(source["ids"].tolist())
                distances.extend(source["distances"].tolist())
        initial_ids=np.asarray(ids+list(range(config["warmup_queries"])),np.int64)
        frechet_optimum=np.asarray(distances+[0.]*config["warmup_queries"])
        info=dict(dataset_sha256=arrays_hash(X),queries_sha256=arrays_hash(Q),raw_curve_bytes=X.nbytes,
                  nba_corpus_sha256=digest(root/"dataset.json"),nba_source_artifacts=corpus["artifacts"],
                  nba_reference_artifacts=config["nba_reference_artifacts"],
                  warmup_source="first ten database multi-curves, untimed",
                  initial_parent_policy="Frechet NN is only a DTW upper-bound candidate")
    if len(initial_ids)!=len(Q) or np.any(initial_ids<0) or np.any(initial_ids>=len(X)):
        raise ValueError("Every initial DTW upper-bound parent must be valid")
    return X,Q,info,initial_ids,frechet_optimum


def specs(config,case):
    k=case["k"]
    if case["kind"]!="nba":
        return family_specs(k,config)
    choices={(factorial(k),1):((k,),False),(1,factorial(k)):((1,)*k,False)}
    for sizes in config["nba_blockings"]:
        if sum(sizes)!=k:
            raise ValueError("NBA blocks must partition the five slots")
        a=prod(factorial(size) for size in sizes);b=factorial(k)//a
        choices.setdefault((a,b),(tuple(sizes),False))
        choices.setdefault((b,a),(tuple(sizes),True))
    return [(a,b,sizes,swapped) for (a,b),(sizes,swapped) in sorted(choices.items())]


@njit(cache=True,parallel=True)
def reference_parent_costs(X,Q,ids):
    result=np.empty(len(Q))
    for q in prange(len(Q)):
        result[q]=dtw_distance(X[ids[q]],Q[q])
    return result


def exact_truth(X,Q,info,initial_ids,output,case):
    cache=output.parent/"ground-truth-cache"
    cache.mkdir(exist_ok=True)
    stem=cache/case["case_id"]
    record_path=stem.with_suffix(".json");arrays_path=stem.with_suffix(".npz")
    kernel_hash=digest(Path(__file__).with_name("dtw.py"))
    initial_hash=arrays_hash(initial_ids)
    if record_path.exists():
        record=json.loads(record_path.read_text())
        if (record["dataset_sha256"]!=info["dataset_sha256"] or
                record["queries_sha256"]!=info["queries_sha256"] or
                record["initial_parents_sha256"]!=initial_hash or
                record["dtw_kernel_sha256"]!=kernel_hash or
                digest(arrays_path)!=record["array_sha256"]):
            raise ValueError("Cached exact DTW ground truth has stale provenance")
        with np.load(arrays_path,allow_pickle=False) as arrays:
            ids=arrays["ids"].copy();distances=arrays["distances"].copy()
            reference=arrays["frechet_parent_dtw"].copy()
        reused=True
    else:
        started=time.perf_counter()
        ids,distances=dtw_ground_truth(X,Q,initial_ids)
        reference=reference_parent_costs(X,Q,initial_ids)
        if (not np.isfinite(distances).all() or np.any(ids<0) or np.any(ids>=len(X))
                or np.any(distances>reference+1e-8)):
            raise AssertionError("Exact DTW scan produced invalid minima")
        np.savez(arrays_path,ids=ids,distances=distances,frechet_parent_dtw=reference)
        write_json(record_path,dict(dataset_sha256=info["dataset_sha256"],queries_sha256=info["queries_sha256"],
                                   initial_parents_sha256=initial_hash,
                                   dtw_kernel_sha256=kernel_hash,array_sha256=digest(arrays_path),
                                   full_collection_size=len(X),preparation_seconds=time.perf_counter()-started))
        reused=False
    np.savez(output/"ground-truth.npz",queries=Q,ids=ids,distances=distances,frechet_parent_dtw=reference)
    return ids,distances,reference,reused


def worker(config,case,output):
    write_json(output/"active_configuration.json",dict(method="DTW-ground-truth",phase="preparation"))
    X,Q,info,initial_ids,frechet_optimum=load_dataset(config,case)
    ids,distances,reference,reused=exact_truth(X,Q,info,initial_ids,output,case)
    h,s=config["tuning_queries"],config["test_queries"]
    Qt,Qtest,Qwarm=Q[:h],Q[h:h+s],Q[h+s:]
    NN=X[ids[:h]]
    write_json(output/"dataset.json",dict(info,**case,metric=config["curve_metric"],
                                         cached_ground_truth=reused,
                                         mean_test_dtw_optimum=float(distances[h:h+s].mean()),
                                         frechet_parent_not_dtw_optimal=float(np.mean(reference[h:h+s]>distances[h:h+s]+1e-8))))
    np.savez(output/"frechet-reference.npz",ids=initial_ids,distances=frechet_optimum)
    manifest=json.loads((output.parent/"manifest.json").read_text())
    seed,k,d=case["index_seed"],case["k"],config["d"]
    trials=[];choices={}

    def state(method,phase,count=None):
        value=dict(method=method,seed=seed,phase=phase)
        if count is not None:value["completed_queries"]=count
        write_json(output/"active_configuration.json",value)

    def exclude(base,reason):
        trials.append(dict(base,status=reason,index_seed=seed))
        write_json(output/"tuning-configurations.json",trials)

    def record(base,index,call):
        warm(call,Qwarm)
        state(base["config_id"],"DTW-tuning")
        rows,complete=timed_queries(call,Qt,distances[:h],config,
                                   lambda count:state(base["config_id"],"DTW-tuning",count))
        successes=sum(row["exact_nn"] for row in rows)
        recall=successes/len(Qt)
        row=dict(base,status="qualified" if complete and recall>=tuning_target(config) else
                 "tuning recall below target" if complete else "tuning timeout",
                 index_seed=seed,tuning_recall=recall,tuning_complete=complete,
                 tuning_median_seconds=float(np.median([r["seconds"] for r in rows])) if complete else None,
                 build_seconds=index.build_seconds)
        for q,result in enumerate(rows):
            append_row(output/"tuning-queries.csv",dict(config_id=base["config_id"],query=q,**result))
        trials.append(row)
        write_json(output/"tuning-configurations.json",trials)
        return row

    def baseline(method,row):
        if row["status"]=="qualified" and (method not in choices or
                row["tuning_median_seconds"]<choices[method]["tuning_median_seconds"]):
            choices[method]=row

    deltas=config["nba_deltas"] if case["kind"]=="nba" else config["deltas"]
    for delta in deltas:
        shifts=np.random.default_rng(seed).uniform(0,delta,(config["Lmax"],k,d))
        for a,b,sizes,swapped in specs(config,case):
            cfg_id=f"joint-d{delta:g}-a{a}-swap{int(swapped)}"
            base=dict(config_id=cfg_id,method="blocking-family",delta=delta,blocks=list(sizes),
                      swapped=swapped,stored=a,probed=b)
            reason=observed_peak_gate(X,len(X),k,a,1,config)
            if reason:
                exclude(base,reason)
                continue
            state(cfg_id,"collision-tuning")
            hit=family_first_hits(Qt,NN,shifts,delta,np.asarray(sizes),swapped)
            L=tables_for_recall(hit,tuning_target(config),config["Lmax"])
            if L is None:
                exclude(base,"no tuned L in grid")
                continue
            base["L"]=L
            reason=observed_peak_gate(X,len(X),k,a,L,config)
            if reason:
                exclude(base,reason)
                continue
            state(cfg_id,"building")
            signal.alarm(config["build_deadline_seconds"])
            try:index=Joint(X,delta,L,seed,sizes,swapped)
            finally:signal.alarm(0)
            base.update(index_bytes=index.index_bytes,entries=index.entries,raw_entries=index.raw_entries,
                        scratch_bytes=index.scratch_bytes)
            if index.index_bytes>config["index_budget_bytes"]:
                exclude(base,"over index storage")
            else:
                call=lambda query:index.query(X,query,config,verifier=dtw_verify)
                row=record(base,index,call)
                if a==1:baseline("probe_all",row)
                if a==factorial(k):baseline("store_all",row)
                del call
            del index
        sp=np.random.default_rng(seed).uniform(0,delta,(config["Lmax"],d))
        L=tables_for_recall(first_hit_per_curve(Qt,NN,sp,delta),tuning_target(config),config["Lmax"])
        base=dict(config_id=f"per_curve-intersection-d{delta:g}",method="per_curve-intersection",
                  delta=delta,L=L,blocks=None,swapped=False)
        if L is None:
            exclude(base,"no tuned L in grid")
            continue
        state(base["config_id"],"building")
        signal.alarm(config["build_deadline_seconds"])
        try:index=PerCurve(X,delta,L,seed)
        finally:signal.alarm(0)
        base.update(index_bytes=index.index_bytes,entries=index.entries,raw_entries=index.raw_entries,
                    scratch_bytes=index.scratch_bytes)
        if index.index_bytes>config["index_budget_bytes"]:
            exclude(base,"over index storage")
        else:
            call=lambda query:index.query(X,query,config,verifier=dtw_verify)
            baseline("per_curve-intersection",record(base,index,call))
            del call
        del index
    selected=choose_blocking(trials,config["blocking_main_latency_weight"],config)
    if selected is not None:choices["blocking"]=selected
    write_json(output/"frozen-main-selections.json",choices)
    evaluated={}
    for cfg_id,row in {row["config_id"]:row for row in choices.values()}.items():
        state(cfg_id,"building-final")
        signal.alarm(config["build_deadline_seconds"])
        try:
            if row["method"]=="blocking-family":
                index=Joint(X,row["delta"],row["L"],seed,row["blocks"],row["swapped"])
                call=lambda query:index.query(X,query,config,verifier=dtw_verify)
            else:
                index=PerCurve(X,row["delta"],row["L"],seed)
                call=lambda query:index.query(X,query,config,verifier=dtw_verify)
        finally:signal.alarm(0)
        if index.index_bytes!=row["index_bytes"]:
            raise AssertionError("DTW index bytes changed between tuning and test")
        warm(call,Qwarm)
        state(cfg_id,"DTW-test")
        rows,complete=timed_queries(call,Qtest,distances[h:h+s],config,
                                   lambda count:state(cfg_id,"DTW-test",count))
        for q,result in enumerate(rows):
            append_row(output/"test-queries.csv",dict(
                config_id=cfg_id,commit=manifest["commit"],config_sha256=manifest["config_sha256"],
                dataset_sha256=info["dataset_sha256"],queries_sha256=info["queries_sha256"],
                index_seed=seed,metric="DTW",phase="test",query=q,nn_distance=float(distances[h+q]),**result))
        evaluated[cfg_id]=dict(row,test_status="complete" if complete else "test timeout",
                               test_queries=len(rows),build_seconds=index.build_seconds)
        del call,index
    main=[]
    for method in METHODS:
        if method in choices:
            main.append(dict(evaluated[choices[method]["config_id"]],display_method=method))
        else:
            if method=="blocking":
                relevant=[r for r in trials if r["method"]=="blocking-family"]
            elif method in ("store_all","probe_all"):
                count=factorial(k) if method=="store_all" else 1
                relevant=[r for r in trials if r["method"]=="blocking-family" and r["stored"]==count]
            else:relevant=[r for r in trials if r["method"]==method]
            main.append(dict(display_method=method,index_seed=seed,test_status="no feasible DTW tuning configuration",
                             exclusion_reasons=sorted({r["status"] for r in relevant})))
    write_json(output/"main-results.json",main)
    print(f"CASE {case['id']} complete",flush=True)


def main(args):
    config_path=Path(args.config).resolve();config=json.loads(config_path.read_text())
    output=Path(args.output).resolve()
    if args.worker is not None:
        worker(config,study_jobs(config)[args.worker],output)
        return
    if subprocess.check_output(["git","status","--porcelain"],text=True).strip():
        raise RuntimeError("Run only from a clean frozen source tree")
    if numba.get_num_threads()!=config["build_threads"] or int(os.environ.get("PIN_CORE","-1"))!=config["pin_core"]:
        raise ValueError("CPU settings differ from the frozen DTW protocol")
    output.mkdir(parents=True,exist_ok=False)
    commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    write_json(output/"config.json",config)
    write_json(output/"manifest.json",dict(commit=commit,config_sha256=digest(config_path),started=time.time(),
                                         python=sys.version,numpy=np.__version__,numba=numba.__version__,
                                         cpu_model=next(line.split(":",1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                                                        if line.startswith("model name")),
                                         metric=config["curve_metric"],pin_core=config["pin_core"],
                                         build_threads=numba.get_num_threads(),dtw_kernel_sha256=digest(Path(__file__).with_name("dtw.py"))))
    started=time.monotonic();cases=[]
    for i,case in enumerate(study_jobs(config)):
        remaining=config["run_cap_seconds"]-(time.monotonic()-started)
        if remaining<=0:raise RuntimeError("DTW study exceeded its run cap")
        folder=output/case["id"];folder.mkdir()
        with (folder/"worker.log").open("w") as log:
            process=subprocess.Popen([sys.executable,"-u","-m","fresh.experiment_dtw",
                                      "--config",str(config_path),"--output",str(folder),"--worker",str(i)],
                                     stdout=log,stderr=subprocess.STDOUT)
            try:code,interruption=monitor_worker(process,folder,config,remaining)
            finally:terminate_worker(process)
        status=interruption or ("complete" if code==0 else "implementation or build failure")
        cases.append(dict(case=case["id"],case_id=case["case_id"],index_seed=case["index_seed"],
                          status=status,returncode=code))
        write_json(output/"cases.json",cases)
        print(f"{case['id']}: {status}",flush=True)
        if code or interruption:raise RuntimeError(f"Stopped at {folder}: {status}")
    write_json(output/"completion.json",dict(completed=time.time(),elapsed_seconds=time.monotonic()-started))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--worker",type=int)
    main(parser.parse_args())
