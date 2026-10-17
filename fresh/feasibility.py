"""Measure object-count feasibility under allocated-index and query deadlines."""
import argparse
import hashlib
import json
from math import ceil, factorial
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import numba

from curveset.bench import pinned
from curveset.tune import first_hit_per_curve, tables_for_recall
from fresh.e1 import append_row, monitor_worker, terminate_worker, write_json
from fresh.poc import Joint, PerCurve, dataset, family_specs, jobs, observed_peak_gate, tuning_target
from fresh.regimes import family_first_hits, ground_truth


METHODS=("blocking","store_all","probe_all","per_curve-intersection","per_curve-rarest")


def normalize_query(result,optimum,elapsed,deadline):
    row=dict(result)
    raw_id=int(result["id"])
    raw_distance=result.get("distance")
    timeout=bool(result["timeout"] or elapsed>deadline)
    found=raw_id>=0 and not timeout
    row.update(seconds=float(elapsed),timeout=timeout,found=found,
               id=raw_id if found else -1,distance=float(raw_distance) if found else None,
               raw_id=raw_id,raw_distance=float(raw_distance) if raw_id>=0 else None,
               exact_nn=bool(found and raw_distance<=optimum+1e-8))
    row["ratio"]=(float(raw_distance/optimum) if optimum>0 else
                  1.0 if raw_distance==0 else float("inf")) if found else None
    return row


def measure_queries(query,queries,optimum,config,tuning=False,progress=None):
    total=len(queries)
    needed=ceil(tuning_target(config)*total)
    rows=[]
    successes=0
    with pinned():
        for q,curve in enumerate(queries):
            started=time.perf_counter()
            result=query(curve)
            elapsed=time.perf_counter()-started
            row=normalize_query(result,optimum[q],elapsed,config["query_deadline_seconds"])
            rows.append(row)
            successes+=int(row["exact_nn"])
            if progress is not None and (q+1)%100==0:
                progress(q+1)
            if tuning and successes+total-len(rows)<needed:
                return rows,False
    return rows,True


def warm_index(index,X,queries,config,rarest=None,active=None):
    relaxed=dict(config,query_deadline_seconds=config["compilation_deadline_seconds"])
    with pinned():
        if isinstance(index,Joint):
            result=index.query(X,queries[0],relaxed)
            call=lambda curve:index.query(X,curve,config)
        else:
            result=index.query(X,queries[0],relaxed,rarest,active)
            call=lambda curve:index.query(X,curve,config,rarest,active)
        if result["timeout"]:
            raise RuntimeError("Untimed kernel compilation query exceeded the compilation deadline")
        for curve in queries:
            call(curve)
    return call


def preflight_joint(X,n,k,a,b,config,L=1):
    if n>=2**31 or n*a>=2**32:
        return "excluded: integer capacity"
    if (a+b)*k>config["index_budget_bytes"]:
        return "excluded: mandatory order arrays exceed serving-index budget"
    reason=observed_peak_gate(X,n,k,a,L,config)
    return "excluded: construction RAM estimate" if reason else None


def select_fastest(rows):
    eligible=[row for row in rows if row["status"]=="qualified"]
    if not eligible:
        return None
    return min(eligible,key=lambda row:(row["tuning_median_seconds"],row["index_bytes"],row["config_id"]))


def tuning_summary(rows,total,complete,config):
    successes=sum(row["exact_nn"] for row in rows)
    qualified=complete and successes/total>=tuning_target(config)
    return dict(status="qualified" if qualified else "rejected: on-time tuning recall",
                tuning_queries=len(rows),tuning_requested_queries=total,
                tuning_complete=complete,tuning_early_rejected=not complete,
                tuning_successes=successes,tuning_recall=successes/total,
                tuning_success_upper_bound=(successes+total-len(rows))/total,
                tuning_median_seconds=float(np.median([row["seconds"] for row in rows])) if complete else None,
                tuning_timeout_count=sum(row["timeout"] for row in rows))


def worker(config,case,output):
    config=dict(config)
    write_json(output/"active_configuration.json",dict(method="ground-truth",phase="preparation"))
    X,Q,info=dataset(config,case)
    started=time.perf_counter()
    ids,distances,relevant=ground_truth(X,Q,config["relevance_radius"])
    h,s=config["tuning_queries"],config["test_queries"]
    Qt,Qtest,Qwarm=Q[:h],Q[h:h+s],Q[h+s:]
    NN=X[ids[:h]]
    write_json(output/"dataset.json",dict(info,**case,preparation_seconds=time.perf_counter()-started,
                                         mean_test_nn_distance=float(distances[h:h+s].mean()),
                                         mean_test_relevant_sets=float(relevant[h:h+s].mean())))
    np.savez(output/"ground-truth.npz",queries=Q,ids=ids,distances=distances,relevant=relevant)
    manifest=json.loads((output.parent/"manifest.json").read_text())
    seed,k,d=case["index_seed"],case["k"],config["d"]
    trials=[]

    def state(method,phase,count=None):
        row=dict(method=method,seed=seed,phase=phase)
        if count is not None:
            row["completed_queries"]=count
        write_json(output/"active_configuration.json",row)

    def exclude(base,reason):
        trials.append(dict(base,status=reason,index_seed=seed))
        write_json(output/"tuning-configurations.json",trials)

    def record(base,index,call):
        memory=base["index_bytes"]
        if memory>config["index_budget_bytes"]:
            exclude(base,"excluded: measured serving-index storage")
            return
        state(base["config_id"],"tuning")
        rows,complete=measure_queries(call,Qt,distances[:h],config,tuning=True,
                                     progress=lambda count:state(base["config_id"],"tuning",count))
        row=dict(base,index_seed=seed,build_seconds=index.build_seconds,
                 **tuning_summary(rows,len(Qt),complete,config))
        for q,result in enumerate(rows):
            append_row(output/"tuning-queries.csv",dict(config_id=base["config_id"],query=q,**result))
        trials.append(row)
        write_json(output/"tuning-configurations.json",trials)

    for delta in config["deltas"]:
        shifts=np.random.default_rng(seed).uniform(0,delta,(config["Lmax"],k,d))
        for a,b,sizes,swapped in family_specs(k,config):
            cfg_id=f"joint-d{delta:g}-a{a}-swap{int(swapped)}"
            base=dict(config_id=cfg_id,method="blocking-family",delta=delta,
                      blocks=list(sizes),swapped=swapped,stored=a,probed=b)
            reason=preflight_joint(X,len(X),k,a,b,config)
            if reason:
                exclude(base,reason)
                continue
            state(cfg_id,"collision-tuning")
            hit=family_first_hits(Qt,NN,shifts,delta,np.asarray(sizes),swapped)
            L=tables_for_recall(hit,tuning_target(config),config["Lmax"])
            if L is None:
                exclude(base,"excluded: no tuned table count")
                continue
            base["L"]=L
            reason=preflight_joint(X,len(X),k,a,b,config,L)
            if reason:
                exclude(base,reason)
                continue
            state(cfg_id,"building")
            signal.alarm(config["build_deadline_seconds"])
            try:
                index=Joint(X,delta,L,seed,sizes,swapped)
            finally:
                signal.alarm(0)
            base.update(index_bytes=index.index_bytes,scratch_bytes=index.scratch_bytes,
                        entries=index.entries,raw_entries=index.raw_entries)
            if index.index_bytes>config["index_budget_bytes"]:
                exclude(base,"excluded: measured serving-index storage")
            else:
                state(cfg_id,"warmup")
                call=warm_index(index,X,Qwarm,config)
                record(base,index,call)
                del call
            del index

        sp=np.random.default_rng(seed).uniform(0,delta,(config["Lmax"],d))
        L=tables_for_recall(first_hit_per_curve(Qt,NN,sp,delta),tuning_target(config),config["Lmax"])
        if L is None:
            for method in METHODS[-2:]:
                exclude(dict(config_id=f"{method}-d{delta:g}",method=method,delta=delta),
                        "excluded: no tuned table count")
            continue
        if (len(X)*k>=2**32 or
                X.nbytes+len(X)*k*(64+13*L)>config["peak_ram_cap_bytes"]):
            for method in METHODS[-2:]:
                exclude(dict(config_id=f"{method}-d{delta:g}",method=method,delta=delta,L=L),
                        "excluded: per-curve construction capacity")
            continue
        state(f"per_curve-d{delta:g}","building")
        signal.alarm(config["build_deadline_seconds"])
        try:
            index=PerCurve(X,delta,L,seed)
        finally:
            signal.alarm(0)
        for method,rarest in (("per_curve-intersection",False),("per_curve-rarest",True)):
            for active in (range(1,L+1) if rarest else (L,)):
                memory=sum(table.nbytes for table in index.tables[:active])+active*d*8
                base=dict(config_id=f"{method}-d{delta:g}-L{active}",method=method,
                          delta=delta,L=active,index_bytes=memory,scratch_bytes=index.scratch_bytes,
                          entries=sum(table.N for table in index.tables[:active]),
                          raw_entries=len(X)*k*active,blocks=None,swapped=False)
                if memory>config["index_budget_bytes"]:
                    exclude(base,"excluded: measured serving-index storage")
                    continue
                state(base["config_id"],"warmup")
                call=warm_index(index,X,Qwarm,config,rarest,active)
                record(base,index,call)
                del call
        del index

    choices={}
    for method in METHODS:
        if method=="blocking":
            options=[row for row in trials if row["method"]=="blocking-family"]
        elif method=="probe_all":
            options=[row for row in trials if row["method"]=="blocking-family" and row["stored"]==1]
        elif method=="store_all":
            options=[row for row in trials if row["method"]=="blocking-family" and row["stored"]==factorial(k)]
        else:
            options=[row for row in trials if row["method"]==method]
        selected=select_fastest(options)
        if selected is not None:
            choices[method]=selected
    write_json(output/"frozen-main-selections.json",choices)
    query_file=output/"test-queries.csv"
    evaluated={}
    for cfg_id,row in {row["config_id"]:row for row in choices.values()}.items():
        state(cfg_id,"building-final")
        signal.alarm(config["build_deadline_seconds"])
        try:
            if row["method"]=="blocking-family":
                index=Joint(X,row["delta"],row["L"],seed,row["blocks"],row["swapped"])
                call=warm_index(index,X,Qwarm,config)
                memory=index.index_bytes
            else:
                index=PerCurve(X,row["delta"],row["L"],seed)
                call=warm_index(index,X,Qwarm,config,row["method"]=="per_curve-rarest",row["L"])
                memory=index.index_bytes
        finally:
            signal.alarm(0)
        if memory!=row["index_bytes"]:
            raise AssertionError("Allocated serving storage changed between tuning and test")
        state(cfg_id,"test")
        rows,complete=measure_queries(call,Qtest,distances[h:h+s],config,
                                     progress=lambda count:state(cfg_id,"test",count))
        if not complete or len(rows)!=s:
            raise AssertionError("Feasibility test must retain every held-out query")
        success=sum(result["exact_nn"] for result in rows)/s
        for q,result in enumerate(rows):
            append_row(query_file,dict(config_id=cfg_id,commit=manifest["commit"],
                                      config_sha256=manifest["config_sha256"],
                                      dataset_sha256=info["dataset_sha256"],index_seed=seed,
                                      phase="test",query=q,nn_distance=float(distances[h+q]),**result))
        evaluated[cfg_id]=dict(row,test_status="complete",test_queries=s,
                               recall_at_1=success,timeout_rate=sum(result["timeout"] for result in rows)/s,
                               feasible=memory<=config["index_budget_bytes"] and success>=config["target_recall"],
                               median_ms=float(np.median([result["seconds"] for result in rows])*1000),
                               p95_ms=float(np.quantile([result["seconds"] for result in rows],.95)*1000))
        del call,index
    if not query_file.exists():
        query_file.write_text("config_id,commit,config_sha256,dataset_sha256,index_seed,phase,query,"
                              "nn_distance,id,distance,found,exact_nn,timeout,seconds,candidates,"
                              "verified,probes,postings\n")
    main=[]
    for method in METHODS:
        if method in choices:
            main.append(dict(evaluated[choices[method]["config_id"]],display_method=method))
        else:
            if method=="blocking":
                options=[row for row in trials if row["method"]=="blocking-family"]
            elif method in ("probe_all","store_all"):
                count=1 if method=="probe_all" else factorial(k)
                options=[row for row in trials if row["method"]=="blocking-family" and row["stored"]==count]
            else:
                options=[row for row in trials if row["method"]==method]
            main.append(dict(display_method=method,index_seed=seed,feasible=False,
                             test_status="no qualifying tuning configuration",
                             exclusion_reasons=sorted({row["status"] for row in options})))
    write_json(output/"main-results.json",main)
    print(f"CASE {case['id']} complete",flush=True)


def main(args):
    config_path=Path(args.config).resolve()
    config=json.loads(config_path.read_text())
    output=Path(args.output).resolve()
    if args.worker is not None:
        worker(config,jobs(config)[args.worker],output)
        return
    if subprocess.check_output(["git","status","--porcelain"],text=True).strip():
        raise RuntimeError("Run only from a clean frozen code commit")
    if int(os.environ.get("PIN_CORE","-1"))!=config["pin_core"] or numba.get_num_threads()!=config["build_threads"]:
        raise ValueError("CPU affinity and construction threads must match the frozen configuration")
    commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    config_hash=hashlib.sha256(config_path.read_bytes()).hexdigest()
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/"config.json",config)
    write_json(output/"manifest.json",dict(
        commit=commit,config_sha256=config_hash,started=time.time(),python=sys.version,
        numpy=np.__version__,numba=numba.__version__,pin_core=config["pin_core"],
        build_threads=numba.get_num_threads(),stage=config["stage"],index_seeds=config["index_seeds"],
        cpu_model=next(line.split(":",1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                       if line.startswith("model name"))))
    started=time.monotonic()
    cases=[]
    for i,case in enumerate(jobs(config)):
        remaining=config["run_cap_seconds"]-(time.monotonic()-started)
        if remaining<=0:
            raise RuntimeError("Feasibility study exceeded its total run cap")
        folder=output/case["id"]
        folder.mkdir()
        with (folder/"worker.log").open("w") as log:
            process=subprocess.Popen([sys.executable,"-u","-m","fresh.feasibility",
                                      "--config",str(config_path),"--output",str(folder),"--worker",str(i)],
                                     stdout=log,stderr=subprocess.STDOUT)
            try:
                code,interruption=monitor_worker(process,folder,config,remaining)
            finally:
                terminate_worker(process)
        status=interruption or ("complete" if code==0 else "implementation or build failure")
        cases.append(dict(case=case["id"],case_id=case["case_id"],index_seed=case["index_seed"],
                          status=status,returncode=code))
        write_json(output/"cases.json",cases)
        print(f"{case['id']}: {status}",flush=True)
        if code or interruption:
            raise RuntimeError(f"Stopped at {folder}: {status}")
    write_json(output/"completion.json",dict(completed=time.time(),elapsed_seconds=time.monotonic()-started))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--worker",type=int)
    main(parser.parse_args())
