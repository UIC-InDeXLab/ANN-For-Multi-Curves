"""A single weighted blocking method, shared posting compression, and tuning-only ablation."""
import argparse
import csv
import hashlib
import json
from math import factorial, prod
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
from numba import njit

from curveset import core, data
from curveset.bench import pinned
from curveset.implicit import Implicit, fill_table
from curveset.index import Tables, _bucket
from curveset.orders import balanced_blocking, blocking_points
from curveset.tune import first_hit_per_curve, tables_for_recall
from fresh.e1 import append_row, arrays_hash, monitor_worker, terminate_worker, write_json
from fresh.regimes import (approximation_summary, family_first_hits, finish_intersection,
                           ground_truth, intersect_postings, posting_ranges, timed_queries,
                           union_postings, verify, warm)


@njit(cache=True)
def compact_count(keys, ids):
    count = 0
    for i in range(len(keys)):
        if i == 0 or keys[i] != keys[i-1] or ids[i] != ids[i-1]:
            count += 1
    return count


@njit(cache=True)
def compact_into(keys, ids, out_keys, out_ids, directory, B):
    out, bucket = 0, 0
    directory[0] = 0
    for i in range(len(keys)):
        if i and keys[i] == keys[i-1] and ids[i] == ids[i-1]:
            continue
        target = _bucket(keys[i], B)
        while bucket < target:
            bucket += 1
            directory[bucket] = out
        out_keys[out], out_ids[out] = keys[i], ids[i]
        out += 1
    while bucket < B:
        bucket += 1
        directory[bucket] = out


def compact_table(keys, divisor):
    """Stable bucket sorting retains increasing parent IDs within every equal fingerprint."""
    temporary = Tables(1, len(keys))
    fill_table(keys, divisor, temporary.B, temporary.keys, temporary.ids, temporary.dir)
    count = compact_count(temporary.keys, temporary.ids)
    result = Tables(1, count)
    compact_into(temporary.keys, temporary.ids, result.keys, result.ids, result.dir, result.B)
    return result


@njit(cache=True)
def probe_table(Q, shift, delta, family, needed, keys, ids, directory, N, B, marks, stamp, candidates, nc):
    k = len(Q)
    fingerprints = np.zeros((k,k),np.uint64)
    prefix = np.empty(k+1,np.uint64)
    prefix[0] = core.SEED
    for i in range(k):
        for j in range(k):
            if needed[i,j]:
                fingerprints[i,j] = core.curve_fp(Q[j],shift[i],delta)
    postings = 0
    for s in range(len(family)):
        common = 0
        if s:
            while common < k and family[s,common] == family[s-1,common]:
                common += 1
        for i in range(common,k):
            prefix[i+1] = core.mix64(prefix[i]+fingerprints[i,family[s,i]])
        fp = prefix[k]
        bucket = _bucket(fp,B)
        lo, hi = np.int64(directory[bucket]), np.int64(directory[bucket+1])
        for t in range(lo,hi):
            if keys[t] > fp:
                break
            if keys[t] == fp:
                postings += 1
                p = ids[t]
                if marks[p] != stamp:
                    marks[p] = stamp
                    candidates[nc] = p
                    nc += 1
    return nc, postings


class Joint:
    def __init__(self, X, delta, L, seed, sizes, swapped):
        from curveset.index import make_index
        started = time.perf_counter()
        self.base = make_index("blocking",X.shape[1],X.shape[3],delta,L,sizes=sizes,swapped=swapped,seed=seed)
        self.tables = []
        for u in range(L):
            fingerprints = core.slot_fps(X,self.base.shifts[u],delta,self.base.need_s)
            keys = core.order_keys(fingerprints,self.base.stored).ravel()
            self.tables.append(compact_table(keys,len(self.base.stored)))
        self.base._buffers(len(X))
        self.build_seconds = time.perf_counter()-started
        self.raw_entries = len(X)*len(self.base.stored)*L

    @property
    def index_bytes(self):
        base = self.base
        return (sum(t.nbytes for t in self.tables)+base.stored.nbytes+base.probed.nbytes+
                base.shifts.nbytes+base.need_s.nbytes+base.need_q.nbytes)

    @property
    def entries(self):
        return sum(t.N for t in self.tables)

    @property
    def scratch_bytes(self):
        return self.base.mark.nbytes+self.base.cand.nbytes

    def query(self,X,Q,config,verifier=verify):
        started = time.perf_counter()
        base = self.base
        nc,postings,probes = 0,0,0
        stamp = base._next_stamp()
        for u,table in enumerate(self.tables):
            for start in range(0,len(base.probed),config["probe_chunk"]):
                family = base.probed[start:start+config["probe_chunk"]]
                nc,read = probe_table(Q,base.shifts[u],base.delta,family,base.need_q,
                                     table.keys,table.ids,table.dir,table.N,table.B,base.mark,stamp,base.cand,nc)
                postings += read
                probes += len(family)
                if time.perf_counter()-started > config["query_deadline_seconds"]:
                    return dict(id=-1,distance=np.inf,timeout=True,candidates=nc,verified=0,
                                probes=probes,postings=postings,seconds=time.perf_counter()-started)
        p,distance,verified,timeout = verifier(X,Q,base.cand[:nc],config,started)
        return dict(id=p,distance=distance,timeout=timeout,candidates=nc,verified=verified,
                    probes=probes,postings=postings,seconds=time.perf_counter()-started)


class PerCurve:
    def __init__(self,X,delta,L,seed):
        started = time.perf_counter()
        self.k,self.delta,self.L = X.shape[1],delta,L
        self.shifts = np.random.default_rng(seed).uniform(0,delta,(L,X.shape[3]))
        self.tables = [compact_table(core.curve_fps_all(X,self.shifts[u],delta).ravel(),self.k) for u in range(L)]
        self.marks = np.zeros(len(X),np.int32)
        self.counts = np.zeros(len(X),np.int16)
        self.touched = np.empty(len(X),np.int64)
        self.candidates = np.empty(len(X),np.int64)
        self.stamp = 0
        self.raw_entries = len(X)*self.k*L
        self.build_seconds = time.perf_counter()-started

    @property
    def index_bytes(self):
        return sum(t.nbytes for t in self.tables)+self.shifts.nbytes

    @property
    def entries(self):
        return sum(t.N for t in self.tables)

    @property
    def scratch_bytes(self):
        return self.marks.nbytes+self.counts.nbytes+self.touched.nbytes+self.candidates.nbytes

    def query(self,X,Q,config,rarest=False,active_L=None,verifier=verify):
        started = time.perf_counter()
        L = self.L if active_L is None else active_L
        self.stamp += self.k+1
        if self.stamp > 2_000_000_000:
            self.marks.fill(0)
            self.stamp = self.k+1
        ranges = np.empty((self.k,L,2),np.int64)
        for u in range(L):
            table = self.tables[u]
            ranges[:,u,:] = posting_ranges(Q,self.shifts[u:u+1],self.delta,
                                           table.keys,table.dir,table.N,table.B,1)[:,0,:]
        chosen = int(np.argmin((ranges[:,:,1]-ranges[:,:,0]).sum(axis=1)))
        nc,nt,postings = 0,0,0
        for i in ([chosen] if rarest else range(self.k)):
            for u in range(L):
                lo,hi = ranges[i,u]
                ids = self.tables[u].ids
                for start in range(lo,hi,config["probe_chunk"]):
                    end = min(hi,start+config["probe_chunk"])
                    if rarest:
                        nc = union_postings(ids,start,end,1,self.marks,self.stamp,self.candidates,nc)
                    else:
                        nt = intersect_postings(ids,start,end,1,self.marks,self.stamp+i,self.counts,self.touched,nt)
                    postings += end-start
                    if time.perf_counter()-started > config["query_deadline_seconds"]:
                        if not rarest:
                            finish_intersection(self.counts,self.touched,nt,self.k,self.candidates)
                        return dict(id=-1,distance=np.inf,timeout=True,candidates=0,verified=0,
                                    probes=self.k*L,postings=postings,seconds=time.perf_counter()-started)
        if not rarest:
            nc = finish_intersection(self.counts,self.touched,nt,self.k,self.candidates)
        p,distance,verified,timeout = verifier(X,Q,self.candidates[:nc],config,started)
        return dict(id=p,distance=distance,timeout=timeout,candidates=nc,verified=verified,
                    probes=self.k*L,postings=postings,seconds=time.perf_counter()-started)


def dataset(config,case):
    k,n = case["k"],case["n"]
    seed = config["dataset_seed"]+100*k+{"conjunctive":1,"rare":2,"repeated":3}[case["kind"]]
    count = config["tuning_queries"]+config["test_queries"]+config["warmup_queries"]
    anchor = config["plant_anchor_n"]
    if n < anchor or count*config["planted_per_query"] > anchor:
        raise ValueError("Planting anchor must fit in all nested datasets")
    pool = case.get("paths")
    if pool is None:
        pool = int(round(1/(1-(1-case["frequency"])**(1/k))))
    paths = data.random_walks(np.random.default_rng(seed),pool,config["m"],config["d"],1000,3)
    ids = np.random.default_rng(seed+1).integers(0,pool,(n,k),dtype=np.uint16)
    qrng = np.random.default_rng(seed+2)
    motifs = np.empty((count,k),np.uint16)
    for q in range(count):
        motifs[q] = qrng.choice(pool,k,replace=case["kind"]=="repeated")
    queries = data.jitter(qrng,paths[motifs],config["rho"])
    positions = np.random.default_rng(seed+3).choice(anchor,count*config["planted_per_query"],replace=False)
    for q in range(count):
        for p in positions[q*config["planted_per_query"]:(q+1)*config["planted_per_query"]]:
            ids[p] = motifs[q,qrng.permutation(k)]
        queries[q] = queries[q,qrng.permutation(k)]
    X = Implicit(paths,ids,seed+4,config["rho"]).materialize(np.arange(n))
    return X,queries,dict(paths=pool,nominal_background_path_frequency=1-(1-1/pool)**k,
                          planted_sets=len(positions),planted_fraction=len(positions)/n,
                          planted_per_query=config["planted_per_query"],dataset_seed=seed,
                          dataset_sha256=arrays_hash(X),queries_sha256=arrays_hash(queries),
                          raw_curve_bytes=X.nbytes)


def weighted_cost(row,weight,config):
    return row["index_bytes"]/config["cost_memory_scale_bytes"]+weight*row["tuning_median_seconds"]/config["cost_latency_scale_seconds"]


def choose_blocking(rows,weight,config):
    eligible = [r for r in rows if r["method"]=="blocking-family" and r["status"]=="qualified"]
    if not eligible:
        return None
    return min(eligible,key=lambda r:(weighted_cost(r,weight,config),r["index_bytes"],r["config_id"]))


def tuning_target(config):
    return config.get("tuning_acceptance_recall",config["target_recall"])


def jobs(config):
    seeds=config.get("index_seeds",[config.get("index_seed")])
    if not seeds or any(seed is None for seed in seeds) or len(set(seeds))!=len(seeds):
        raise ValueError("Index seeds must be distinct and explicitly configured")
    return [dict(case,index_seed=seed,case_id=case["id"],
                 id=f"{case['id']}-seed{seed}" if "index_seeds" in config else case["id"])
            for case in config["cases"] for seed in seeds]


def resume_completed(output,commit,config_hash):
    if (output/"completion.json").exists():
        raise ValueError("Evaluation already finished; do not resume a completed run")
    manifest=json.loads((output/"manifest.json").read_text())
    if manifest["commit"]!=commit or manifest["config_sha256"]!=config_hash:
        raise ValueError("Resume requires the original frozen commit and config hash")
    completed={row["case"]:row for row in json.loads((output/"cases.json").read_text())
               if row["status"]=="complete"}
    for name in completed:
        for artifact in ("main-results.json","ablation-results.json","test-queries.csv"):
            if not (output/name/artifact).is_file():
                raise ValueError(f"Completed job has a missing result artifact: {name}/{artifact}")
    return completed


def family_specs(k,config):
    a,b,sizes,swapped = balanced_blocking(k)
    desired = set(config["blocking_storage_multipliers"])
    if config["include_balanced_in_ablation"]:
        desired.add(a)
    desired.update((1,factorial(k)))
    specs = [(aa,bb,ss,swap) for aa,bb,ss,swap in blocking_points(k) if aa in desired]
    return specs


def observed_peak_gate(X,n,k,a,L,config):
    raw_entries = n*a
    if n >= 2**31 or raw_entries >= 2**32:
        return "not run: integer capacity"
    # Includes uncompressed construction arrays even when the final postings compress well.
    estimated = X.nbytes+raw_entries*64+13*raw_entries*L+(a+factorial(k)//a)*k
    return "not run: peak RAM estimate" if estimated > config["peak_ram_cap_bytes"] else None


def worker(config,case,output):
    write_json(output/"active_configuration.json",dict(method="ground-truth",phase="preparation"))
    X,Q,info = dataset(config,case)
    started = time.perf_counter()
    ids,distances,relevant = ground_truth(X,Q,config["relevance_radius"])
    h,s = config["tuning_queries"],config["test_queries"]
    Qt,Qtest,Qwarm = Q[:h],Q[h:h+s],Q[h+s:]
    NN = X[ids[:h]]
    write_json(output/"dataset.json",dict(info,**case,preparation_seconds=time.perf_counter()-started,
                                         mean_test_nn_distance=float(distances[h:h+s].mean()),
                                         mean_test_relevant_sets=float(relevant[h:h+s].mean())))
    np.savez(output/"ground-truth.npz",queries=Q,ids=ids,distances=distances,relevant=relevant)
    manifest = json.loads((output.parent/"manifest.json").read_text())
    tuning,chosen = [],{}
    seed,k,d = case.get("index_seed",config.get("index_seed")),case["k"],config["d"]
    if seed is None:
        raise ValueError("Worker requires an index seed")

    def record(row,query,build):
        warm(query,Qwarm)
        progress=lambda count:write_json(output/"active_configuration.json",
                                         dict(method=row["config_id"],seed=seed,phase="tuning",completed_queries=count))
        rows,complete = timed_queries(query,Qt,distances[:h],config,progress)
        success = sum(r["exact_nn"] for r in rows)/len(rows)
        row.update(status="qualified" if complete and success>=tuning_target(config) else
                   "tuning recall below target" if complete else "tuning timeout",
                   tuning_recall=success,
                   index_seed=seed,tuning_acceptance_recall=tuning_target(config),
                   tuning_median_seconds=float(np.median([r["seconds"] for r in rows])) if complete else None,
                   build_seconds=build)
        for q,r in enumerate(rows):
            append_row(output/"tuning-queries.csv",dict(config_id=row["config_id"],query=q,**r))
        tuning.append(row)
        write_json(output/"tuning-configurations.json",tuning)
        return row

    def select_baseline(method,row):
        if row["status"]=="qualified" and (method not in chosen or
                                          row["tuning_median_seconds"]<chosen[method]["tuning_median_seconds"]):
            chosen[method] = row

    for delta in config["deltas"]:
        shifts = np.random.default_rng(seed).uniform(0,delta,(config["Lmax"],k,d))
        for a,b,sizes,swapped in family_specs(k,config):
            cfg_id = f"joint-d{delta:g}-a{a}-swap{int(swapped)}"
            base = dict(config_id=cfg_id,method="blocking-family",delta=delta,blocks=list(sizes),
                        swapped=swapped,stored=a,probed=b)
            hit = family_first_hits(Qt,NN,shifts,delta,np.asarray(sizes),swapped)
            L = tables_for_recall(hit,tuning_target(config),config["Lmax"])
            base["L"] = L
            reason = "no tuned L in grid" if L is None else observed_peak_gate(X,len(X),k,a,L,config)
            if reason:
                tuning.append(dict(base,status=reason))
                write_json(output/"tuning-configurations.json",tuning)
                continue
            write_json(output/"active_configuration.json",dict(method=cfg_id,phase="building"))
            signal.alarm(config["build_deadline_seconds"])
            try:
                index = Joint(X,delta,L,seed,sizes,swapped)
            finally:
                signal.alarm(0)
            base.update(index_bytes=index.index_bytes,entries=index.entries,raw_entries=index.raw_entries,
                        scratch_bytes=index.scratch_bytes)
            if index.index_bytes > config["index_budget_bytes"]:
                tuning.append(dict(base,status="over index storage"))
                write_json(output/"tuning-configurations.json",tuning)
            else:
                write_json(output/"active_configuration.json",dict(method=cfg_id,phase="tuning"))
                row = record(base,lambda query:index.query(X,query,config),index.build_seconds)
                if a==1:
                    select_baseline("probe_all",row)
                if a==factorial(k):
                    select_baseline("store_all",row)
            del index
        sp = np.random.default_rng(seed).uniform(0,delta,(config["Lmax"],d))
        L = tables_for_recall(first_hit_per_curve(Qt,NN,sp,delta),tuning_target(config),config["Lmax"])
        if L is None:
            for method in ("per_curve-intersection","per_curve-rarest"):
                tuning.append(dict(config_id=f"{method}-d{delta:g}",method=method,delta=delta,L=None,status="no tuned L in grid"))
            write_json(output/"tuning-configurations.json",tuning)
            continue
        write_json(output/"active_configuration.json",dict(method="per_curve",phase="building"))
        signal.alarm(config["build_deadline_seconds"])
        try:
            index = PerCurve(X,delta,L,seed)
        finally:
            signal.alarm(0)
        for method,rarest in (("per_curve-intersection",False),("per_curve-rarest",True)):
            for active in (range(1,L+1) if rarest else (L,)):
                memory = sum(t.nbytes for t in index.tables[:active])+active*d*8
                base = dict(config_id=f"{method}-d{delta:g}-L{active}",method=method,delta=delta,L=active,
                            index_bytes=memory,entries=sum(t.N for t in index.tables[:active]),
                            raw_entries=len(X)*k*active,scratch_bytes=index.scratch_bytes,
                            blocks=None,swapped=False,stored=None,probed=k)
                if memory>config["index_budget_bytes"]:
                    tuning.append(dict(base,status="over index storage"))
                    write_json(output/"tuning-configurations.json",tuning)
                    break
                write_json(output/"active_configuration.json",dict(method=method,phase="tuning"))
                row = record(base,lambda query:index.query(X,query,config,rarest,active),index.build_seconds)
                select_baseline(method,row)
                if row["status"]=="qualified" or row["status"]=="tuning timeout":
                    break
        del index

    selections = {}
    for weight in config["ablation_latency_weights"]:
        row = choose_blocking(tuning,weight,config)
        if row:
            selections[str(weight)] = dict(row,latency_weight=weight,weighted_tuning_cost=weighted_cost(row,weight,config))
    main_weight = str(config["blocking_main_latency_weight"])
    if main_weight in selections:
        chosen["blocking"] = selections[main_weight]
    write_json(output/"frozen-main-selections.json",chosen)
    write_json(output/"frozen-weight-selections.json",selections)
    requested = {r["config_id"]:r for r in [*chosen.values(),*selections.values()]}
    evaluated = {}
    for cfg_id,row in requested.items():
        write_json(output/"active_configuration.json",dict(method=cfg_id,phase="building-final"))
        signal.alarm(config["build_deadline_seconds"])
        try:
            if row["method"]=="blocking-family":
                index = Joint(X,row["delta"],row["L"],seed,row["blocks"],row["swapped"])
                call = lambda query:index.query(X,query,config)
            else:
                index = PerCurve(X,row["delta"],row["L"],seed)
                call = lambda query:index.query(X,query,config,row["method"]=="per_curve-rarest")
        finally:
            signal.alarm(0)
        if index.index_bytes != row["index_bytes"]:
            raise AssertionError("Compressed index storage changed between selection and test")
        warm(call,Qwarm)
        write_json(output/"active_configuration.json",dict(method=cfg_id,phase="test"))
        progress=lambda count:write_json(output/"active_configuration.json",
                                         dict(method=cfg_id,seed=seed,phase="test",completed_queries=count))
        rows,complete = timed_queries(call,Qtest,distances[h:h+s],config,progress)
        for q,r in enumerate(rows):
            append_row(output/"test-queries.csv",dict(
                config_id=cfg_id,commit=manifest["commit"],config_sha256=manifest["config_sha256"],
                dataset_sha256=info["dataset_sha256"],index_seed=seed,phase="test",query=q,
                nn_distance=float(distances[h+q]),**r))
        evaluated[cfg_id] = dict(row,test_status="complete" if complete else "test timeout",
                                 test_queries=len(rows),build_seconds=index.build_seconds)
        write_json(output/"evaluated-configurations.json",evaluated)
        del index
    main = []
    for method in ("blocking","probe_all","store_all","per_curve-intersection","per_curve-rarest"):
        if method in chosen:
            main.append(dict(evaluated[chosen[method]["config_id"]],display_method=method))
        else:
            main.append(dict(display_method=method,test_status="no feasible tuned configuration"))
    all_parents = np.arange(len(X),dtype=np.int64)

    def brute(query):
        started = time.perf_counter()
        p,distance,checked,timeout = verify(X,query,all_parents,config,started)
        return dict(id=p,distance=distance,timeout=timeout,candidates=len(X),verified=checked,
                    probes=0,postings=0,seconds=time.perf_counter()-started)

    warm(brute,Qwarm)
    write_json(output/"active_configuration.json",dict(method="brute-force",seed=seed,phase="test"))
    progress=lambda count:write_json(output/"active_configuration.json",
                                     dict(method="brute-force",seed=seed,phase="test",completed_queries=count))
    rows,complete = timed_queries(brute,Qtest,distances[h:h+s],config,progress)
    if not complete or not all(r["exact_nn"] for r in rows):
        raise AssertionError("Brute-force reference failed")
    for q,r in enumerate(rows):
        append_row(output/"test-queries.csv",dict(
            config_id="brute-force",commit=manifest["commit"],config_sha256=manifest["config_sha256"],
            dataset_sha256=info["dataset_sha256"],index_seed=seed,phase="test",query=q,
            nn_distance=float(distances[h+q]),**r))
    main.append(dict(display_method="brute-force",config_id="brute-force",index_bytes=0,
                     scratch_bytes=all_parents.nbytes,index_seed=seed,test_status="complete",test_queries=len(rows)))
    write_json(output/"main-results.json",main)
    write_json(output/"ablation-results.json",[
        dict(evaluated[r["config_id"]],latency_weight=float(weight)) for weight,r in selections.items()])
    print(f"CASE {case['id']} complete",flush=True)


def main(args):
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    output = Path(args.output).resolve()
    if args.worker is not None:
        worker(config,jobs(config)[args.worker],output)
        return
    if subprocess.check_output(["git","status","--porcelain"],text=True).strip():
        raise RuntimeError("Run only from a clean frozen code commit")
    commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    config_hash=hashlib.sha256(config_path.read_bytes()).hexdigest()
    completed={}
    if args.resume:
        completed=resume_completed(output,commit,config_hash)
        write_json(output/f"resume-{time.time_ns()}.json",dict(time=time.time(),complete_jobs=len(completed),
                                                             reason="Explicit resume; preserve interrupted attempts"))
    else:
        output.mkdir(parents=True,exist_ok=False)
    import numba
    if not args.resume:
        write_json(output/"config.json",config)
        write_json(output/"manifest.json",dict(
            commit=commit,config_sha256=config_hash,started=time.time(),
            python=sys.version,numpy=np.__version__,numba=numba.__version__,pin_core=config["pin_core"],
            build_threads=numba.get_num_threads(),stage=config["stage"],
            index_seeds=config.get("index_seeds",[config.get("index_seed")]),
            cpu_model=next(line.split(":",1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                           if line.startswith("model name"))))
    started = time.monotonic()
    cases = []
    for i,case in enumerate(jobs(config)):
        if case["id"] in completed:
            cases.append(completed[case["id"]])
            continue
        remaining = config["run_cap_seconds"]-(time.monotonic()-started)
        if remaining <= 0:
            cases.append(dict(case=case["id"],case_id=case["case_id"],index_seed=case["index_seed"],
                              status="not run: run cap"))
            write_json(output/"cases.json",cases)
            continue
        folder = output/case["id"]
        if folder.exists():
            if not args.resume:
                raise ValueError(f"Output already exists: {folder}")
            preserved=folder.with_name(folder.name+f".interrupted-{time.time_ns()}")
            folder.rename(preserved)
        folder.mkdir()
        with (folder/"worker.log").open("w") as log:
            process = subprocess.Popen([sys.executable,"-m","fresh.poc","--config",str(config_path),
                                        "--output",str(folder),"--worker",str(i)],stdout=log,stderr=subprocess.STDOUT)
            try:
                code,interruption = monitor_worker(process,folder,config,remaining)
            finally:
                terminate_worker(process)
        status = interruption or ("complete" if code==0 else "implementation or build failure")
        cases.append(dict(case=case["id"],case_id=case["case_id"],index_seed=case["index_seed"],
                          status=status,returncode=code))
        write_json(output/"cases.json",cases)
        print(f"{case['id']}: {status}",flush=True)
        if code or interruption:
            raise RuntimeError(f"Stopped at {folder}: {status}")
    write_json(output/"completion.json",dict(completed=time.time(),elapsed_seconds=time.monotonic()-started))


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--worker",type=int)
    parser.add_argument("--resume",action="store_true")
    main(parser.parse_args())
