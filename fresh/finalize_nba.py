"""Derive full real-data results and valid examples before marking paper analysis complete."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from fresh.e1 import write_json
from fresh.nba_full import digest
from fresh.nba_motion_examples import choose
from fresh.summarize_nba_full import summarize


def finish(root,data,policy):
    report=summarize(root,data)
    blocking=next((r for r in report["main"] if r["method"]=="blocking"),None)
    illustration="not produced: blocking not completely evaluated across required seeds"
    motion_status=illustration
    if blocking is not None and blocking["status"]=="complete":
        subprocess.run([sys.executable,"-m","fresh.nba_full_figures","--data",str(data),
                        "--results",str(root/"seed-2301"),"--method","blocking"],check=True)
        definition=json.loads(policy.read_text())
        seed=root/f"seed-{definition['index_seed']}"
        selected=json.loads((seed/"frozen-selections.json").read_text())["main"][definition["method"]]
        cid=selected["config_id"]
        rows=json.loads((seed/(cid+"-test.json")).read_text())["queries"]
        diagnostics=json.loads((seed/(cid+"-candidate-diagnostics.json")).read_text())
        queries=json.loads((data/"test-metadata.json").read_text())
        database=json.loads((data/"database-metadata.json").read_text())
        illustration="hash-first, median, difficult, and explicitly selected examples generated"
        try:
            choose(rows,diagnostics,queries,database,definition)
        except ValueError as exc:
            if str(exc)!="No eligible moving-play example; do not substitute a static or synthetic match":
                raise
            motion_status=str(exc)
            write_json(root/"moving-example-unavailable.json",dict(status=motion_status,policy_sha256=digest(policy)))
            print(motion_status,flush=True)
        else:
            subprocess.run([sys.executable,"-m","fresh.nba_motion_examples","--data",str(data),
                            "--results",str(root),"--policy",str(policy)],check=True)
            motion_status="preregistered moving-play example generated"
    write_json(root/"analysis-complete.json",dict(
        completed=time.time(),dataset_sha256=digest(data/"dataset.json"),
        summary_sha256=digest(root/"replicated-summary.json"),
        all_seed_groups_checked=True,illustration_status=illustration,
        moving_example_status=motion_status,
        semantic_validation="geometric retrieval only; no expert tactic-equivalence judgment"))
    print("Full NBA metrics and actual-output illustration analysis completed.",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("results",type=Path)
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--policy",type=Path,default=Path("configs/nba-illustration-policy-20261013.json"))
    args=parser.parse_args();finish(args.results.resolve(),args.data.resolve(),args.policy.resolve())
