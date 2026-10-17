"""Complete first-hit k=10 quality cells separately from server timing experiments."""
import argparse
from itertools import product
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

from fresh.e1 import terminate_worker,write_json


def main(args):
    path=args.config.resolve()
    config=json.loads(path.read_text())
    parent=Path(config["parent_results"]).resolve()
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/"config.json",config)
    write_json(output/"manifest.json",dict(
        commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        config_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        stage=config["stage"],python=sys.version,numpy=np.__version__,started=time.time(),
        analysis_host=platform.node(),
        calibration_origin=str(parent),timings_excluded_from_server_comparison=True))
    for d in config["dimensions"]:
        (output/f"calibration-d{d}.json").write_bytes((parent/f"calibration-d{d}.json").read_bytes())
    started=time.monotonic()
    cases=[]
    for d,k,workload in product(config["dimensions"],config["index_k"],config["workloads"]):
        folder=output/f"{workload}-d{d}-k{k}"
        folder.mkdir()
        remaining=config["run_cap_seconds"]-(time.monotonic()-started)
        if remaining<=0:
            raise RuntimeError("Quality completion run cap reached")
        command=[sys.executable,"-m","fresh.e1","--worker","--config",str(path),"--output",str(folder),
                 "--workload",workload,"--k",str(k),"--d",str(d)]
        with (folder/"worker.log").open("w") as log:
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            try:
                code=process.wait(timeout=remaining)
            finally:
                terminate_worker(process)
        if code:
            raise RuntimeError(f"Quality worker failed ({code}); inspect {folder/'worker.log'}")
        statuses=json.loads((folder/"configurations.json").read_text())
        if len(statuses)!=len(config["index_seeds"]) or any(r["status"]!="complete" for r in statuses):
            raise RuntimeError(f"Not all required first-hit seeds completed: {folder}")
        cases.append(dict(case=folder.name,status="complete"))
        write_json(output/"cases.json",cases)
        print(f"QUALITY {folder.name} complete",flush=True)
    write_json(output/"completion.json",dict(completed=time.time(),elapsed_seconds=time.monotonic()-started))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    main(parser.parse_args())
