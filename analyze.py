"""Summaries of the experiment CSVs and pgfplots data files for the paper.

Run on the machine that holds results/:  python analyze.py [--out DIR]
Writes exp_*.dat into DIR (default: results/dat) and prints the summary tables used in the text.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

METHODS = ["store_all", "probe_all", "blocking", "sorted", "per_curve"]


def load(path):
    p = Path(path)
    if not p.exists():
        return None
    df = pd.read_csv(p)
    return df[df.status == "ok"].copy()


def write_dat(path, df, cols):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(" ".join(cols) + "\n")
        for _, r in df.iterrows():
            f.write(" ".join("nan" if pd.isna(r[c]) else (f"{r[c]:.6g}" if isinstance(r[c], float) else str(r[c]))
                             for c in cols) + "\n")


def scaling(df, key, out, name):
    """Wide table: one row per key value (k or n), memory (MB), mean verify time (ms), probes per method."""
    if df is None:
        return
    rows = []
    for v, g in df.groupby(key):
        row = {key: v, "bf_ms": 1000 * g.bf_seconds.mean()}
        for m in METHODS:
            gm = g[(g.method == m) & (g["mode"] == "verify")]
            row[f"{m}_mb"] = gm.index_bytes.iloc[0] / 1e6 if len(gm) else np.nan
            row[f"{m}_ms"] = 1000 * gm.seconds.mean() if len(gm) else np.nan
            row[f"{m}_probes"] = gm.probes.mean() if len(gm) else np.nan
            row[f"{m}_build_s"] = gm.build_seconds.iloc[0] if len(gm) else np.nan
        rows.append(row)
    t = pd.DataFrame(rows).sort_values(key)
    cols = [key, "bf_ms"] + [f"{m}_{s}" for m in METHODS for s in ("mb", "ms", "probes", "build_s")]
    write_dat(out / f"exp_{name}.dat", t, cols)
    print(f"\n== {name}\n" + t[[key, "bf_ms"] + [f"{m}_ms" for m in METHODS] + [f"{m}_mb" for m in METHODS]]
          .round(3).to_string(index=False))


def pareto(df, out):
    if df is None:
        return
    for k, g in df.groupby("k"):
        t = g.groupby(["stored", "probed", "blocks"]).agg(
            mb=("index_bytes", lambda s: s.iloc[0] / 1e6),
            ms_verify=("seconds", lambda s: 1000 * s[g.loc[s.index, "mode"] == "verify"].mean()),
            ms_first=("seconds", lambda s: 1000 * s[g.loc[s.index, "mode"] == "first"].mean()),
            L=("L", "first"), n=("n", "first")).reset_index().sort_values("stored")
        write_dat(out / f"exp_pareto_k{k}.dat", t, ["stored", "probed", "mb", "ms_verify", "ms_first", "L", "n"])
        print(f"\n== pareto k={k}\n" + t.round(3).to_string(index=False))


def guarantees(df, out):
    if df is None:
        return
    g = df.groupby(["d", "k", "method", "mode"]).agg(
        success=("within_cr", "mean"), found=("found", "mean"), planted=("planted_hit", "mean"),
        ratio_p50=("ratio", "median"), ratio_p90=("ratio", lambda s: s.quantile(0.9)), ratio_max=("ratio", "max"),
        beyond=("within_cr", lambda s: int(((df.loc[s.index, "found"] == 1) & (s == 0)).sum())))
    print("\n== guarantees\n" + g.round(3).to_string())
    t = g.reset_index()
    t = t[t.method == "blocking"]
    for d, gd in t.groupby("d"):
        wide = gd.pivot(index="k", columns="mode", values=["success", "ratio_p50", "ratio_p90", "ratio_max"])
        wide.columns = [f"{a}_{b}" for a, b in wide.columns]
        wide = wide.reset_index()
        write_dat(out / f"exp_guarantees_d{d}.dat", wide, list(wide.columns))


def motif(df, out):
    if df is None:
        return
    for k, gk in df.groupby("k"):
        rows = []
        for n, g in gk.groupby("n"):
            row = {"n": n, "bf_ms": 1000 * g.bf_seconds.mean()}
            for m in METHODS:
                gm = g[g.method == m]
                row[f"{m}_ms"] = 1000 * gm.seconds.mean() if len(gm) else np.nan
                row[f"{m}_cand"] = gm.candidates.mean() if len(gm) else np.nan
                row[f"{m}_recall"] = (gm.d_ret <= gm.d_nn * (1 + 1e-9)).mean() if len(gm) else np.nan
            pc = gk[(gk.n == n) & (gk.method == "per_curve")]
            row["per_curve_listed"] = pd.to_numeric(pc.listed, errors="coerce").mean() if len(pc) else np.nan
            rows.append(row)
        t = pd.DataFrame(rows).sort_values("n")
        write_dat(out / f"exp_motif_k{k}.dat", t, list(t.columns))
        print(f"\n== motif k={k}\n" + t.round(3).to_string(index=False))


def sorted_key(df, out):
    if df is None:
        return
    g = df.groupby(["dataset", "k", "method"]).planted_hit.mean().unstack(["dataset", "method"])
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    t = g.reset_index()
    write_dat(out / "exp_sorted.dat", t, list(t.columns))
    print("\n== sorted key: collision rate per table\n" + t.round(3).to_string(index=False))


def mot_sweep(df, out):
    if df is None:
        return
    df = df.copy()
    df["exact"] = (df.found == 1) & (df.d_ret <= df.d_nn * (1 + 1e-9))
    for (ds, exp, d), g in df.groupby(["dataset", "exp", "d"]):
        t = g.groupby(["k", "delta"]).agg(n=("n", "first"), exact=("exact", "mean"), found=("found", "mean"),
                                          cand=("candidates", "mean"), ms=("seconds", lambda s: 1000 * s.mean()),
                                          bf_ms=("bf_seconds", lambda s: 1000 * s.mean())).reset_index()
        t["cand_frac"] = t.cand / t.n
        tag = f"{exp.replace('sweep-', '')}_d{d}_m{ds.split('m=')[-1]}"
        for k, tk in t.groupby("k"):
            write_dat(out / f"exp_mot_sweep_{tag}_k{k}.dat", tk, ["delta", "exact", "found", "cand_frac", "ms",
                                                                   "bf_ms", "n"])
        print(f"\n== MOT sweep {ds} {exp} d={d}\n" + t.round(4).to_string(index=False))


def mot_methods(df, out):
    if df is None:
        return
    df = df.copy()
    df["exact"] = (df.found == 1) & (df.d_ret <= df.d_nn * (1 + 1e-9))
    g = df[df["mode"] == "verify"].groupby(["dataset", "exp", "d", "k", "method"]).agg(
        n=("n", "first"), exact=("exact", "mean"), cand=("candidates", "mean"),
        ms=("seconds", lambda s: 1000 * s.mean()), bf_ms=("bf_seconds", lambda s: 1000 * s.mean()),
        mb=("index_bytes", lambda s: s.iloc[0] / 1e6), build_s=("build_seconds", "first")).reset_index()
    print("\n== MOT methods\n" + g.round(3).to_string(index=False))
    for (exp, d), gd in g.groupby(["exp", "d"]):
        rows = []
        for k, gk in gd.groupby("k"):
            row = {"k": k, "n": gk.n.iloc[0], "bf_ms": gk.bf_ms.mean()}
            for m in METHODS:
                gm = gk[gk.method == m]
                for col in ("mb", "ms", "exact"):
                    row[f"{m}_{col}"] = gm[col].iloc[0] if len(gm) else np.nan
            rows.append(row)
        t = pd.DataFrame(rows)
        write_dat(out / f"exp_mot_methods_{exp.replace('methods-', '')}_d{d}.dat", t, list(t.columns))


def mot_table(df, out):
    """LaTeX rows for the real-data table: exact-NN rate (%) and query time (ms) per method, natural queries."""
    if df is None:
        return
    df = df[(df["mode"] == "verify") & (df.exp == "methods-natural")].copy()
    df["exact"] = (df.found == 1) & (df.d_ret <= df.d_nn * (1 + 1e-9))
    lines = []
    for (d, k), g in df.groupby(["d", "k"]):
        if k not in (2, 4, 6, 8):
            continue
        cells = [f"{d}-D" if k == 2 else "", str(k), f"{g.n.iloc[0]:,}".replace(",", "{,}"),
                 f"{1000 * g.bf_seconds.mean():.1f}"]
        for m in ("blocking", "sorted", "store_all", "probe_all", "per_curve"):
            gm = g[g.method == m]
            if len(gm):
                cells += [f"{100 * gm.exact.mean():.0f}", f"{1000 * gm.seconds.mean():.1f}"]
            else:
                cells += ["--", "--"]
        mb = g[g.method == "blocking"].index_bytes.iloc[0] / 1e6
        cells.append(f"{mb:.0f}" if mb >= 10 else f"{mb:.1f}")
        lines.append(" & ".join(cells) + r"\\")
    (out / "exp_mot_table.tex").write_text("\n".join(lines) + "\n")
    print("\n== MOT table rows\n" + "\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/dat")
    a = ap.parse_args()
    R, out = Path(a.results), Path(a.out)
    pd.set_option("display.width", 250)
    guarantees(load(R / "guarantees.csv"), out)
    scaling(load(R / "scaling_k.csv"), "k", out, "scaling_k")
    scaling(load(R / "scaling_n.csv"), "n", out, "scaling_n")
    pareto(load(R / "pareto.csv"), out)
    motif(load(R / "motif.csv"), out)
    sorted_key(load(R / "sorted.csv"), out)
    mot_sweep(load(R / "mot_sweep.csv"), out)
    mot_methods(load(R / "mot_methods.csv"), out)
    mot_table(load(R / "mot_methods.csv"), out)
