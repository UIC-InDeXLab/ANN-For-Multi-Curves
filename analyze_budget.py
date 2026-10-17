"""Summaries of the budgeted experiments (24 GB of index, 60 s per query) and pgfplots data for the paper.

Run where results/ lives:  python analyze_budget.py [--results results] [--out results/dat]
Per configuration: status (ok, over memory budget, over time budget), median query time over the queries answered,
index size, fraction of queries whose exact nearest neighbor was returned, and mean candidates verified.
Configurations over a budget get estimates: memory from the index size formula, time from the probes and the
per-probe cost measured for the same method at the largest feasible k.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

METHODS = ["store_all", "probe_all", "blocking", "sorted", "per_curve"]


def load(path):
    p = Path(path)
    return pd.read_csv(p, dtype={"blocks": str}) if p.exists() else None


def fmt(v):
    if isinstance(v, str):
        return v
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "nan"
    return f"{v:.6g}"


def write_dat(path, df, cols):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(" ".join(cols) + "\n")
        for _, r in df.iterrows():
            f.write(" ".join(fmt(r[c]) for c in cols) + "\n")


def summarize(df, keys):
    """One row per configuration (keys + method + blocks)."""
    rows = []
    for key, g in df.groupby(keys + ["method", "blocks"], dropna=False):
        ok = g[g.status == "ok"]
        status = "ok" if len(ok) and not (g.status == "over time budget").any() else g.status.iloc[-1]
        method, L, b, k = g.method.iloc[0], g.L.iloc[0], g.probed.iloc[0], g.k.iloc[0]
        total = L * k if method == "per_curve" else (L if method == "sorted" else L * b)
        rows.append(dict(zip(keys + ["method", "blocks"], key), status=status, stored=g.stored.iloc[0],
                         probed=b, L=L, gb=g.index_bytes.iloc[0] / 1e9, probes=total, queries=len(ok),
                         t_med=ok.seconds.median() if len(ok) else np.nan,
                         t_mean=ok.seconds.mean() if len(ok) else np.nan,
                         t_p90=ok.seconds.quantile(0.9) if len(ok) else np.nan,
                         bf_med=g.bf_seconds.median(), exact=ok.exact.mean() if len(ok) else np.nan,
                         cand=ok.candidates.mean() if len(ok) else np.nan,
                         build_s=g.build_seconds.iloc[0] if "build_seconds" in g else np.nan))
    return pd.DataFrame(rows)


def wide(s, key, out, name):
    """One row per key value: median time (s), memory (GB), exact rate per method, and estimates beyond the
    budgets (dotted lines in the figures)."""
    ns_per_probe = {}
    for m in METHODS:
        ok = s[(s.method == m) & (s.status == "ok")]
        if len(ok):
            last = ok.sort_values(key).iloc[-1]
            ns_per_probe[m] = last.t_med / last.probes
    rows = []
    for v, g in s.groupby(key):
        row = {key: v, "bf_s": g.bf_med.median()}
        for m in METHODS:
            gm = g[g.method == m]
            if not len(gm):
                continue
            r = gm.iloc[0]
            ok = r.status == "ok"
            row[f"{m}_s"] = r.t_med if ok else np.nan
            row[f"{m}_gb"] = r.gb if ok else np.nan
            row[f"{m}_exact"] = r.exact if ok else np.nan
            row[f"{m}_gb_est"] = r.gb
            row[f"{m}_s_est"] = r.t_med if ok else r.probes * ns_per_probe.get(m, np.nan)
            row[f"{m}_status"] = r.status.replace(" ", "_")
            row[f"{m}_blocks"] = r.blocks if isinstance(r.blocks, str) else ""
        rows.append(row)
    t = pd.DataFrame(rows).sort_values(key)
    cols = [key, "bf_s"] + [c for m in METHODS for c in (f"{m}_s", f"{m}_gb", f"{m}_exact", f"{m}_gb_est",
                                                         f"{m}_s_est") if c in t]
    write_dat(out / f"exp_{name}.dat", t, cols)
    show = [key, "bf_s"] + [c for m in METHODS for c in (f"{m}_s", f"{m}_gb", f"{m}_status") if c in t]
    print(f"\n== {name} (median seconds per query, GB)\n" + t[show].to_string(index=False))
    return t


def budget_k(df, out):
    s = summarize(df, ["k"])
    wide(s, "k", out, "budget_k")
    blk = s[s.method == "blocking"][["k", "blocks", "stored", "probed", "gb", "t_med", "status"]]
    print("\nblocking chosen for the memory budget:\n" + blk.to_string(index=False))


def budget_n(df, out):
    wide(summarize(df, ["n"]), "n", out, "budget_n")


def pareto(df, out):
    s = summarize(df, ["k"])
    for k, g in s.groupby("k"):
        g = g.sort_values("stored").copy()
        g["t_plot"] = np.where(g.status == "ok", g.t_med, np.nan)
        g["over_time"] = (g.status != "ok") & (g.status != "over memory budget")
        write_dat(out / f"exp_budget_pareto_k{k}.dat", g[g.status == "ok"], ["stored", "probed", "gb", "t_med",
                                                                              "exact"])
        write_dat(out / f"exp_budget_pareto_k{k}_overtime.dat", g[g.over_time], ["stored", "probed", "gb"])
        print(f"\n== pareto k={k}\n" + g[["blocks", "stored", "probed", "gb", "t_med", "status"]]
              .to_string(index=False))


def motif(df, out):
    s = summarize(df, ["k", "n"])
    for k, g in s.groupby("k"):
        rows = []
        for n, gn in g.groupby("n"):
            row = {"n": n, "bf_s": gn.bf_med.median()}
            for m in METHODS:
                gm = gn[(gn.method == m) & (gn.status == "ok")]
                row[f"{m}_s"] = gm.t_med.iloc[0] if len(gm) else np.nan
                row[f"{m}_cand"] = gm.cand.iloc[0] if len(gm) else np.nan
            rows.append(row)
        t = pd.DataFrame(rows).sort_values("n")
        pc = df[(df.k == k) & (df.method == "per_curve") & (df.status == "ok")]
        t["per_curve_listed"] = t.n.map(pc.groupby("n").listed.mean())
        write_dat(out / f"exp_budget_motif_k{k}.dat", t, list(t.columns))
        print(f"\n== motif k={k}\n" + t.to_string(index=False))


def mot(df, out):
    s = summarize(df, ["d", "k", "n"])
    lines = []
    for d, gd in s.groupby("d"):
        rows = []
        for i, ((k, n), g) in enumerate(gd.groupby(["k", "n"])):
            row = {"k": k, "n": n, "bf_s": g.bf_med.median()}
            cells = [f"{d}-D" if i == 0 else "", str(k), f"{n:,}".replace(",", "{,}"), sec(g.bf_med.median())]
            for m in ["blocking", "sorted", "store_all", "probe_all", "per_curve"]:
                gm = g[g.method == m]
                r = gm.iloc[0] if len(gm) else None
                ok = r is not None and r.status == "ok"
                row[f"{m}_s"] = r.t_med if ok else np.nan
                row[f"{m}_exact"] = r.exact if ok else np.nan
                row[f"{m}_gb"] = r.gb if ok else np.nan
                if ok:
                    cells += [f"{100 * r.exact:.0f}", sec(r.t_med)]
                elif r is None:
                    cells += ["--", "--"]
                elif r.status == "over memory budget":
                    cells += [r"\multicolumn{2}{c}{over 24\,GB}"]
                else:
                    cells += [r"\multicolumn{2}{c}{over 60\,s}"]
            gb = row["blocking_gb"]
            cells.append("--" if np.isnan(gb) else (f"{gb:.2g}" if gb < 10 else f"{gb:.0f}"))
            rows.append(row)
            lines.append(" & ".join(cells) + r"\\")
        t = pd.DataFrame(rows)
        write_dat(out / f"exp_mot_budget_d{d}.dat", t, list(t.columns))
        print(f"\n== MOT d={d}\n" + t.round(4).to_string(index=False))
        lines.append(r"\midrule")
    (out / "exp_mot_budget_table.tex").write_text("\n".join(lines[:-1]) + "\n")
    print("\n== MOT table rows\n" + "\n".join(lines[:-1]))
    blk = s[s.method == "blocking"][["d", "k", "blocks", "stored", "probed", "gb", "t_med", "status"]]
    print("\nMOT blocking chosen for the memory budget:\n" + blk.to_string(index=False))


def sec(v):
    """Seconds in a readable unit: three digits at most, one decimal below 10."""
    if v is None or np.isnan(v):
        return "--"
    for scale, unit in ((1, "s"), (1e-3, "ms"), (1e-6, "$\\mu$s")):
        if v >= scale or unit == "$\\mu$s":
            x = v / scale
            return (f"{x:.1f}" if x < 10 else f"{x:.0f}") + "\\," + unit
    return "--"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/dat")
    a = ap.parse_args()
    R, out = Path(a.results), Path(a.out)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    for name, fn in (("budget_k", budget_k), ("budget_n", budget_n), ("budget_pareto", pareto),
                     ("budget_motif", motif), ("mot_budget", mot)):
        df = load(R / f"{name}.csv")
        if df is not None:
            fn(df, out)
