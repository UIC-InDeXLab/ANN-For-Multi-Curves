"""Summaries of the large-scale experiments (run_large.py) and pgfplots data for the paper.

Run where results/ lives:  python analyze_large.py [--results results] [--out results/dat]
Per (workload, n, k, method, variant): status, tables L, median query time, index size, mean recall and precision
with respect to the relevant sets (d_match < r), and the fraction of queries whose exact nearest neighbor is found.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_budget import sec, write_dat

ORDER = ["blocking", "probe_all", "store_all", "per_curve", "per_curve_capped", "per_curve_topk", "brute_force"]


def load(results, workload):
    p = Path(results) / f"large_{workload}.csv"
    if not p.exists() or p.stat().st_size == 0:
        return None
    return pd.read_csv(p, dtype={"variant": str, "blocks": str})


def summarize(df):
    rows = []
    keys = ["workload", "n", "k", "method", "variant"]
    for key, g in df.groupby(keys, dropna=False):
        ok = g[g.status == "ok"]
        bad = g[g.status != "ok"]
        status = "ok" if len(ok) and not (g.status == "over time budget").any() else (
            bad.status.iloc[0] if len(bad) else "ok")
        if len(ok) and (g.status == "over time budget").any():
            status = "over time budget"
        rows.append(dict(zip(keys, key), status=status, L=g.L.iloc[0], L_tuned=g.L_tuned.iloc[0],
                         blocks=g.blocks.iloc[0], stored=g.stored.iloc[0], probed=g.probed.iloc[0],
                         queries=len(ok), t=ok.seconds.median() if len(ok) else (bad.seconds.max() if len(bad) else np.nan),
                         gb=g.index_bytes.iloc[0] / 1e9 if pd.notna(g.index_bytes.iloc[0]) else np.nan,
                         recall=ok.recall.mean() if len(ok) else np.nan,
                         precision=ok.precision.mean() if len(ok) else np.nan,
                         empty=(ok.candidates == 0).mean() if len(ok) else np.nan,
                         nn=ok.found_nn.mean() if len(ok) else np.nan,
                         cand=ok.candidates.mean() if len(ok) else np.nan,
                         listed=pd.to_numeric(ok.listed, errors="coerce").mean() if len(ok) else np.nan,
                         build=g.build_seconds.iloc[0] if "build_seconds" in g else np.nan,
                         relevant=g.relevant.median()))
    s = pd.DataFrame(rows)
    s["order"] = s.method.map({m: i for i, m in enumerate(ORDER)})
    return s.sort_values(["workload", "n", "k", "order", "variant"]).drop(columns="order")


def _pick(g, m, var=None):
    gm = g[(g.method == m) & ((g.variant.map(lambda v: str(v).split(" ")[0]) == var) if var else True)]
    return gm.iloc[0] if len(gm) else None


def _status_cells(r, width):
    if r is None:
        return [r"\multicolumn{%d}{c}{--}" % width]
    if r.status.startswith("no blocking"):
        return [r"\multicolumn{%d}{c}{over 24\,GB or 60\,s}" % width]
    if r.status.startswith("over memory"):
        return [r"\multicolumn{%d}{c}{over 24\,GB}" % width]
    if r.status.startswith("over time"):
        return [r"\multicolumn{%d}{c}{over 60\,s}" % width]
    return [r"\multicolumn{%d}{c}{infeasible}" % width]


def _pct(x):
    return "--" if pd.isna(x) else f"{100 * x:.0f}"


def _gb(x):
    return "--" if pd.isna(x) else (f"{x:.2f}" if x < 1 else f"{x:.1f}" if x < 10 else f"{x:.0f}")


def table_rows(s, workload, out):
    """LaTeX rows (one per n, k): brute force; blocking (time, recall, precision, GB); probe all (time);
    per-curve complete lists (time, recall, precision, GB); top-K lists with K = 10^4 (time, recall)."""
    lines = []
    g0 = s[s.workload == workload]
    for i, ((n, k), g) in enumerate(g0.groupby(["n", "k"])):
        bf = _pick(g, "brute_force")
        e = int(np.log10(n))
        cells = [workload if i == 0 else "", f"$10^{e}$" if n == 10 ** e else f"{n:,}", str(k),
                 sec(bf.t) if bf is not None and bf.status == "ok" else (r"over 60\,s" if bf is not None else "--")]
        r = _pick(g, "blocking")
        cells += [sec(r.t), _pct(r.recall), _pct(r.precision), _gb(r.gb)] if r is not None and r.status == "ok" \
            else _status_cells(r, 4)
        r = _pick(g, "probe_all")
        cells += [sec(r.t)] if r is not None and r.status == "ok" else _status_cells(r, 1)
        r = _pick(g, "per_curve", "complete")
        if r is not None and r.status == "ok":
            mark = r"$^\ast$" if "reduced" in str(r.variant) else ""
            cells += [sec(r.t), _pct(r.recall) + mark, _pct(r.precision), _gb(r.gb)]
        else:
            cells += _status_cells(r, 4)
        r = _pick(g, "per_curve_capped", "K=10000")
        cells += [sec(r.t), _pct(r.recall)] if r is not None and r.status == "ok" else _status_cells(r, 2)
        lines.append(" & ".join(cells) + r"\\")
    (out / f"exp_large_{workload}_table.tex").write_text("\n".join(lines) + "\n")
    return lines


def scaling(s, workload, k, out):
    g = s[(s.workload == workload) & (s.k == k)]
    t = pd.DataFrame([dict(n=n, **_times(gn)) for n, gn in g.groupby("n")])
    if len(t):
        write_dat(out / f"exp_large_{workload}_k{k}_scaling.dat", t, list(t.columns))
    return t


def capped(s, workload, n, out):
    rows = []
    for k, g in s[(s.workload == workload) & (s.n == n)].groupby("k"):
        for _, r in g[g.method.isin(["per_curve_capped", "per_curve_topk"])].iterrows():
            rows.append(dict(k=k, method=r.method, K=int(str(r.variant).split("=")[1]), t=r.t, recall=r.recall,
                             nn=r.nn, empty=r.empty, status=r.status))
    t = pd.DataFrame(rows)
    for (k, m), g in t.groupby(["k", "method"]) if len(t) else []:
        write_dat(out / f"exp_large_{workload}_n{n}_k{k}_{m}.dat", g.sort_values("K"), ["K", "t", "recall", "nn"])
    return t


def _times(gk):
    """Median query times of one (n, k) for the plots. Per-curve LSH with fewer tables than its recall target
    needs (to fit 24 GB) goes to a separate column, and brute force is kept even beyond 60 s."""
    row = {}
    for m, var in (("blocking", None), ("probe_all", None), ("per_curve", "complete"), ("brute_force", None)):
        gm = gk[(gk.method == m) & ((gk.variant.map(lambda v: str(v).split(" ")[0]) == var) if var else True)]
        r = gm.iloc[0] if len(gm) else None
        ok = r is not None and r.status == "ok"
        if m == "per_curve":
            reduced = ok and r.L < r.L_tuned
            row["per_curve_t"] = r.t if ok and not reduced else np.nan
            row["per_curve_reduced_t"] = r.t if reduced else np.nan
        elif m == "brute_force":
            row["brute_force_t"] = r.t if r is not None and pd.notna(r.t) else np.nan
        else:
            row[f"{m}_t"] = r.t if ok else np.nan
    return row


def by_k(s, workload, n, out):
    """Median query time per method as k grows, at one n."""
    g = s[(s.workload == workload) & (s.n == n)]
    t = pd.DataFrame([dict(k=k, **_times(gk)) for k, gk in g.groupby("k")])
    if len(t):
        write_dat(out / f"exp_large_{workload}_n{n}_byk.dat", t, list(t.columns))
    return t


def points(s, workload, n, k, out):
    """Recall against median query time of every method and variant, at one (n, k)."""
    g = s[(s.workload == workload) & (s.n == n) & (s.k == k) & (s.status == "ok")].copy()
    g["recall"] = g.recall.fillna(1.0)  # brute force is exact
    for m in g.method.unique():
        write_dat(out / f"exp_large_{workload}_n{n}_k{k}_pts_{m}.dat", g[g.method == m], ["t", "recall", "nn"])
    return g[["method", "variant", "t", "recall", "nn"]]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/dat")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    for wl in ("popular", "crowd"):
        df = load(a.results, wl)
        if df is None:
            continue
        s = summarize(df)
        show = s.copy()
        show["t_ms"] = 1000 * show.pop("t")
        print(f"\n== {wl}\n" + show[["n", "k", "method", "variant", "status", "L", "L_tuned", "blocks", "queries",
                                     "t_ms", "gb", "recall", "precision", "empty", "nn", "cand", "listed", "build"]]
              .round(4).to_string(index=False))
        lines = table_rows(s, wl, out)
        print(f"\n== {wl} table rows\n" + "\n".join(lines))
        for n in sorted(s.n.unique()):
            capped(s, wl, n, out)
        for n in sorted(s.n.unique()):
            t = by_k(s, wl, n, out)
            if len(t) > 1:
                print(f"\n== {wl} n={n} by k\n" + t.to_string(index=False))
            for k in sorted(s[s.n == n].k.unique()):
                points(s, wl, n, k, out)
        for k in sorted(s.k.unique()):
            t = scaling(s, wl, k, out)
            if len(t) > 1:
                print(f"\n== {wl} k={k} scaling\n" + t.to_string(index=False))
