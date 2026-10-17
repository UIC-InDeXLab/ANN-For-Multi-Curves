"""Summaries of the tuned experiments and pgfplots data for the paper.

Run where results/ lives:  python analyze_tuned.py [--results results] [--out results/dat]
Every method ran with its own tuned tables at every grid size delta. For each method we report the grid size with
the lowest cost (fraction of the memory budget plus fraction of the time budget), its median query time, index size,
recall (exact nearest neighbor returned, on held-out queries), and mean ratio of the returned distance to the
nearest-neighbor distance.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_budget import sec, write_dat

METHODS = ["store_all", "probe_all", "blocking", "per_curve"]
MEM, TIME = 24e9, 60.0


def load(path):
    p = Path(path)
    return pd.read_csv(p, dtype={"blocks": str}) if p.exists() and p.stat().st_size > 0 else None


def configs(df, keys):
    """One row per (keys, method, delta, blocks): status, median time, memory, recall, ratio, cost."""
    rows = []
    for key, g in df.groupby(keys + ["method", "delta", "blocks"], dropna=False):
        ok = g[g.status == "ok"]
        timeout = (g.status == "over time budget").any()
        status = "ok" if len(ok) and not timeout else g.status.iloc[-1]
        t = ok.seconds.median() if len(ok) else np.nan
        gb = g.index_bytes.iloc[0] / 1e9
        found = ok[ok.found == 1]
        rows.append(dict(zip(keys + ["method", "delta", "blocks"], key), status=status, L=g.L.iloc[0],
                         stored=g.stored.iloc[0], probed=g.probed.iloc[0], t=t, gb=gb,
                         recall=ok.exact.mean() if len(ok) else np.nan,
                         ratio=(found.d_ret / found.d_nn).mean() if len(found) else np.nan,
                         cand=ok.candidates.mean() if len(ok) else np.nan, bf=g.bf_seconds.median(),
                         listed=pd.to_numeric(ok.listed, errors="coerce").mean() if len(ok) else np.nan,
                         cost=gb * 1e9 / MEM + t / TIME if status == "ok" else np.nan,
                         build=g.build_seconds.iloc[0] if "build_seconds" in g else np.nan))
    return pd.DataFrame(rows)


def best(c, keys):
    """Per (keys, method): the feasible configuration with the lowest cost, or the reason none is feasible."""
    out = []
    for key, g in c.groupby(keys + ["method"]):
        ok = g[g.status == "ok"]
        if len(ok):
            out.append(ok.sort_values("cost").iloc[0])
            continue
        r = g.iloc[0].copy()
        # Grid sizes that cannot reach the recall target do not decide which budget binds.
        st = {s for s in g.status if not s.startswith("recall target")} or set(g.status)
        if st == {"over memory budget"}:
            r["status"] = "over memory budget"
        elif all(s.startswith("over time budget") for s in st):
            r["status"] = "over time budget"
        elif all(s.startswith("recall target") for s in st):
            r["status"] = "recall target not reached"
        elif all(s.startswith("over") or s.startswith("no blocking") for s in st):
            r["status"] = "over memory or time budget"
        else:
            r["status"] = "infeasible"
        out.append(r)
    return pd.DataFrame(out)


def wide(b, key, name, out):
    rows = []
    for v, g in b.groupby(key):
        row = {key: v, "bf_s": g.bf.median()}
        for m in METHODS:
            gm = g[g.method == m]
            if not len(gm):
                continue
            r = gm.iloc[0]
            ok = r.status == "ok"
            for col in ("t", "gb", "recall", "ratio", "cost", "delta", "L", "cand", "listed"):
                row[f"{m}_{col}"] = r[col] if ok else np.nan
            row[f"{m}_status"] = r.status.replace(" ", "_")
            row[f"{m}_blocks"] = r.blocks if isinstance(r.blocks, str) else ""
        rows.append(row)
    t = pd.DataFrame(rows).sort_values(key)
    num = [c for c in t.columns if not c.endswith("_status") and not c.endswith("_blocks")]
    write_dat(out / f"exp_{name}.dat", t, num)
    return t


def compact(b, keys, title):
    """One line per configuration: the chosen grid size, tables, blocks, median time, memory, recall, ratio."""
    cols = keys + ["method", "status", "delta", "L", "blocks", "t", "gb", "recall", "ratio", "cand", "bf"]
    t = b[cols].copy()
    t["t_ms"] = 1000 * t.pop("t")
    t["bf_ms"] = 1000 * t.pop("bf")
    print(f"\n== {title}\n" + t.sort_values(keys + ["method"]).round(4).to_string(index=False))


def synthetic(df, tun, out, name):
    c = configs(df, ["n", "k"])
    b = best(c, ["n", "k"])
    compact(b, ["n", "k"], f"{name}: cheapest configuration per method (cost = memory/24 GB + time/60 s)")
    for n, g in b.groupby("n"):
        wide(g, "k", f"tuned_{name}_n{n}", out)
    kfix = b[b.k == 8]
    if kfix.n.nunique() > 1:
        wide(kfix, "n", f"tuned_{name}_k8", out)


def mot(df, tun, out):
    c = configs(df, ["d", "k", "n"])
    b = best(c, ["d", "k", "n"])
    lines = []
    for d, gd in b.groupby("d"):
        t = wide(gd, "k", f"tuned_mot_d{d}", out)
        print(f"\n== MOT d={d}: best configuration per method\n" + t.round(4).to_string(index=False))
        for i, ((k, n), g) in enumerate(gd.groupby(["k", "n"])):
            cells = [f"{d}-D" if i == 0 else "", str(k), f"{n:,}".replace(",", "{,}"), sec(g.bf.median())]
            for m, width in (("blocking", 3), ("probe_all", 2), ("store_all", 2), ("per_curve", 3)):
                gm = g[g.method == m]
                r = gm.iloc[0] if len(gm) else None
                if r is None:
                    cells += ["--"] * width
                elif r.status == "ok":
                    vals = [f"{100 * r.recall:.0f}", sec(r.t)]
                    if width == 3:
                        mb = 1000 * r.gb
                        vals.append(f"{mb:.2f}" if mb < 1 else f"{mb:.1f}" if mb < 10 else f"{mb:,.0f}".replace(",", "{,}"))
                    cells += vals
                elif r.status == "over memory budget":
                    cells += [r"\multicolumn{%d}{c}{over 24\,GB}" % width]
                elif r.status == "over time budget":
                    cells += [r"\multicolumn{%d}{c}{over 60\,s}" % width]
                elif r.status == "recall target not reached":
                    cells += [r"\multicolumn{%d}{c}{below 90\%%}" % width]
                else:
                    cells += [r"\multicolumn{%d}{c}{over budget}" % width]
            lines.append(" & ".join(cells) + r"\\")
        lines.append(r"\midrule")
    (out / "exp_tuned_mot_table.tex").write_text("\n".join(lines[:-1]) + "\n")
    print("\n== MOT table rows\n" + "\n".join(lines[:-1]))
    if tun is not None:
        print("\n== MOT tuning\n" + tun.to_string(index=False))


def pareto(df, out):
    c = configs(df, ["dataset", "n", "k"])
    for (ds, k), g in c.groupby(["dataset", "k"]):
        blk = g[(g.method == "blocking") & (g.status == "ok")].sort_values("stored").copy()
        blk["t_ms"] = 1000 * blk.t
        write_dat(out / f"exp_tuned_pareto_{ds}_k{k}.dat", blk, ["stored", "probed", "gb", "t", "t_ms", "recall"])
        over = g[(g.method == "blocking") & (g.status != "ok") & (g.status != "over memory budget")]
        write_dat(out / f"exp_tuned_pareto_{ds}_k{k}_overtime.dat", over, ["stored", "probed", "gb"])
        other = g[(g.method != "blocking") & (g.status == "ok")].copy()
        other["t_ms"] = 1000 * other.t
        for _, r in other.iterrows():
            write_dat(out / f"exp_tuned_pareto_{ds}_k{k}_{r.method}.dat", other[other.method == r.method],
                      ["gb", "t", "t_ms", "recall"])
        print(f"\n== pareto {ds} k={k}\n" + g.sort_values(["method", "stored"])[
            ["method", "blocks", "stored", "probed", "gb", "t", "recall", "cost", "status"]].round(4).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/dat")
    a = ap.parse_args()
    R, out = Path(a.results), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)
    pd.set_option("display.max_rows", 500)
    for name in ("popular", "crowd", "lanes"):
        df = load(R / f"tuned_{name}.csv")
        if df is not None:
            synthetic(df, load(R / f"tuned_{name}_tuning.csv"), out, name)
    df = load(R / "tuned_mot.csv")
    if df is not None:
        mot(df, load(R / "tuned_mot_tuning.csv"), out)
    df = load(R / "tuned_pareto.csv")
    if df is not None:
        pareto(df, out)
