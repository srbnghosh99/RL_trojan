#!/usr/bin/env python3
"""
harness_from_list.py
Build the cross-agent attack comparison from a CURATED runs CSV.

Input CSV must have at least: file, agent, task  (task in {clean, attack}).
Optional: seed, root (folder holding the DTL file for convergence curves).

Workflow:
  1. run the folder scanner once  -> out/runs.csv
  2. open runs.csv, DELETE rows you don't trust (stubs, dupes, wrong config),
     fix the 'task' labels, and fill 'seed' where you know it. Save as
     runs_curated.csv
  3. python3 harness_from_list.py --runs runs_curated.csv --root . --outdir out2

Only the rows in the CSV are ever used. Only needs numpy/pandas/matplotlib.
"""
import argparse, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_dtl_test(path):
    """Return the TEST-phase curve of one DTL file, or None."""
    rows = []
    with open(path) as f:
        for ln in f:
            t = ln.split()
            if len(t) < 4 or t[1].upper() != "TEST":
                continue
            try:
                rows.append((int(t[2]), float(t[3])))
            except ValueError:
                continue
    if not rows:
        return None
    return pd.DataFrame(rows, columns=["episode", "travel_time"]).sort_values("episode")


def load_runs(csv, root, last_n):
    df = pd.read_csv(csv)
    need = {"file", "agent", "task"}
    if not need.issubset(df.columns):
        raise SystemExit(f"runs CSV must have columns {need}; has {list(df.columns)}")
    if "seed" not in df.columns:
        df["seed"] = -1

    # If final_tt_lastN isn't already in the CSV, recompute it from the file.
    curves = []
    final_tt = []
    for _, r in df.iterrows():
        val = r.get("final_tt_lastN", np.nan)
        fp = r["file"]
        cand = fp if os.path.isabs(fp) or os.path.exists(fp) else os.path.join(root, fp)
        # find the file if only a basename was given
        if not os.path.exists(cand):
            hits = [os.path.join(dp, fn) for dp, _, fns in os.walk(root)
                    for fn in fns if fn == os.path.basename(fp)]
            cand = hits[0] if hits else None
        curve = read_dtl_test(cand) if cand else None
        if curve is not None:
            curves.append(curve.assign(agent=r["agent"], task=r["task"], seed=r["seed"]))
            if pd.isna(val):
                val = curve["travel_time"].tail(last_n).mean()
        final_tt.append(val)
    df["final_tt_lastN"] = final_tt
    if df["final_tt_lastN"].isna().any():
        miss = df[df["final_tt_lastN"].isna()]["file"].tolist()
        print(f"[warn] no travel time for: {miss} (file not found and no value in CSV)")
    curves = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()
    return df.dropna(subset=["final_tt_lastN"]), curves


def summarize(df):
    g = (df.groupby(["agent", "task"])
           .agg(TT=("final_tt_lastN", "mean"), TT_std=("final_tt_lastN", "std"),
                n=("final_tt_lastN", "size")).reset_index())
    g["TT_std"] = g["TT_std"].fillna(0.0)
    return g


def clean_vs_attack(summ):
    c = summ[summ.task == "clean"].set_index("agent")
    a = summ[summ.task == "attack"].set_index("agent")
    out = []
    for ag in sorted(set(c.index) | set(a.index)):
        cc = c.loc[ag, "TT"] if ag in c.index else np.nan
        aa = a.loc[ag, "TT"] if ag in a.index else np.nan
        cs = c.loc[ag, "TT_std"] if ag in c.index else np.nan
        as_ = a.loc[ag, "TT_std"] if ag in a.index else np.nan
        pct = (aa - cc) / cc * 100 if (cc == cc and aa == aa and cc) else np.nan
        out.append(dict(agent=ag, clean_TT=cc, clean_std=cs,
                        clean_n=int(c.loc[ag, "n"]) if ag in c.index else 0,
                        attack_TT=aa, attack_std=as_,
                        attack_n=int(a.loc[ag, "n"]) if ag in a.index else 0,
                        degradation_pct=pct))
    return pd.DataFrame(out).sort_values("degradation_pct", na_position="last").round(1)


def plot_slope(tbl, out):
    both = tbl.dropna(subset=["clean_TT", "attack_TT"])
    if both.empty:
        print("[skip] slopegraph: need agents with both conditions"); return
    plt.figure(figsize=(7.5, 5))
    for _, r in both.iterrows():
        col = "#D85A30" if r.attack_TT > r.clean_TT else "#1D9E75"
        plt.plot([0, 1], [r.clean_TT, r.attack_TT], "-o", color=col, lw=2)
        plt.text(-0.02, r.clean_TT, f"{r.agent} {r.clean_TT:.0f}", ha="right", va="center", fontsize=9)
        plt.text(1.02, r.attack_TT, f"{r.agent} {r.attack_TT:.0f}", ha="left", va="center", fontsize=9)
    plt.xlim(-0.6, 1.6); plt.gca().invert_yaxis()
    plt.xticks([0, 1], ["clean", "under attack"])
    plt.ylabel("final travel time (lower = better)")
    plt.title("Same dataset, same attack \u2014 travel time by agent")
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close(); print(f"[out] {out}")


def plot_degradation(tbl, out):
    d = tbl.dropna(subset=["degradation_pct"]).sort_values("degradation_pct")
    if d.empty:
        print("[skip] degradation bars"); return
    cols = ["#1D9E75" if v <= 0 else "#D85A30" for v in d.degradation_pct]
    err = d["attack_std"].fillna(0).values
    plt.figure(figsize=(7.5, 4.5))
    plt.barh(d.agent, d.degradation_pct, color=cols)
    plt.axvline(0, color="#888", lw=0.8)
    plt.xlabel("travel-time change under attack (%) \u2014 higher = more fragile")
    plt.title("Attack fragility by agent")
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close(); print(f"[out] {out}")


def plot_convergence(curves, out):
    if curves.empty:
        return
    plt.figure(figsize=(8, 5))
    agents = sorted(curves.agent.unique())
    palette = ["#1D9E75", "#D85A30", "#534AB7", "#BA7517", "#185FA5", "#993556"]
    cmap = {a: palette[i % len(palette)] for i, a in enumerate(agents)}
    ls = {"clean": "-", "attack": "--"}
    seen = set()
    for (agent, task, seed), g in curves.groupby(["agent", "task", "seed"]):
        lab = f"{agent} / {task}"
        plt.plot(g.episode, g.travel_time, color=cmap[agent], ls=ls.get(task, ":"),
                 alpha=0.85, label=lab if lab not in seen else None)
        seen.add(lab)
    plt.xlabel("episode"); plt.ylabel("TEST travel time")
    plt.title("Convergence (solid=clean, dashed=attack)")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close(); print(f"[out] {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="curated runs CSV (file,agent,task[,seed])")
    ap.add_argument("--root", default=".", help="folder to resolve DTL files for curves")
    ap.add_argument("--outdir", default="out2")
    ap.add_argument("--last-n", type=int, default=5)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    runs, curves = load_runs(a.runs, a.root, a.last_n)
    print(f"[info] using {len(runs)} runs")
    print(runs[["file", "agent", "task", "seed", "final_tt_lastN"]].to_string(index=False))

    summ = summarize(runs)
    tbl = clean_vs_attack(summ)
    print("\n=== per-agent clean vs attack ===")
    print(tbl.to_string(index=False))
    if (summ["n"] == 1).any():
        print("\n[note] some (agent,task) have n=1 -> no error bars.")

    summ.to_csv(os.path.join(a.outdir, "summary_agent_task.csv"), index=False)
    tbl.to_csv(os.path.join(a.outdir, "clean_vs_attack.csv"), index=False)
    plot_slope(tbl, os.path.join(a.outdir, "slopegraph.png"))
    plot_degradation(tbl, os.path.join(a.outdir, "degradation_bars.png"))
    plot_convergence(curves, os.path.join(a.outdir, "convergence.png"))
    print(f"\n[done] {a.outdir}/")


if __name__ == "__main__":
    main()
