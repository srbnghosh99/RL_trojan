#!/usr/bin/env python3
"""
dtl_attack_harness.py
Cross-agent attack analysis from LibSignal DTL logs.

Walks a folder of *DTL.log files. Each file = one run's full curve, tab/space
separated, columns:
  agent | phase(TRAIN/TEST) | episode | travel_time | ... | throughput
Agent is column 1. TASK (clean vs attack) is NOT in the rows, so it's inferred
from the FILE PATH. Verify the printed labels.

Run:
  python3 dtl_attack_harness.py --root . --outdir out

Task inference (in order):
  1. --manifest manifest.csv   (columns: file,task[,seed,agent]) exact overrides
  2. --attack-keywords / --clean-keywords  (comma lists matched in the path)
     defaults: attack = adversarial,attack,trojan ; clean = tsc,clean,benign
  If a file matches neither, it's labeled 'unknown' and excluded from the
  clean-vs-attack comparison (but still shown in convergence).

Only needs: numpy, pandas, matplotlib.
"""
import argparse, glob, os, re, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEF_ATTACK = ["adversarial", "attack", "trojan"]
DEF_CLEAN  = ["tsc", "clean", "benign"]


def parse_dtl(path):
    """Return DataFrame with agent, phase, episode, travel_time, reward,
    queue, delay, throughput (reward/queue/delay/throughput only reliable
    for TEST rows; travel_time reliable for all)."""
    rows = []
    with open(path) as f:
        for ln in f:
            t = ln.split()
            if len(t) < 4:
                continue
            agent, phase = t[0], t[1].upper()
            if phase not in ("TRAIN", "TEST"):
                continue
            try:
                ep = int(t[2]); tt = float(t[3])
            except ValueError:
                continue
            rec = dict(agent=agent, phase=phase, episode=ep, travel_time=tt,
                       reward=np.nan, queue=np.nan, delay=np.nan, throughput=np.nan)
            # TEST layout: 4=100,5=100,6=reward,7=queue,8=delay,9=throughput
            if phase == "TEST" and len(t) >= 10:
                try:
                    rec.update(reward=float(t[6]), queue=float(t[7]),
                               delay=float(t[8]), throughput=float(t[9]))
                except ValueError:
                    pass
            rows.append(rec)
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["file"] = path
    return df


def infer_task(path, attack_kw, clean_kw):
    p = path.lower()
    if any(k in p for k in attack_kw):   # check attack FIRST ('tsc' is in 'tsc_rl_adversarial')
        return "attack"
    if any(k in p for k in clean_kw):
        return "clean"
    return "unknown"


def infer_seed(path):
    m = re.search(r"seed[_\-]?(\d+)", path.lower())
    return int(m.group(1)) if m else -1


def load_all(root, pattern, manifest, attack_kw, clean_kw):
    files = sorted(glob.glob(os.path.join(root, "**", pattern), recursive=True))
    if not files:
        sys.exit(f"no files matching {pattern} under {root}")
    man = {}
    if manifest:
        mdf = pd.read_csv(manifest)
        for _, r in mdf.iterrows():
            man[os.path.basename(str(r["file"]))] = r.to_dict()

    per_run, curves = [], []
    print(f"{'file':<45} {'agent':<20} {'task':<8} seed")
    print("-" * 82)
    for fp in files:
        df = parse_dtl(fp)
        if df is None:
            print(f"{os.path.basename(fp):<45} (no parseable rows)")
            continue
        base = os.path.basename(fp)
        agent = df["agent"].iloc[0]
        task = infer_task(fp, attack_kw, clean_kw)
        seed = infer_seed(fp)
        if base in man:                                   # manifest overrides
            m = man[base]
            task = str(m.get("task", task))
            if "seed" in m and not pd.isna(m["seed"]): seed = int(m["seed"])
            if "agent" in m and not pd.isna(m["agent"]): agent = str(m["agent"])
        print(f"{base:<45} {agent:<20} {task:<8} {seed}")

        test = df[df.phase == "TEST"].sort_values("episode")
        curves.append(test.assign(agent=agent, task=task, seed=seed))
        if len(test) == 0:
            continue
        per_run.append(dict(
            file=base, agent=agent, task=task, seed=seed,
            n_test_ep=len(test),
            final_tt_last=test["travel_time"].iloc[-1],
            final_tt_lastN=test["travel_time"].tail(ARGS.last_n).mean(),
            final_thru=test["throughput"].tail(ARGS.last_n).mean(),
            final_queue=test["queue"].tail(ARGS.last_n).mean(),
        ))
    runs = pd.DataFrame(per_run)
    curves = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()
    return runs, curves


def summarize(runs):
    g = (runs.groupby(["agent", "task"])
             .agg(TT=("final_tt_lastN", "mean"),
                  TT_std=("final_tt_lastN", "std"),
                  thru=("final_thru", "mean"),
                  n=("final_tt_lastN", "size")).reset_index())
    g["TT_std"] = g["TT_std"].fillna(0.0)
    return g


def clean_vs_attack(summ):
    c = summ[summ.task == "clean"].set_index("agent")
    a = summ[summ.task == "attack"].set_index("agent")
    rows = []
    for ag in sorted(set(c.index) | set(a.index)):
        cc = c.loc[ag, "TT"] if ag in c.index else np.nan
        aa = a.loc[ag, "TT"] if ag in a.index else np.nan
        pct = (aa - cc) / cc * 100 if (cc == cc and aa == aa and cc) else np.nan
        rows.append(dict(agent=ag, clean_TT=cc,
                         clean_n=int(c.loc[ag, "n"]) if ag in c.index else 0,
                         attack_TT=aa,
                         attack_n=int(a.loc[ag, "n"]) if ag in a.index else 0,
                         degradation_pct=pct))
    return pd.DataFrame(rows).sort_values("degradation_pct", na_position="last").round(1)


def plot_slope(tbl, out):
    both = tbl.dropna(subset=["clean_TT", "attack_TT"])
    if both.empty:
        print("[skip] slopegraph: need agents with BOTH clean and attack runs"); return
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
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    print(f"[out] {out}")


def plot_degradation(tbl, out):
    d = tbl.dropna(subset=["degradation_pct"]).sort_values("degradation_pct")
    if d.empty:
        print("[skip] degradation bars: no agent has both conditions"); return
    cols = ["#1D9E75" if v <= 0 else "#D85A30" for v in d.degradation_pct]
    plt.figure(figsize=(7.5, 4.5))
    plt.barh(d.agent, d.degradation_pct, color=cols)
    plt.axvline(0, color="#888", lw=0.8)
    plt.xlabel("travel-time change under attack (%) \u2014 higher = more fragile")
    plt.title("Attack fragility by agent")
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    print(f"[out] {out}")


def plot_convergence(curves, out):
    if curves.empty:
        return
    plt.figure(figsize=(8, 5))
    agents = sorted(curves.agent.unique())
    palette = ["#1D9E75", "#D85A30", "#534AB7", "#BA7517", "#185FA5", "#993556"]
    cmap = {a: palette[i % len(palette)] for i, a in enumerate(agents)}
    ls = {"clean": "-", "attack": "--", "unknown": ":"}
    seen = set()
    for (agent, task, seed), g in curves.groupby(["agent", "task", "seed"]):
        g = g.sort_values("episode")
        lab = f"{agent} / {task}"
        plt.plot(g.episode, g.travel_time, color=cmap[agent],
                 ls=ls.get(task, ":"), alpha=0.85,
                 label=lab if lab not in seen else None)
        seen.add(lab)
    plt.xlabel("episode"); plt.ylabel("TEST travel time")
    plt.title("Convergence per run (solid=clean, dashed=attack)")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    print(f"[out] {out}")


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--glob", default="*DTL.log")
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--last-n", type=int, default=5,
                    help="avg final travel time over last N TEST episodes")
    ap.add_argument("--attack-keywords", default=",".join(DEF_ATTACK))
    ap.add_argument("--clean-keywords", default=",".join(DEF_CLEAN))
    ARGS = ap.parse_args()

    os.makedirs(ARGS.outdir, exist_ok=True)
    atk = [k.strip().lower() for k in ARGS.attack_keywords.split(",") if k.strip()]
    cln = [k.strip().lower() for k in ARGS.clean_keywords.split(",") if k.strip()]

    runs, curves = load_all(ARGS.root, ARGS.glob, ARGS.manifest, atk, cln)
    if runs.empty:
        sys.exit("no TEST rows found in any file")

    n_unknown = (runs.task == "unknown").sum()
    if n_unknown:
        print(f"\n[warn] {n_unknown} run(s) labeled 'unknown' (task not found in path). "
              "Fix with --attack-keywords/--clean-keywords or --manifest; "
              "they're excluded from the clean-vs-attack comparison.")

    runs.to_csv(os.path.join(ARGS.outdir, "runs.csv"), index=False)
    summ = summarize(runs)
    summ.to_csv(os.path.join(ARGS.outdir, "summary_agent_task.csv"), index=False)
    tbl = clean_vs_attack(summ)

    print("\n=== per-agent clean vs attack (final travel time) ===")
    print(tbl.to_string(index=False))
    single = runs.groupby(["agent", "task"]).size()
    if (single == 1).any():
        print("\n[note] some (agent,task) have a single run -> no variance; add seeds.")

    tbl.to_csv(os.path.join(ARGS.outdir, "clean_vs_attack.csv"), index=False)
    plot_slope(tbl, os.path.join(ARGS.outdir, "slopegraph.png"))
    plot_degradation(tbl, os.path.join(ARGS.outdir, "degradation_bars.png"))
    plot_convergence(curves, os.path.join(ARGS.outdir, "convergence.png"))
    print(f"\n[done] outputs in {ARGS.outdir}/")


if __name__ == "__main__":
    main()