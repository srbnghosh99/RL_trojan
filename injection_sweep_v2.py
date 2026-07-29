#!/usr/bin/env python3
"""
injection_sweep.py
================================================================================
Q: For the SAME injection budget, does RL break at FEWER injected vehicles than
   heuristics? At a given injection count, which controller degrades more?

For each scenario (intersection x flow level) and each controller, sweep the
number of injected fake vehicles and record final travel time. One panel per
scenario: travel time vs injection count, one line per controller. The line
that rises earliest/steepest is the most attack-sensitive.

Controllers:  heuristics = maxqueue, maxpressure ;  RL = mplight, colight
Simulator  :  SUMO (libsumo)

MODES
-----
  --synthetic     no SUMO; fake model, just to verify plotting works.
  (default/real)  shell out to run.py per (scenario, agent, n_inject, seed),
                  then read the final TEST travel time out of the newest
                  *DTL.log the run produced. Results cached per run so an
                  interrupted sweep can resume.

!! REQUIRED WIRING (real mode) !!
--------------------------------
This script exports ATK_N_INJECT (and ATK_APPROACH / ATK_CKPT_NETWORK) as
environment variables. Your attacker must READ them, otherwise every sweep
point runs the same hardcoded injection count and the x-axis is meaningless.
In your attacker, where the vehicle count is set:

    import os
    n_inject = int(os.environ.get("ATK_N_INJECT", 10))
    approach = os.environ.get("ATK_APPROACH", "random")

Verify with --verify-injection before trusting any curve.

Typical real run:
    python3 injection_sweep.py --agents mplight --scenarios 1x1_normal \
        --max-inject 20 --step 4 --seeds 4 --device cpu --ngpu 0

Outputs (in --outdir):
    sweep_raw.csv, sweep_summary.csv, break_points.csv
    injection_sweep.png, injections_to_break.png, summary.md
"""
import argparse
import glob
import itertools
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# This script lives in the repo root, right next to run.py.
REPO = os.path.dirname(os.path.abspath(__file__))

# ---- controllers -------------------------------------------------------------
# Full set. Subset at runtime with --agents (e.g. --agents mplight).
AGENTS = ["maxqueue", "maxpressure", "mplight", "colight"]
IS_RL = {"maxqueue": False, "maxpressure": False,   # heuristics
         "mplight": True, "colight": True}          # RL
COLORS = {"maxqueue": "#BA7517", "maxpressure": "#D85A30",   # heuristics: warm
          "mplight": "#1D9E75", "colight": "#534AB7"}        # RL: cool
STYLE = {False: "--", True: "-"}                    # heuristic dashed, RL solid

# ---- scenarios ---------------------------------------------------------------
# 'network' = value passed to --network.  'ckpt' = base network whose trained
# checkpoint to load (flow variants share topology).
SCENARIOS = [
    dict(name="1x1_low",    network="cityflow1x1_low",  flow="low",    ckpt="cityflow1x1"),
    dict(name="1x1_normal", network="cityflow1x1",      flow="normal", ckpt="cityflow1x1"),
    dict(name="1x1_high",   network="cityflow1x1_high", flow="high",   ckpt="cityflow1x1"),
    dict(name="4x4_normal", network="cityflow4x4",      flow="normal", ckpt="cityflow4x4"),
]


# --------------------------------------------------------------------------- #
#  Evaluation back-ends                                                        #
# --------------------------------------------------------------------------- #
def synth_eval(agent, scenario, n_inject, seed=0):
    """Fake but plausible. NOTE: rigged so RL breaks earlier -- plumbing test only."""
    rng = np.random.default_rng(hash((agent, scenario["name"], n_inject, seed)) % 2**32)
    base = {"low": 90, "normal": 110, "high": 150}[scenario["flow"]]
    if IS_RL[agent]:
        knee, steep, base = 6, 30, base * 0.9
    else:
        knee, steep = 12, 14
    breakage = steep * max(0, n_inject - knee) ** 1.3 / 6
    return base + breakage + rng.normal(0, 4)


def _newest_dtl(since_mtime=0.0):
    """Newest *DTL.log under the repo, optionally newer than a timestamp."""
    logs = glob.glob(os.path.join(REPO, "data", "output_data", "**", "*DTL.log"),
                     recursive=True)
    logs = [p for p in logs if os.path.getmtime(p) > since_mtime]
    if not logs:
        return None
    return max(logs, key=os.path.getmtime)


def _final_test_travel_time(dtl_path, last_n=1):
    """Mean travel time over the last `last_n` TEST rows of a DTL log."""
    vals = []
    with open(dtl_path) as f:
        for ln in f:
            t = ln.split()
            if len(t) >= 4 and t[1].upper() == "TEST":
                try:
                    vals.append(float(t[3]))
                except ValueError:
                    pass
    if not vals:
        return None
    return float(np.mean(vals[-last_n:]))


def real_eval(agent, scenario, n_inject, seed, args):
    """Run one SUMO eval via run.py; return final travel time (float) or None."""
    run_dir = args.outdir
    os.makedirs(run_dir, exist_ok=True)
    # n_inject = int(os.environ.get("ATK_N_INJECT", 10))
    tag = f"{scenario['name']}__{agent}__n{n_inject}__s{seed}"
    jpath = os.path.join(run_dir, tag + ".json")

    # cached?
    if os.path.exists(jpath) and not args.force:
        try:
            return json.load(open(jpath))["travel_time"]
        except Exception:
            pass

    env = os.environ.copy()
    env.update(
        ATK_MODE="sweep",
        ATK_N_INJECT=str(n_inject),
        ATK_APPROACH=args.approach,
        ATK_OUT=os.path.abspath(os.path.join(run_dir, tag)),
    )
    if scenario.get("ckpt"):
        env["ATK_CKPT_NETWORK"] = scenario["ckpt"]

    cmd = [sys.executable, "run.py",
           "--agent", agent,
           "--world", "sumo",
           "--interface", args.interface,
           "--task", args.task,
           "--controller_source_network", scenario.get("ckpt", scenario["network"]),
           "--attacker_source_network", scenario.get("ckpt", scenario["network"]),
           "--network", scenario["network"],
           "--seed", str(seed),
           "--thread", str(args.thread),
           "--device", args.device,
           "--ngpu", str(args.ngpu)]

    print(f"  -> {scenario['name']:12s} {agent:12s} n_inject={n_inject:<3d} seed={seed} ...",
          flush=True)

    t0 = _now()
    try:
        proc = subprocess.run(cmd, cwd=REPO, env=env, timeout=args.timeout,
                              capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        print(f"     [timeout after {args.timeout}s] skipped")
        return None

    if proc.returncode != 0:
        print(f"     [run.py exit {proc.returncode}]")
        _print_tail(proc, args.tail)
        return None

    dtl = _newest_dtl(since_mtime=t0)
    if dtl is None:
        print("     [no result] no new *DTL.log was written")
        _print_tail(proc, args.tail)
        return None

    tt = _final_test_travel_time(dtl, args.last_n)
    if tt is None:
        print(f"     [no result] no TEST rows in {os.path.basename(dtl)}")
        return None

    try:
        json.dump(dict(travel_time=tt, dtl=os.path.relpath(dtl, REPO),
                       agent=agent, scenario=scenario["name"],
                       n_inject=n_inject, seed=seed), open(jpath, "w"))
    except Exception:
        pass
    print(f"     travel_time={tt:.2f}   ({os.path.basename(dtl)})")
    return tt


def _now():
    import time
    return time.time()


def _print_tail(proc, n):
    out = (proc.stdout or "").strip().splitlines()[-n:]
    err = (proc.stderr or "").strip().splitlines()[-n:]
    if out:
        print("     --- stdout tail ---")
        for l in out:
            print("     " + l)
    if err:
        print("     --- stderr tail ---")
        for l in err:
            print("     " + l)


# --------------------------------------------------------------------------- #
def verify_injection(agents, scens, args):
    """
    Sanity check BEFORE a full sweep: run n_inject=0 and n_inject=max for one
    (agent, scenario). If travel time is identical, ATK_N_INJECT is being
    ignored by the attacker and every sweep point would be the same run.
    """
    agent, sc = agents[0], scens[0]
    print(f"\n[verify] does the attacker read ATK_N_INJECT?  ({agent} / {sc['name']})")
    lo = real_eval(agent, sc, 0, int(args.seeds.split(",")[0]), args)
    hi = real_eval(agent, sc, args.max_inject, int(args.seeds.split(",")[0]), args)
    if lo is None or hi is None:
        print("[verify] FAILED to obtain both runs -- fix run.py errors above first.")
        return False
    if abs(hi - lo) < 1e-6:
        print(f"[verify] travel time IDENTICAL at n=0 and n={args.max_inject} ({lo:.2f}).")
        print("[verify] -> the attacker is NOT reading ATK_N_INJECT. Wire it "
              "(os.environ['ATK_N_INJECT']) before sweeping, or the x-axis is fake.")
        return False
    print(f"[verify] OK: n=0 -> {lo:.2f} ; n={args.max_inject} -> {hi:.2f} "
          f"(delta {hi - lo:+.2f}). Injection count has an effect.")
    return True


# --------------------------------------------------------------------------- #
def main():
    global REPO
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true", help="no SUMO; test plotting only")
    ap.add_argument("--verify-injection", action="store_true",
                    help="check ATK_N_INJECT actually changes the result, then exit")
    ap.add_argument("--max-inject", type=int, default=20)
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--seeds", default="4")
    ap.add_argument("--agents", default=",".join(AGENTS),
                    help="comma list subset, e.g. mplight or mplight,maxqueue")
    ap.add_argument("--scenarios", default=",".join(s["name"] for s in SCENARIOS),
                    help="comma list subset of scenario names")
    ap.add_argument("--task", default="tsc_rl_adversarial")
    ap.add_argument("--break-threshold", type=float, default=0.25,
                    help="fractional travel-time increase that counts as 'broken'")
    ap.add_argument("--approach", default="random", help="random|N|E|S|W")
    ap.add_argument("--interface", default="libsumo", choices=["libsumo", "traci"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--ngpu", default="0")
    ap.add_argument("--thread", default="8")
    ap.add_argument("--last-n", type=int, default=1,
                    help="avg over last N TEST rows of the DTL log")
    ap.add_argument("--timeout", type=int, default=7200, help="per-run seconds")
    ap.add_argument("--tail", type=int, default=15, help="lines of stdout/stderr on failure")
    ap.add_argument("--force", action="store_true", help="ignore cached run json")
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--outdir", default="attack_out")
    a = ap.parse_args()

    REPO = os.path.abspath(a.repo)
    if not a.synthetic and not os.path.exists(os.path.join(REPO, "run.py")):
        print(f"[error] run.py not found in --repo '{REPO}'. Pass --repo /path/to/repo.")
        return
    if not os.path.isabs(a.outdir):
        a.outdir = os.path.join(REPO, a.outdir)
    os.makedirs(a.outdir, exist_ok=True)
    print(f"[repo]   {REPO}\n[outdir] {a.outdir}")

    seeds = [int(s) for s in a.seeds.split(",") if s.strip() != ""]
    injects = list(range(0, a.max_inject + 1, a.step))
    want_agents = [x.strip() for x in a.agents.split(",") if x.strip()]
    agents = [x for x in want_agents if x in AGENTS]
    unknown = [x for x in want_agents if x not in AGENTS]
    if unknown:
        print(f"[warn] unknown agents ignored: {unknown}")
    want_scen = [x.strip() for x in a.scenarios.split(",") if x.strip()]
    scens = [s for s in SCENARIOS if s["name"] in want_scen]
    if not agents or not scens:
        print("[error] no valid agents or scenarios selected.")
        return
    print(f"[plan]   agents={agents}  scenarios={[s['name'] for s in scens]}  "
          f"n_inject={injects}  seeds={seeds}")

    if a.verify_injection and not a.synthetic:
        verify_injection(agents, scens, a)
        return

    rows = []
    for sc, agent, n, seed in itertools.product(scens, agents, injects, seeds):
        tt = synth_eval(agent, sc, n, seed) if a.synthetic else real_eval(agent, sc, n, seed, a)
        if tt is None:
            continue
        rows.append(dict(scenario=sc["name"], flow=sc["flow"], agent=agent,
                         is_rl=IS_RL[agent], n_inject=n, seed=seed, travel_time=tt))

    if not rows:
        print("\nNo results collected. Check run.py errors above "
              "(checkpoints present? SUMO working? correct --task?).")
        return

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.outdir, "sweep_raw.csv"), index=False)

    g = (df.groupby(["scenario", "agent", "n_inject"])
           .travel_time.agg(["mean", "std"]).reset_index())
    g["std"] = g["std"].fillna(0.0)
    g.to_csv(os.path.join(a.outdir, "sweep_summary.csv"), index=False)

    # flat-line warning: injection count had no effect
    for (scn, ag), s in g.groupby(["scenario", "agent"]):
        if s["mean"].std() < 1e-6 and len(s) > 1:
            print(f"[warn] {ag}/{scn}: travel time identical at every injection "
                  "count -> ATK_N_INJECT is probably not wired into the attacker.")

    _plot_curves(g, scens, agents, a.outdir)
    bt = _break_points(g, scens, agents, a.break_threshold, a.outdir)
    _plot_break_bars(bt, scens, agents, a.outdir, a.break_threshold)
    _write_summary(g, bt, scens, agents, a)
    print(f"\n[done] outputs in {a.outdir}")


# --------------------------------------------------------------------------- #
def _plot_curves(g, scens, agents, outdir):
    names = [s["name"] for s in scens if s["name"] in set(g.scenario)]
    if not names:
        return
    ncol = 2 if len(names) > 1 else 1
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.5 * ncol, 4.2 * nrow), squeeze=False)
    for ax, scn in zip(axes.flat, names):
        sub = g[g.scenario == scn]
        for agent in agents:
            s = sub[sub.agent == agent].sort_values("n_inject")
            if s.empty:
                continue
            ax.plot(s.n_inject, s["mean"], STYLE[IS_RL[agent]],
                    color=COLORS[agent], marker="o", ms=4, label=agent)
            ax.fill_between(s.n_inject, s["mean"] - s["std"], s["mean"] + s["std"],
                            color=COLORS[agent], alpha=0.12)
        ax.set_title(scn)
        ax.set_xlabel("fake vehicles injected")
        ax.set_ylabel("final travel time (s)")
        ax.legend(fontsize=8)
    for ax in axes.flat[len(names):]:
        ax.axis("off")
    fig.suptitle("Same injection budget: how few vehicles break each controller "
                 "(solid = RL, dashed = heuristic)", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "injection_sweep_{agent}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {outdir}/injection_sweep_{agent}.png")


def _break_points(g, scens, agents, thr, outdir):
    out = []
    for scn in [s["name"] for s in scens]:
        for agent in agents:
            s = g[(g.scenario == scn) & (g.agent == agent)].sort_values("n_inject")
            if s.empty:
                continue
            clean = s["mean"].iloc[0]
            hit = s[s["mean"] >= clean * (1.0 + thr)]
            n_break = int(hit.n_inject.iloc[0]) if len(hit) else None
            out.append(dict(scenario=scn, agent=agent, is_rl=IS_RL[agent],
                            clean_TT=round(float(clean), 1),
                            inject_to_break=n_break))
    bt = pd.DataFrame(out)
    if bt.empty:
        return bt
    bt.to_csv(os.path.join(outdir, "break_points.csv"), index=False)
    print(f"\n=== injections to reach +{int(thr*100)}% over each controller's "
          f"own clean travel time ===")
    print(bt.to_string(index=False))
    return bt


def _plot_break_bars(bt, scens, agents, outdir, thr):
    if bt.empty:
        return
    names = [s["name"] for s in scens if s["name"] in set(bt.scenario)]
    if not names:
        return
    x = np.arange(len(names))
    w = 0.8 / max(1, len(agents))
    fig, ax = plt.subplots(figsize=(1.8 * len(names) + 3, 4.5))
    for k, agent in enumerate(agents):
        vals = []
        for scn in names:
            r = bt[(bt.scenario == scn) & (bt.agent == agent)]
            v = r.inject_to_break.iloc[0] if len(r) else None
            vals.append(np.nan if v is None else v)
        ax.bar(x + k * w, vals, w, color=COLORS[agent],
               hatch="" if IS_RL[agent] else "//",
               edgecolor="white", label=agent)
    ax.set_xticks(x + 0.4 - w / 2)
    ax.set_xticklabels(names)
    ax.set_ylabel("fake vehicles to break")
    ax.set_title(f"Fewest injections to degrade travel time +{int(thr*100)}%\n"
                 "(shorter bar = easier to attack;  hatched = heuristic, solid = RL)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "injections_to_break.png"), dpi=150)
    plt.close(fig)
    print(f"[plot] {outdir}/injections_to_break.png")


def _write_summary(g, bt, scens, agents, a):
    if bt.empty:
        return
    lines = ["# Injection sweep - readout\n",
             f"Break threshold: +{int(a.break_threshold*100)}% over each "
             "controller's own clean travel time.\n",
             "\n## Fewest fake vehicles to break each controller\n",
             "| scenario | " + " | ".join(agents) + " |",
             "|" + "---|" * (len(agents) + 1)]
    for scn in [s["name"] for s in scens]:
        cells = []
        for agent in agents:
            r = bt[(bt.scenario == scn) & (bt.agent == agent)]
            v = r.inject_to_break.iloc[0] if len(r) else None
            cells.append("never" if (v is None or (isinstance(v, float) and np.isnan(v)))
                         else str(int(v)))
        lines.append(f"| {scn} | " + " | ".join(cells) + " |")

    lines.append("\n## RL vs heuristic (mean injections-to-break)\n")
    for scn in [s["name"] for s in scens]:
        def _vals(rl_flag):
            out = []
            for ag in agents:
                if IS_RL[ag] != rl_flag:
                    continue
                r = bt[(bt.scenario == scn) & (bt.agent == ag)]
                if len(r):
                    v = r.inject_to_break.iloc[0]
                    if v is not None and not (isinstance(v, float) and np.isnan(v)):
                        out.append(v)
            return out
        rl, he = _vals(True), _vals(False)
        if rl and he:
            lines.append(f"- **{scn}**: RL breaks at ~{np.mean(rl):.1f} veh, "
                         f"heuristics at ~{np.mean(he):.1f} veh "
                         f"({'RL more fragile' if np.mean(rl) < np.mean(he) else 'heuristic more fragile'}).")
    with open(os.path.join(a.outdir, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[summary] {a.outdir}/summary.md")


if __name__ == "__main__":
    main()
