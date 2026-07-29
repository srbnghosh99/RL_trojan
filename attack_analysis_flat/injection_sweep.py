#!/usr/bin/env python3
"""
injection_sweep.py
================================================================================
Q: For the SAME injection budget, does RL break at FEWER injected vehicles than
   heuristics? And at a given injection count, which controller degrades more?

For each scenario (intersection x flow level) and each controller, sweep the
number of injected fake vehicles (0,2,4,...,N per segment) and record final
travel time. One panel per scenario: travel time vs injection count, one line
per controller. The controller whose line rises earliest/steepest is the most
attack-sensitive. A break-point table + bar chart report the fewest injections
needed to push each controller's travel time +X% over its own clean baseline.

Controllers:  heuristics = maxqueue, maxpressure ;  RL = mplight, colight
Simulator  :  SUMO (libsumo)

MODES
-----
  --synthetic     use a plausible fake model (no SUMO) to test the plotting.
  (default/real)  shell out to run.py --task tsc_attack_analysis per run, which
                  injects fake vehicles via world.inject_fake_vehicles and
                  reports final travel time. Results are cached per run so you
                  can resume an interrupted sweep.

Typical real run (after training checkpoints exist and flow variants are made):
    python analysis/make_flow_variants.py --net cityflow1x1 --low 0.5 --high 1.6
    python analysis/injection_sweep.py --max-inject 20 --step 2 --seeds 0 \
           --device cuda:0 --ngpu 0

Outputs (in --outdir, default analysis/results/sweep):
    sweep_raw.csv, sweep_summary.csv, break_points.csv
    injection_sweep.png            (travel time vs injections, per scenario)
    injections_to_break.png        (bar chart: fewest injections to break)
    summary.md                     (plain-language readout)
"""
import argparse
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

AGENTS = ["maxqueue", "maxpressure", "mplight", "colight"]
IS_RL = {"maxqueue": False, "maxpressure": False, "mplight": True, "colight": True}
COLORS = {"maxqueue": "#BA7517", "maxpressure": "#D85A30",   # heuristics: warm
          "mplight": "#1D9E75", "colight": "#534AB7"}          # RL: cool
STYLE = {False: "--", True: "-"}                               # heuristic dashed, RL solid

# scenarios: intersection x flow level.  'network' = configs/sim_sumo/<network>.cfg
# 'ckpt' = base network whose trained checkpoint to load (flow variants share topology).
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
    """Fake but plausible: RL degrades at fewer injections than heuristics."""
    rng = np.random.default_rng(hash((agent, scenario["name"], n_inject, seed)) % 2**32)
    base = {"low": 90, "normal": 110, "high": 150}[scenario["flow"]]
    if IS_RL[agent]:
        knee, steep, base = 6, 30, base * 0.9
    else:
        knee, steep = 12, 14
    breakage = steep * max(0, n_inject - knee) ** 1.3 / 6
    return base + breakage + rng.normal(0, 4)


def real_eval(agent, scenario, n_inject, seed, args):
    """Run one SUMO eval through run.py and return final travel time (float)."""
    run_dir = args.outdir
    os.makedirs(run_dir, exist_ok=True)
    out_prefix = os.path.join(run_dir, f"{scenario['name']}__{agent}__n{n_inject}__s{seed}")
    jpath = out_prefix + ".json"
    if os.path.exists(jpath) and not args.force:
        try:
            return json.load(open(jpath))["travel_time"]
        except Exception:  # noqa
            pass

    env = os.environ.copy()
    env.update(
        ATK_MODE="sweep",
        ATK_N_INJECT=str(n_inject),
        ATK_OUT=os.path.abspath(out_prefix),
        ATK_APPROACH=args.approach,
    )
    if scenario.get("ckpt"):
        env["ATK_CKPT_NETWORK"] = scenario["ckpt"]

    cmd = [sys.executable, "run.py",
           "--task", "tsc_attack_analysis",
           "--agent", agent,
           "--network", scenario["network"],
           "--world", "sumo",
           "--interface", args.interface,
           "--seed", str(seed),
           "--thread_num", "1",
           "--ngpu", args.ngpu,
           "--device", args.device]
    print(f"  -> {scenario['name']:11s} {agent:11s} n_inject={n_inject:<3d} seed={seed} ...",
          flush=True)
    try:
        proc = subprocess.run(cmd, cwd=REPO, env=env, timeout=args.timeout,
                              capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        print(f"     [timeout after {args.timeout}s] skipped")
        return None
    if not os.path.exists(jpath):
        tail = "\n".join(proc.stderr.strip().splitlines()[-8:])
        print(f"     [no result] run.py failed. stderr tail:\n{tail}")
        return None
    try:
        return json.load(open(jpath))["travel_time"]
    except Exception as exc:  # noqa
        print(f"     [bad json] {exc}")
        return None


# --------------------------------------------------------------------------- #
def main():
    global REPO
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true", help="no SUMO; test plotting only")
    ap.add_argument("--max-inject", type=int, default=20, help="max fake vehicles PER SEGMENT")
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--agents", default=",".join(AGENTS), help="comma list subset of controllers")
    ap.add_argument("--scenarios", default=",".join(s["name"] for s in SCENARIOS),
                    help="comma list subset of scenario names")
    ap.add_argument("--break-threshold", type=float, default=0.25,
                    help="fractional travel-time increase that counts as 'broken' (0.25 = +25%%)")
    ap.add_argument("--approach", default="random", help="random|N|E|S|W")
    ap.add_argument("--interface", default="libsumo", choices=["libsumo", "traci"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--ngpu", default="0")
    ap.add_argument("--timeout", type=int, default=3600, help="per-run seconds")
    ap.add_argument("--force", action="store_true", help="ignore cached run json")
    ap.add_argument("--repo", default=REPO,
                    help="repo root containing run.py (auto-detected by default)")
    ap.add_argument("--outdir", default="attack_out")
    a = ap.parse_args()
    REPO = os.path.abspath(a.repo)
    if not os.path.exists(os.path.join(REPO, "run.py")):
        print(f"[error] run.py not found in --repo '{REPO}'. "
              f"Pass --repo /path/to/RL_TSC_Backdoor.")
        return
    # keep results next to run.py unless an absolute outdir was given
    if not os.path.isabs(a.outdir):
        a.outdir = os.path.join(REPO, a.outdir)
    os.makedirs(a.outdir, exist_ok=True)
    print(f"[repo] {REPO}\n[outdir] {a.outdir}")

    seeds = [int(s) for s in a.seeds.split(",") if s != ""]
    injects = list(range(0, a.max_inject + 1, a.step))
    agents = [x for x in a.agents.split(",") if x in AGENTS]
    scens = [s for s in SCENARIOS if s["name"] in a.scenarios.split(",")]

    rows = []
    for sc, agent, n, seed in itertools.product(scens, agents, injects, seeds):
        if a.synthetic:
            tt = synth_eval(agent, sc, n, seed)
        else:
            tt = real_eval(agent, sc, n, seed, a)
        if tt is None:
            continue
        rows.append(dict(scenario=sc["name"], flow=sc["flow"], agent=agent,
                         is_rl=IS_RL[agent], n_inject=n, seed=seed, travel_time=tt))

    if not rows:
        print("No results collected. (Real mode: check checkpoints, SUMO, and run.py errors above.)")
        return
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.outdir, "sweep_raw.csv"), index=False)

    g = (df.groupby(["scenario", "agent", "n_inject"])
           .travel_time.agg(["mean", "std"]).reset_index())
    g["std"] = g["std"].fillna(0.0)
    g.to_csv(os.path.join(a.outdir, "sweep_summary.csv"), index=False)

    _plot_curves(g, scens, agents, a.outdir)
    bt = _break_points(g, scens, agents, a.break_threshold, a.outdir)
    _plot_break_bars(bt, scens, agents, a.outdir, a.break_threshold)
    _write_summary(g, bt, scens, agents, a)
    print(f"\n[done] outputs in {a.outdir}")


# --------------------------------------------------------------------------- #
def _plot_curves(g, scens, agents, outdir):
    names = [s["name"] for s in scens]
    ncol = 2
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
        ax.set_xlabel("fake vehicles injected (per segment)")
        ax.set_ylabel("final travel time (s)")
        ax.legend(fontsize=8)
    for ax in axes.flat[len(names):]:
        ax.axis("off")
    fig.suptitle("Same injection budget: how few vehicles break each controller "
                 "(solid = RL, dashed = heuristic)", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "injection_sweep.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {outdir}/injection_sweep.png")


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
    bt.to_csv(os.path.join(outdir, "break_points.csv"), index=False)
    print(f"\n=== injections (per segment) to reach +{int(thr*100)}% over each "
          f"controller's own clean travel time ===")
    print(bt.to_string(index=False))
    return bt


def _plot_break_bars(bt, scens, agents, outdir, thr):
    names = [s["name"] for s in scens]
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
    ax.set_ylabel("fake vehicles / segment to break")
    ax.set_title(f"Fewest injections to degrade travel time +{int(thr*100)}%\n"
                 "(shorter bar = easier to attack;  hatched = heuristic, solid = RL)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "injections_to_break.png"), dpi=150)
    plt.close(fig)
    print(f"[plot] {outdir}/injections_to_break.png")


def _write_summary(g, bt, scens, agents, a):
    lines = ["# Injection sweep — readout\n",
             f"Break threshold: +{int(a.break_threshold*100)}% over each controller's own clean travel time.\n",
             "\n## Fewest fake vehicles (per segment) to break each controller\n"]
    lines.append("| scenario | " + " | ".join(agents) + " |")
    lines.append("|" + "---|" * (len(agents) + 1))
    for scn in [s["name"] for s in scens]:
        cells = []
        for agent in agents:
            r = bt[(bt.scenario == scn) & (bt.agent == agent)]
            v = r.inject_to_break.iloc[0] if len(r) else None
            cells.append("never" if (v is None or (isinstance(v, float) and np.isnan(v))) else str(int(v)))
        lines.append(f"| {scn} | " + " | ".join(cells) + " |")
    # RL vs heuristic headline
    lines.append("\n## RL vs heuristic (mean injections-to-break)\n")
    for scn in [s["name"] for s in scens]:
        rl = [bt[(bt.scenario == scn) & (bt.agent == ag)].inject_to_break.iloc[0]
              for ag in agents if IS_RL[ag] and len(bt[(bt.scenario == scn) & (bt.agent == ag)])]
        he = [bt[(bt.scenario == scn) & (bt.agent == ag)].inject_to_break.iloc[0]
              for ag in agents if not IS_RL[ag] and len(bt[(bt.scenario == scn) & (bt.agent == ag)])]
        rl = [v for v in rl if v is not None and not (isinstance(v, float) and np.isnan(v))]
        he = [v for v in he if v is not None and not (isinstance(v, float) and np.isnan(v))]
        if rl and he:
            lines.append(f"- **{scn}**: RL breaks at ~{np.mean(rl):.1f} veh/seg, "
                         f"heuristics at ~{np.mean(he):.1f} veh/seg "
                         f"({'RL more fragile' if np.mean(rl) < np.mean(he) else 'heuristic more fragile'}).")
    with open(os.path.join(a.outdir, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[summary] {a.outdir}/summary.md")


if __name__ == "__main__":
    main()
