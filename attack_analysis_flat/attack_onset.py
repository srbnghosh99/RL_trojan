#!/usr/bin/env python3
"""
attack_onset.py
================================================================================
Q: WHEN does the attack start biting, and does RL absorb it LONGER than a
   heuristic? "From what time is the attack actually working?"

For a scenario, run each controller twice with the SAME seed:
  * clean    (0 injected)                 -> baseline queue vs time
  * attacked (N injected, from onset step)-> queue vs time under attack
Injection starts at a known simulation step (--onset). We log per-step total
queue. The gap between onset and the moment the attacked queue departs from its
own clean baseline = how long that controller ABSORBS the attack. Longer
absorption on RL = adaptivity, made visible, and reported in SECONDS.

Controllers:  heuristics = maxqueue, maxpressure ;  RL = mplight, colight
Simulator  :  SUMO (libsumo)

MODES
-----
  --synthetic     no SUMO; plausible fake curves to test the plotting.
  (default/real)  shell out to run.py --task tsc_attack_analysis (ATK_MODE=onset)
                  which writes <prefix>_queue.csv (sim_step, sim_time_seconds,
                  total_queue). Runs are cached so you can resume.

Typical real run:
    python analysis/attack_onset.py --scenarios 1x1_normal --n-inject 10 \
           --onset 900 --steps 3600 --device cuda:0 --ngpu 0

Outputs (in --outdir, default analysis/results/onset):
    onset__<scenario>.png       (queue vs time, onset marker, departure dots)
    absorption.csv              (steps + seconds each controller absorbs)
    absorption_bars.png         (seconds absorbed per controller)
"""
import argparse
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
COLORS = {"maxqueue": "#BA7517", "maxpressure": "#D85A30",
          "mplight": "#1D9E75", "colight": "#534AB7"}
STYLE = {False: "--", True: "-"}

SCENARIOS = {
    "1x1_low":    dict(network="cityflow1x1_low",  flow="low",    ckpt="cityflow1x1"),
    "1x1_normal": dict(network="cityflow1x1",      flow="normal", ckpt="cityflow1x1"),
    "1x1_high":   dict(network="cityflow1x1_high", flow="high",   ckpt="cityflow1x1"),
    "4x4_normal": dict(network="cityflow4x4",      flow="normal", ckpt="cityflow4x4"),
}


# --------------------------------------------------------------------------- #
def synth_series(agent, n_inject, onset, steps, interval, seed=0):
    rng = np.random.default_rng(hash((agent, n_inject, seed)) % 2**32)
    t = np.arange(steps)
    base = 25 + 3 * np.sin(t / 40) + rng.normal(0, 1.5, steps)
    if n_inject <= 0:
        return t * interval, base
    delay = 60 if IS_RL[agent] else 10          # RL absorbs longer
    rate = 0.15 if IS_RL[agent] else 0.5
    ramp = np.clip(t - (onset + delay), 0, None) * rate * (n_inject / 10.0)
    return t * interval, base + ramp


def real_series(agent, scenario, n_inject, onset, seed, args):
    """Run one onset eval through run.py; return (time_seconds, queue) arrays."""
    run_dir = args.outdir
    os.makedirs(run_dir, exist_ok=True)
    tag = "clean" if n_inject == 0 else f"n{n_inject}"
    out_prefix = os.path.join(run_dir, f"{scenario}__{agent}__{tag}__on{onset}__s{seed}")
    qcsv = out_prefix + "_queue.csv"
    if not (os.path.exists(qcsv) and not args.force):
        env = os.environ.copy()
        env.update(
            ATK_MODE="onset",
            ATK_N_INJECT=str(n_inject),
            ATK_ONSET=str(onset),
            ATK_OUT=os.path.abspath(out_prefix),
            ATK_APPROACH=args.approach,
        )
        sc = SCENARIOS[scenario]
        if sc.get("ckpt"):
            env["ATK_CKPT_NETWORK"] = sc["ckpt"]
        cmd = [sys.executable, "run.py",
               "--task", "tsc_attack_analysis", "--agent", agent,
               "--network", sc["network"], "--world", "sumo",
               "--interface", args.interface, "--seed", str(seed),
               "--thread_num", "1", "--ngpu", args.ngpu, "--device", args.device]
        print(f"  -> {scenario:11s} {agent:11s} {tag:5s} onset={onset} seed={seed} ...",
              flush=True)
        try:
            proc = subprocess.run(cmd, cwd=REPO, env=env, timeout=args.timeout,
                                  capture_output=True, text=True)
        except subprocess.TimeoutExpired:
            print(f"     [timeout after {args.timeout}s]")
            return None, None
        if not os.path.exists(qcsv):
            tail = "\n".join(proc.stderr.strip().splitlines()[-8:])
            print(f"     [no result] run.py failed. stderr tail:\n{tail}")
            return None, None
    d = pd.read_csv(qcsv)
    return d["sim_time_seconds"].to_numpy(), d["total_queue"].to_numpy()


def departure(clean_q, atk_q, time_s, onset_s, tol):
    """First time (s) after onset where attacked queue exceeds clean+tol."""
    m = time_s >= onset_s
    idx = np.where(m & ((atk_q - clean_q) > tol))[0]
    return float(time_s[idx[0]]) if len(idx) else None


# --------------------------------------------------------------------------- #
def main():
    global REPO
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--scenarios", default="1x1_normal",
                    help="comma list of: " + ", ".join(SCENARIOS))
    ap.add_argument("--agents", default=",".join(AGENTS))
    ap.add_argument("--n-inject", type=int, default=10, help="fake vehicles per segment when attacking")
    ap.add_argument("--onset", type=int, default=900, help="sim STEP at which injection starts")
    ap.add_argument("--steps", type=int, default=3600, help="only used for synthetic curves")
    ap.add_argument("--interval", type=float, default=1.0, help="seconds per sim step (synthetic)")
    ap.add_argument("--tol", type=float, default=8.0, help="queue delta that counts as 'degrading'")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--approach", default="random")
    ap.add_argument("--interface", default="libsumo", choices=["libsumo", "traci"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--ngpu", default="0")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--repo", default=REPO,
                    help="repo root containing run.py (auto-detected by default)")
    ap.add_argument("--outdir", default="attack_out")
    a = ap.parse_args()
    REPO = os.path.abspath(a.repo)
    if not os.path.exists(os.path.join(REPO, "run.py")):
        print(f"[error] run.py not found in --repo '{REPO}'. "
              f"Pass --repo /path/to/RL_TSC_Backdoor.")
        return
    if not os.path.isabs(a.outdir):
        a.outdir = os.path.join(REPO, a.outdir)
    os.makedirs(a.outdir, exist_ok=True)
    print(f"[repo] {REPO}\n[outdir] {a.outdir}")
    agents = [x for x in a.agents.split(",") if x in AGENTS]
    scens = [x for x in a.scenarios.split(",") if x in SCENARIOS]

    absorption_rows = []
    for scn in scens:
        plt.figure(figsize=(9.5, 5.2))
        onset_s = a.onset * a.interval
        for agent in agents:
            if a.synthetic:
                t, clean = synth_series(agent, 0, a.onset, a.steps, a.interval, a.seed)
                _, atk = synth_series(agent, a.n_inject, a.onset, a.steps, a.interval, a.seed)
            else:
                t, clean = real_series(agent, scn, 0, a.onset, a.seed, a)
                t2, atk = real_series(agent, scn, a.n_inject, a.onset, a.seed, a)
                if clean is None or atk is None:
                    continue
                n = min(len(clean), len(atk), len(t))
                t, clean, atk = t[:n], clean[:n], atk[:n]
                onset_s = a.onset * (t[1] - t[0] if len(t) > 1 else 1.0)

            plt.plot(t, atk, STYLE[IS_RL[agent]], color=COLORS[agent], lw=1.6, label=agent)
            plt.plot(t, clean, STYLE[IS_RL[agent]], color=COLORS[agent], lw=0.8, alpha=0.35)
            dep = departure(clean, atk, t, onset_s, a.tol)
            if dep is not None:
                yi = int(np.argmin(np.abs(t - dep)))
                plt.plot(dep, atk[yi], "o", color=COLORS[agent], ms=8, zorder=5)
                absorption_rows.append(dict(
                    scenario=scn, agent=agent, is_rl=IS_RL[agent],
                    onset_s=round(onset_s, 1), departs_s=round(dep, 1),
                    absorbed_s=round(dep - onset_s, 1)))
            else:
                absorption_rows.append(dict(
                    scenario=scn, agent=agent, is_rl=IS_RL[agent],
                    onset_s=round(onset_s, 1), departs_s=None, absorbed_s=None))

        plt.axvline(onset_s, color="#444", ls=":", lw=1.2)
        ymax = plt.ylim()[1]
        plt.text(onset_s + 5, ymax * 0.95, "injection starts", fontsize=9)
        plt.xlabel("simulation time (seconds)")
        plt.ylabel("total standing queue (vehicles)")
        plt.title(f"Attack onset — {scn} ({a.n_inject} veh/seg)\n"
                  "dots = when each controller starts degrading   "
                  "(solid = RL, dashed = heuristic; faint = clean baseline)")
        plt.legend(fontsize=8)
        plt.tight_layout()
        fp = os.path.join(a.outdir, f"onset__{scn}.png")
        plt.savefig(fp, dpi=150)
        plt.close()
        print(f"[plot] {fp}")

    if not absorption_rows:
        print("No series collected. (Real mode: check checkpoints, SUMO, run.py errors above.)")
        return
    adf = pd.DataFrame(absorption_rows)
    adf.to_csv(os.path.join(a.outdir, "absorption.csv"), index=False)
    print("\n=== absorption time (seconds from injection start to degradation) ===")
    print(adf.to_string(index=False))
    _plot_absorption_bars(adf, scens, agents, a.outdir)
    print(f"\n[done] outputs in {a.outdir}")


def _plot_absorption_bars(adf, scens, agents, outdir):
    x = np.arange(len(scens))
    w = 0.8 / max(1, len(agents))
    fig, ax = plt.subplots(figsize=(1.8 * len(scens) + 3, 4.5))
    for k, agent in enumerate(agents):
        vals = []
        for scn in scens:
            r = adf[(adf.scenario == scn) & (adf.agent == agent)]
            v = r.absorbed_s.iloc[0] if len(r) else None
            vals.append(np.nan if v is None else v)
        ax.bar(x + k * w, vals, w, color=COLORS[agent],
               hatch="" if IS_RL[agent] else "//", edgecolor="white", label=agent)
    ax.set_xticks(x + 0.4 - w / 2)
    ax.set_xticklabels(scens)
    ax.set_ylabel("seconds absorbed before degrading")
    ax.set_title("How long each controller absorbs the attack\n"
                 "(taller bar = more adaptive;  hatched = heuristic, solid = RL)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fp = os.path.join(outdir, "absorption_bars.png")
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    print(f"[plot] {fp}")


if __name__ == "__main__":
    main()
