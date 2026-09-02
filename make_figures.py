#!/usr/bin/env python3
"""
Generate the full figure set from whatever attack traces are present.

Discovers the lanes each attacker actually injects into, then renders:

  figures/space_time__<attacker>__<lane>.png        one per injected lane
  figures/space_time__<attacker>__<lane>_zoom.png   0-180s zoom of the same
  figures/space_time_compare.png                    FGSM vs RL, shared lane
  figures/green_share.png / .csv                    per-lane green time
  figures/fgsm_timeseries.png                       injections / vehicles / wait
  figures/summary.txt                               what was found and drawn

Nothing is required to exist -- every stage is skipped cleanly if its inputs
are missing, so this works with RL only, FGSM only, or both.

Usage
-----
  python3 make_figures.py
  python3 make_figures.py --outdir figures --zoom 0,180
  python3 make_figures.py --clean-signal clean_signal_state.csv
"""
import argparse
import os
import subprocess
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PLOT = os.path.join(HERE, "plot_space_time_attack.py")
GREEN = os.path.join(HERE, "green_share.py")

TRACES = [("fgsm", "White-box FGSM", "fgsm_attack"),
          ("rl",   "RL (multi-PPO)", "rl_attack")]


def log(msg, fh=None):
    print(msg)
    if fh:
        fh.write(msg + "\n")


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("   ! failed:", " ".join(cmd))
        print("   ", (r.stderr or r.stdout).strip().splitlines()[-1:])
        return False
    return True


def injected_lanes(prefix):
    """Lanes this attacker injected into, most-injected first."""
    p = f"{prefix}_positions.csv"
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p)
    if "is_fake" in d.columns:
        d["is_fake"] = d.is_fake.astype(str).str.lower().isin(["true", "1", "yes"])
    else:
        d["is_fake"] = d.vehicle_id.astype(str).str.contains("fake")
    f = d[d.is_fake]
    return f.lane.value_counts().to_dict() if len(f) else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--zoom", default="0,180", help="start,end seconds; '' to skip")
    ap.add_argument("--clean-signal", default=None,
                    help="signal_state.csv from the epsilon=0 control run")
    ap.add_argument("--max-lanes", type=int, default=4,
                    help="cap on per-lane figures per attacker")
    a = ap.parse_args()

    for script in (PLOT, GREEN):
        if not os.path.exists(script):
            print(f"ERROR: {os.path.basename(script)} not found next to this file.",
                  file=sys.stderr)
            return 1

    os.makedirs(a.outdir, exist_ok=True)
    fh = open(os.path.join(a.outdir, "summary.txt"), "w")
    log("=" * 66, fh)
    log("FIGURE GENERATION", fh)
    log("=" * 66, fh)

    present = []
    for key, label, prefix in TRACES:
        lanes = injected_lanes(prefix)
        if lanes is None:
            log(f"[skip] {prefix}_positions.csv not found", fh)
            continue
        if not lanes:
            log(f"[warn] {label}: trace exists but contains NO fake vehicles", fh)
            continue
        present.append((key, label, prefix, lanes))
        top = ", ".join(f"{L}={n}" for L, n in
                        sorted(lanes.items(), key=lambda x: -x[1])[:6])
        log(f"[ok]   {label}: {len(lanes)} injected lane(s) -> {top}", fh)

    if not present:
        log("\nNo usable traces. Nothing to draw.", fh)
        fh.close()
        return 1

    # --- 1. per-lane space-time, per attacker ------------------------------
    log("\n--- per-lane space-time ---", fh)
    for key, label, prefix, lanes in present:
        ordered = sorted(lanes, key=lambda L: -lanes[L])[: a.max_lanes]
        for L in ordered:
            flag = f"--{'fgsm' if key == 'fgsm' else 'rl'}-prefix"
            other = "--rl-prefix" if key == "fgsm" else "--fgsm-prefix"
            base = [sys.executable, PLOT, flag, prefix, other, "__none__", "--lane", L]
            out = os.path.join(a.outdir, f"space_time__{key}__{L}.png")
            if run(base + ["--out", out]):
                log(f"  {os.path.basename(out)}   ({lanes[L]} injections)", fh)
            if a.zoom:
                outz = os.path.join(a.outdir, f"space_time__{key}__{L}_zoom.png")
                if run(base + ["--time-window", a.zoom, "--out", outz]):
                    log(f"  {os.path.basename(outz)}", fh)

    # --- 2. side-by-side comparison ---------------------------------------
    if len(present) > 1:
        log("\n--- FGSM vs RL comparison ---", fh)
        out = os.path.join(a.outdir, "space_time_compare.png")
        if run([sys.executable, PLOT, "--out", out]):
            log(f"  {os.path.basename(out)}", fh)
        if a.zoom:
            outz = os.path.join(a.outdir, "space_time_compare_zoom.png")
            if run([sys.executable, PLOT, "--time-window", a.zoom, "--out", outz]):
                log(f"  {os.path.basename(outz)}", fh)
    else:
        log("\n[skip] comparison figure -- only one trace present", fh)

    # --- 3. green share ----------------------------------------------------
    log("\n--- green share ---", fh)
    args, posargs = [], []
    if a.clean_signal and os.path.exists(a.clean_signal):
        args.append(f"clean={a.clean_signal}")
    for key, label, prefix, _lanes in present:
        s = f"{prefix}_signal_state.csv"
        if os.path.exists(s):
            args.append(f"{label.split()[0]}={s}")
            posargs.append(f"{label.split()[0]}={prefix}_positions.csv")
    if args:
        out = os.path.join(a.outdir, "green_share.png")
        cmd = [sys.executable, GREEN] + args + ["--out", out]
        if posargs:
            cmd += ["--positions", ",".join(posargs)]
        if run(cmd):
            log(f"  {os.path.basename(out)} (+ .csv)", fh)
    else:
        log("  [skip] no *_signal_state.csv found", fh)

    # --- 4. FGSM per-decision time series ---------------------------------
    if os.path.exists("fgsm_timeseries.csv") and os.path.exists("plot_fgsm_timeseries.py"):
        log("\n--- FGSM time series ---", fh)
        if run([sys.executable, "plot_fgsm_timeseries.py", "fgsm_timeseries.csv"]):
            if os.path.exists("fgsm_timeseries.png"):
                os.replace("fgsm_timeseries.png",
                           os.path.join(a.outdir, "fgsm_timeseries.png"))
                log("  fgsm_timeseries.png", fh)

    log("\n" + "=" * 66, fh)
    log(f"done -> {a.outdir}/", fh)
    fh.close()

    for f in sorted(os.listdir(a.outdir)):
        print("   ", f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
