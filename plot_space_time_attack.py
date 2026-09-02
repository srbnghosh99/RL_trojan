#!/usr/bin/env python3
"""
Space-time diagram for fake-vehicle injection attacks on TSC.

Plots, for one approach lane: real vehicle trajectories (position along lane vs
time) as thin lines, injected fake vehicles as points, and the traffic-signal
state for that lane as a band at the stop bar.

Works for either attacker, and renders them side by side when both are present:

  RL / multi-PPO :  rl_attack_positions.csv    + rl_attack_signal_state.csv
  White-box FGSM :  fgsm_attack_positions.csv  + fgsm_attack_signal_state.csv

(The FGSM CSVs only exist after applying patch_whitebox_trajectory.py, which
adds the same trajectory samplers to trainer/tsc_trainer_whitebox.py.)

Usage
-----
  python3 plot_space_time_attack.py                        # auto-detect both
  python3 plot_space_time_attack.py --lane road_1_2_3_0
  python3 plot_space_time_attack.py --rl-prefix rl_attack --fgsm-prefix fgsm_attack
  python3 plot_space_time_attack.py --out space_time_compare.png --time-window 0,300
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

# ---------------------------------------------------------------------------
# Palette — validated categorical slots 1 & 2 (light surface #fcfcfb).
# Signal colors are domain-semantic (a traffic light really is R/A/G).
# ---------------------------------------------------------------------------
SURFACE      = "#fcfcfb"
INK          = "#0b0b0b"
INK_SECOND   = "#52514e"
INK_MUTED    = "#8a8880"
GRID         = "#e6e5e1"

C_REAL       = "#2a78d6"   # slot 1 blue
C_FAKE       = "#eb6834"   # slot 2 orange
C_GREEN      = "#008300"
C_YELLOW     = "#eda100"
C_RED        = "#e34948"


def signal_color(ch):
    c = ch.lower()
    if c == "g":
        return C_GREEN
    if c == "y":
        return C_YELLOW
    if c == "r":
        return C_RED
    return "#cccccc"


def load_pair(prefix):
    """Return (positions_df, signal_df) for a prefix, or (None, None)."""
    ppath = f"{prefix}_positions.csv"
    spath = f"{prefix}_signal_state.csv"
    if not os.path.exists(ppath):
        return None, None
    pos = pd.read_csv(ppath)
    if "is_fake" in pos.columns:
        pos["is_fake"] = pos["is_fake"].astype(str).str.lower().isin(["true", "1", "yes"])
    else:
        pos["is_fake"] = pos["vehicle_id"].astype(str).str.contains("fake")
    sig = pd.read_csv(spath) if os.path.exists(spath) else None
    return pos, sig


def pick_lane(pos, preferred=None):
    """Choose the lane with the most fake injections, tie-broken by real volume."""
    if preferred:
        return preferred
    fakes = pos[pos.is_fake]
    if len(fakes):
        counts = fakes.lane.value_counts()
        top = counts[counts == counts.max()].index.tolist()
        if len(top) > 1:
            real = pos[(~pos.is_fake) & (pos.lane.isin(top))].lane.value_counts()
            return real.index[0] if len(real) else top[0]
        return top[0]
    return pos.lane.value_counts().index[0]


def lane_signal_series(sig, lane):
    """Extract (time, state_char) for one lane from the TLS log."""
    if sig is None or not len(sig):
        return None
    out = []
    for _, row in sig.iterrows():
        lanes = str(row["controlled_lanes"]).split(";")
        state = str(row["state"])
        idx = next((i for i, L in enumerate(lanes) if L == lane and i < len(state)), None)
        if idx is not None:
            out.append((row["time"], state[idx]))
    if not out:
        return None
    return pd.DataFrame(out, columns=["time", "state"]).sort_values("time")


def draw_panel(ax, pos, sig, lane, title, window=None, show_ylabel=True):
    d = pos[pos.lane == lane].copy()
    if window:
        d = d[(d.time >= window[0]) & (d.time <= window[1])]
    if not len(d):
        ax.text(0.5, 0.5, f"no data for {lane}", ha="center", va="center",
                transform=ax.transAxes, color=INK_MUTED, fontsize=10)
        ax.set_title(title, fontsize=11, color=INK, loc="left", pad=10)
        return 0, 0

    real = d[~d.is_fake]
    fake = d[d.is_fake]
    ymax = float(d.position.max())

    # --- signal band at the stop bar (top of the lane) ---
    band_h = ymax * 0.045
    band_y = ymax + band_h * 0.55
    ss = lane_signal_series(sig, lane)
    if ss is not None and len(ss):
        if window:
            ss = ss[(ss.time >= window[0]) & (ss.time <= window[1])]
        t = ss.time.values
        st = ss.state.values
        if len(t) > 1:
            dt = np.median(np.diff(t)) or 1.0
            for i in range(len(t)):
                ax.add_patch(Rectangle((t[i], band_y), dt, band_h,
                                       facecolor=signal_color(st[i]),
                                       edgecolor="none", zorder=1))
        ax.text(t[0] if len(t) else 0, band_y + band_h * 1.5, "signal",
                fontsize=7.5, color=INK_MUTED, va="bottom")

    # --- real vehicle trajectories: thin lines, one per vehicle ---
    for _, g in real.groupby("vehicle_id"):
        if len(g) < 2:
            continue
        g = g.sort_values("time")
        ax.plot(g.time.values, g.position.values, color=C_REAL,
                linewidth=0.7, alpha=0.55, solid_capstyle="round", zorder=2)

    # --- fake vehicles: single-instant ghosts, so they are points ---
    if len(fake):
        ax.scatter(fake.time.values, fake.position.values, s=14, color=C_FAKE,
                   alpha=0.85, linewidths=0.4, edgecolors=SURFACE, zorder=3)
        # If every fake lands at essentially the same coordinate, say so on the
        # figure -- that is a physical-plausibility (and stealth) problem, not
        # a plotting artifact.
        spread = float(fake.position.std() or 0.0)
        if spread < 5.0:
            py = float(fake.position.mean())
            ax.annotate(
                f"all injections at {py:.0f} m  (spread {spread:.1f} m)",
                xy=(float(fake.time.max()), py),
                xytext=(0.985, (py / (ymax * 1.12)) - 0.10),
                textcoords="axes fraction", ha="right", va="top",
                fontsize=7.5, color=C_FAKE,
                arrowprops=dict(arrowstyle="-", color=C_FAKE, lw=0.7,
                                shrinkA=0, shrinkB=2, alpha=0.7),
            )

    ax.set_xlabel("simulation time (s)", fontsize=9, color=INK_SECOND)
    if show_ylabel:
        ax.set_ylabel("position along lane (m)   →  stop bar", fontsize=9, color=INK_SECOND)
    ax.set_title(title, fontsize=11, color=INK, loc="left", pad=10)
    ax.set_ylim(0, band_y + band_h * 2.6)
    ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_SECOND, labelsize=8.5, length=3)
    return real.vehicle_id.nunique(), len(fake)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rl-prefix", default="rl_attack")
    ap.add_argument("--fgsm-prefix", default="fgsm_attack")
    ap.add_argument("--lane", default=None, help="lane id; default = lane with most injections")
    ap.add_argument("--time-window", default=None, help="start,end in seconds")
    ap.add_argument("--out", default="space_time_attack.png")
    ap.add_argument("--dpi", type=int, default=200)
    a = ap.parse_args()

    window = None
    if a.time_window:
        lo, hi = a.time_window.split(",")
        window = (float(lo), float(hi))

    panels = []
    for prefix, label in ((a.fgsm_prefix, "White-box FGSM"), (a.rl_prefix, "RL (multi-PPO)")):
        pos, sig = load_pair(prefix)
        if pos is None:
            print(f"[skip] {prefix}_positions.csv not found")
            continue
        panels.append((label, pos, sig, prefix))

    if not panels:
        print("No position CSVs found. Nothing to plot.", file=sys.stderr)
        return 1

    lane = a.lane or pick_lane(panels[0][1])
    print(f"lane: {lane}")

    fig, axes = plt.subplots(1, len(panels), figsize=(7.2 * len(panels), 4.6),
                             squeeze=False, sharey=False)
    fig.patch.set_facecolor(SURFACE)
    axes = axes[0]

    for k, (label, pos, sig, prefix) in enumerate(panels):
        ax = axes[k]
        ax.set_facecolor(SURFACE)
        nreal, nfake = draw_panel(ax, pos, sig, lane, label, window, show_ylabel=(k == 0))
        print(f"  {label}: {nreal} real vehicles, {nfake} fake injections")

    handles = [
        Line2D([0], [0], color=C_REAL, lw=1.6, label="real vehicle trajectory"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C_FAKE,
               markeredgecolor=SURFACE, markersize=6, label="injected fake vehicle"),
        Line2D([0], [0], color=C_GREEN, lw=5, label="green"),
        Line2D([0], [0], color=C_YELLOW, lw=5, label="yellow"),
        Line2D([0], [0], color=C_RED, lw=5, label="red"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               fontsize=8.5, labelcolor=INK_SECOND, bbox_to_anchor=(0.5, -0.015))

    fig.suptitle(f"Space-time diagram under fake-vehicle injection  ·  lane {lane}",
                 fontsize=12.5, color=INK, x=0.008, ha="left", y=0.99)
    fig.tight_layout(rect=[0, 0.055, 1, 0.95])
    fig.savefig(a.out, dpi=a.dpi, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
