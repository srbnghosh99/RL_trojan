#!/usr/bin/env python3
"""
FGSM vs RL space-time diagrams for EVERY lane.

Renders one two-panel figure per lane (FGSM left, RL right, shared axes) plus a
single small-multiples contact sheet with all lanes at once.

Inputs (all four, from the two attack runs):
    fgsm_attack_positions.csv   fgsm_attack_signal_state.csv
    rl_attack_positions.csv     rl_attack_signal_state.csv

Usage
-----
  python3 space_time_all_lanes.py
  python3 space_time_all_lanes.py --match-times          # fair when the two runs
                                                         # sampled at different rates
  python3 space_time_all_lanes.py --outdir st_all --time-window 0,600
  python3 space_time_all_lanes.py --a-prefix fgsm_attack --a-label "White-box FGSM" \
                                  --b-prefix rl_attack   --b-label "RL (multi-PPO)"

--match-times restricts BOTH traces to timestamps they share. Use it whenever one
run sampled per decision (60 points) and the other per second (600+) -- otherwise
the denser trace looks smoother purely because it has more samples.
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

SURFACE, INK, INK_SECOND, INK_MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8880", "#e6e5e1"
C_REAL, C_FAKE = "#2a78d6", "#eb6834"
C_GREEN, C_YELLOW, C_RED = "#008300", "#eda100", "#e34948"


def signal_color(ch):
    c = str(ch).lower()
    return {"g": C_GREEN, "y": C_YELLOW, "r": C_RED}.get(c, "#cccccc")


def load_pair(prefix):
    p, s = f"{prefix}_positions.csv", f"{prefix}_signal_state.csv"
    if not os.path.exists(p):
        return None, None
    pos = pd.read_csv(p)
    if "is_fake" in pos.columns:
        pos["is_fake"] = pos.is_fake.astype(str).str.lower().isin(["true", "1", "yes"])
    else:
        pos["is_fake"] = pos.vehicle_id.astype(str).str.contains("fake")
    sig = pd.read_csv(s) if os.path.exists(s) else None
    return pos, sig


def lane_signal(sig, lane):
    if sig is None or not len(sig):
        return None
    out = []
    for _, r in sig.iterrows():
        lanes = str(r["controlled_lanes"]).split(";")
        st = str(r["state"])
        idx = next((i for i, L in enumerate(lanes) if L == lane and i < len(st)), None)
        if idx is not None:
            out.append((r["time"], st[idx]))
    if not out:
        return None
    df = pd.DataFrame(out, columns=["time", "state"]).sort_values("time")
    # The RL trainer samples each decision step TWICE (once in the decision
    # block, once at the top of the rollout), so timestamps repeat. Duplicates
    # corrupt the median-gap estimate below, so drop them.
    return df.drop_duplicates(subset=["time"], keep="first")


def draw(ax, pos, sig, lane, title, window, ylim, xlim, compact=False):
    d = pos[pos.lane == lane]
    if window:
        d = d[(d.time >= window[0]) & (d.time <= window[1])]
    if not len(d):
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes,
                color=INK_MUTED, fontsize=8)
        ax.set_title(title, fontsize=9 if compact else 11, color=INK, loc="left", pad=6)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        return 0, 0

    real, fake = d[~d.is_fake], d[d.is_fake]
    top = ylim[1]
    band_h = top * 0.040
    band_y = top - band_h * 1.25

    ss = lane_signal(sig, lane)
    if ss is not None and len(ss):
        if window:
            ss = ss[(ss.time >= window[0]) & (ss.time <= window[1])]
        t, st = ss.time.values.astype(float), ss.state.values
        if len(t) > 1:
            # Width of each segment = gap to the NEXT sample, so the band tiles
            # continuously whatever the sampling cadence. Using one median width
            # for every rectangle leaves gaps when a trace is sampled per second
            # but another per decision -- which looked like a signal difference
            # and was purely a rendering artifact.
            gaps = np.diff(t)
            nz = gaps[gaps > 0]
            last = float(np.median(nz)) if len(nz) else 1.0
            widths = np.append(gaps, last)
            for i in range(len(t)):
                w = widths[i] if widths[i] > 0 else last
                ax.add_patch(Rectangle((t[i], band_y), w, band_h,
                                       facecolor=signal_color(st[i]), edgecolor="none", zorder=1))

    for _, g in real.groupby("vehicle_id"):
        if len(g) < 2:
            continue
        g = g.sort_values("time")
        ax.plot(g.time.values, g.position.values, color=C_REAL,
                linewidth=0.6 if compact else 0.75, alpha=0.55,
                solid_capstyle="round", zorder=2)

    if len(fake):
        ax.scatter(fake.time.values, fake.position.values, s=6 if compact else 14,
                   color=C_FAKE, alpha=0.85, linewidths=0.3,
                   edgecolors=SURFACE, zorder=3)

    ax.set_title(title, fontsize=9 if compact else 11, color=INK, loc="left", pad=6)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.grid(True, color=GRID, linewidth=0.5, zorder=0); ax.set_axisbelow(True)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    for s_ in ("left", "bottom"):
        ax.spines[s_].set_color(GRID)
    ax.tick_params(colors=INK_SECOND, labelsize=7 if compact else 8.5, length=3)
    return real.vehicle_id.nunique(), len(fake)


LEGEND = [
    Line2D([0], [0], color=C_REAL, lw=1.6, label="real vehicle trajectory"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor=C_FAKE,
           markeredgecolor=SURFACE, markersize=6, label="injected fake vehicle"),
    Line2D([0], [0], color=C_GREEN, lw=5, label="green"),
    Line2D([0], [0], color=C_YELLOW, lw=5, label="yellow"),
    Line2D([0], [0], color=C_RED, lw=5, label="red"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-prefix", default="fgsm_attack")
    ap.add_argument("--a-label", default="White-box FGSM")
    ap.add_argument("--b-prefix", default="rl_attack")
    ap.add_argument("--b-label", default="RL (multi-PPO)")
    ap.add_argument("--outdir", default="space_time_all")
    ap.add_argument("--time-window", default=None, help="start,end seconds")
    ap.add_argument("--match-times", action="store_true",
                    help="restrict both traces to shared timestamps")
    ap.add_argument("--include-internal", action="store_true",
                    help="also plot ':junction' internal lanes")
    ap.add_argument("--controlled-only", action="store_true",
                    help="only lanes the traffic light controls (drops outgoing roads)")
    ap.add_argument("--dpi", type=int, default=170)
    a = ap.parse_args()

    A_pos, A_sig = load_pair(a.a_prefix)
    B_pos, B_sig = load_pair(a.b_prefix)
    missing = [p for p, v in ((a.a_prefix, A_pos), (a.b_prefix, B_pos)) if v is None]
    if missing:
        print(f"ERROR: missing {', '.join(f'{m}_positions.csv' for m in missing)}",
              file=sys.stderr)
        return 1

    if a.match_times:
        common = np.intersect1d(A_pos.time.unique(), B_pos.time.unique())
        A_pos = A_pos[A_pos.time.isin(common)]
        B_pos = B_pos[B_pos.time.isin(common)]
        if A_sig is not None:
            A_sig = A_sig[A_sig.time.isin(common)]
        if B_sig is not None:
            B_sig = B_sig[B_sig.time.isin(common)]
        print(f"--match-times: {len(common)} shared timestamps "
              f"({common.min():.0f}-{common.max():.0f}s)")

    window = None
    if a.time_window:
        lo, hi = a.time_window.split(",")
        window = (float(lo), float(hi))

    lanes = sorted(set(A_pos.lane.unique()) | set(B_pos.lane.unique()))
    if not a.include_internal:
        lanes = [L for L in lanes if not str(L).startswith(":")]
    if a.controlled_only:
        ctrl = set()
        for sig in (A_sig, B_sig):
            if sig is not None:
                for v in sig["controlled_lanes"].astype(str):
                    ctrl.update(v.split(";"))
        keep = [L for L in lanes if L in ctrl]
        if keep:
            print(f"--controlled-only: {len(keep)} of {len(lanes)} lanes are signal-controlled")
            lanes = keep
    if not lanes:
        print("no lanes found", file=sys.stderr)
        return 1

    os.makedirs(a.outdir, exist_ok=True)
    both = pd.concat([A_pos, B_pos])
    if window:
        both = both[(both.time >= window[0]) & (both.time <= window[1])]
    xlim = (both.time.min(), both.time.max())

    print(f"{len(lanes)} lanes: {', '.join(lanes)}\n")
    rows = []

    # ---- one figure per lane ------------------------------------------------
    for L in lanes:
        sub = both[both.lane == L]
        top = float(sub.position.max()) if len(sub) else 100.0
        ylim = (0, top * 1.10)
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.3))
        fig.patch.set_facecolor(SURFACE)
        for ax in axes:
            ax.set_facecolor(SURFACE)
        na = draw(axes[0], A_pos, A_sig, L, a.a_label, window, ylim, xlim)
        nb = draw(axes[1], B_pos, B_sig, L, a.b_label, window, ylim, xlim)
        axes[0].set_ylabel("position along lane (m)  →  stop bar", fontsize=9, color=INK_SECOND)
        for ax in axes:
            ax.set_xlabel("simulation time (s)", fontsize=9, color=INK_SECOND)
        fig.legend(handles=LEGEND, loc="lower center", ncol=5, frameon=False,
                   fontsize=8.5, labelcolor=INK_SECOND, bbox_to_anchor=(0.5, -0.02))
        fig.suptitle(f"Space-time  ·  lane {L}", fontsize=12.5, color=INK, x=0.006,
                     ha="left", y=0.99)
        fig.tight_layout(rect=[0, 0.06, 1, 0.94])
        out = os.path.join(a.outdir, f"space_time__{L}.png")
        fig.savefig(out, dpi=a.dpi, facecolor=SURFACE, bbox_inches="tight")
        plt.close(fig)
        rows.append({"lane": L, f"{a.a_label}_real": na[0], f"{a.a_label}_fake": na[1],
                     f"{a.b_label}_real": nb[0], f"{a.b_label}_fake": nb[1]})
        print(f"  {os.path.basename(out):<38} "
              f"{a.a_label}: {na[0]:>3} real / {na[1]:>4} fake   "
              f"{a.b_label}: {nb[0]:>3} real / {nb[1]:>4} fake")

    # ---- contact sheet: all lanes, both attacks -----------------------------
    n = len(lanes)
    fig, axes = plt.subplots(n, 2, figsize=(12, 2.15 * n), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for i, L in enumerate(lanes):
        sub = both[both.lane == L]
        top = float(sub.position.max()) if len(sub) else 100.0
        ylim = (0, top * 1.10)
        for j, (pos, sig, lab) in enumerate(((A_pos, A_sig, a.a_label),
                                             (B_pos, B_sig, a.b_label))):
            ax = axes[i][j]
            ax.set_facecolor(SURFACE)
            draw(ax, pos, sig, L, f"{L}   ·   {lab}" if i == 0 else L,
                 window, ylim, xlim, compact=True)
            if i < n - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("simulation time (s)", fontsize=8, color=INK_SECOND)
            if j == 1:
                ax.set_yticklabels([])
    fig.legend(handles=LEGEND, loc="lower center", ncol=5, frameon=False,
               fontsize=8.5, labelcolor=INK_SECOND, bbox_to_anchor=(0.5, -0.004))
    fig.suptitle(f"Space-time, all lanes   ·   left: {a.a_label}   right: {a.b_label}",
                 fontsize=13, color=INK, x=0.006, ha="left", y=0.997)
    fig.tight_layout(rect=[0, 0.022, 1, 0.985])
    sheet = os.path.join(a.outdir, "space_time__ALL_LANES.png")
    fig.savefig(sheet, dpi=a.dpi, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(rows).to_csv(os.path.join(a.outdir, "per_lane_counts.csv"), index=False)
    print(f"\ncontact sheet -> {sheet}")
    print(f"counts        -> {os.path.join(a.outdir, 'per_lane_counts.csv')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())