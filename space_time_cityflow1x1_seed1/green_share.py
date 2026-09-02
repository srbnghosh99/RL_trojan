#!/usr/bin/env python3
"""
Per-lane green-time share under different attacks.

The fake-vehicle attack works by making the victim over-extend the phase serving
the approach it injects into, starving the others. This quantifies that: for each
lane, the fraction of sampled time its signal shows green, compared across traces.

Usage
-----
  python3 green_share.py \
      clean=fgsm_attack_signal_state.csv \
      FGSM=fgsm_attack_signal_state.csv \
      RL=rl_attack_signal_state.csv \
      --positions RL=rl_attack_positions.csv,FGSM=fgsm_attack_positions.csv \
      --out green_share.png

Any number of label=path pairs. --positions is optional; when given, lanes that
trace actually injected into are marked, and an injected-vs-other summary printed.
"""
import argparse
import os
import sys

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK_SECOND, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]   # validated all-pairs for <=3 series


def green_share(path):
    sig = pd.read_csv(path)
    rows = []
    for _, r in sig.iterrows():
        lanes = str(r["controlled_lanes"]).split(";")
        st = str(r["state"])
        seen = set()
        for i, L in enumerate(lanes):
            if i < len(st) and L not in seen:
                seen.add(L)
                rows.append((L, st[i].lower() == "g"))
    if not rows:
        return pd.Series(dtype=float)
    d = pd.DataFrame(rows, columns=["lane", "g"])
    return d.groupby("lane").g.mean() * 100.0


def injected_lanes(path):
    d = pd.read_csv(path)
    if "is_fake" not in d.columns:
        return set()
    d["is_fake"] = d.is_fake.astype(str).str.lower().isin(["true", "1", "yes"])
    return set(d[d.is_fake].lane.unique())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+", help="label=path_to_signal_state.csv")
    ap.add_argument("--positions", default=None, help="label=positions.csv,label=...")
    ap.add_argument("--out", default="green_share.png")
    ap.add_argument("--dpi", type=int, default=200)
    a = ap.parse_args()

    series = {}
    for spec in a.traces:
        if "=" not in spec:
            print(f"skip {spec!r}: expected label=path", file=sys.stderr)
            continue
        label, path = spec.split("=", 1)
        if not os.path.exists(path):
            print(f"skip {label}: {path} not found", file=sys.stderr)
            continue
        s = green_share(path)
        if len(s):
            series[label] = s

    if not series:
        print("no usable traces", file=sys.stderr)
        return 1

    inj = {}
    if a.positions:
        for spec in a.positions.split(","):
            if "=" in spec:
                label, path = spec.split("=", 1)
                if os.path.exists(path):
                    inj[label] = injected_lanes(path)

    df = pd.DataFrame(series)
    df = df.sort_values(df.columns[-1], ascending=False)

    print("\n=== green share (% of sampled time) ===")
    show = df.round(1).copy()
    for label, lanes in inj.items():
        show[f"{label}_injected"] = ["yes" if L in lanes else "" for L in show.index]
    print(show.to_string())

    for label, lanes in inj.items():
        if label in df.columns and lanes:
            a_ = df.loc[df.index.isin(lanes), label].mean()
            b_ = df.loc[~df.index.isin(lanes), label].mean()
            print(f"\n{label}: injected lanes {a_:.1f}% green vs others {b_:.1f}%  "
                  f"({a_ / max(b_, 1e-9):.2f}x)")

    n = len(df.columns)
    x = range(len(df))
    w = 0.8 / n
    fig, ax = plt.subplots(figsize=(max(7, 0.85 * len(df) * n), 4.2))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for k, col in enumerate(df.columns):
        ax.bar([i + k * w - 0.4 + w / 2 for i in x], df[col].values, width=w * 0.92,
               color=SERIES[k % len(SERIES)], label=col, zorder=3)
    ax.set_xticks(list(x))
    ax.set_xticklabels(df.index, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("green share (% of time)", fontsize=9, color=INK_SECOND)
    ax.set_title("Per-lane green time under fake-vehicle injection",
                 fontsize=12, color=INK, loc="left", pad=10)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_SECOND, labelsize=8.5, length=3)
    if n >= 2:
        ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_SECOND, ncol=n)
    fig.tight_layout()
    fig.savefig(a.out, dpi=a.dpi, facecolor=SURFACE, bbox_inches="tight")
    df.round(2).to_csv(os.path.splitext(a.out)[0] + ".csv")
    print(f"\nwrote {a.out} and {os.path.splitext(a.out)[0]}.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
