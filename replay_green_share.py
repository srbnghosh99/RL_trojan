#!/usr/bin/env python3
"""
Per-road green-time share straight out of CityFlow replay files.

The fake-vehicle attack works by making the victim over-extend the phase serving
the approach it injects into, starving the rest. This measures that directly from
the replay's own TLS section -- no simulator, no extra instrumentation.

Compare a clean run against attacked runs to make the effect attributable:

  python3 replay_green_share.py clean=replay_clean.txt RL=replay_rl.txt \\
      --injected road_1_2_3 --out green_share_replay.png

Replay line format is "<vehicles>;<tls>" where the tls section is
"road_id c c,road_id c c,..." with one char per lane.
"""
import argparse
import collections
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SURFACE, INK, INK_SECOND, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]   # validated all-pairs for <=3 series


def parse_replay(path):
    """-> {(road, lane_idx): [state_char, ...]}"""
    per = collections.defaultdict(list)
    with open(path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split(";")
            if len(parts) < 2:
                continue
            for ent in parts[1].split(","):
                ent = ent.strip()
                if not ent:
                    continue
                tok = ent.split()
                for i, ch in enumerate(tok[1:]):
                    per[(tok[0], i)].append(ch)
    return per


def summarise(per):
    rows = []
    for (road, i), chars in per.items():
        if not chars:
            continue
        green = sum(1 for c in chars if c == "g") / len(chars) * 100
        best = cur = 0
        for c in chars:
            if c != "g":
                cur += 1
                best = max(best, cur)
            else:
                cur = 0
        rows.append({"road": road, "lane": i, "green_pct": round(green, 1),
                     "longest_red_s": best, "steps": len(chars)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("replays", nargs="+", help="label=replay.txt")
    ap.add_argument("--injected", default=None,
                    help="comma-separated road ids the attacker injected into")
    ap.add_argument("--out", default="green_share_replay.png")
    ap.add_argument("--dpi", type=int, default=200)
    a = ap.parse_args()

    inj = set(a.injected.split(",")) if a.injected else set()
    frames = {}
    for spec in a.replays:
        label, path = spec.split("=", 1) if "=" in spec else ("replay", spec)
        if not os.path.exists(path):
            print(f"skip {label}: {path} not found", file=sys.stderr)
            continue
        df = summarise(parse_replay(path))
        if df.empty:
            print(f"skip {label}: no TLS section found", file=sys.stderr)
            continue
        df["key"] = df.road + "_" + df.lane.astype(str)
        frames[label] = df
        print(f"\n=== {label} ===")
        out = df.sort_values("green_pct", ascending=False)
        if inj:
            out = out.assign(injected=["yes" if r in inj else "" for r in out.road])
        print(out.to_string(index=False))
        if inj:
            i_ = df[df.road.isin(inj)].green_pct.mean()
            o_ = df[~df.road.isin(inj)].green_pct.mean()
            print(f"\n  injected roads {i_:.1f}% green vs others {o_:.1f}%  "
                  f"({i_ / max(o_, 1e-9):.2f}x)")
            print(f"  worst starvation: {df.longest_red_s.max()} s continuous red "
                  f"on {df.loc[df.longest_red_s.idxmax(), 'key']}")

    if not frames:
        return 1

    merged = None
    for label, df in frames.items():
        s = df.set_index("key").green_pct.rename(label)
        merged = s.to_frame() if merged is None else merged.join(s, how="outer")
    merged = merged.sort_values(merged.columns[-1], ascending=False)

    n = len(merged.columns)
    x = range(len(merged))
    w = 0.8 / n
    fig, ax = plt.subplots(figsize=(max(7, 0.9 * len(merged) * n), 4.2))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for k, col in enumerate(merged.columns):
        ax.bar([i + k * w - 0.4 + w / 2 for i in x], merged[col].values,
               width=w * 0.92, color=SERIES[k % len(SERIES)], label=col, zorder=3)
    ax.set_xticks(list(x))
    labels = [f"{t}{' *' if t.rsplit('_', 1)[0] in inj else ''}" for t in merged.index]
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("green share (% of simulation)", fontsize=9, color=INK_SECOND)
    ax.set_title("Per-lane green time" + ("   (* = injected approach)" if inj else ""),
                 fontsize=12, color=INK, loc="left", pad=10)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    for s_ in ("left", "bottom"):
        ax.spines[s_].set_color(GRID)
    ax.tick_params(colors=INK_SECOND, labelsize=8.5, length=3)
    if n >= 2:
        ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_SECOND, ncol=n)
    fig.tight_layout()
    fig.savefig(a.out, dpi=a.dpi, facecolor=SURFACE, bbox_inches="tight")
    merged.round(2).to_csv(os.path.splitext(a.out)[0] + ".csv")
    print(f"\nwrote {a.out} and {os.path.splitext(a.out)[0]}.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
