#!/usr/bin/env python3
"""
lane_metrics.py
================================================================================
Per-LANE and per-SIGNAL throughput / queue, which the DTL logs do not contain.

Two parts:

  A) LaneRecorder  -- drop into your test step loop. Each step it records,
     per lane: vehicles present, vehicles waiting, and vehicles that LEFT the
     lane this step (= throughput). Writes one tidy CSV.

  B) plotting      -- run this file on that CSV to produce:
        lane_throughput_heatmap.png   lanes x time, colour = departures
        lane_throughput_bars.png      total throughput per lane
        signal_throughput.png         throughput per intersection over time
        lane_queue_heatmap.png        lanes x time, colour = queue

HOW TO RECORD (add ~6 lines to the trainer's test loop)
-------------------------------------------------------
    from lane_metrics import LaneRecorder
    rec = LaneRecorder(self.world, out="lane_metrics.csv", every=1)

    for i in range(self.test_steps):
        ...
        self.world.step(actions)
        rec.step()                      # <-- after each world.step
    rec.close()

THEN PLOT
---------
    python3 lane_metrics.py --csv lane_metrics.csv --outdir lane_out
    # mark when the attack starts:
    python3 lane_metrics.py --csv lane_metrics.csv --onset 300 --outdir lane_out
    # compare clean vs attack:
    python3 lane_metrics.py --csv attack.csv --baseline clean.csv --outdir lane_out

NOTE ON "THROUGHPUT": departures are counted as vehicle IDs present on a lane
at step t and absent at t+1. A lane change also looks like a departure, so on
multi-lane approaches treat per-lane numbers as approximate and prefer the
per-signal (aggregated) view.
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
#  A) RECORDER  (used inside your repo, not by the plotting CLI)
# --------------------------------------------------------------------------- #
class LaneRecorder:
    """
    Records per-lane presence / waiting / departures each simulation step.

    Works with the SUMO (libsumo/traci) world used in this repo. Falls back to
    world.get_info(...) when the direct lane API is unavailable.
    """

    def __init__(self, world, out="lane_metrics.csv", every=1, lanes=None):
        self.world = world
        self.out = out
        self.every = int(every)
        self.t = 0
        self.rows = []
        self._prev_ids = {}

        self.lanes = list(lanes) if lanes else self._discover_lanes()
        self.lane2inter = self._map_lanes_to_intersections()
        print(f"[LaneRecorder] tracking {len(self.lanes)} lanes -> {out}")

    # ---- discovery ----
    def _discover_lanes(self):
        for attr in ("all_lanes", "lanes"):
            v = getattr(self.world, attr, None)
            if v:
                return list(v)
        try:
            return list(self.world.eng.lane.getIDList())
        except Exception:
            return []

    def _map_lanes_to_intersections(self):
        """lane id -> intersection id, using each intersection's in_roads."""
        m = {}
        try:
            for inter in self.world.intersections:
                iid = getattr(inter, "id", None)
                roads = getattr(inter, "in_roads", []) or []
                for road in roads:
                    rid = road["id"] if isinstance(road, dict) else road
                    for ln in self.lanes:
                        if str(ln).startswith(str(rid)):
                            m[ln] = iid
        except Exception:
            pass
        return m

    # ---- per-step readings ----
    def _lane_vehicle_ids(self, lane):
        try:
            return set(self.world.eng.lane.getLastStepVehicleIDs(lane))
        except Exception:
            return None

    def _fallback_counts(self):
        try:
            return (self.world.get_info("lane_count") or {},
                    self.world.get_info("lane_waiting_count") or {})
        except Exception:
            return {}, {}

    def step(self):
        """Call once after every world.step()."""
        self.t += 1
        if self.t % self.every:
            return

        ids_now = {}
        use_ids = True
        for ln in self.lanes:
            s = self._lane_vehicle_ids(ln)
            if s is None:
                use_ids = False
                break
            ids_now[ln] = s

        if use_ids:
            wait = {}
            try:
                wait = self.world.get_info("lane_waiting_count") or {}
            except Exception:
                pass
            for ln in self.lanes:
                prev = self._prev_ids.get(ln, set())
                departures = len(prev - ids_now[ln])
                self.rows.append(dict(
                    t=self.t, lane=ln,
                    intersection=self.lane2inter.get(ln, "NA"),
                    present=len(ids_now[ln]),
                    waiting=float(wait.get(ln, np.nan)),
                    departures=departures))
            self._prev_ids = ids_now
        else:
            counts, wait = self._fallback_counts()
            for ln in self.lanes:
                self.rows.append(dict(
                    t=self.t, lane=ln,
                    intersection=self.lane2inter.get(ln, "NA"),
                    present=float(counts.get(ln, np.nan)),
                    waiting=float(wait.get(ln, np.nan)),
                    departures=np.nan))

    def close(self):
        if not self.rows:
            print("[LaneRecorder] nothing recorded")
            return None
        df = pd.DataFrame(self.rows)
        df.to_csv(self.out, index=False)
        print(f"[LaneRecorder] {len(df)} rows -> {self.out}")
        return df


# --------------------------------------------------------------------------- #
#  B) PLOTTING
# --------------------------------------------------------------------------- #
def _load(csv):
    df = pd.read_csv(csv)
    need = {"t", "lane", "present"}
    if not need.issubset(df.columns):
        raise SystemExit(f"CSV needs columns {need}; has {list(df.columns)}")
    if "departures" not in df:
        df["departures"] = np.nan
    if "intersection" not in df:
        df["intersection"] = "NA"
    if "waiting" not in df:
        df["waiting"] = np.nan
    return df


def _heatmap(df, value, outpath, title, onset=None, bins=120, cmap="magma"):
    p = df.pivot_table(index="lane", columns="t", values=value, aggfunc="mean")
    if p.empty or p.isna().all().all():
        print(f"[skip] {value}: no data")
        return
    # bin time so wide runs stay readable
    if p.shape[1] > bins:
        edges = np.linspace(p.columns.min(), p.columns.max(), bins + 1)
        grp = pd.cut(p.columns, edges, labels=False, include_lowest=True)
        p = p.T.groupby(grp).mean().T
        tmin, tmax = edges[0], edges[-1]
    else:
        tmin, tmax = p.columns.min(), p.columns.max()

    h = max(3.0, 0.22 * len(p.index))
    fig, ax = plt.subplots(figsize=(11, h))
    im = ax.imshow(p.values, aspect="auto", cmap=cmap,
                   extent=[tmin, tmax, len(p.index) - 0.5, -0.5],
                   interpolation="nearest")
    ax.set_yticks(range(len(p.index)))
    ax.set_yticklabels(p.index, fontsize=7)
    ax.set_xlabel("simulation step")
    ax.set_ylabel("lane")
    ax.set_title(title)
    if onset is not None:
        ax.axvline(onset, color="#00e5ff", ls="--", lw=1.4)
        ax.text(onset, -1.1, "attack starts", color="#00a0b0", fontsize=8)
    fig.colorbar(im, ax=ax, label=value)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"[plot] {outpath}")


def _lane_bars(df, outpath, baseline=None, top=25):
    tot = df.groupby("lane").departures.sum().sort_values(ascending=False)
    if tot.isna().all():
        print("[skip] lane throughput bars: no departure data")
        return
    tot = tot.head(top)[::-1]
    if baseline is not None:
        b = baseline.groupby("lane").departures.sum()
        b = b.reindex(tot.index)
        y = np.arange(len(tot))
        fig, ax = plt.subplots(figsize=(8, max(3.5, 0.3 * len(tot))))
        ax.barh(y - 0.2, b.values, 0.4, color="#1D9E75", label="clean")
        ax.barh(y + 0.2, tot.values, 0.4, color="#D85A30", label="under attack")
        ax.set_yticks(y)
        ax.set_yticklabels(tot.index, fontsize=7)
        ax.legend(fontsize=8)
    else:
        fig, ax = plt.subplots(figsize=(8, max(3.5, 0.3 * len(tot))))
        ax.barh(tot.index, tot.values, color="#534AB7")
        ax.tick_params(axis="y", labelsize=7)
    ax.set_xlabel("vehicles that left the lane (throughput)")
    ax.set_title("Throughput per lane")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"[plot] {outpath}")


def _signal_series(df, outpath, onset=None, window=30, baseline=None):
    if df.departures.isna().all():
        print("[skip] per-signal throughput: no departure data")
        return
    g = df.groupby(["intersection", "t"]).departures.sum().reset_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    palette = ["#1D9E75", "#D85A30", "#534AB7", "#BA7517", "#185FA5", "#993556"]
    for k, (iid, s) in enumerate(g.groupby("intersection")):
        s = s.sort_values("t")
        y = s.departures.rolling(window, min_periods=1).mean()
        ax.plot(s.t, y, color=palette[k % len(palette)], lw=1.6, label=str(iid))
    if baseline is not None and not baseline.departures.isna().all():
        b = baseline.groupby("t").departures.sum().sort_index()
        ax.plot(b.index, b.rolling(window, min_periods=1).mean(),
                color="#888", ls=":", lw=1.4, label="clean (all signals)")
    if onset is not None:
        ax.axvline(onset, color="#444", ls="--", lw=1.2)
        ax.text(onset, ax.get_ylim()[1] * 0.95, "attack starts", fontsize=8)
    ax.set_xlabel("simulation step")
    ax.set_ylabel(f"throughput (rolling mean, {window} steps)")
    ax.set_title("Throughput per signal over time")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"[plot] {outpath}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV written by LaneRecorder")
    ap.add_argument("--baseline", default=None, help="clean-run CSV to compare against")
    ap.add_argument("--onset", type=int, default=None, help="step the attack starts")
    ap.add_argument("--outdir", default="lane_out")
    ap.add_argument("--top", type=int, default=25, help="lanes shown in the bar chart")
    ap.add_argument("--window", type=int, default=30, help="rolling window for signal plot")
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    df = _load(a.csv)
    base = _load(a.baseline) if a.baseline else None
    print(f"[info] {df.lane.nunique()} lanes, {df.t.nunique()} steps, "
          f"{df.intersection.nunique()} signal(s)")

    _heatmap(df, "departures", os.path.join(a.outdir, "lane_throughput_heatmap.png"),
             "Per-lane throughput over time (bright = vehicles getting through)",
             onset=a.onset)
    _heatmap(df, "waiting", os.path.join(a.outdir, "lane_queue_heatmap.png"),
             "Per-lane queue over time (bright = piling up)",
             onset=a.onset, cmap="inferno")
    _lane_bars(df, os.path.join(a.outdir, "lane_throughput_bars.png"),
               baseline=base, top=a.top)
    _signal_series(df, os.path.join(a.outdir, "signal_throughput.png"),
                   onset=a.onset, window=a.window, baseline=base)

    tot = df.groupby("lane").departures.sum().sort_values()
    if not tot.isna().all():
        print("\n=== lowest-throughput lanes (starved) ===")
        print(tot.head(5).to_string())
        print("\n=== highest-throughput lanes (served) ===")
        print(tot.tail(5).to_string())
    print(f"\n[done] {a.outdir}/")


if __name__ == "__main__":
    main()
