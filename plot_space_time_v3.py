"""
plot_space_time_v2.py

The REAL space-time diagram: x = time, y = continuous position along one
lane, each vehicle drawn as a connected line through its own position
samples. A moving vehicle draws a near-diagonal line (slope = speed); a
stopped/queued vehicle draws a flat line; a shockwave/jam front shows up as
a visible bend where many trajectories flatten out together.

DIRECTION OF TRAVEL: SUMO's getLanePosition() reports distance from the
START of the lane (position 0), increasing toward the lane's END. For an
incoming approach lane, position 0 is upstream (far from the intersection)
and the lane's end is the stop line -- so INCREASING position over time
means a vehicle moving TOWARD the intersection. This is the standard SUMO
convention; verify against your own data by checking that real vehicles'
positions trend upward over time (they should, approaching the signal).

STOP LINE marker: drawn as a dashed horizontal line at the maximum position
actually observed for any vehicle on this lane in the data -- an
approximation (the true lane length isn't available to this script,
which only reads the CSV), not a value queried live from SUMO.

SIGNAL STATE overlay (optional, --signal-csv): colored background bands
showing SUMO's own ground-truth red/yellow/green state for this specific
lane, using getControlledLanes()'s ordering to line lanes up with
characters in getRedYellowGreenState()'s output -- not re-derived from our
own phase_pairs bookkeeping.

Fake vehicles (frozen in place, briefly) will show up as short, flat, nearly
point-like marks -- they never move, matching how inject_fake_vehicles /
reset_fake_vehicles actually work (moveTo a fixed position, setSpeed(0), then
removed a few steps later).

Reads a continuous-positions CSV (time, vehicle_id, is_fake, lane, position)
-- only covers the first N steps of the run (see trainer.trajectory_sample_limit),
not the whole simulation.

SINGLE-SCENARIO usage:
    python3 plot_space_time_v2.py /path/to/positions.csv
    python3 plot_space_time_v2.py /path/to/positions.csv --lane road_0_1_0_0
    python3 plot_space_time_v2.py /path/to/positions.csv --signal-csv /path/to/signal_state.csv

ATTACK vs NO-ATTACK comparison: pass the attack run's CSV as the positional
argument, and the no-attack run's CSV via --compare-to. Add --signal-csv /
--compare-signal-csv for the overlay on each panel:
    python3 plot_space_time_v2.py attack/positions.csv --compare-to noattack/positions.csv \
        --signal-csv attack/signal_state.csv --compare-signal-csv noattack/signal_state.csv
"""

import csv
import argparse
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches


SIGNAL_COLORS = {
    'g': '#d5f5e3',  # pale green
    'y': '#fdebd0',  # pale yellow/orange
    'r': '#fadbd8',  # pale red
}


def load_rows(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "time": float(r["time"]),
                "vehicle_id": r["vehicle_id"],
                "is_fake": r["is_fake"].strip().lower() in ("true", "1", "yes"),
                "lane": r["lane"],
                "position": float(r["position"]),
            })
    return rows


def load_signal_rows(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "time": float(r["time"]),
                "tls_id": r["tls_id"],
                "controlled_lanes": r["controlled_lanes"].split(";") if r["controlled_lanes"] else [],
                "state": r["state"],
            })
    return rows


def lane_signal_timeline(signal_rows, lane):
    """(time, state_char) pairs for one specific lane, found via whichever
    traffic light's controlled_lanes list contains it."""
    tls_id, lane_idx = None, None
    for r in signal_rows:
        if lane in r["controlled_lanes"]:
            tls_id = r["tls_id"]
            lane_idx = r["controlled_lanes"].index(lane)
            break
    if tls_id is None:
        return []
    timeline = []
    for r in signal_rows:
        if r["tls_id"] != tls_id or lane_idx >= len(r["state"]):
            continue
        timeline.append((r["time"], r["state"][lane_idx].lower()))
    timeline.sort(key=lambda p: p[0])
    return timeline


def draw_signal_bands(ax, timeline):
    """Shade the background in consecutive same-color intervals."""
    if not timeline:
        return False
    drawn = False
    start_t, cur_char = timeline[0]
    for i in range(1, len(timeline)):
        t, char = timeline[i]
        if char != cur_char:
            color = SIGNAL_COLORS.get(cur_char)
            if color:
                ax.axvspan(start_t, t, color=color, alpha=0.6, zorder=0)
                drawn = True
            start_t, cur_char = t, char
    color = SIGNAL_COLORS.get(cur_char)
    if color:
        ax.axvspan(start_t, timeline[-1][0], color=color, alpha=0.6, zorder=0)
        drawn = True
    return drawn


def pick_lane(rows, requested_lane):
    if requested_lane:
        return requested_lane
    fake_counts_by_lane = defaultdict(int)
    for r in rows:
        if r["is_fake"]:
            fake_counts_by_lane[r["lane"]] += 1
    if fake_counts_by_lane:
        return max(fake_counts_by_lane, key=fake_counts_by_lane.get)
    counts_by_lane = defaultdict(int)
    for r in rows:
        counts_by_lane[r["lane"]] += 1
    return max(counts_by_lane, key=counts_by_lane.get)


def plot_scenario(ax, rows, lane, title, signal_rows=None):
    lane_rows = [r for r in rows if r["lane"] == lane]
    if not lane_rows:
        available = sorted(set(r["lane"] for r in rows))
        ax.text(0.5, 0.5, f"No rows for lane '{lane}'.\nAvailable: {available[:10]}",
                ha="center", va="center", transform=ax.transAxes)
        return False, False, False

    signal_drawn = False
    if signal_rows:
        timeline = lane_signal_timeline(signal_rows, lane)
        signal_drawn = draw_signal_bands(ax, timeline)

    by_vehicle = defaultdict(list)
    is_fake_by_vehicle = {}
    for r in lane_rows:
        by_vehicle[r["vehicle_id"]].append((r["time"], r["position"]))
        is_fake_by_vehicle[r["vehicle_id"]] = is_fake_by_vehicle.get(r["vehicle_id"], False) or r["is_fake"]
    for v in by_vehicle:
        by_vehicle[v].sort(key=lambda p: p[0])

    real_plotted, fake_plotted = False, False
    max_position = 0.0
    for vehicle_id, points in by_vehicle.items():
        times = [p[0] for p in points]
        positions = [p[1] for p in points]
        max_position = max(max_position, max(positions))
        if is_fake_by_vehicle[vehicle_id]:
            ax.plot(times, positions, color="#c0392b", linewidth=2.5, marker="o", markersize=4, alpha=0.9, zorder=3)
            fake_plotted = True
        else:
            ax.plot(times, positions, color="#2c3e50", linewidth=0.9, alpha=0.7, zorder=2)
            real_plotted = True

    # approximate stop-line marker: the furthest any vehicle got recorded at,
    # not a value queried live from SUMO (this script only reads the CSV)
    ax.axhline(y=max_position, color="black", linestyle="--", linewidth=1, alpha=0.6, zorder=1)
    ax.text(ax.get_xlim()[1] if ax.get_xlim()[1] > 0 else 1, max_position, "  approx. stop line",
            fontsize=8, color="black", va="bottom", alpha=0.7)

    ax.set_ylabel(f"position on '{lane}' (m)\n(increasing = toward intersection)")
    ax.set_title(title)
    return real_plotted, fake_plotted, signal_drawn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="attack (or single-scenario) run's continuous-positions CSV")
    parser.add_argument("--compare-to", default=None,
                         help="no-attack run's continuous-positions CSV, for a two-panel side-by-side comparison")
    parser.add_argument("--lane", default=None,
                         help="which lane to plot (default: the lane with the most fake-vehicle samples, from the attack CSV)")
    parser.add_argument("--signal-csv", default=None, help="signal-state CSV for the attack (or single) scenario")
    parser.add_argument("--compare-signal-csv", default=None, help="signal-state CSV for the no-attack scenario")
    parser.add_argument("--time-window", default=None,
                         help="start,end -- zoom into this time range (e.g. to check for a short yellow phase "
                              "that's too thin to see at full scale)")
    args = parser.parse_args()

    rows_attack = load_rows(args.csv_path)
    if not rows_attack:
        print(f"No rows found in {args.csv_path}.")
        return

    if args.time_window:
        start, end = (float(x) for x in args.time_window.split(","))
        rows_attack = [r for r in rows_attack if start <= r["time"] <= end]

    target_lane = pick_lane(rows_attack, args.lane)
    print(f"Plotting lane: {target_lane}")

    signal_attack = load_signal_rows(args.signal_csv) if args.signal_csv else None
    signal_noattack = load_signal_rows(args.compare_signal_csv) if args.compare_signal_csv else None

    if args.time_window:
        start, end = (float(x) for x in args.time_window.split(","))
        if signal_attack:
            signal_attack = [r for r in signal_attack if start <= r["time"] <= end]
        if signal_noattack:
            signal_noattack = [r for r in signal_noattack if start <= r["time"] <= end]

    if args.compare_to:
        rows_noattack = load_rows(args.compare_to)
        if args.time_window:
            start, end = (float(x) for x in args.time_window.split(","))
            rows_noattack = [r for r in rows_noattack if start <= r["time"] <= end]
        if not rows_noattack:
            print(f"No rows found in {args.compare_to}.")
            return

        fig, axes = plt.subplots(2, 1, figsize=(12, 11), sharex=True, sharey=True)
        r1, f1, s1 = plot_scenario(axes[0], rows_noattack, target_lane, "NO ATTACK (baseline)", signal_noattack)
        r2, f2, s2 = plot_scenario(axes[1], rows_attack, target_lane, "UNDER ATTACK", signal_attack)
        axes[1].set_xlabel("simulation time (s)")

        handles = []
        if r1 or r2:
            handles.append(mlines.Line2D([], [], color="#2c3e50", label="real vehicle"))
        if f1 or f2:
            handles.append(mlines.Line2D([], [], color="#c0392b", marker="o", label="fake vehicle (attacker-injected)"))
        if s1 or s2:
            handles.append(mpatches.Patch(color=SIGNAL_COLORS['g'], label="green"))
            handles.append(mpatches.Patch(color=SIGNAL_COLORS['y'], label="yellow"))
            handles.append(mpatches.Patch(color=SIGNAL_COLORS['r'], label="red"))
        if handles:
            axes[0].legend(handles=handles, loc="upper left", fontsize=8)

        fig.suptitle(f"Space-time diagram: attack vs. no-attack, lane '{target_lane}'\n"
                     "(diagonal = moving, flat = stopped/queued -- compare where/how much flattening happens)")
        fig.tight_layout()
        out_path = "space_time_comparison.png"
    else:
        fig, ax = plt.subplots(figsize=(12, 7))
        r1, f1, s1 = plot_scenario(ax, rows_attack, target_lane,
                               "Space-time diagram: vehicle trajectories\n"
                               "(diagonal = moving, flat = stopped/queued, red = fake/injected)",
                               signal_attack)
        ax.set_xlabel("simulation time (s)")
        handles = []
        if r1:
            handles.append(mlines.Line2D([], [], color="#2c3e50", label="real vehicle"))
        if f1:
            handles.append(mlines.Line2D([], [], color="#c0392b", marker="o", label="fake vehicle (attacker-injected)"))
        if s1:
            handles.append(mpatches.Patch(color=SIGNAL_COLORS['g'], label="green"))
            handles.append(mpatches.Patch(color=SIGNAL_COLORS['y'], label="yellow"))
            handles.append(mpatches.Patch(color=SIGNAL_COLORS['r'], label="red"))
        if handles:
            ax.legend(handles=handles, loc="upper left", fontsize=8)
        fig.tight_layout()
        out_path = "space_time_v2.png"

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
