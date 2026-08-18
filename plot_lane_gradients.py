"""
plot_lane_gradients.py

Reads the fgsm_lane_gradients.csv produced by trainer/tsc_trainer_whitebox.py
(one row per lane per decision: sim_time, agent_idx, lane, gradient,
fake_vehicles_injected) and produces two views:

  1. Averaged over the whole run: mean gradient per lane vs. total fake
     vehicles that lane received. This directly answers "does a higher
     gradient lane actually get more fake vehicles" -- they should track
     each other closely, since that's exactly what
     gradient_to_fake_vehicle_plan() is designed to do.

  2. A single-decision snapshot (first decision by default, or pick one with
     --sim-time): the raw per-lane gradient bars, colored by whether that
     lane was positive (a candidate) or negative (excluded), annotated with
     how many fake vehicles actually landed there.

Usage:
    python3 plot_lane_gradients.py /path/to/fgsm_lane_gradients.csv
    python3 plot_lane_gradients.py /path/to/fgsm_lane_gradients.csv --sim-time 120
"""

import sys
import csv
import argparse
from collections import defaultdict

import matplotlib.pyplot as plt


def load_rows(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "sim_time": int(r["sim_time"]),
                "agent_idx": int(r["agent_idx"]),
                "lane": r["lane"],
                "gradient": float(r["gradient"]),
                "fake_vehicles_injected": int(r["fake_vehicles_injected"]),
            })
    return rows


def plot_averaged(rows, ax_grad, ax_fake):
    grad_sum = defaultdict(float)
    fake_sum = defaultdict(int)
    count = defaultdict(int)
    for r in rows:
        grad_sum[r["lane"]] += r["gradient"]
        fake_sum[r["lane"]] += r["fake_vehicles_injected"]
        count[r["lane"]] += 1

    lanes = sorted(grad_sum.keys(), key=lambda l: grad_sum[l] / count[l], reverse=True)
    avg_grad = [grad_sum[l] / count[l] for l in lanes]
    total_fake = [fake_sum[l] for l in lanes]

    colors = ["#c0392b" if g > 0 else "#7f8c8d" for g in avg_grad]
    ax_grad.bar(lanes, avg_grad, color=colors)
    ax_grad.axhline(0, color="black", linewidth=0.8)
    ax_grad.set_ylabel("avg gradient")
    ax_grad.set_title("Average gradient per lane, across the whole run\n(red = positive/candidate, gray = negative/excluded)")
    ax_grad.tick_params(axis="x", rotation=45)

    ax_fake.bar(lanes, total_fake, color="#2980b9")
    ax_fake.set_ylabel("total fake vehicles\ninjected (whole run)")
    ax_fake.set_title("Total fake vehicles actually placed per lane")
    ax_fake.tick_params(axis="x", rotation=45)


def plot_snapshot(rows, sim_time, ax):
    snapshot = [r for r in rows if r["sim_time"] == sim_time]
    if not snapshot:
        available = sorted(set(r["sim_time"] for r in rows))
        ax.text(0.5, 0.5, f"No rows at sim_time={sim_time}.\nAvailable: {available[:10]}...",
                ha="center", va="center", transform=ax.transAxes)
        return

    snapshot.sort(key=lambda r: r["gradient"], reverse=True)
    lanes = [r["lane"] for r in snapshot]
    grads = [r["gradient"] for r in snapshot]
    fakes = [r["fake_vehicles_injected"] for r in snapshot]

    colors = ["#c0392b" if g > 0 else "#7f8c8d" for g in grads]
    bars = ax.bar(lanes, grads, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    for bar, fake in zip(bars, fakes):
        if fake > 0:
            ax.annotate(f"+{fake}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha="center", va="bottom" if bar.get_height() >= 0 else "top", fontsize=9, color="#c0392b")
    ax.set_ylabel("gradient")
    ax.set_title(f"Single decision snapshot at sim_time={sim_time}\n(red bars = attacked, label = fake vehicles injected there)")
    ax.tick_params(axis="x", rotation=45)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", nargs="?", default="fgsm_lane_gradients.csv")
    parser.add_argument("--sim-time", type=int, default=None,
                         help="which decision's sim_time to snapshot (default: first one in the file)")
    args = parser.parse_args()

    rows = load_rows(args.csv_path)
    if not rows:
        print(f"No rows found in {args.csv_path}")
        return

    sim_time = args.sim_time if args.sim_time is not None else rows[0]["sim_time"]

    fig, axes = plt.subplots(3, 1, figsize=(11, 12))
    plot_averaged(rows, axes[0], axes[1])
    plot_snapshot(rows, sim_time, axes[2])

    fig.tight_layout()
    out_path = "lane_gradients.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
