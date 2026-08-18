"""
plot_fgsm_timeseries.py

Reads the fgsm_timeseries.csv produced by trainer/tsc_trainer_whitebox.py's
test() run and plots three things, stacked on a shared time axis:

  1. Fake vehicles injected per decision (~every 10 sim-seconds, i.e.
     action_interval steps)
  2. Total REAL vehicles active in the network at that same moment
     (fake ones excluded -- this is real traffic only)
  3. Average waiting time of those real vehicles at that moment

Usage:
    python3 plot_fgsm_timeseries.py /path/to/fgsm_timeseries.csv
    python3 plot_fgsm_timeseries.py   # looks for ./fgsm_timeseries.csv
"""

import sys
import csv

import matplotlib.pyplot as plt


def load_rows(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "sim_time": int(r["sim_time"]),
                "fake_vehicles_injected": int(r["fake_vehicles_injected"]),
                "total_vehicles": None if r["total_vehicles"] in ("", "None") else int(r["total_vehicles"]),
                "avg_wait_time": None if r["avg_wait_time"] in ("", "None") else float(r["avg_wait_time"]),
            })
    return rows


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "fgsm_timeseries.csv"
    rows = load_rows(path)

    if not rows:
        print(f"No rows found in {path}")
        return

    if rows[0]["total_vehicles"] is None:
        print(
            "Warning: total_vehicles/avg_wait_time are empty (None) for every row. "
            "This means _snapshot_real_traffic() couldn't reach the SUMO vehicle "
            "API on that run (e.g. running under CityFlow instead of SUMO, or "
            "self.world.eng.vehicle wasn't available). Fake-vehicle-injection "
            "counts are still valid and will still plot."
        )

    t = [r["sim_time"] for r in rows]
    fake = [r["fake_vehicles_injected"] for r in rows]
    total = [r["total_vehicles"] for r in rows]
    wait = [r["avg_wait_time"] for r in rows]

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    axes[0].bar(t, fake, width=8, color="#c0392b")
    axes[0].set_ylabel("fake vehicles\ninjected")
    axes[0].set_title("Fake vehicles injected per decision (~every 10s)")

    axes[1].plot(t, total, color="#2980b9", marker="o", markersize=3)
    axes[1].set_ylabel("total real\nvehicles")
    axes[1].set_title("Total real vehicles in network")

    axes[2].plot(t, wait, color="#27ae60", marker="o", markersize=3)
    axes[2].set_ylabel("avg wait\ntime (s)")
    axes[2].set_title("Average wait time of real vehicles")
    axes[2].set_xlabel("simulation time (s)")

    for ax in axes:
        ax.grid(alpha=0.3)

    fig.suptitle("FGSM white-box attack: per-decision time series", y=1.02)
    fig.tight_layout()
    out_path = "fgsm_timeseries.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
