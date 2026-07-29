 #!/usr/bin/env python3
"""
stealth_analysis.py
================================================================================
How DIFFERENT is the flow of cars through the signal under attack vs normal --
and therefore, how detectable is the attack?

Framing: a monitor watching the intersection sees vehicles passing per timestep.
If the attacked distribution looks like the normal one, the attack is stealthy.
If it shifts, an alarm fires. This script quantifies that gap and converts it
into an actual detection rate.

INPUT: two CSVs from lane_metrics.py (columns: t, lane, intersection,
       present, waiting, departures) -- one clean run, one attacked run.

OUTPUT (in --outdir):
    throughput_distribution.png   histogram + CDF overlay, clean vs attack
    throughput_timeseries.png     per-step flow with detection band + alarms
    detection_roc.png             detection rate vs false-alarm rate
    stealth_metrics.csv           KS / Wasserstein / JS / detection numbers
    summary.md                    plain-language readout

USAGE
    python3 stealth_analysis.py --clean clean.csv --attack attack.csv \
        --onset 300 --outdir stealth_out

    # per-signal on a 4x4 grid:
    python3 stealth_analysis.py --clean clean.csv --attack attack.csv --per-signal

READING IT
    Small KS / small Wasserstein / detection rate near the false-alarm rate
        -> the attack hides inside normal variation (STEALTHY)
    Large KS / detection rate near 1.0
        -> a simple throughput monitor catches it (NOT stealthy)

Stealth is only interesting alongside damage: an attack that changes nothing is
trivially stealthy and also useless. Report stealth next to the travel-time
degradation from injection_sweep.
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLEAN_C = "#1D9E75"
ATTACK_C = "#D85A30"


# --------------------------------------------------------------------------- #
def load_series(csv, value="departures", signal=None, after=None):
    """CSV -> per-timestep total flow (one number per simulation step)."""
    df = pd.read_csv(csv)
    if value not in df.columns:
        raise SystemExit(f"{csv}: no '{value}' column (has {list(df.columns)})")
    if signal is not None and "intersection" in df.columns:
        df = df[df.intersection.astype(str) == str(signal)]
    if after is not None:
        df = df[df.t >= after]
    s = df.groupby("t")[value].sum().sort_index()
    return s.dropna()


def js_divergence(a, b, bins):
    """Jensen-Shannon divergence between two samples, on shared bins."""
    pa, _ = np.histogram(a, bins=bins, density=False)
    pb, _ = np.histogram(b, bins=bins, density=False)
    pa = pa.astype(float) + 1e-12
    pb = pb.astype(float) + 1e-12
    pa /= pa.sum()
    pb /= pb.sum()
    m = 0.5 * (pa + pb)
    kl = lambda p, q: np.sum(p * np.log2(p / q))
    return 0.5 * kl(pa, m) + 0.5 * kl(pb, m)


def detection_stats(clean, attack, k=3.0):
    """
    Simple monitor: alarm when a step's flow deviates from the clean baseline
    by more than k standard deviations.
      false_alarm_rate = fraction of CLEAN steps that alarm
      detection_rate   = fraction of ATTACK steps that alarm
    A stealthy attack has detection_rate close to false_alarm_rate.
    """
    mu, sd = float(np.mean(clean)), float(np.std(clean))
    if sd == 0:
        return dict(threshold_k=k, false_alarm_rate=np.nan,
                    detection_rate=np.nan, lo=mu, hi=mu)
    lo, hi = mu - k * sd, mu + k * sd
    fa = float(np.mean((clean < lo) | (clean > hi)))
    dr = float(np.mean((attack < lo) | (attack > hi)))
    return dict(threshold_k=k, false_alarm_rate=fa, detection_rate=dr, lo=lo, hi=hi)


def cusum_detection_delay(clean, attack, onset_index, k_sigma=0.5, h_sigma=5.0):
    """
    CUSUM on the attacked series using clean mean/std. Returns how many steps
    after onset the alarm first fires (None if it never does).
    """
    mu, sd = float(np.mean(clean)), float(np.std(clean))
    if sd == 0:
        return None
    k, h = k_sigma * sd, h_sigma * sd
    s_hi = s_lo = 0.0
    x = np.asarray(attack, dtype=float)
    for i, v in enumerate(x):
        s_hi = max(0.0, s_hi + (v - mu) - k)
        s_lo = max(0.0, s_lo + (mu - v) - k)
        if (s_hi > h or s_lo > h) and i >= onset_index:
            return i - onset_index
    return None


def roc_curve(clean, attack, n=60):
    """Sweep the k-sigma threshold; return (false_alarm, detection) arrays."""
    mu, sd = float(np.mean(clean)), float(np.std(clean))
    if sd == 0:
        return np.array([0, 1]), np.array([0, 1])
    ks = np.linspace(0.0, 6.0, n)
    fa, dr = [], []
    for k in ks:
        lo, hi = mu - k * sd, mu + k * sd
        fa.append(np.mean((clean < lo) | (clean > hi)))
        dr.append(np.mean((attack < lo) | (attack > hi)))
    return np.array(fa), np.array(dr)


# --------------------------------------------------------------------------- #
def analyze(clean, attack, label, args, outdir):
    """Compute all stealth metrics for one (clean, attack) pair."""
    lo = min(clean.min(), attack.min())
    hi = max(clean.max(), attack.max())
    bins = np.linspace(lo, hi, 25)

    ks_stat, ks_p = stats.ks_2samp(clean.values, attack.values)
    w = stats.wasserstein_distance(clean.values, attack.values)
    js = js_divergence(clean.values, attack.values, bins)
    det = detection_stats(clean.values, attack.values, k=args.k)
    fa, dr = roc_curve(clean.values, attack.values)
    auc = float(np.trapezoid(dr[np.argsort(fa)], np.sort(fa)))

    onset_idx = 0
    if args.onset is not None:
        idx = np.where(attack.index.values >= args.onset)[0]
        onset_idx = int(idx[0]) if len(idx) else 0
    delay = cusum_detection_delay(clean.values, attack.values, onset_idx)

    rec = dict(
        signal=label,
        clean_mean=float(clean.mean()), attack_mean=float(attack.mean()),
        clean_std=float(clean.std()), attack_std=float(attack.std()),
        mean_shift_pct=float((attack.mean() - clean.mean()) / clean.mean() * 100)
        if clean.mean() else np.nan,
        ks_stat=ks_stat, ks_pvalue=ks_p,
        wasserstein=w, js_divergence=js,
        false_alarm_rate=det["false_alarm_rate"],
        detection_rate=det["detection_rate"],
        detection_auc=auc,
        cusum_steps_to_detect=delay if delay is not None else np.nan,
    )
    return rec, (bins, det, fa, dr)


def plot_distribution(clean, attack, bins, outpath, label=""):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    ax = axes[0]
    ax.hist(clean, bins=bins, alpha=0.55, color=CLEAN_C, label="normal", density=True)
    ax.hist(attack, bins=bins, alpha=0.55, color=ATTACK_C, label="under attack", density=True)
    ax.set_xlabel("vehicles through the signal per step")
    ax.set_ylabel("density")
    ax.set_title("Flow distribution")
    ax.legend(fontsize=8)

    ax = axes[1]
    for s, c, lb in ((clean, CLEAN_C, "normal"), (attack, ATTACK_C, "under attack")):
        x = np.sort(np.asarray(s))
        ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=c, lw=1.8, label=lb)
    ax.set_xlabel("vehicles per step")
    ax.set_ylabel("cumulative probability")
    ax.set_title("CDF (gap between curves = KS statistic)")
    ax.legend(fontsize=8)

    fig.suptitle(f"Per-step flow: normal vs attacked {label}".strip())
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"[plot] {outpath}")


def plot_timeseries(clean, attack, det, outpath, onset=None, window=20, label=""):
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.axhspan(det["lo"], det["hi"], color=CLEAN_C, alpha=0.12,
               label="normal band (mean ± k·σ)")
    ax.plot(clean.index, clean.rolling(window, min_periods=1).mean(),
            color=CLEAN_C, lw=1.5, label="normal")
    ax.plot(attack.index, attack.rolling(window, min_periods=1).mean(),
            color=ATTACK_C, lw=1.6, label="under attack")
    out = attack[(attack < det["lo"]) | (attack > det["hi"])]
    # if len(out):
    #     ax.plot(out.index, out.values, "x", color="#B3261E", ms=5,
    #             label=f"alarms ({len(out)} steps)")
    if onset is not None:
        ax.axvline(onset, color="#444", ls="--", lw=1.2)
        ax.text(onset, ax.get_ylim()[1] * 0.96, "attack starts", fontsize=8)
    ax.set_xlabel("simulation step")
    ax.set_ylabel(f"vehicles per step (rolling {window})")
    ax.set_title(f"Flow through the signal over time {label}".strip())
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"[plot] {outpath}")


def plot_roc(fa, dr, outpath, auc, label=""):
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot(fa, dr, color=ATTACK_C, lw=2)
    ax.plot([0, 1], [0, 1], ls=":", color="#888", lw=1.2, label="undetectable (chance)")
    ax.set_xlabel("false-alarm rate on normal traffic")
    ax.set_ylabel("detection rate on attacked traffic")
    ax.set_title(f"Detectability {label}\nAUC = {auc:.3f} "
                 f"(0.5 = perfectly stealthy, 1.0 = trivially caught)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"[plot] {outpath}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", required=False, help="CSV from a normal run")
    ap.add_argument("--attack", required=True, help="CSV from an attacked run")
    ap.add_argument("--value", default="departures",
                    choices=["departures", "present", "waiting"])
    ap.add_argument("--onset", type=int, default=None, help="step the attack starts")
    ap.add_argument("--after", type=int, default=None,
                    help="ignore steps before this (e.g. warmup)")
    ap.add_argument("--k", type=float, default=3.0, help="alarm threshold in sigmas")
    ap.add_argument("--window", type=int, default=20, help="rolling window for plots")
    ap.add_argument("--per-signal", action="store_true",
                    help="also analyse each intersection separately")
    ap.add_argument("--outdir", default="stealth_out")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    clean = load_series(a.clean, a.value, after=a.after)
    attack = load_series(a.attack, a.value, after=a.after)
    if len(clean) < 5 or len(attack) < 5:
        raise SystemExit("too few timesteps to compare")
    print(f"[info] normal: {len(clean)} steps, attacked: {len(attack)} steps "
          f"(value='{a.value}')")

    rows = []
    rec, (bins, det, fa, dr) = analyze(clean, attack, "ALL", a, a.outdir)
    rows.append(rec)
    plot_distribution(clean, attack, bins,
                      os.path.join(a.outdir, "throughput_distribution.png"))
    plot_timeseries(clean, attack, det,
                    os.path.join(a.outdir, "throughput_timeseries.png"),
                    onset=a.onset, window=a.window)
    plot_roc(fa, dr, os.path.join(a.outdir, "detection_roc.png"), rec["detection_auc"])

    if a.per_signal:
        sigs = sorted(set(pd.read_csv(a.attack).get("intersection", pd.Series(dtype=str))
                          .astype(str).unique()))
        for s in sigs:
            if s in ("NA", "nan"):
                continue
            c = load_series(a.clean, a.value, signal=s, after=a.after)
            k = load_series(a.attack, a.value, signal=s, after=a.after)
            if len(c) < 5 or len(k) < 5:
                continue
            r, _ = analyze(c, k, s, a, a.outdir)
            rows.append(r)

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(a.outdir, "stealth_metrics.csv"), index=False)

    print("\n=== stealth / detectability ===")
    show = ["signal", "clean_mean", "attack_mean", "mean_shift_pct", "ks_stat",
            "ks_pvalue", "wasserstein", "detection_rate", "false_alarm_rate",
            "detection_auc", "cusum_steps_to_detect"]
    print(out[show].round(4).to_string(index=False))

    r = rows[0]
    verdict = ("STEALTHY - hides inside normal variation"
               if r["detection_auc"] < 0.65 else
               "PARTLY DETECTABLE - a tuned monitor would notice"
               if r["detection_auc"] < 0.85 else
               "EASILY DETECTED - flow shifts obviously")
    lines = [
        "# Stealth readout\n",
        f"Metric analysed: per-step `{a.value}` through the signal.\n",
        f"- Normal flow: {r['clean_mean']:.2f} +/- {r['clean_std']:.2f} veh/step",
        f"- Attacked flow: {r['attack_mean']:.2f} +/- {r['attack_std']:.2f} veh/step "
        f"({r['mean_shift_pct']:+.1f}%)",
        f"- KS statistic {r['ks_stat']:.3f} (p={r['ks_pvalue']:.3g})",
        f"- Wasserstein distance {r['wasserstein']:.3f}, JS divergence {r['js_divergence']:.3f}",
        f"- At k={a.k} sigma: detection rate {r['detection_rate']:.1%} "
        f"vs false-alarm rate {r['false_alarm_rate']:.1%}",
        f"- Detection AUC {r['detection_auc']:.3f}",
        f"- CUSUM steps to detect: {r['cusum_steps_to_detect']}",
        f"\n**Verdict: {verdict}.**\n",
        "Note: stealth is only meaningful next to damage. Pair this with the "
        "travel-time degradation from injection_sweep -- an attack that shifts "
        "nothing is stealthy and useless.\n",
    ]
    with open(os.path.join(a.outdir, "summary.md"), "w") as f:
        f.write("\n".join(lines))
    print(f"\n{verdict}")
    print(f"[done] {a.outdir}/")


if __name__ == "__main__":
    main()
