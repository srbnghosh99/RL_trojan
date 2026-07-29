#!/usr/bin/env python3
"""
attack_coverage.py
================================================================================
WHY USE RL TO GENERATE THE ATTACK?

Claim: a gradient attacker (FGSM) is a one-shot greedy rule -- it takes the
gradient at the current state and injects there. It has no exploration, so its
attacks concentrate in a narrow band of the action space and it only works in
the situations its gradient happens to favour. An RL attacker (MultiPPO) is
trained by exploration, so it discovers a wider, state-dependent repertoire and
lands damage in more situations.

This script measures that, in two parts:

  MECHANISM (this file)  -- how much of the attack space each attacker uses:
      * approach coverage      : which approaches it ever attacks
      * magnitude coverage     : the spread of injection sizes it uses
      * action entropy         : how varied its choices are (bits)
      * unique action combos   : distinct (approach, magnitude-bin) pairs
      * state-space coverage   : how many distinct traffic states it attacks in
      * state-dependence       : does its action change with the state, or is it
                                 the same move regardless? (mutual information)

  OUTCOME (separate)     -- run injection_sweep_v3.py for each attacker across
      scenarios and count how many it actually breaks. Coverage only matters if
      it converts into more scenarios broken; report the two together.

--------------------------------------------------------------------------------
STEP 1 - RECORD  (one edit, captures BOTH attackers)

Both attackers end up calling SDSMInjector.inject_approach_vehicles(approach,
segment_counts), so instrument that single method in attacker/sdsm_injector.py:

    # at the top of the file
    from attack_coverage import AttackRecorder
    _AREC = AttackRecorder()          # module-level, writes on exit

    # first lines inside inject_approach_vehicles(self, approach, segment_counts)
    _AREC.record(self.world, self.intersection_id, approach, segment_counts)

Then run each attacker once, naming the output:

    ATK_LOG=fgsm.csv      python3 run.py --agent maxqueue --task tsc_rl_adversarial ...
    ATK_LOG=multippo.csv  python3 run.py --agent maxqueue --task tsc_rl_adversarial ...

STEP 2 - COMPARE

    python3 attack_coverage.py --a fgsm.csv --a-label FGSM \\
                               --b multippo.csv --b-label MultiPPO \\
                               --outdir coverage_out

OUTPUTS
    approach_coverage.png     which approaches each attacker uses
    magnitude_distribution.png spread of injection sizes
    action_heatmap.png        approach x magnitude, side by side
    state_coverage.png        traffic states each attacker attacks in
    coverage_metrics.csv      all numbers
    summary.md                readout
================================================================================
"""
import argparse
import atexit
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

A_COLOR = "#D85A30"     # attacker A (gradient)
B_COLOR = "#1D9E75"     # attacker B (RL)


# --------------------------------------------------------------------------- #
#  RECORDER  (used inside the repo)
# --------------------------------------------------------------------------- #
class AttackRecorder:
    """
    Logs every injection decision: which approach, how many vehicles per
    segment, and a summary of the traffic state at that moment.

    Output file comes from the ATK_LOG env var (default attack_actions.csv),
    so you can name each attacker's log at run time. Flushes on process exit.
    """

    def __init__(self, out=None, flush_every=200):
        self.out = out or os.environ.get("ATK_LOG", "attack_actions.csv")
        self.flush_every = int(flush_every)
        self.rows = []
        self._n = 0
        self._header_written = False
        atexit.register(self.close)
        print(f"[AttackRecorder] logging attack decisions -> {self.out}")

    def _state_summary(self, world):
        """Cheap, engine-agnostic snapshot of the traffic state."""
        tot_q = tot_c = np.nan
        try:
            wait = world.get_info("lane_waiting_count") or {}
            cnt = world.get_info("lane_count") or {}
            vals_w = list(wait.values()) if isinstance(wait, dict) else list(wait)
            vals_c = list(cnt.values()) if isinstance(cnt, dict) else list(cnt)
            tot_q = float(np.nansum([v for v in vals_w if v is not None]))
            tot_c = float(np.nansum([v for v in vals_c if v is not None]))
        except Exception:
            pass
        return tot_q, tot_c

    def record(self, world, intersection_id, approach, segment_counts, phase=None):
        try:
            counts = [float(c) for c in np.ravel(segment_counts)]
        except Exception:
            counts = []
        tot_q, tot_c = self._state_summary(world)
        if phase is None:
            try:
                phase = int(world.id2intersection[intersection_id].current_phase)
            except Exception:
                phase = -1
        try:
            t = float(world.get_current_time())
        except Exception:
            t = self._n

        row = dict(t=t, intersection=intersection_id, approach=str(approach),
                   total_vehicles=float(np.sum(counts)) if counts else 0.0,
                   max_segment=float(np.max(counts)) if counts else 0.0,
                   n_segments=len(counts), phase=phase,
                   state_queue=tot_q, state_count=tot_c)
        for i, c in enumerate(counts):
            row[f"seg_{i}"] = c
        self.rows.append(row)
        self._n += 1
        if self._n % self.flush_every == 0:
            self._flush()

    def _flush(self):
        if not self.rows:
            return
        df = pd.DataFrame(self.rows)
        mode = "a" if self._header_written else "w"
        df.to_csv(self.out, mode=mode, header=not self._header_written, index=False)
        self._header_written = True
        self.rows = []

    def close(self):
        self._flush()
        if self._header_written:
            print(f"[AttackRecorder] wrote {self.out}")


# --------------------------------------------------------------------------- #
#  ANALYSIS
# --------------------------------------------------------------------------- #
def entropy(labels):
    """Shannon entropy in bits of a categorical series."""
    v = pd.Series(labels).value_counts(normalize=True)
    v = v[v > 0]
    return float(-(v * np.log2(v)).sum())


def mutual_information(x, y, bins=8):
    """MI in bits between a state variable and the chosen action."""
    x = pd.Series(x).astype(float)
    if x.notna().sum() < 5:
        return np.nan
    xb = pd.qcut(x.rank(method="first"), min(bins, x.nunique()),
                 labels=False, duplicates="drop")
    tab = pd.crosstab(xb, pd.Series(y).astype(str))
    if tab.size == 0:
        return np.nan
    p = tab.values.astype(float)
    p /= p.sum()
    px = p.sum(1, keepdims=True)
    py = p.sum(0, keepdims=True)
    nz = p > 0
    return float(np.sum(p[nz] * np.log2(p[nz] / (px @ py)[nz])))


def load(csv, label):
    df = pd.read_csv(csv)
    need = {"approach", "total_vehicles"}
    if not need.issubset(df.columns):
        raise SystemExit(f"{csv}: needs {need}, has {list(df.columns)}")
    df["attacker"] = label
    if "state_queue" not in df:
        df["state_queue"] = np.nan
    if "phase" not in df:
        df["phase"] = -1
    return df


def metrics(df, label, mag_bins=6):
    m = df["total_vehicles"].astype(float)
    edges = np.linspace(0, max(1.0, m.max()), mag_bins + 1)
    mbin = np.clip(np.digitize(m, edges) - 1, 0, mag_bins - 1)
    combo = df["approach"].astype(str) + "|" + pd.Series(mbin, index=df.index).astype(str)

    # state-space coverage: distinct (queue-decile, phase) cells attacked in
    cells = np.nan
    if df["state_queue"].notna().sum() > 5:
        qb = pd.qcut(df["state_queue"].rank(method="first"), 10,
                     labels=False, duplicates="drop")
        cells = int(pd.Series(list(zip(qb, df["phase"]))).nunique())

    return dict(
        attacker=label,
        n_decisions=len(df),
        approaches_used=int(df["approach"].nunique()),
        approach_entropy_bits=entropy(df["approach"]),
        unique_action_combos=int(combo.nunique()),
        action_entropy_bits=entropy(combo),
        mean_vehicles=float(m.mean()),
        std_vehicles=float(m.std()),
        magnitude_range=float(m.max() - m.min()),
        state_cells_attacked=cells,
        state_dependence_MI_bits=mutual_information(df["state_queue"], df["approach"]),
    )


def plot_approach(a, b, la, lb, out):
    ap = sorted(set(a.approach.astype(str)) | set(b.approach.astype(str)))
    fa = a.approach.astype(str).value_counts(normalize=True).reindex(ap).fillna(0)
    fb = b.approach.astype(str).value_counts(normalize=True).reindex(ap).fillna(0)
    x = np.arange(len(ap))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(x - 0.2, fa.values, 0.4, color=A_COLOR, label=la)
    ax.bar(x + 0.2, fb.values, 0.4, color=B_COLOR, label=lb)
    ax.set_xticks(x)
    ax.set_xticklabels(ap)
    ax.set_xlabel("approach attacked")
    ax.set_ylabel("share of decisions")
    ax.set_title("Which approaches each attacker uses\n"
                 "(one tall bar = narrow; spread = broad coverage)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] {out}")


def plot_magnitude(a, b, la, lb, out):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    lo = 0
    hi = max(a.total_vehicles.max(), b.total_vehicles.max())
    bins = np.linspace(lo, hi, 20)
    ax.hist(a.total_vehicles, bins=bins, alpha=0.55, color=A_COLOR,
            density=True, label=la)
    ax.hist(b.total_vehicles, bins=bins, alpha=0.55, color=B_COLOR,
            density=True, label=lb)
    ax.set_xlabel("vehicles injected per decision")
    ax.set_ylabel("density")
    ax.set_title("Injection magnitudes used\n(narrow spike = one move repeated)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] {out}")


def plot_action_heatmap(a, b, la, lb, out, mag_bins=6):
    ap = sorted(set(a.approach.astype(str)) | set(b.approach.astype(str)))
    hi = max(a.total_vehicles.max(), b.total_vehicles.max(), 1.0)
    edges = np.linspace(0, hi, mag_bins + 1)

    def grid(df):
        g = np.zeros((len(ap), mag_bins))
        mb = np.clip(np.digitize(df.total_vehicles, edges) - 1, 0, mag_bins - 1)
        for appr, k in zip(df.approach.astype(str), mb):
            g[ap.index(appr), k] += 1
        return g / max(1, g.sum())

    ga, gb = grid(a), grid(b)
    vmax = max(ga.max(), gb.max())
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6 + 0.3 * len(ap)))
    for ax, g, lb_ in ((axes[0], ga, la), (axes[1], gb, lb)):
        im = ax.imshow(g, aspect="auto", cmap="magma", vmin=0, vmax=vmax)
        ax.set_yticks(range(len(ap)))
        ax.set_yticklabels(ap)
        ax.set_xticks(range(mag_bins))
        ax.set_xticklabels([f"{edges[i]:.0f}-{edges[i+1]:.0f}" for i in range(mag_bins)],
                           fontsize=7, rotation=45)
        ax.set_xlabel("vehicles injected")
        cov = float((g > 0).sum()) / g.size * 100
        ax.set_title(f"{lb_}\n{cov:.0f}% of the action grid used")
        fig.colorbar(im, ax=ax, label="share of decisions")
    axes[0].set_ylabel("approach")
    fig.suptitle("Attack-space coverage (bright cells = used; dark = never tried)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] {out}")


def plot_state_coverage(a, b, la, lb, out):
    if a.state_queue.isna().all() or b.state_queue.isna().all():
        print("[skip] state coverage: no state_queue recorded")
        return
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.scatter(a.state_queue, a.total_vehicles, s=12, alpha=0.35,
               color=A_COLOR, label=la)
    ax.scatter(b.state_queue, b.total_vehicles, s=12, alpha=0.35,
               color=B_COLOR, label=lb)
    ax.set_xlabel("network queue when the attack was launched (traffic state)")
    ax.set_ylabel("vehicles injected")
    ax.set_title("Which traffic states each attacker acts in\n"
                 "(a tight cluster = only works in one situation)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="CSV from attacker A (e.g. FGSM)")
    ap.add_argument("--b", required=True, help="CSV from attacker B (e.g. MultiPPO)")
    ap.add_argument("--a-label", default="FGSM")
    ap.add_argument("--b-label", default="MultiPPO")
    ap.add_argument("--mag-bins", type=int, default=6)
    ap.add_argument("--outdir", default="coverage_out")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    A = load(args.a, args.a_label)
    B = load(args.b, args.b_label)
    print(f"[info] {args.a_label}: {len(A)} decisions | "
          f"{args.b_label}: {len(B)} decisions")

    rows = [metrics(A, args.a_label, args.mag_bins),
            metrics(B, args.b_label, args.mag_bins)]
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(args.outdir, "coverage_metrics.csv"), index=False)

    plot_approach(A, B, args.a_label, args.b_label,
                  os.path.join(args.outdir, "approach_coverage.png"))
    plot_magnitude(A, B, args.a_label, args.b_label,
                   os.path.join(args.outdir, "magnitude_distribution.png"))
    plot_action_heatmap(A, B, args.a_label, args.b_label,
                        os.path.join(args.outdir, "action_heatmap.png"), args.mag_bins)
    plot_state_coverage(A, B, args.a_label, args.b_label,
                        os.path.join(args.outdir, "state_coverage.png"))

    print("\n=== attack-space coverage ===")
    print(out.round(3).to_string(index=False))

    ra, rb = rows[0], rows[1]
    broader = (rb["action_entropy_bits"] > ra["action_entropy_bits"] and
               rb["unique_action_combos"] >= ra["unique_action_combos"])
    verdict = (f"{args.b_label} explores a BROADER attack space than {args.a_label}"
               if broader else
               f"{args.b_label} does NOT show broader coverage than {args.a_label} here")
    lines = [
        "# Attack-space coverage readout\n",
        f"| metric | {args.a_label} | {args.b_label} |", "|---|---|---|",
        f"| decisions logged | {ra['n_decisions']} | {rb['n_decisions']} |",
        f"| approaches used | {ra['approaches_used']} | {rb['approaches_used']} |",
        f"| unique action combos | {ra['unique_action_combos']} | {rb['unique_action_combos']} |",
        f"| action entropy (bits) | {ra['action_entropy_bits']:.2f} | {rb['action_entropy_bits']:.2f} |",
        f"| magnitude spread (std) | {ra['std_vehicles']:.2f} | {rb['std_vehicles']:.2f} |",
        f"| state cells attacked | {ra['state_cells_attacked']} | {rb['state_cells_attacked']} |",
        f"| state-dependence MI (bits) | {ra['state_dependence_MI_bits']:.3f} | {rb['state_dependence_MI_bits']:.3f} |",
        f"\n**{verdict}.**\n",
        "Higher action entropy and more unique combos mean the attacker varies its "
        "move instead of repeating one. Higher state-dependence MI means it *adapts* "
        "the move to the traffic state rather than acting blind.\n",
        "IMPORTANT: coverage is the mechanism, not the result. Pair this with "
        "injection_sweep_v3.py across scenarios and report how many scenarios each "
        "attacker actually breaks -- wider exploration only matters if it converts "
        "into more damage in more situations.\n",
    ]
    with open(os.path.join(args.outdir, "summary.md"), "w") as f:
        f.write("\n".join(lines))
    print(f"\n{verdict}")
    print(f"[done] {args.outdir}/")


if __name__ == "__main__":
    main()
