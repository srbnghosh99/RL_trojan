"""
Saturation-detection check for MPLight under attack.

For each decision moment, compute the "inflation" each phase received (attack pressure
minus no-attack pressure) and check whether MPLight tends to pick the phase with the
LEAST inflation. If yes → MPLight has learned to detect saturation/anomalies and
prefer realistic-looking phases. If no → MPLight is just confused by corrupted input.

Usage:
    python analyze_saturation_detection.py NO_ATTACK.xlsx ATTACK.xlsx
"""

import sys
import pandas as pd
import numpy as np


def analyze(noattack_path, attack_path):
    na = pd.read_excel(noattack_path)
    at = pd.read_excel(attack_path)

    p_cols = [f'pressure_phase_{i}' for i in range(8)]
    q_cols = [f'q_phase_{i}' for i in range(8)]

    # Merge on time to compare same-moment decisions
    merged = na.merge(
        at,
        on='time',
        suffixes=('_na', '_at')
    )

    n_total = len(merged)
    n_phases = 8

    # Build matrices of per-phase pressure inflation
    pressure_na = merged[[f'{c}_na' for c in p_cols]].values
    pressure_at = merged[[f'{c}_at' for c in p_cols]].values
    inflation = pressure_at - pressure_na   # how much each phase got "boosted" by fakes

    # Identify the least-inflated phase (smallest absolute inflation) per row
    least_inflated_phase = np.argmin(inflation, axis=1)

    # What MPLight chose under attack
    chosen_at = merged['chosen_phase_at'].values

    # What MaxPressure picked under attack (its argmax of corrupted pressures)
    max_p_phase_at = merged['max_pressure_phase_at'].values

    # ------------------- METRICS -------------------
    pct_least_inflated = (chosen_at == least_inflated_phase).mean() * 100
    pct_max_pressure = (chosen_at == max_p_phase_at).mean() * 100
    pct_random = 100.0 / n_phases  # baseline = 12.5% for 8 phases

    print("=" * 70)
    print("SATURATION-DETECTION ANALYSIS")
    print("=" * 70)
    print(f"Total decisions compared: {n_total}")
    print(f"Random-baseline (1/n_phases): {pct_random:.1f}%")
    print()
    print(f"Of {n_total} decisions under attack:")
    print(f"  MPLight picked LEAST-inflated phase:  {pct_least_inflated:.1f}%  "
          f"({int(pct_least_inflated/100*n_total)}/{n_total})")
    print(f"  MPLight picked MaxPressure's pick:    {pct_max_pressure:.1f}%  "
          f"({int(pct_max_pressure/100*n_total)}/{n_total})")
    print()

    if pct_least_inflated > 3 * pct_random:
        print("INTERPRETATION 1 (likely): MPLight has learned saturation detection.")
        print("It picks the least-inflated phase far more than random would.")
    elif pct_least_inflated > 1.5 * pct_random:
        print("INTERPRETATION 1 (partial): MPLight shows some saturation-aware behavior.")
    else:
        print("INTERPRETATION 2: MPLight is NOT preferring untouched phases.")
        print("Decisions appear to be driven by Q-values in a more complex way.")
    print()

    # ------------------- BREAKDOWN BY INFLATION LEVEL -------------------
    print("=" * 70)
    print("BREAKDOWN: when there is a CLEAR untouched phase, does MPLight pick it?")
    print("=" * 70)
    print("(A 'clear untouched phase' = max inflation across phases > 20, and the")
    print("least-inflated phase has < 5 inflation.)")
    print()

    max_inflation = inflation.max(axis=1)
    min_inflation = inflation.min(axis=1)
    has_clear_target = (max_inflation > 20) & (min_inflation < 5)
    sub = np.where(has_clear_target)[0]

    if len(sub) > 0:
        sub_pct = (chosen_at[sub] == least_inflated_phase[sub]).mean() * 100
        print(f"Decisions with a clearly-untouched phase: {len(sub)}/{n_total} "
              f"({100*len(sub)/n_total:.1f}%)")
        print(f"  Of those, MPLight picked the untouched phase: {sub_pct:.1f}% "
              f"({int(sub_pct/100*len(sub))}/{len(sub)})")
    else:
        print("No decisions met the 'clear untouched phase' criterion.")
    print()

    # ------------------- BREAKDOWN: SATURATION CEILING -------------------
    print("=" * 70)
    print("How often does the attacker saturate >= 3 phases simultaneously?")
    print("=" * 70)

    # Find what pressures the attacker is hitting (ceiling-like values)
    flat_pressures = pressure_at.flatten()
    # Saturation = phases at the 95th percentile under attack
    threshold = np.percentile(flat_pressures, 95)
    saturated_count = (pressure_at >= threshold).sum(axis=1)
    pct_3plus_saturated = (saturated_count >= 3).mean() * 100
    print(f"95th percentile pressure value (estimated ceiling): {threshold:.1f}")
    print(f"Decisions with >= 3 phases at saturation:  {pct_3plus_saturated:.1f}%")
    print(f"Decisions with >= 4 phases at saturation:  {(saturated_count >= 4).mean()*100:.1f}%")
    print(f"Decisions with >= 5 phases at saturation:  {(saturated_count >= 5).mean()*100:.1f}%")
    print()

    # When multiple phases are saturated, does MPLight prefer the unsaturated ones?
    multi_sat_idx = np.where(saturated_count >= 3)[0]
    if len(multi_sat_idx) > 0:
        sat_mask = (pressure_at[multi_sat_idx] >= threshold)
        chosen_in_multisat = chosen_at[multi_sat_idx]
        # chosen phase WAS one of the saturated ones
        chosen_is_saturated = sat_mask[np.arange(len(multi_sat_idx)), chosen_in_multisat]
        pct_picked_unsat = (1 - chosen_is_saturated.mean()) * 100
        print(f"When >= 3 phases are saturated, MPLight picked an UNsaturated phase "
              f"{pct_picked_unsat:.1f}% of the time")
        print(f"  (Random baseline would be ~"
              f"{100 - 100*saturated_count[multi_sat_idx].mean()/n_phases:.1f}%)")
    print()

    # ------------------- BREAKDOWN: Q-VALUE BEHAVIOR -------------------
    print("=" * 70)
    print("Q-VALUE ANALYSIS")
    print("=" * 70)
    qvals_at = merged[[f'{c}_at' for c in q_cols]].values
    qvals_na = merged[[f'{c}_na' for c in q_cols]].values
    print(f"Mean Q-value under no-attack: {qvals_na.mean():.2f}")
    print(f"Mean Q-value under attack:    {qvals_at.mean():.2f}")
    print(f"Q-spread per decision (max - min):")
    print(f"  No-attack: mean spread = {(qvals_na.max(axis=1) - qvals_na.min(axis=1)).mean():.2f}")
    print(f"  Attack:    mean spread = {(qvals_at.max(axis=1) - qvals_at.min(axis=1)).mean():.2f}")
    print()
    print("A larger spread = controller is more decisive about which phase is best.")
    print("A tiny spread = Q-values are clustered, decision is essentially noise-driven.")

    return {
        'pct_least_inflated': pct_least_inflated,
        'pct_max_pressure_match': pct_max_pressure,
        'random_baseline': pct_random,
    }


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python analyze_saturation_detection.py NO_ATTACK.xlsx ATTACK.xlsx")
        sys.exit(1)
    analyze(sys.argv[1], sys.argv[2])
