# Stealth readout

Metric analysed: per-step `departures` through the signal.

- Normal flow: 3.86 +/- 3.12 veh/step
- Attacked flow: 22.26 +/- 9.91 veh/step (+476.7%)
- KS statistic 0.869 (p=0)
- Wasserstein distance 18.401, JS divergence 0.726
- At k=3.0 sigma: detection rate 83.2% vs false-alarm rate 0.6%
- Detection AUC 0.966
- CUSUM steps to detect: 13

**Verdict: EASILY DETECTED - flow shifts obviously.**

Note: stealth is only meaningful next to damage. Pair this with the travel-time degradation from injection_sweep -- an attack that shifts nothing is stealthy and useless.
