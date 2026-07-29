# Stealth readout

Metric analysed: per-step `departures` through the signal.

- Normal flow: 3.86 +/- 3.12 veh/step
- Attacked flow: 4.02 +/- 3.53 veh/step (+4.0%)
- KS statistic 0.023 (p=0.294)
- Wasserstein distance 0.232, JS divergence 0.004
- At k=3.0 sigma: detection rate 1.1% vs false-alarm rate 0.6%
- Detection AUC 0.523
- CUSUM steps to detect: 6

**Verdict: STEALTHY - hides inside normal variation.**

Note: stealth is only meaningful next to damage. Pair this with the travel-time degradation from injection_sweep -- an attack that shifts nothing is stealthy and useless.
