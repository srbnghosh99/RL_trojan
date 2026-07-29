# Attack analysis: RL vs heuristic controllers under fake-vehicle injection

This adds an **attack-analysis** experiment on top of your existing SUMO
max-adversarial pipeline. It answers, with figures:

1. **Is RL adaptive?** How long each controller *absorbs* the attack before its
   queue departs from the clean baseline — reported in **seconds**
   (`attack_onset.py`).
2. **Same injection budget — is RL better or worse?** Final travel time vs
   number of injected fake vehicles, one line per controller
   (`injection_sweep.py`).
3. **Can fewer injections break RL than a heuristic?** The fewest fake vehicles
   needed to push each controller +25% over its own clean travel time — table +
   bar chart (`injection_sweep.py`).
4. **From what time does the attack actually work?** The onset marker + the
   departure dots show exactly when degradation begins for each controller.

Controllers compared: **heuristics** `maxqueue`, `maxpressure` — **RL**
`mplight`, `colight`. Scenarios (intersection × flow): `1x1_low`, `1x1_normal`,
`1x1_high`, `4x4_normal`.

The attack is your existing one: `world.inject_fake_vehicles(...)` injects
stopped fake vehicles on an approach so the victim *sees* extra demand at the
decision moment, then `world.reset_fake_vehicles()` removes them before the
rollout. The only new knobs are **how many** vehicles and **when**.

---

## What was added (nothing existing was rewritten)

| File | Purpose |
|---|---|
| `trainer/tsc_trainer_attack_analysis.py` | New trainer+task `tsc_attack_analysis`. Subclasses `TSCTrainerMaxAdversarial`; only the eval loop changes. Reads `ATK_*` env vars, writes result JSON / per-step queue CSV. |
| `trainer/__init__.py` | One added import line so the new trainer/task register. |
| `configs/tsc_max_adversarial/maxqueue.yml` | Config for the `maxqueue` heuristic (includes `signal_config` phase_pairs it needs). |
| `configs/tsc_attack_analysis/*.yml` (+ `configs/agents/tsc_attack_analysis/*.yml`) | Analysis configs for the 4 controllers; each just includes its `tsc_max_adversarial` counterpart. Copies live under both config roots so it works whichever path your `build_config` uses. |
| `analysis/make_flow_variants.py` | Builds low/high demand `.rou.xml` + matching `sim_sumo` `.cfg` for a network. |
| `analysis/injection_sweep.py` | Orchestrator for questions 2 & 3. |
| `analysis/attack_onset.py` | Orchestrator for questions 1 & 4. |
| `analysis/run_all.sh` | Convenience wrapper. |

> **Config path note.** Your in-repo `common/utils.build_config` looks under
> `configs/agents/<task>/`, while the committed agent configs live under
> `configs/<task>/`. I placed the new analysis configs in **both** locations.
> If your working copy uses a different root, drop the four
> `tsc_attack_analysis/*.yml` files next to wherever your existing
> `tsc_max_adversarial/*.yml` files are read from.

---

## Prerequisites

* Your normal SUMO setup (`libsumo`, `sumolib`) and Python deps, plus
  `numpy pandas matplotlib` for the plots.
* **Trained controller checkpoints** for `mplight` and `colight` at the usual
  place, i.e. `data/output_data/tsc/<model>/<prefix>/model/best_<rank>`
  (same ones your `tsc_max_adversarial` eval loads). Heuristics need none.
  Flow variants (`1x1_low`, `1x1_high`) reuse the **base** `cityflow1x1`
  checkpoint automatically (`ATK_CKPT_NETWORK`), since only the demand changes.

## Run it

```bash
# 0) from the repo root
cd /path/to/RL_TSC_Backdoor
pip install numpy pandas matplotlib      # if not already present

# 1) make low/high flow variants for the 1x1 network (normal = untouched base)
python analysis/make_flow_variants.py --net cityflow1x1 --low 0.5 --high 1.6

# 2) ONSET / adaptivity (question 1 & 4) — one figure per scenario
python analysis/attack_onset.py --scenarios 1x1_normal,1x1_high \
       --n-inject 10 --onset 900 --device cuda:0 --ngpu 0

# 3) SWEEP / injection efficiency (question 2 & 3)
python analysis/injection_sweep.py --max-inject 20 --step 2 --seeds 0 \
       --device cuda:0 --ngpu 0
```

Add `--synthetic` to either script to preview the plots with fake data (no SUMO,
no checkpoints) — useful to confirm the figures before committing to full runs.

Set `--device` / `--ngpu` to match your machine (`--device cpu` for CPU-only).
Every run is cached under `analysis/results/*/runs/`, so an interrupted sweep
resumes where it left off; pass `--force` to recompute.

## Outputs

`analysis/results/sweep/`
* `injection_sweep.png` — travel time vs fake vehicles, one panel per scenario.
* `injections_to_break.png` — fewest injections to break each controller.
* `break_points.csv`, `sweep_summary.csv`, `sweep_raw.csv`, `summary.md`.

`analysis/results/onset/`
* `onset__<scenario>.png` — queue vs time, injection-start line, departure dots.
* `absorption_bars.png` — seconds each controller absorbs before degrading.
* `absorption.csv`.

## Key knobs

* `--n-inject` / `--max-inject` — fake vehicles **per segment** per attacked
  approach per decision (your current setup uses 10). Total actually injected is
  recorded in each run's JSON (`total_fake_vehicles_injected`).
* `--onset` — simulation step at which injection begins (onset study).
* `--break-threshold` — fractional travel-time rise that counts as "broken"
  (default 0.25 = +25%).
* `--approach` — `random` (default) or a fixed `N/E/S/W`.

## Assumptions / notes

* "Flow level" is created by thinning (low) or duplicating (high) vehicles in
  the route file; tune `--low` / `--high` in `make_flow_variants.py` to hit the
  demand levels you want.
* These scripts compare **controllers** under a fixed injection budget (matching
  your uploaded scaffolds). If you also want RL-*attacker* vs heuristic-*attacker*
  (attack-generation quality/speed), that's a different experiment — say the word
  and I'll add it against `attacker/multi_ppo_attacker.py`.
