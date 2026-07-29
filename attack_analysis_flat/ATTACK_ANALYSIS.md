# Attack analysis: RL vs heuristic controllers under fake-vehicle injection

All the scripts you run live in the **repo root** (this folder, next to `run.py`).
Outputs go into a single `attack_out/` folder. Nothing is deeply nested.

Answers, with figures:

1. **Is RL adaptive / "from what time does the attack work"?** — `attack_onset.py`
   plots total queue vs time with the injection-start line and a dot where each
   controller departs from its clean baseline. The gap is reported in **seconds**.
2. **Same injection budget — is RL better or worse?** — `injection_sweep.py`
   plots final travel time vs injected fake vehicles, one line per controller.
3. **Can fewer injections break RL than a heuristic?** — `injection_sweep.py`
   break-point table + bar chart (fewest vehicles to push a controller +25% over
   its own clean travel time).

Controllers: heuristics `maxqueue`, `maxpressure` — RL `mplight`, `colight`.
Scenarios: `1x1_low`, `1x1_normal`, `1x1_high`, `4x4_normal`. Simulator: SUMO.

## Files

Runnable, in the repo root:
`make_flow_variants.py`, `attack_onset.py`, `injection_sweep.py`, `run_all.sh`.

Framework files (must stay where `run.py` loads them — these dirs already exist):
* `trainer/tsc_trainer_attack_analysis.py` — new task/trainer `tsc_attack_analysis`
  (subclasses your max-adversarial trainer; only the eval loop changes).
* `trainer/__init__.py` — one added import line.
* `configs/tsc_max_adversarial/maxqueue.yml` — config for the `maxqueue` heuristic.
* `configs/tsc_attack_analysis/*.yml` (+ `configs/agents/tsc_attack_analysis/*.yml`)
  — analysis configs; copies in both roots so it works whichever your
  `build_config` reads.

## Prerequisites

* Your normal SUMO setup (`libsumo`, `sumolib`) plus `numpy pandas matplotlib`.
* Trained `mplight` / `colight` checkpoints under
  `data/output_data/tsc/<model>/<prefix>/model/best_<rank>` (same ones your
  `tsc_max_adversarial` eval loads). Heuristics need none. Flow variants reuse
  the base `cityflow1x1` checkpoint automatically.

## Run it

```bash
cd RL_TSC_Backdoor-trojdrlshra        # this folder, where run.py is
pip install numpy pandas matplotlib   # if needed

bash run_all.sh cuda:0 0              # DEVICE NGPU   (use: bash run_all.sh cpu 0)
```

Or step by step:

```bash
python make_flow_variants.py --net cityflow1x1 --low 0.5 --high 1.6   # run once
python attack_onset.py --scenarios 1x1_normal,1x1_high --n-inject 10 --onset 900 --device cuda:0
python injection_sweep.py --max-inject 20 --step 2 --seeds 0 --device cuda:0
```

Add `--synthetic` to preview the plots without SUMO/checkpoints.
Runs are cached in `attack_out/`; an interrupted sweep resumes. `--force` recomputes.

## Outputs (all in `attack_out/`)

* `injection_sweep.png`, `injections_to_break.png`, `break_points.csv`,
  `sweep_summary.csv`, `sweep_raw.csv`, `summary.md`
* `onset__<scenario>.png`, `absorption_bars.png`, `absorption.csv`
* per-run cache: `<scenario>__<agent>__n<k>__s<seed>.json` (+ `_queue.csv` for onset)

## Key knobs

* `--n-inject` / `--max-inject` — fake vehicles **per segment** per attacked
  approach per decision (your current setup = 10). Actual total is logged per run.
* `--onset` — sim step injection begins. `--break-threshold` — default 0.25 (+25%).
* `--approach` — `random` (default) or fixed `N/E/S/W`.
* `--device` / `--ngpu` — match your machine. `--repo PATH` if you ever run a
  script from another folder.
