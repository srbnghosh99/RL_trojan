"""
tsc_trainer_attack_analysis.py
================================================================================
A thin, non-invasive extension of the existing max-adversarial pipeline used
*only* for the attack-analysis experiments (attack_onset.py / injection_sweep.py).

It reuses everything the working `tsc_max_adversarial` trainer already sets up
(world, victim agents + checkpoint loading, attacker agents, metrics, env) and
replaces ONLY the evaluation loop so that we can control, per run:

  * ATK_MODE      'sweep' | 'onset' | 'clean'
  * ATK_N_INJECT  fake vehicles injected PER SEGMENT (0 == clean baseline)
  * ATK_ONSET     simulation step at which injection starts   (onset mode)
  * ATK_APPROACH  'random' (default) | 'N' | 'E' | 'S' | 'W'  (fixed approach)
  * ATK_OUT       output path prefix (writes <prefix>.json and, for onset,
                  <prefix>_queue.csv)

Injection uses the SAME primitive the max-adversarial trainer uses:
    world.inject_fake_vehicles(intersection_id, approach, [n]*num_segments)
    world.reset_fake_vehicles()
so the attack the controllers experience here is identical to your existing one,
only with a knob on *how many* vehicles and *when*.

Nothing else in the repo is modified except one import line in trainer/__init__.py.
Drive it exactly like your normal runs, e.g.:

    ATK_MODE=sweep ATK_N_INJECT=4 ATK_OUT=out/mplight_1x1_n4 \
    python run.py --task tsc_attack_analysis --agent mplight \
                  --network cityflow1x1 --world sumo --interface libsumo \
                  --seed 0 --thread_num 1
"""
import os
import csv
import json
import time

import numpy as np

from common.registry import Registry
from trainer.tsc_trainer_adversarial_max import TSCTrainerMaxAdversarial
from attacker.multi_ppo_attacker import MultiPPOAttacker


# --------------------------------------------------------------------------- #
#  Task: just call test() once (no training).                                 #
# --------------------------------------------------------------------------- #
@Registry.register_task("tsc_attack_analysis")
class TSCAttackAnalysisTask:
    def __init__(self, trainer):
        self.trainer = trainer

    def run(self):
        self.trainer.test(drop_load=False)


# --------------------------------------------------------------------------- #
#  Trainer                                                                     #
# --------------------------------------------------------------------------- #
@Registry.register_trainer("tsc_attack_analysis")
class TSCTrainerAttackAnalysis(TSCTrainerMaxAdversarial):

    # -- helpers ------------------------------------------------------------ #
    def _env_int(self, key, default):
        try:
            return int(os.environ.get(key, default))
        except (TypeError, ValueError):
            return default

    def _checkpoint_path(self):
        """Map this run's output dir back to the tsc-trained checkpoint dir.

        Logger path looks like  data/output_data/<task>/<model>/<prefix>
        and checkpoints live in  data/output_data/tsc/<model>/<prefix>/model/best_<rank>
        (exactly how the max-adversarial trainer resolves them, but robust to
        our custom task name).

        For flow-level variants (e.g. cityflow1x1_low / cityflow1x1_high) the
        network topology is identical to the base network, so the base network's
        checkpoint is used. Set ATK_CKPT_NETWORK=cityflow1x1 to point RL loading
        at the base checkpoint while the sim runs the scaled-demand cfg.
        """
        task_name = Registry.mapping['command_mapping']['setting'].param['task']
        run_net = Registry.mapping['command_mapping']['setting'].param['network']
        logger_path = Registry.mapping['logger_mapping']['path'].path
        path = logger_path.replace(task_name, 'tsc')
        ckpt_net = os.environ.get('ATK_CKPT_NETWORK', '').strip()
        if ckpt_net and ckpt_net != run_net:
            path = path.replace(run_net, ckpt_net)
        return path

    def _total_queue(self):
        """Instantaneous total standing queue across all controlled intersections."""
        total = 0.0
        for ag in self.agents:
            q = ag.get_queue()
            total += float(np.sum(np.asarray(q, dtype=float)))
        return total

    # -- override checkpoint loading so it works under our task name -------- #
    def create_agents(self):
        """Same construction as TSCTrainerMaxAdversarial.create_agents, but the
        victim checkpoint path is resolved from OUR task name, and missing
        checkpoints warn instead of crashing."""
        self.agents = []
        agent = Registry.mapping['model_mapping'][
            Registry.mapping['command_mapping']['setting'].param['agent']](self.world, 0)
        num_agent = int(len(self.world.intersections) / agent.sub_agents)
        self.agents.append(agent)
        for i in range(1, num_agent):
            self.agents.append(Registry.mapping['model_mapping'][
                Registry.mapping['command_mapping']['setting'].param['agent']](self.world, i))

        if Registry.mapping['model_mapping']['setting'].param['name'] == 'magd':
            for ag in self.agents:
                ag.link_agents(self.agents)

        # ---- attacker agents (one per intersection) ----
        self.attacker_agents = []
        attacker_config = Registry.mapping['attacker_mapping']['setting'].param
        if attacker_config.get('intersection', True):
            self.num_attacker_agent = len(self.world.intersections)
        else:
            self.num_attacker_agent = int(len(self.world.intersections) / agent.sub_agents)

        if Registry.mapping['command_mapping']['setting'].param['network'] == 'cityflow1x1':
            num_segments = 2
            num_approaches = 4
        else:
            num_segments = attacker_config.get('num_segments', 3)
            num_approaches = attacker_config.get('num_approaches', 4)
        learning_rate = attacker_config.get('learning_rate', 1e-4)
        gamma = attacker_config.get('gamma', 0.99)
        self.penalty_lambda = attacker_config.get('penalty_lambda', 0.01)
        max_vehicles_per_segment = attacker_config.get('max_vehicles_per_segment', 10)
        device = Registry.mapping['command_mapping']['setting'].param.get('device', 'cpu')
        self.max_vehicles_per_segment = max_vehicles_per_segment

        for i in range(self.num_attacker_agent):
            attacker_kwargs = {'param': {
                'learning_rate': float(learning_rate),
                'gamma': float(gamma),
                'penalty_lambda': float(self.penalty_lambda),
                'max_vehicles_per_segment': float(max_vehicles_per_segment),
                'num_segments': int(num_segments),
                'num_approaches': int(num_approaches),
                'device': device,
            }}
            self.attacker_agents.append(MultiPPOAttacker(self.world, i, **attacker_kwargs))

        self.n_agents = len(self.attacker_agents)

        # ---- load victim (controller) checkpoints ----
        model_path = self._checkpoint_path()
        for ag in self.agents:
            try:
                ag.load_model(e=-1, model_path=model_path)
            except FileNotFoundError as exc:
                self.logger.warning(
                    f"[attack_analysis] checkpoint not found for "
                    f"{Registry.mapping['command_mapping']['setting'].param['agent']} "
                    f"at {model_path} ({exc}). Heuristics ignore this; RL agents "
                    f"would run UNTRAINED -- check the path.")
            except Exception as exc:  # noqa
                self.logger.warning(f"[attack_analysis] load_model warning: {exc}")

    # -- controllable evaluation loop --------------------------------------- #
    def test(self, drop_load=False):
        mode = os.environ.get('ATK_MODE', 'sweep').lower()
        n_inject = self._env_int('ATK_N_INJECT', 0)
        onset = self._env_int('ATK_ONSET', 0)
        approach_mode = os.environ.get('ATK_APPROACH', 'random').upper()
        out_prefix = os.environ.get('ATK_OUT', 'attack_result')
        seed = Registry.mapping['command_mapping']['setting'].param.get('seed', 0)
        rng = np.random.default_rng(seed)

        cmd = Registry.mapping['command_mapping']['setting'].param
        world_interval = float(Registry.mapping['world_mapping']['setting'].param.get('interval', 1.0))

        # cityflow replay guard (SUMO ignores this branch)
        if cmd['world'] == 'cityflow':
            try:
                self.env.eng.set_save_replay(False)
            except Exception:  # noqa
                pass

        self.metric.clear()
        obs = self.env.reset()
        for a in self.agents:
            a.reset()
        dones = [False] * self.n_agents

        approaches = ['N', 'E', 'S', 'W']
        num_decisions = self.test_steps // self.action_interval
        total_injected = 0
        per_step_queue = []          # (sim_step, total_queue)
        first_attack_step = None
        get_time = time.process_time
        decision_time = 0.0
        wall_t0 = time.time()
        sim_step = 0

        for d in range(num_decisions):
            phases = np.stack([ag.get_phase() for ag in self.agents])

            attack_active = (n_inject > 0) and (mode != 'clean') and \
                            (mode != 'onset' or sim_step >= onset)

            if attack_active:
                if first_attack_step is None:
                    first_attack_step = sim_step
                for idx, _ in enumerate(self.attacker_agents):
                    if approach_mode in approaches:
                        approach = approach_mode
                    else:
                        approach = approaches[int(rng.integers(0, 4))]
                    counts = [int(n_inject)] * self.attacker_agents[idx].num_segments
                    try:
                        total_injected += self.world.inject_fake_vehicles(
                            self.attacker_agents[idx].intersection_id, approach, counts)
                    except Exception as exc:  # noqa
                        self.logger.warning(f"[attack_analysis] inject failed: {exc}")

            # victim observes (fake vehicles included) and decides
            obs = [ag.get_ob() for ag in self.agents]
            t_dec = get_time()
            actions = [ag.get_action(obs[i], phases[i], test=True)
                       for i, ag in enumerate(self.agents)]
            decision_time += get_time() - t_dec

            if attack_active:
                # fake vehicles only bias the DECISION, then are removed before rollout
                self.world.reset_fake_vehicles()

            actions = np.stack(actions)
            rewards_list = []
            for j in range(self.action_interval):
                obs, rewards, dones, _ = self.env.step(actions.flatten())
                sim_step += 1
                rewards_list.append(np.stack(rewards))
                per_step_queue.append((sim_step, self._total_queue()))
            self.metric.update(np.mean(rewards_list, axis=0))
            if all(dones):
                break

        wall_time = time.time() - wall_t0

        travel_time = float(self.metric.real_average_travel_time())
        try:
            queue_mean = float(self.metric.queue())
        except Exception:  # noqa
            queue_mean = float('nan')
        try:
            delay_mean = float(self.metric.delay())
        except Exception:  # noqa
            delay_mean = float('nan')
        try:
            throughput = int(self.metric.throughput())
        except Exception:  # noqa
            throughput = -1

        summary = {
            'mode': mode,
            'agent': cmd['agent'],
            'network': cmd['network'],
            'world': cmd['world'],
            'seed': int(seed),
            'n_inject_per_segment': int(n_inject),
            'onset_step': int(onset) if mode == 'onset' else None,
            'approach_mode': approach_mode,
            'total_fake_vehicles_injected': int(total_injected),
            'first_attack_step': first_attack_step,
            'travel_time': travel_time,
            'queue': queue_mean,
            'delay': delay_mean,
            'throughput': throughput,
            'test_steps': int(self.test_steps),
            'action_interval': int(self.action_interval),
            'sim_interval_seconds': world_interval,
            'decision_time_seconds': round(decision_time, 4),
            'wall_time_seconds': round(wall_time, 2),
        }

        out_dir = os.path.dirname(out_prefix)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_prefix + '.json', 'w') as f:
            json.dump(summary, f, indent=2)

        if mode == 'onset':
            with open(out_prefix + '_queue.csv', 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['sim_step', 'sim_time_seconds', 'total_queue'])
                for s, q in per_step_queue:
                    w.writerow([s, s * world_interval, q])

        self.logger.info(
            "[attack_analysis] mode=%s agent=%s net=%s n_inject=%d injected=%d "
            "TravelTime=%.4f queue=%.4f delay=%.4f throughput=%d -> %s.json" % (
                mode, cmd['agent'], cmd['network'], n_inject, total_injected,
                travel_time, queue_mean, delay_mean, throughput, out_prefix))
        return self.metric
