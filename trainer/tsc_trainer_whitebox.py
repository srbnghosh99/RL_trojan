import os

import numpy as np

from common.metrics import Metrics
from environment import TSCEnv
from common.registry import Registry
from trainer.base_trainer import BaseTrainer
from attacker.FGSM import FGSM


@Registry.register_trainer("tsc_whitebox")
class TSCTrainerWhitebox(BaseTrainer):
    '''
    Evaluate a PRETRAINED victim controller (e.g. mplight) under a white-box
    FGSM attack. This never trains the victim -- it only loads the checkpoint
    produced by the plain "tsc" task and measures its performance while an
    attacker with full access to its network computes gradient-driven
    fake-vehicle injections each decision step.

    Mirrors trainer/tsc_trainer_adversarial_max.py's test() loop, but swaps
    the learned MultiPPOAttacker for the white-box FGSM attacker.
    '''

    def __init__(
        self,
        logger,
        gpu=0,
        cpu=False,
        name="tsc",
        wandb=None,
        comet=None,
    ):
        super().__init__(
            logger=logger,
            gpu=gpu,
            cpu=cpu,
            name=name,
        )
        self.episodes = Registry.mapping['trainer_mapping']['setting'].param['episodes']
        self.steps = Registry.mapping['trainer_mapping']['setting'].param['steps']
        self.test_steps = Registry.mapping['trainer_mapping']['setting'].param['test_steps']
        self.action_interval = Registry.mapping['trainer_mapping']['setting'].param['action_interval']
        self.wandb = wandb
        self.comet = comet

    def create_world(self):
        self.world = Registry.mapping['world_mapping'][Registry.mapping['command_mapping']['setting'].param['world']](
            self.path,
            Registry.mapping['command_mapping']['setting'].param['thread_num'],
            interface=Registry.mapping['command_mapping']['setting'].param['interface'],
        )

    def create_metrics(self):
        if Registry.mapping['command_mapping']['setting'].param['delay_type'] == 'apx':
            lane_metrics = ['rewards', 'queue', 'delay']
            world_metrics = ['real avg travel time', 'throughput']
        else:
            lane_metrics = ['rewards', 'queue']
            world_metrics = ['delay', 'real avg travel time', 'throughput']
        self.metric = Metrics(lane_metrics, world_metrics, self.world, self.agents)

    def create_agents(self):
        self.agents = []
        agent = Registry.mapping['model_mapping'][Registry.mapping['command_mapping']['setting'].param['agent']](self.world, 0)
        num_agent = int(len(self.world.intersections) / agent.sub_agents)
        self.agents.append(agent)
        for i in range(1, num_agent):
            self.agents.append(
                Registry.mapping['model_mapping'][Registry.mapping['command_mapping']['setting'].param['agent']](self.world, i)
            )

        # This task is EVAL-ONLY: always load the checkpoint trained under the
        # plain "tsc" task, never train here. By default that's the checkpoint
        # for the SAME network being simulated (--network). If
        # --controller_source_network is given, load the checkpoint trained on
        # THAT network instead, while still simulating on --network -- this is
        # the cross-domain test: does a controller trained on network A still
        # behave/get attacked the same way when actually deployed on network B?
        command_param = Registry.mapping['command_mapping']['setting'].param
        target_network = command_param['network']
        controller_source_network = command_param.get('controller_source_network', None)
        attacker_source_network = command_param.get('attacker_source_network', None)

        model_path = Registry.mapping['logger_mapping']['path'].path.replace('tsc_whitebox', 'tsc')
        if controller_source_network is not None and controller_source_network != target_network:
            model_path = model_path.replace(target_network, controller_source_network)
            self.logger.info(
                f"[cross-domain] Loading victim controller checkpoint from network "
                f"'{controller_source_network}', but simulating live on '{target_network}'."
            )

        if attacker_source_network is not None:
            self.logger.warning(
                f"--attacker_source_network='{attacker_source_network}' was given, but FGSM has no "
                f"trained parameters of its own to source from another network -- it always computes "
                f"a fresh gradient from whatever victim checkpoint is actually loaded. This flag is "
                f"currently a NO-OP for the tsc_whitebox task; it does not do anything. "
                f"(It's meaningful for the learned tsc_rl_adversarial/tsc_max_adversarial PPO attacker, "
                f"which does have its own trained weights -- but not for FGSM as currently implemented.)"
            )

        for ag in self.agents:
            ag.load_model(e=-1, model_path=model_path)

        attacker_config = Registry.mapping['attacker_mapping']['setting'].param
        self.attacker = FGSM(
            epsilon=float(attacker_config.get('epsilon', 0.15)),
            max_vehicles_per_lane=int(attacker_config.get('max_vehicles_per_lane', 8)),
            max_total_vehicles=attacker_config.get('max_total_vehicles', None),
            allocation=attacker_config.get('allocation', 'proportional'),
            pgd_steps=int(attacker_config.get('pgd_steps', 1)),
            top_k_lanes=attacker_config.get('top_k_lanes', None),
            fallback_to_largest_abs_grad=bool(attacker_config.get('fallback_to_largest_abs_grad', True)),
            min_vehicles_per_selected_lane=int(attacker_config.get('min_vehicles_per_selected_lane', 1)),
            targeted=bool(attacker_config.get('targeted', False)),
            model_attr=attacker_config.get('model_attr', None),
            device=Registry.mapping['command_mapping']['setting'].param.get('device', None),
            strict=bool(attacker_config.get('strict', False)),
            logger=self.logger,
        )

        self.n_agents = len(self.agents)

    def create_env(self):
        # No attacker_agents passed -- FGSM is not a learned policy inside the
        # env's reward loop, it's applied directly to the world before each
        # decision, exactly like tsc_max_adversarial does for its PPO attacker.
        self.env = TSCEnv(self.world, self.agents, self.metric)

    def _snapshot_real_traffic(self):
        '''
        Query the simulator directly for (a) how many REAL vehicles are
        currently active in the network and (b) their average accumulated
        waiting time, at this instant. Excludes anything with 'fake_' in its
        ID -- this is meant to describe real traffic, not the attack's own
        injections. Uses the same libsumo calls already used elsewhere in
        world_sumo.py (get_vehicle_trajectory, inject_fake_vehicles), so
        nothing new is assumed about the API.

        Returns (total_vehicles, avg_wait_time), or (None, None) if this
        isn't a SUMO/libsumo world (e.g. plain CityFlow -- not implemented
        here, since the SUMO vehicle-level API this relies on doesn't apply).
        '''
        eng = getattr(self.world, 'eng', None)
        if eng is None or not hasattr(eng, 'vehicle'):
            return None, None
        try:
            all_ids = eng.vehicle.getIDList()
            real_ids = [v for v in all_ids if 'fake_' not in str(v)]
            if not real_ids:
                return 0, 0.0
            wait_times = [eng.vehicle.getWaitingTime(v) for v in real_ids]
            return len(real_ids), float(np.mean(wait_times))
        except Exception as e:
            self.logger.info(f"_snapshot_real_traffic failed (non-fatal): {e}")
            return None, None

    def _write_csv(self, rows, filename):
        '''
        Write a list of dict rows (all rows must share the same keys) to a
        CSV file in the current working directory (wherever `python3 run.py`
        was invoked from) -- not nested inside the run's output folder.
        '''
        if not rows:
            return None
        try:
            import csv
            csv_path = os.path.abspath(filename)
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            return csv_path
        except Exception as e:
            self.logger.info(f"Failed to write {filename} (non-fatal): {e}")
            return None

    def _write_timeseries_csv(self, rows):
        return self._write_csv(rows, 'fgsm_timeseries.csv')



    def train(self):
        pass

    def train_test(self, e):
        pass


    def _sample_signal_state(self, sim_time):
        """SUMO ground-truth signal state (ported from tsc_trainer_adversarial_rl.py)."""
        eng = getattr(self.world, 'eng', None)
        if eng is None or not hasattr(eng, 'trafficlight'):
            return []
        rows = []
        for inter in getattr(self.world, 'intersections', []):
            try:
                state = eng.trafficlight.getRedYellowGreenState(inter.id)
                controlled_lanes = eng.trafficlight.getControlledLanes(inter.id)
            except Exception:
                continue
            rows.append({
                'time': sim_time,
                'tls_id': inter.id,
                'controlled_lanes': ';'.join(controlled_lanes),
                'state': state,
            })
        return rows

    def _sample_vehicle_positions(self, sim_time):
        """Per-vehicle lane position (ported from tsc_trainer_adversarial_rl.py)."""
        eng = getattr(self.world, 'eng', None)
        if eng is None or not hasattr(eng, 'vehicle'):
            return []
        try:
            vehicle_lane, _ = self.world.get_vehicle_lane()
        except Exception:
            return []
        rows = []
        for v, lane in vehicle_lane.items():
            try:
                pos = eng.vehicle.getLanePosition(v)
            except Exception:
                continue
            rows.append({
                'time': sim_time,
                'vehicle_id': v,
                'is_fake': 'fake_' in str(v),
                'lane': lane,
                'position': pos,
            })
        return rows

    def test(self, drop_load=True):
        '''
        test
        Evaluate the pretrained victim under white-box FGSM attack.

        :param drop_load: kept for interface compatibility with other
            trainers; this task always loads the pretrained victim in
            create_agents(), so drop_load has no effect here.
        :return self.metric: real avg travel time, queue, delay, throughput
        '''
        if Registry.mapping['command_mapping']['setting'].param['world'] == 'cityflow':
            if self.save_replay:
                self.env.eng.set_save_replay(True)
                self.env.eng.set_replay_file(os.path.join(self.replay_file_dir, "final.txt"))
            else:
                self.env.eng.set_save_replay(False)
        self.metric.clear()

        obs = self.env.reset()
        dones = [False] * self.n_agents
        for a in self.agents:
            a.reset()

        total_fake_injected = 0
        total_attack_calls = 0
        total_attack_successes = 0
        timeseries_rows = []  # one row per decision: (sim_time, fake_injected, total_vehicles, avg_wait_time)
        gradient_rows = []    # one row per (decision, agent, lane): raw gradient + fake vehicles placed there
        pgd_step_rows = []    # one row per (sampled decision, agent, step, lane): PGD refinement trace
        position_rows = []    # one row per (sampled step, vehicle): space-time trajectory data
        signal_rows = []      # one row per (sampled step, traffic light): ground-truth signal state
        self.trajectory_sample_limit = 600  # only sample the first N raw steps -- keeps file size sane
        pgd_sample_every = 20  # only log full step traces for every Nth decision -- avoids huge files
                               # when pgd_steps > 1, since this logs every lane at every step

        for i in range(self.test_steps):
            if i % self.action_interval == 0:
                phases = np.stack([ag.get_phase() for ag in self.agents])
                actions = []
                fake_injected_this_decision = 0
                decision_idx = i // self.action_interval
                sample_this_decision = (
                    self.attacker.pgd_steps > 1 and decision_idx % pgd_sample_every == 0
                )

                # --- White-box FGSM: compute gradient through each victim's own
                # network and inject fake vehicles into the world BEFORE the
                # victim reads its observation for this decision. ---
                for idx, ag in enumerate(self.agents):
                    _plan, info = self.attacker.attack(
                        agent=ag,
                        obs=obs[idx],
                        phase=phases[idx],
                        world=self.world,
                        inject=True,
                        return_info=True,
                    )
                    total_attack_calls += 1
                    total_fake_injected += info['fake_vehicle_total']
                    fake_injected_this_decision += info['fake_vehicle_total']
                    if info['success']:
                        total_attack_successes += 1

                    fake_by_lane = info['fake_vehicle_plan'].get('lane_counts', {})
                    for lane_name, grad_value in info['lane_gradients'].items():
                        gradient_rows.append({
                            'sim_time': i,
                            'agent_idx': idx,
                            'lane': lane_name,
                            'gradient': grad_value,
                            'fake_vehicles_injected': fake_by_lane.get(lane_name, 0),
                        })

                    if sample_this_decision:
                        for step_record in info.get('step_history', []):
                            step_idx = step_record['step']
                            cumulative_before = step_record['cumulative_before']
                            step_allocation = step_record['step_allocation']
                            for lane_name, grad_value in step_record['lane_gradients'].items():
                                pgd_step_rows.append({
                                    'sim_time': i,
                                    'agent_idx': idx,
                                    'step': step_idx,
                                    'lane': lane_name,
                                    'gradient': grad_value,
                                    'cumulative_fake_before_this_step': cumulative_before.get(lane_name, 0),
                                    'fake_vehicles_this_step': step_allocation.get(lane_name, 0),
                                })

                # --- sample the space-time state WHILE the fakes still exist ---
                if i < self.trajectory_sample_limit:
                    position_rows.extend(self._sample_vehicle_positions(i))
                    signal_rows.extend(self._sample_signal_state(i))

                # victim reads its (now poisoned) observation and decides
                obs = [ag.get_ob() for ag in self.agents]
                for idx, ag in enumerate(self.agents):
                    actions.append(ag.get_action(obs[idx], phases[idx], test=True))

                # fake vehicles only exist to bias this decision -- remove them
                # before the real rollout, matching tsc_max_adversarial
                if hasattr(self.world, 'reset_fake_vehicles'):
                    self.world.reset_fake_vehicles()

                # --- snapshot real-traffic state for this decision, AFTER the
                # fake vehicles are gone, so these numbers reflect real traffic
                # only, not the injected ones. ---
                total_vehicles, avg_wait_time = self._snapshot_real_traffic()
                timeseries_rows.append({
                    'sim_time': i,
                    'fake_vehicles_injected': fake_injected_this_decision,
                    'total_vehicles': total_vehicles,
                    'avg_wait_time': avg_wait_time,
                })

                actions = np.stack(actions)
                rewards_list = []
                for j in range(self.action_interval):
                    obs, rewards, dones, _ = self.env.step(actions.flatten())
                    i += 1
                    rewards_list.append(np.stack(rewards))

                rewards = np.mean(rewards_list, axis=0)
                self.metric.update(rewards)
            if all(dones):
                break

        self.logger.info(
            "Final Travel Time is %.4f, mean rewards: %.4f, queue: %.4f, delay: %.4f, "
            "throughput: %d, fake vehicles injected: %d (attack calls: %d, successful: %d)" % (
                self.metric.real_average_travel_time(),
                self.metric.rewards(),
                self.metric.queue(),
                self.metric.delay(),
                self.metric.throughput(),
                total_fake_injected,
                total_attack_calls,
                total_attack_successes,
            )
        )

        csv_path = self._write_timeseries_csv(timeseries_rows)
        if csv_path:
            self.logger.info(f"Per-decision time series written to {csv_path}")

        grad_csv_path = self._write_csv(gradient_rows, 'fgsm_lane_gradients.csv')
        if grad_csv_path:
            self.logger.info(f"Per-lane gradient log written to {grad_csv_path}")

        pos_csv_path = self._write_csv(position_rows, 'fgsm_attack_positions.csv')
        if pos_csv_path:
            self.logger.info(f"Vehicle position trace written to {pos_csv_path}")

        sig_csv_path = self._write_csv(signal_rows, 'fgsm_attack_signal_state.csv')
        if sig_csv_path:
            self.logger.info(f"Signal state trace written to {sig_csv_path}")

        pgd_csv_path = self._write_csv(pgd_step_rows, 'fgsm_pgd_steps.csv')
        if pgd_csv_path:
            self.logger.info(f"PGD per-step refinement trace written to {pgd_csv_path}")

        if self.wandb is not None:
            self.wandb.log({
                'Test/Travel Time': self.metric.real_average_travel_time(),
                'Test/Mean Reward': self.metric.rewards(),
                'Test/Mean Queue': self.metric.queue(),
                'Test/Mean Delay': self.metric.delay(),
                'Test/Throughput': self.metric.throughput(),
                'Test/FakeVehiclesInjected': total_fake_injected,
            })
        elif self.comet is not None:
            self.comet.log_metrics({
                'Test/Travel Time': self.metric.real_average_travel_time(),
                'Test/Mean Reward': self.metric.rewards(),
                'Test/Mean Queue': self.metric.queue(),
                'Test/Mean Delay': self.metric.delay(),
                'Test/Throughput': self.metric.throughput(),
                'Test/FakeVehiclesInjected': total_fake_injected,
            })

        return self.metric

