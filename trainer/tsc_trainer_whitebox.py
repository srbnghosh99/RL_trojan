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
        # plain "tsc" task (same network/world/seed), never train here.
        model_path = Registry.mapping['logger_mapping']['path'].path.replace('tsc_whitebox', 'tsc')
        for ag in self.agents:
            ag.load_model(e=-1, model_path=model_path)

        attacker_config = Registry.mapping['attacker_mapping']['setting'].param
        self.attacker = FGSM(
            epsilon=float(attacker_config.get('epsilon', 0.15)),
            max_vehicles_per_lane=int(attacker_config.get('max_vehicles_per_lane', 8)),
            max_total_vehicles=attacker_config.get('max_total_vehicles', None),
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

    def train(self):
        pass

    def train_test(self, e):
        pass

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

        for i in range(self.test_steps):
            if i % self.action_interval == 0:
                phases = np.stack([ag.get_phase() for ag in self.agents])
                actions = []

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
                    if info['success']:
                        total_attack_successes += 1

                # victim reads its (now poisoned) observation and decides
                obs = [ag.get_ob() for ag in self.agents]
                for idx, ag in enumerate(self.agents):
                    actions.append(ag.get_action(obs[idx], phases[idx], test=True))

                # fake vehicles only exist to bias this decision -- remove them
                # before the real rollout, matching tsc_max_adversarial
                if hasattr(self.world, 'reset_fake_vehicles'):
                    self.world.reset_fake_vehicles()

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
