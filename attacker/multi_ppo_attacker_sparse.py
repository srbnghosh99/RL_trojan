"""
Multi-PPO Attacker - Sparse, On-Policy PPO Version.

This module implements a multi-action PPO attacker that:
1. Observes traffic state from the victim environment.
2. Selects an approach (N/E/S/W).
3. Selects an integer fake-vehicle count for every segment.
4. Injects fake vehicles to poison the victim controller's perception.
5. Learns from an externally computed reward, such as delay impact minus
   fake-vehicle cost.

Important changes from the previous implementation:
1. Scale actions are categorical integer counts in [0, max_vehicles].
   This removes Gaussian clipping and keeps executed actions consistent with
   their stored PPO log probabilities.
2. Generalized Advantage Estimation (GAE) is computed once for the ordered
   rollout before minibatch shuffling.
3. The rollout is discarded after PPO updates, preserving on-policy training.
4. Raw rewards are preserved. Only advantages are normalized, so the
   fake-vehicle penalty in the reward keeps a consistent meaning.
"""

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.registry import Registry


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Initialize a linear layer with orthogonal weights."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


@Registry.register_model("multi_ppo_attacker")
class MultiPPOAttacker:
    """
    Multi-action PPO attacker integrated with the TSC environment.

    Policies:
    - Approach policy: categorical choice among N/E/S/W.
    - Scale policy: one categorical integer vehicle count per segment.

    The reward is calculated outside this class, for example by
    ``SDSMInjector.calculate_reward()``, and passed to ``observe()``.
    """

    def __init__(self, world, rank, state_dim=None, **kwargs):
        self.world = world
        self.rank = rank
        self.intersection_id = world.intersection_ids[rank]

        param = kwargs.get("param", {})

        # PPO hyperparameters.
        self.actor_learning_rate = param.get("actor_learning_rate", 1e-4)
        self.critic_learning_rate = param.get("critic_learning_rate", 1e-4)
        self.gamma = param.get("gamma", 0.99)
        self.gae_lambda = param.get("gae_lambda", 0.95)
        self.clip_epsilon = param.get("clip_epsilon", 0.1)
        self.entropy_coef = param.get("entropy_coef", 0.01)
        self.max_grad_norm = param.get("max_grad_norm", 0.5)

        # Rollout and optimization settings.
        self.max_buffer_size = int(param.get("max_buffer_size", 256))
        self.ppo_epochs = int(param.get("ppo_epochs", 4))
        self.minibatch_size = int(param.get("minibatch_size", 64))
        self.target_kl = param.get("target_kl", None)
        if self.target_kl is not None:
            self.target_kl = float(self.target_kl)

        # Attack action settings.
        self.num_segments = int(param.get("num_segments", 2))
        self.num_approaches = int(param.get("num_approaches", 4))
        self.max_vehicles_per_segment = int(
            os.environ.get(
                "ATK_N_INJECT",
                param.get("max_vehicles_per_segment", 10),
            )
        )
        # self.penalty_lambda = param.get("penalty_lambda", 0.01)
        self.penalty_lambda = param.get("penalty_lambda", 0.50)

        device_name = Registry.mapping["command_mapping"]["setting"].param[
            "device"
        ]
        if device_name != "cpu" and not torch.cuda.is_available():
            device_name = "cpu"
        self.device = torch.device(device_name)

        from attacker.state_generator import AttackerStateGenerator

        self.state_gen = AttackerStateGenerator(
            world,
            world.id2intersection[self.intersection_id],
            num_segments=self.num_segments,
            num_approaches=self.num_approaches,
        )

        from attacker.sdsm_injector import SDSMInjector

        self.injector = SDSMInjector(
            world,
            self.intersection_id,
            max_vehicles_per_segment=self.max_vehicles_per_segment,
            num_segments=self.num_segments,
            penalty_lambda=self.penalty_lambda,
        )

        self.state_dim = self.state_gen.ob_length

        self.actor = MultiPPOActor(
            state_dim=self.state_dim,
            num_approaches=self.num_approaches,
            max_vehicles=self.max_vehicles_per_segment,
            num_segments=self.num_segments,
        ).to(self.device)

        self.critic = MultiPPOCritic(
            state_dim=self.state_dim,
        ).to(self.device)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.actor_learning_rate,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.critic_learning_rate,
        )

        # PPO uses a rollout buffer, not an off-policy replay buffer.
        # Tuple format:
        # (state, action, reward, next_state, done, old_log_prob, old_value)
        self.replay_buffer: List[Tuple] = []

        self.current_plan = None
        self._approaches = ["N", "E", "S", "W"]

    def reset(self):
        """Reset episode-specific attacker state."""
        self.replay_buffer = []
        self.current_plan = None
        self.injector.reset()

    def get_state(self):
        """Return the current attacker observation."""
        return self.state_gen.generate()

    def _get_distributions(self, states: torch.Tensor):
        """Construct the approach and per-segment scale distributions."""
        approach_logits, scale_logits = self.actor(states)

        approach_dist = torch.distributions.Categorical(
            logits=approach_logits,
        )
        scale_dist = torch.distributions.Categorical(
            logits=scale_logits,
        )

        return approach_dist, scale_dist, approach_logits, scale_logits

    def get_action(self, state, test=False):
        """
        Select an attack action.

        Returns:
            ((approach_action, scale_action), log_prob, value)

            approach_action:
                Integer approach index.
            scale_action:
                NumPy integer array of shape ``(num_segments,)``. Every value
                is in ``[0, max_vehicles_per_segment]``.
            log_prob:
                Log probability of exactly the returned joint action.
            value:
                Critic estimate V(state).
        """
        state_tensor = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            (
                approach_dist,
                scale_dist,
                approach_logits,
                scale_logits,
            ) = self._get_distributions(state_tensor)

            if test:
                approach_action = approach_logits.argmax(dim=-1)
                scale_action = scale_logits.argmax(dim=-1)
            else:
                approach_action = approach_dist.sample()
                scale_action = scale_dist.sample()

            # Joint log probability of the exact action returned to the caller.
            log_prob = (
                approach_dist.log_prob(approach_action)
                + scale_dist.log_prob(scale_action).sum(dim=-1)
            )

            value = self.critic(state_tensor).squeeze(-1)

        approach_action_int = int(approach_action.item())
        scale_action_array = (
            scale_action.squeeze(0)
            .cpu()
            .numpy()
            .astype(np.int64)
        )

        return (
            approach_action_int,
            scale_action_array,
        ), float(log_prob.item()), float(value.item())

    def get_value(self, state):
        """Return the critic estimate for one state."""
        with torch.no_grad():
            state_tensor = torch.as_tensor(
                state,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)
            return self.critic(state_tensor).squeeze(0).cpu().numpy()

    def compute_advantage(
        self,
        rewards: Sequence[float],
        values: Sequence[float],
        next_values: Sequence[float],
        dones: Sequence[bool],
    ):
        """
        Compute Generalized Advantage Estimation over an ordered rollout.

        This method keeps the original public name for compatibility, but now
        performs true GAE using ``gae_lambda``.

        Returns:
            NumPy array of advantages.
        """
        rewards_array = np.asarray(rewards, dtype=np.float32)
        values_array = np.asarray(values, dtype=np.float32)
        next_values_array = np.asarray(next_values, dtype=np.float32)
        dones_array = np.asarray(dones, dtype=np.float32)

        if not (
            len(rewards_array)
            == len(values_array)
            == len(next_values_array)
            == len(dones_array)
        ):
            raise ValueError(
                "rewards, values, next_values, and dones must have equal length"
            )

        advantages = np.zeros_like(rewards_array, dtype=np.float32)
        gae = 0.0

        for index in reversed(range(len(rewards_array))):
            nonterminal = 1.0 - dones_array[index]
            delta = (
                rewards_array[index]
                + self.gamma * next_values_array[index] * nonterminal
                - values_array[index]
            )
            gae = (
                delta
                + self.gamma
                * self.gae_lambda
                * nonterminal
                * gae
            )
            advantages[index] = gae

        return advantages

    def _prepare_rollout(self, rollout: Sequence[Tuple]) -> Dict[str, torch.Tensor]:
        """Create fixed PPO targets from one ordered on-policy rollout."""
        if len(rollout) == 0:
            raise ValueError("Cannot prepare an empty rollout")

        states_np = np.asarray([transition[0] for transition in rollout])
        next_states_np = np.asarray([transition[3] for transition in rollout])

        rewards_np = np.asarray(
            [transition[2] for transition in rollout],
            dtype=np.float32,
        )
        dones_np = np.asarray(
            [transition[4] for transition in rollout],
            dtype=np.float32,
        )
        old_log_probs_np = np.asarray(
            [transition[5] for transition in rollout],
            dtype=np.float32,
        )
        old_values_np = np.asarray(
            [transition[6] for transition in rollout],
            dtype=np.float32,
        )

        approach_actions_np = np.asarray(
            [transition[1][0] for transition in rollout],
            dtype=np.int64,
        )
        scale_actions_np = np.asarray(
            [transition[1][1] for transition in rollout],
            dtype=np.int64,
        )

        next_states_tensor = torch.as_tensor(
            next_states_np,
            dtype=torch.float32,
            device=self.device,
        )

        # Values must be fixed before any optimizer updates.
        with torch.no_grad():
            next_values_np = (
                self.critic(next_states_tensor)
                .squeeze(-1)
                .cpu()
                .numpy()
                .astype(np.float32)
            )

        advantages_np = self.compute_advantage(
            rewards=rewards_np,
            values=old_values_np,
            next_values=next_values_np,
            dones=dones_np,
        )
        returns_np = advantages_np + old_values_np

        # Preserve raw rewards and returns. Normalize only the advantages.
        advantage_mean = float(advantages_np.mean())
        advantage_std = float(advantages_np.std())
        advantages_np = (
            advantages_np - advantage_mean
        ) / (advantage_std + 1e-8)

        return {
            "states": torch.as_tensor(
                states_np,
                dtype=torch.float32,
                device=self.device,
            ),
            "approach_actions": torch.as_tensor(
                approach_actions_np,
                dtype=torch.long,
                device=self.device,
            ),
            "scale_actions": torch.as_tensor(
                scale_actions_np,
                dtype=torch.long,
                device=self.device,
            ),
            "old_log_probs": torch.as_tensor(
                old_log_probs_np,
                dtype=torch.float32,
                device=self.device,
            ),
            "old_values": torch.as_tensor(
                old_values_np,
                dtype=torch.float32,
                device=self.device,
            ),
            "advantages": torch.as_tensor(
                advantages_np,
                dtype=torch.float32,
                device=self.device,
            ),
            "returns": torch.as_tensor(
                returns_np,
                dtype=torch.float32,
                device=self.device,
            ),
            "rewards": torch.as_tensor(
                rewards_np,
                dtype=torch.float32,
                device=self.device,
            ),
        }

    def _train_minibatch(
        self,
        rollout_data: Dict[str, torch.Tensor],
        indices: np.ndarray,
    ) -> Dict[str, float]:
        """Perform one PPO actor update and one critic update."""
        index_tensor = torch.as_tensor(
            indices,
            dtype=torch.long,
            device=self.device,
        )

        states = rollout_data["states"][index_tensor]
        approach_actions = rollout_data["approach_actions"][index_tensor]
        scale_actions = rollout_data["scale_actions"][index_tensor]
        old_log_probs = rollout_data["old_log_probs"][index_tensor]
        advantages = rollout_data["advantages"][index_tensor]
        returns = rollout_data["returns"][index_tensor]

        # ----- Critic update -----
        values = self.critic(states).squeeze(-1)
        critic_loss = F.mse_loss(values, returns)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(),
            self.max_grad_norm,
        )
        self.critic_optimizer.step()

        # ----- Actor update -----
        (
            approach_dist,
            scale_dist,
            _,
            _,
        ) = self._get_distributions(states)

        new_log_probs = (
            approach_dist.log_prob(approach_actions)
            + scale_dist.log_prob(scale_actions).sum(dim=-1)
        )

        log_ratio = new_log_probs - old_log_probs
        ratios = torch.exp(log_ratio)

        surrogate_unclipped = ratios * advantages
        surrogate_clipped = torch.clamp(
            ratios,
            1.0 - self.clip_epsilon,
            1.0 + self.clip_epsilon,
        ) * advantages

        policy_loss = -torch.min(
            surrogate_unclipped,
            surrogate_clipped,
        ).mean()

        # Mean scale entropy prevents the number of segments from changing the
        # effective magnitude of entropy_coef.
        approach_entropy = approach_dist.entropy().mean()
        scale_entropy = scale_dist.entropy().mean()
        entropy = approach_entropy + scale_entropy

        actor_loss = policy_loss - self.entropy_coef * entropy

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(),
            self.max_grad_norm,
        )
        self.actor_optimizer.step()

        with torch.no_grad():
            approximate_kl = ((ratios - 1.0) - log_ratio).mean()
            clip_fraction = (
                torch.abs(ratios - 1.0) > self.clip_epsilon
            ).float().mean()

            returns_variance = torch.var(returns, unbiased=False)
            explained_variance = 1.0 - (
                torch.var(returns - values, unbiased=False)
                / (returns_variance + 1e-8)
            )

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "policy_loss": float(policy_loss.item()),
            "entropy": float(entropy.item()),
            "approx_kl": float(approximate_kl.item()),
            "clip_fraction": float(clip_fraction.item()),
            "explained_variance": float(explained_variance.item()),
            "ratio_mean": float(ratios.mean().item()),
            "value_mean": float(values.mean().item()),
            "return_mean": float(returns.mean().item()),
        }

    def train(self, batch):
        """
        Backward-compatible one-pass training method.

        ``update_policy()`` should normally be used because it computes one
        ordered rollout and performs multiple shuffled PPO minibatch updates.
        """
        if len(batch) == 0:
            return {
                "critic_loss": 0.0,
                "actor_loss": 0.0,
                "policy_loss": 0.0,
                "entropy": 0.0,
                "approx_kl": 0.0,
                "clip_fraction": 0.0,
                "explained_variance": 0.0,
                "ratio_mean": 1.0,
                "value_mean": 0.0,
                "return_mean": 0.0,
            }

        rollout_data = self._prepare_rollout(batch)
        indices = np.arange(len(batch), dtype=np.int64)
        return self._train_minibatch(rollout_data, indices)

    def update_policy(
        self,
        num_updates: Optional[int] = None,
    ):
        """
        Update PPO from one fresh rollout and then discard that rollout.

        Args:
            num_updates:
                Number of PPO epochs. When omitted, ``ppo_epochs`` from the
                configuration is used.

        Returns:
            Mean diagnostics across PPO minibatch updates, or ``None`` until
            enough rollout samples have been collected.
        """
        if len(self.replay_buffer) < self.max_buffer_size:
            return None

        # All samples were collected before this update, so they belong to the
        # same behavior-policy version. Train on them and then discard them.
        rollout = list(self.replay_buffer)
        rollout_data = self._prepare_rollout(rollout)

        epochs = self.ppo_epochs if num_updates is None else int(num_updates)
        epochs = max(1, epochs)
        minibatch_size = max(
            1,
            min(self.minibatch_size, len(rollout)),
        )

        losses: List[Dict[str, float]] = []
        stop_early = False

        for _ in range(epochs):
            permutation = np.random.permutation(len(rollout))

            for start in range(0, len(rollout), minibatch_size):
                minibatch_indices = permutation[
                    start : start + minibatch_size
                ]
                loss_dict = self._train_minibatch(
                    rollout_data,
                    minibatch_indices,
                )
                losses.append(loss_dict)

                if (
                    self.target_kl is not None
                    and loss_dict["approx_kl"] > self.target_kl
                ):
                    stop_early = True
                    break

            if stop_early:
                break

        # Critical on-policy step: never reuse this rollout after the policy
        # has been updated.
        self.replay_buffer.clear()

        if not losses:
            return None

        result = {
            key: float(np.mean([loss[key] for loss in losses]))
            for key in losses[0]
        }
        result["num_gradient_updates"] = len(losses)
        result["early_stop_kl"] = float(stop_early)
        result["rollout_reward_mean"] = float(
            rollout_data["rewards"].mean().item()
        )
        result["rollout_fake_count_mean"] = float(
            np.mean(
                [
                    np.asarray(transition[1][1], dtype=np.float32).sum()
                    for transition in rollout
                ]
            )
        )
        return result

    def observe(
        self,
        state,
        action,
        log_prob_value,
        reward,
        next_state,
        done,
    ):
        """Store one transition in the current on-policy rollout."""
        log_prob, value = log_prob_value

        approach_action = int(action[0])
        scale_action = np.asarray(action[1], dtype=np.int64)

        if scale_action.shape != (self.num_segments,):
            raise ValueError(
                "scale_action must have shape "
                f"({self.num_segments},), got {scale_action.shape}"
            )
        if np.any(scale_action < 0) or np.any(
            scale_action > self.max_vehicles_per_segment
        ):
            raise ValueError(
                "scale_action contains an invalid fake-vehicle count"
            )

        self.replay_buffer.append(
            (
                np.asarray(state, dtype=np.float32),
                (approach_action, scale_action.copy()),
                float(reward),
                np.asarray(next_state, dtype=np.float32),
                bool(done),
                float(log_prob),
                float(value),
            )
        )

    def save_model(self, path):
        """Save actor, critic, and optimizer states."""
        if not os.path.exists(path):
            os.makedirs(path)

        checkpoint_path = os.path.join(path, f"best_{self.rank}.pth")
        torch.save(
            {
                "format_version": 2,
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "max_vehicles_per_segment": self.max_vehicles_per_segment,
                "num_segments": self.num_segments,
            },
            checkpoint_path,
        )

    def load_model(self, path):
        """Load a checkpoint created by this categorical-scale version."""
        checkpoint = torch.load(path, map_location="cpu")

        try:
            self.actor.load_state_dict(checkpoint["actor"])
        except RuntimeError as error:
            raise RuntimeError(
                "This actor now uses categorical integer scale actions. "
                "A checkpoint created by the previous Gaussian-scale actor "
                "is not architecture-compatible; train a new attacker model."
            ) from error

        self.critic.load_state_dict(checkpoint["critic"])

        if "optimizer" in checkpoint:
            try:
                self.actor_optimizer.load_state_dict(
                    checkpoint["optimizer"]
                )
            except (ValueError, RuntimeError):
                # Model weights remain usable even if an optimizer state was
                # created under a different optimizer configuration.
                pass

        if "critic_optimizer" in checkpoint:
            try:
                self.critic_optimizer.load_state_dict(
                    checkpoint["critic_optimizer"]
                )
            except (ValueError, RuntimeError):
                pass

        self.actor.to(self.device)
        self.critic.to(self.device)

    def cleanup(self):
        """Remove all injected fake vehicles from the simulation."""
        self.injector.cleanup_injected_vehicles()


class MultiPPOActor(nn.Module):
    """
    Actor with separate categorical approach and scale heads.

    - Approach logits shape: ``[batch, num_approaches]``.
    - Scale logits shape:
      ``[batch, num_segments, max_vehicles + 1]``.

    Each scale category is the exact integer count to inject into that segment.
    """

    def __init__(
        self,
        state_dim,
        num_approaches,
        max_vehicles,
        num_segments,
    ):
        super().__init__()
        self.num_approaches = num_approaches
        self.max_vehicles = max_vehicles
        self.num_segments = num_segments
        self.num_scale_actions = max_vehicles + 1

        self.encoder_approach = nn.Sequential(
            layer_init(nn.Linear(state_dim, state_dim * 3)),
            nn.ReLU(),
            layer_init(nn.Linear(state_dim * 3, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
        )

        self.encoder_actor = nn.Sequential(
            layer_init(nn.Linear(state_dim, state_dim * 3)),
            nn.ReLU(),
            layer_init(nn.Linear(state_dim * 3, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
        )

        self.approach_actor = nn.Sequential(
            layer_init(nn.Linear(256, 128)),
            nn.ReLU(),
            layer_init(
                nn.Linear(128, num_approaches),
                std=0.01,
            ),
        )

        # One categorical distribution per segment. Categories correspond to
        # exact counts: 0, 1, ..., max_vehicles.
        self.scale_actor = nn.Sequential(
            layer_init(nn.Linear(256, 128)),
            nn.ReLU(),
            layer_init(
                nn.Linear(
                    128,
                    num_segments * self.num_scale_actions,
                ),
                std=0.01,
            ),
        )

    def forward(self, state):
        approach_features = self.encoder_approach(state)
        scale_features = self.encoder_actor(state)

        approach_logits = self.approach_actor(approach_features)
        scale_logits = self.scale_actor(scale_features)
        scale_logits = scale_logits.view(
            -1,
            self.num_segments,
            self.num_scale_actions,
        )

        return approach_logits, scale_logits


class MultiPPOCritic(nn.Module):
    """Critic network for estimating V(state)."""

    def __init__(self, state_dim):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Linear(state_dim, state_dim * 3)),
            nn.ReLU(),
            layer_init(nn.Linear(state_dim * 3, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 1), std=1.0),
        )

    def forward(self, state):
        return self.network(state)
