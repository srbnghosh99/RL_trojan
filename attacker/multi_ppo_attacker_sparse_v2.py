"""
Multi-PPO Attacker - Sparse Hierarchical PPO Version.

This version keeps the public attacker action format compatible with the
existing runner:

    ((approach_index, segment_counts), log_probability, value)

The main changes relative to the previous categorical PPO version are:

1. Hierarchical attack gate:
   The policy first chooses ATTACK or NO-ATTACK. A no-attack decision produces
   zero injected vehicles and its log probability depends only on the gate.

2. Sparse scale prior:
   Scale-head logits are initialized to prefer smaller integer counts. This is
   an initialization preference, not a hard action restriction.

3. Expected-count regularization:
   The actor loss contains a small differentiable penalty on the expected
   number of fake vehicles. This complements the environment reward and helps
   the policy discover low-count attacks when the reward signal is noisy.

4. Separate entropy coefficients and clipped value loss:
   Exploration of attack timing, approach, and scale can be controlled
   independently. The scale entropy coefficient can be kept low so entropy
   does not continually encourage large vehicle counts. PPO-style value
   clipping improves critic stability.

The environment reward is still calculated externally, for example by
SDSMInjector.calculate_reward(), and passed through observe().
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
    """Sparse hierarchical PPO attacker for traffic-signal perception attacks."""

    def __init__(self, world, rank, state_dim=None, **kwargs):
        self.world = world
        self.rank = rank
        self.intersection_id = world.intersection_ids[rank]

        param = kwargs.get("param", {})

        # Core PPO hyperparameters.
        self.actor_learning_rate = float(
            param.get("actor_learning_rate", 1e-4)
        )
        self.critic_learning_rate = float(
            param.get("critic_learning_rate", 1e-4)
        )
        self.gamma = float(param.get("gamma", 0.99))
        self.gae_lambda = float(param.get("gae_lambda", 0.95))
        self.clip_epsilon = float(param.get("clip_epsilon", 0.1))
        self.value_clip_epsilon = float(
            param.get("value_clip_epsilon", self.clip_epsilon)
        )
        self.max_grad_norm = float(param.get("max_grad_norm", 0.5))

        # Separate exploration strengths. Scale entropy is intentionally lower
        # by default because a large scale entropy continuously encourages all
        # count categories, including unnecessarily high counts.
        legacy_entropy_coef = float(param.get("entropy_coef", 0.01))
        self.attack_entropy_coef = float(
            param.get("attack_entropy_coef", legacy_entropy_coef * 0.5)
        )
        self.approach_entropy_coef = float(
            param.get("approach_entropy_coef", legacy_entropy_coef)
        )
        self.scale_entropy_coef = float(
            param.get("scale_entropy_coef", legacy_entropy_coef * 0.1)
        )

        # Direct differentiable pressure toward lower expected fake counts.
        # Keep this small because the environment reward already penalizes
        # actual fake vehicles.
        self.sparsity_coef = float(param.get("sparsity_coef", 0.01))

        # Initial policy biases. These do not hard-code a solution; training
        # can overcome them when larger attacks produce enough additional reward.
        self.sparse_prior_strength = float(
            param.get("sparse_prior_strength", 1.0)
        )
        self.no_attack_prior_strength = float(
            param.get("no_attack_prior_strength", 0.5)
        )

        # Rollout and optimization settings.
        self.max_buffer_size = int(param.get("max_buffer_size", 256))
        self.ppo_epochs = int(param.get("ppo_epochs", 4))
        self.minibatch_size = int(param.get("minibatch_size", 64))
        self.target_kl = param.get("target_kl", 0.02)
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
        self.penalty_lambda = float(param.get("penalty_lambda", 0.01))

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
            sparse_prior_strength=self.sparse_prior_strength,
            no_attack_prior_strength=self.no_attack_prior_strength,
        ).to(self.device)

        self.critic = MultiPPOCritic(state_dim=self.state_dim).to(self.device)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.actor_learning_rate,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.critic_learning_rate,
        )

        # Stored transition format:
        # (
        #   state,
        #   (approach_action, scale_action),
        #   attack_action,
        #   reward,
        #   next_state,
        #   done,
        #   old_log_probability,
        #   old_value,
        # )
        self.replay_buffer: List[Tuple] = []

        # get_action() and observe() are assumed to be called sequentially.
        # The public action remains backward-compatible, while this internal
        # field preserves the hierarchical gate action for PPO training.
        self._pending_attack_action: Optional[int] = None

        self.current_plan = None
        self._approaches = ["N", "E", "S", "W"]

    def reset(self):
        """Reset episode-specific state."""
        self.replay_buffer = []
        self._pending_attack_action = None
        self.current_plan = None
        self.injector.reset()

    def get_state(self):
        """Return the current attacker observation."""
        return self.state_gen.generate()

    def _get_distributions(self, states: torch.Tensor):
        """Create attack-gate, approach, and segment-count distributions."""
        attack_logits, approach_logits, scale_logits = self.actor(states)

        attack_dist = torch.distributions.Categorical(logits=attack_logits)
        approach_dist = torch.distributions.Categorical(logits=approach_logits)
        scale_dist = torch.distributions.Categorical(logits=scale_logits)

        return (
            attack_dist,
            approach_dist,
            scale_dist,
            attack_logits,
            approach_logits,
            scale_logits,
        )

    def get_action(self, state, test=False):
        """
        Select an attack action while preserving the original public format.

        Returns:
            ((approach_index, integer_segment_counts), log_probability, value)

        Hierarchical probability:
            P(no attack | state)

        or

            P(attack | state)
            * P(approach | state)
            * product_j P(segment_count_j | state)
        """
        state_tensor = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            (
                attack_dist,
                approach_dist,
                scale_dist,
                attack_logits,
                approach_logits,
                scale_logits,
            ) = self._get_distributions(state_tensor)

            if test:
                attack_action = attack_logits.argmax(dim=-1)
            else:
                attack_action = attack_dist.sample()

            attack_action_int = int(attack_action.item())

            if attack_action_int == 0:
                # No-attack branch is a single degenerate action. Approach and
                # scale probabilities are intentionally excluded from log_prob.
                approach_action = torch.zeros(
                    1,
                    dtype=torch.long,
                    device=self.device,
                )
                scale_action = torch.zeros(
                    (1, self.num_segments),
                    dtype=torch.long,
                    device=self.device,
                )
                log_prob = attack_dist.log_prob(attack_action)
            else:
                if test:
                    approach_action = approach_logits.argmax(dim=-1)
                    scale_action = scale_logits.argmax(dim=-1)
                else:
                    approach_action = approach_dist.sample()
                    scale_action = scale_dist.sample()

                log_prob = (
                    attack_dist.log_prob(attack_action)
                    + approach_dist.log_prob(approach_action)
                    + scale_dist.log_prob(scale_action).sum(dim=-1)
                )

            value = self.critic(state_tensor).squeeze(-1)

        self._pending_attack_action = attack_action_int

        approach_action_int = int(approach_action.item())
        scale_action_array = (
            scale_action.squeeze(0).cpu().numpy().astype(np.int64)
        )

        return (
            approach_action_int,
            scale_action_array,
        ), float(log_prob.item()), float(value.item())

    def get_value(self, state):
        """Return V(state) for one observation."""
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
        """Compute Generalized Advantage Estimation on an ordered rollout."""
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

    @staticmethod
    def _decode_transition(transition):
        """
        Decode new and previous categorical-rollout tuple formats.

        New format has eight entries and stores attack_action explicitly.
        Seven-entry transitions are accepted for limited backward compatibility;
        their gate is inferred from whether any fake vehicles were requested.
        """
        if len(transition) == 8:
            (
                state,
                action,
                attack_action,
                reward,
                next_state,
                done,
                old_log_prob,
                old_value,
            ) = transition
        elif len(transition) == 7:
            (
                state,
                action,
                reward,
                next_state,
                done,
                old_log_prob,
                old_value,
            ) = transition
            attack_action = int(np.asarray(action[1]).sum() > 0)
        else:
            raise ValueError(
                "Unexpected transition format; expected 7 or 8 entries"
            )

        return {
            "state": state,
            "action": action,
            "attack_action": int(attack_action),
            "reward": float(reward),
            "next_state": next_state,
            "done": bool(done),
            "old_log_prob": float(old_log_prob),
            "old_value": float(old_value),
        }

    def _prepare_rollout(self, rollout: Sequence[Tuple]) -> Dict[str, torch.Tensor]:
        """Build fixed returns and advantages from one on-policy rollout."""
        if len(rollout) == 0:
            raise ValueError("Cannot prepare an empty rollout")

        decoded = [self._decode_transition(t) for t in rollout]

        states_np = np.asarray([t["state"] for t in decoded])
        next_states_np = np.asarray([t["next_state"] for t in decoded])
        rewards_np = np.asarray(
            [t["reward"] for t in decoded], dtype=np.float32
        )
        dones_np = np.asarray(
            [t["done"] for t in decoded], dtype=np.float32
        )
        old_log_probs_np = np.asarray(
            [t["old_log_prob"] for t in decoded], dtype=np.float32
        )
        old_values_np = np.asarray(
            [t["old_value"] for t in decoded], dtype=np.float32
        )
        attack_actions_np = np.asarray(
            [t["attack_action"] for t in decoded], dtype=np.int64
        )
        approach_actions_np = np.asarray(
            [t["action"][0] for t in decoded], dtype=np.int64
        )
        scale_actions_np = np.asarray(
            [t["action"][1] for t in decoded], dtype=np.int64
        )

        next_states_tensor = torch.as_tensor(
            next_states_np,
            dtype=torch.float32,
            device=self.device,
        )

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
            "attack_actions": torch.as_tensor(
                attack_actions_np,
                dtype=torch.long,
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
        """Perform one PPO actor update and one clipped critic update."""
        index_tensor = torch.as_tensor(
            indices,
            dtype=torch.long,
            device=self.device,
        )

        states = rollout_data["states"][index_tensor]
        attack_actions = rollout_data["attack_actions"][index_tensor]
        approach_actions = rollout_data["approach_actions"][index_tensor]
        scale_actions = rollout_data["scale_actions"][index_tensor]
        old_log_probs = rollout_data["old_log_probs"][index_tensor]
        old_values = rollout_data["old_values"][index_tensor]
        advantages = rollout_data["advantages"][index_tensor]
        returns = rollout_data["returns"][index_tensor]

        # ----- Critic update with PPO-style value clipping -----
        values = self.critic(states).squeeze(-1)
        value_loss_unclipped = torch.square(values - returns)
        values_clipped = old_values + torch.clamp(
            values - old_values,
            -self.value_clip_epsilon,
            self.value_clip_epsilon,
        )
        value_loss_clipped = torch.square(values_clipped - returns)
        critic_loss = 0.5 * torch.max(
            value_loss_unclipped,
            value_loss_clipped,
        ).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(),
            self.max_grad_norm,
        )
        self.critic_optimizer.step()

        # ----- Hierarchical actor update -----
        (
            attack_dist,
            approach_dist,
            scale_dist,
            _,
            _,
            _,
        ) = self._get_distributions(states)

        attack_log_probs = attack_dist.log_prob(attack_actions)
        conditional_log_probs = (
            approach_dist.log_prob(approach_actions)
            + scale_dist.log_prob(scale_actions).sum(dim=-1)
        )

        attack_mask = attack_actions.float()
        new_log_probs = (
            attack_log_probs
            + attack_mask * conditional_log_probs
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

        # Hierarchical entropy. Conditional entropies matter only in proportion
        # to the current probability of choosing the attack branch.
        attack_probability = attack_dist.probs[:, 1]
        attack_entropy = attack_dist.entropy().mean()
        approach_entropy_per_sample = approach_dist.entropy()
        scale_entropy_per_sample = scale_dist.entropy().sum(dim=-1)

        weighted_approach_entropy = (
            attack_probability * approach_entropy_per_sample
        ).mean()
        weighted_scale_entropy = (
            attack_probability * scale_entropy_per_sample
        ).mean()

        entropy_bonus = (
            self.attack_entropy_coef * attack_entropy
            + self.approach_entropy_coef * weighted_approach_entropy
            + self.scale_entropy_coef * weighted_scale_entropy
        )

        # Differentiable expected fake-count regularizer.
        count_values = torch.arange(
            self.max_vehicles_per_segment + 1,
            dtype=torch.float32,
            device=self.device,
        )
        expected_counts_per_segment = (
            scale_dist.probs * count_values
        ).sum(dim=-1)
        expected_total_if_attack = expected_counts_per_segment.sum(dim=-1)
        max_total = max(
            self.num_segments * self.max_vehicles_per_segment,
            1,
        )
        expected_fake_fraction = (
            attack_probability * expected_total_if_attack / max_total
        ).mean()

        actor_loss = (
            policy_loss
            - entropy_bonus
            + self.sparsity_coef * expected_fake_fraction
        )

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

            selected_fake_count = scale_actions.sum(dim=-1).float()
            selected_fake_count = selected_fake_count * attack_mask

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "policy_loss": float(policy_loss.item()),
            "entropy_bonus": float(entropy_bonus.item()),
            "attack_entropy": float(attack_entropy.item()),
            "approach_entropy": float(weighted_approach_entropy.item()),
            "scale_entropy": float(weighted_scale_entropy.item()),
            "sparsity_loss": float(
                (self.sparsity_coef * expected_fake_fraction).item()
            ),
            "expected_fake_fraction": float(
                expected_fake_fraction.item()
            ),
            "selected_fake_count_mean": float(
                selected_fake_count.mean().item()
            ),
            "attack_probability_mean": float(
                attack_probability.mean().item()
            ),
            "attack_rate_batch": float(attack_mask.mean().item()),
            "approx_kl": float(approximate_kl.item()),
            "clip_fraction": float(clip_fraction.item()),
            "explained_variance": float(explained_variance.item()),
            "ratio_mean": float(ratios.mean().item()),
            "value_mean": float(values.mean().item()),
            "return_mean": float(returns.mean().item()),
        }

    def train(self, batch):
        """Backward-compatible one-pass training entry point."""
        if len(batch) == 0:
            return {
                "critic_loss": 0.0,
                "actor_loss": 0.0,
                "policy_loss": 0.0,
                "entropy_bonus": 0.0,
                "attack_entropy": 0.0,
                "approach_entropy": 0.0,
                "scale_entropy": 0.0,
                "sparsity_loss": 0.0,
                "expected_fake_fraction": 0.0,
                "selected_fake_count_mean": 0.0,
                "attack_probability_mean": 0.0,
                "attack_rate_batch": 0.0,
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

    def update_policy(self, num_updates: Optional[int] = None):
        """Train on one fresh rollout, then discard it."""
        if len(self.replay_buffer) < self.max_buffer_size:
            return None

        rollout = list(self.replay_buffer)
        rollout_data = self._prepare_rollout(rollout)

        epochs = self.ppo_epochs if num_updates is None else int(num_updates)
        epochs = max(1, epochs)
        minibatch_size = max(1, min(self.minibatch_size, len(rollout)))

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
        result["rollout_attack_rate"] = float(
            rollout_data["attack_actions"].float().mean().item()
        )
        result["rollout_fake_count_mean"] = float(
            (
                rollout_data["scale_actions"].sum(dim=-1).float()
                * rollout_data["attack_actions"].float()
            ).mean().item()
        )
        result["rollout_reward_per_fake"] = float(
            rollout_data["rewards"].sum().item()
            / max(
                (
                    rollout_data["scale_actions"].sum(dim=-1).float()
                    * rollout_data["attack_actions"].float()
                ).sum().item(),
                1.0,
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

        if self._pending_attack_action is None:
            # Safe fallback for nonstandard runners that do not call observe()
            # immediately after get_action().
            attack_action = int(scale_action.sum() > 0)
        else:
            attack_action = int(self._pending_attack_action)

        if attack_action == 0 and scale_action.sum() != 0:
            raise ValueError(
                "No-attack gate produced a nonzero scale action"
            )

        self.replay_buffer.append(
            (
                np.asarray(state, dtype=np.float32),
                (approach_action, scale_action.copy()),
                attack_action,
                float(reward),
                np.asarray(next_state, dtype=np.float32),
                bool(done),
                float(log_prob),
                float(value),
            )
        )

        self._pending_attack_action = None

    def save_model(self, path):
        """Save actor, critic, optimizer, and sparse-policy metadata."""
        if not os.path.exists(path):
            os.makedirs(path)

        checkpoint_path = os.path.join(path, f"best_{self.rank}.pth")
        torch.save(
            {
                "format_version": 3,
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "max_vehicles_per_segment": self.max_vehicles_per_segment,
                "num_segments": self.num_segments,
                "sparsity_coef": self.sparsity_coef,
                "sparse_prior_strength": self.sparse_prior_strength,
                "no_attack_prior_strength": self.no_attack_prior_strength,
            },
            checkpoint_path,
        )

    def load_model(self, path):
        """Load a checkpoint created by this hierarchical sparse version."""
        checkpoint = torch.load(path, map_location="cpu")

        try:
            self.actor.load_state_dict(checkpoint["actor"])
        except RuntimeError as error:
            raise RuntimeError(
                "This actor contains a hierarchical attack gate and sparse "
                "categorical scale head. Older Gaussian or non-gated "
                "checkpoints are not architecture-compatible; train a new "
                "attacker checkpoint."
            ) from error

        self.critic.load_state_dict(checkpoint["critic"])

        if "optimizer" in checkpoint:
            try:
                self.actor_optimizer.load_state_dict(checkpoint["optimizer"])
            except (ValueError, RuntimeError):
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
    """Hierarchical actor: attack gate, approach, and integer segment counts."""

    def __init__(
        self,
        state_dim,
        num_approaches,
        max_vehicles,
        num_segments,
        sparse_prior_strength=1.0,
        no_attack_prior_strength=0.5,
    ):
        super().__init__()
        self.num_approaches = int(num_approaches)
        self.max_vehicles = int(max_vehicles)
        self.num_segments = int(num_segments)
        self.num_scale_actions = self.max_vehicles + 1

        # Smaller, shared representation is generally more data-efficient for
        # a short 256-step on-policy rollout than the previous deep networks.
        self.shared_encoder = nn.Sequential(
            layer_init(nn.Linear(state_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
        )

        self.attack_head = nn.Sequential(
            layer_init(nn.Linear(128, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 2), std=0.01),
        )

        self.approach_head = nn.Sequential(
            layer_init(nn.Linear(128, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, self.num_approaches), std=0.01),
        )

        self.scale_hidden = nn.Sequential(
            layer_init(nn.Linear(128, 64)),
            nn.Tanh(),
        )
        self.scale_output = layer_init(
            nn.Linear(
                64,
                self.num_segments * self.num_scale_actions,
            ),
            std=0.01,
        )

        # Initial preference for NO-ATTACK without preventing the policy from
        # learning frequent attacks when they are profitable.
        with torch.no_grad():
            final_attack_layer = self.attack_head[-1]
            final_attack_layer.bias[0] = float(no_attack_prior_strength)
            final_attack_layer.bias[1] = 0.0

            # Repeated bias vector [0, -k, -2k, ...] for every segment.
            if self.max_vehicles > 0:
                normalized_counts = torch.arange(
                    self.num_scale_actions,
                    dtype=torch.float32,
                ) / float(self.max_vehicles)
            else:
                normalized_counts = torch.zeros(
                    self.num_scale_actions,
                    dtype=torch.float32,
                )
            count_bias = -float(sparse_prior_strength) * normalized_counts
            repeated_bias = count_bias.repeat(self.num_segments)
            self.scale_output.bias.copy_(repeated_bias)

    def forward(self, state):
        features = self.shared_encoder(state)

        attack_logits = self.attack_head(features)
        approach_logits = self.approach_head(features)

        scale_features = self.scale_hidden(features)
        scale_logits = self.scale_output(scale_features)
        scale_logits = scale_logits.view(
            -1,
            self.num_segments,
            self.num_scale_actions,
        )

        return attack_logits, approach_logits, scale_logits


class MultiPPOCritic(nn.Module):
    """Compact critic for estimating V(state)."""

    def __init__(self, state_dim):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Linear(state_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 1), std=1.0),
        )

    def forward(self, state):
        return self.network(state)
