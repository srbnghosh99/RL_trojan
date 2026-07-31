"""
fgsm_attack.py

Simple white-box FGSM attack on the single-intersection DQN victim.

Standard FGSM:
    x_adv = x + epsilon * sign( grad_x  CE(Q(x), a_clean) )

Interpretation here: "a_clean" (the victim's own chosen phase) is treated as
the label. Nudging x in the direction that INCREASES this loss pushes the
observation away from whatever supports the clean decision -- i.e. it's an
untargeted attack whose goal is simply to flip the phase the agent picks.

Physicalization: the attacker's real-world action space is "add fake
vehicles to a lane's queue," not "add arbitrary signed noise to a feature
vector." So:
    1. Compute the raw FGSM perturbation in feature space.
    2. Keep only the positive part (you can't un-inject a real vehicle).
    3. Round up to whole vehicles and cap at a max-injectable-per-lane budget.
    4. Apply the resulting integer vehicle counts to the real queue state.
    5. Re-run the victim on the poisoned state and see whether its phase
       decision changed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from single_intersection_env import LANE_NAMES, NUM_LANES
from train_victim import QNetwork


@dataclass
class FGSMResult:
    clean_action: int
    adv_action: int
    flipped: bool
    epsilon: float
    fake_vehicles_by_lane: dict = field(default_factory=dict)
    total_fake_vehicles: int = 0
    clean_state: np.ndarray = None
    adv_state: np.ndarray = None


class SimpleFGSM:
    def __init__(
        self,
        q_net: QNetwork,
        state_norm: float,
        epsilon: float = 0.15,
        max_vehicles_per_lane: int = 8,
        device: str = "cpu",
    ):
        self.q_net = q_net
        self.state_norm = state_norm
        self.epsilon = epsilon
        self.max_vehicles_per_lane = max_vehicles_per_lane
        self.device = device

    def attack(self, raw_state: np.ndarray) -> FGSMResult:
        """
        raw_state: real (un-normalized) lane vehicle counts, shape (NUM_LANES,)
        """
        self.q_net.eval()

        x = torch.tensor(
            raw_state / self.state_norm, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        x.requires_grad_(True)

        # 1. clean forward pass
        q_values = self.q_net(x)
        clean_action = int(q_values.argmax(dim=1).item())

        # 2. white-box gradient of the loss w.r.t. the clean decision
        label = torch.tensor([clean_action], dtype=torch.int64, device=self.device)
        loss = F.cross_entropy(q_values, label)
        self.q_net.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad.zero_()
        loss.backward()
        grad_sign = x.grad.sign().squeeze(0).detach().cpu().numpy()  # shape (NUM_LANES,)

        # 3. physicalize: only positive (add-vehicle) perturbations are realizable
        raw_perturbation = self.epsilon * grad_sign  # in normalized-feature units
        positive_perturbation = np.clip(raw_perturbation, a_min=0.0, a_max=None)

        # convert normalized perturbation back into a vehicle count and round UP
        fake_vehicle_counts = np.ceil(positive_perturbation * self.state_norm).astype(int)
        fake_vehicle_counts = np.clip(fake_vehicle_counts, 0, self.max_vehicles_per_lane)

        # 4. apply to the real (un-normalized) state
        adv_raw_state = raw_state + fake_vehicle_counts

        # 5. re-run victim on poisoned observation
        with torch.no_grad():
            x_adv = torch.tensor(
                adv_raw_state / self.state_norm, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            adv_action = int(self.q_net(x_adv).argmax(dim=1).item())

        fake_by_lane = {
            LANE_NAMES[i]: int(fake_vehicle_counts[i])
            for i in range(NUM_LANES)
            if fake_vehicle_counts[i] > 0
        }

        return FGSMResult(
            clean_action=clean_action,
            adv_action=adv_action,
            flipped=(adv_action != clean_action),
            epsilon=self.epsilon,
            fake_vehicles_by_lane=fake_by_lane,
            total_fake_vehicles=int(fake_vehicle_counts.sum()),
            clean_state=raw_state.copy(),
            adv_state=adv_raw_state.copy(),
        )
