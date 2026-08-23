"""HMVFE coordinator network for MATE — variant D (Tier B: critic reduction).

Same FM + Mixture-of-Experts coordinator as ``hmvfe_mate_b`` (7 discretised
fields per pair, sigmoid per-pair actor), but the **state-value critic is
configurable** instead of the paper's fixed parameter-free ``max`` over
interaction scores:

    critic_reduction =
      'max'     -> v = max_ij(z_ij)         # paper default (Sec. 4.1.3), actor-coupled
      'mean'    -> v = mean_ij(z_ij)         # smoother param-free baseline, actor-coupled
      'learned' -> v = ValueHead(pool(e*, u*))  # separate head on the shared trunk (DEFAULT)

Motivation (Tier B): the ``max`` reduction is a lossy, high-variance baseline and
shares *all* parameters with the actor (pushing v up saturates one pair's
sigmoid), which can distort the policy. The ``learned`` head reads a mean-pooled
summary of the (reweighted) field embeddings and produces the value with its own
parameters -- decoupled from the actor's per-pair scores while still sharing the
embedding/MoE/FM trunk -- giving a lower-variance advantage baseline. The paper
reports ``max`` as best on its own DSN env; this variant tests that choice on MATE.

The forward interface still mirrors ``HitMACCoordinator`` (``act`` / ``value_only``
/ ``forward``) so the A2C+GAE trainer and evaluator are reused unchanged. ``obs``
is the discretised index tensor ``[N_cam, N_tgt, num_fields]``.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['HMVFECoordinator']


class VitalFeatureExpert(nn.Module):
    """One MoE expert: VFE (Hadamard kernel) -> LN -> MLP -> project to R^{fields}."""

    def __init__(self, num_fields: int, embedding_dim: int, mlp_hidden: int, mlp_layers: int) -> None:
        super().__init__()
        self.kernel = nn.Parameter(torch.empty(num_fields, embedding_dim))
        nn.init.normal_(self.kernel, mean=0.0, std=0.05)

        flat = num_fields * embedding_dim
        self.input_ln = nn.LayerNorm(flat)

        layers = []
        dim = flat
        for _ in range(max(1, mlp_layers)):
            linear = nn.Linear(dim, mlp_hidden)
            nn.init.normal_(linear.weight, mean=0.0, std=1e-4)
            nn.init.constant_(linear.bias, 0.0)
            layers += [linear, nn.LayerNorm(mlp_hidden), nn.ReLU(inplace=True)]
            dim = mlp_hidden
        self.mlp = nn.Sequential(*layers)

        self.project = nn.Linear(dim, num_fields)
        nn.init.normal_(self.project.weight, mean=0.0, std=1e-4)
        nn.init.constant_(self.project.bias, 0.0)

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        refined = e * self.kernel                     # Hadamard, broadcast over pairs
        flat = refined.reshape(e.shape[0], -1)        # [pairs, num_fields * d]
        h = self.input_ln(flat)
        h = self.mlp(h)
        return self.project(h)


class Gate(nn.Module):
    """MoE routing network: one hidden layer over the flattened embedding."""

    def __init__(self, num_fields: int, embedding_dim: int, hidden: int, num_experts: int) -> None:
        super().__init__()
        flat = num_fields * embedding_dim
        self.net = nn.Sequential(
            nn.Linear(flat, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_experts),
        )

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        return self.net(e.reshape(e.shape[0], -1))


class HMVFECoordinator(nn.Module):
    """High-level orchestrator with a configurable state-value critic (Tier B)."""

    def __init__(
        self,
        num_cameras: int,
        num_targets: int,
        table_size: int,
        *,
        embedding_dim: int = 10,
        num_experts: int = 4,
        top_k: int = 2,
        gating_hidden: int = 128,
        mlp_hidden: int = 128,
        mlp_layers: int = 2,
        num_fields: int = 5,
        critic_reduction: str = 'learned',
        value_head_hidden: int = 128,
        global_state_dim: int = 0,
        belief_enabled: bool = False,
        belief_hidden_dim: int = 128,
        critic_use_global_state: bool = False,
    ) -> None:
        super().__init__()
        self.num_cameras = int(num_cameras)
        self.num_targets = int(num_targets)
        self.num_fields = int(num_fields)
        self.num_experts = int(num_experts)
        self.top_k = max(1, min(int(top_k), int(num_experts)))
        self.critic_reduction = str(critic_reduction).lower()
        self.global_state_dim = int(global_state_dim)
        self.belief_enabled = bool(belief_enabled) and self.global_state_dim > 0
        self.belief_hidden_dim = int(belief_hidden_dim)
        self.critic_use_global_state = bool(critic_use_global_state) and self.belief_enabled
        self.last_belief_state = None
        if self.critic_reduction not in ('max', 'mean', 'learned'):
            raise ValueError(f"critic_reduction must be max|mean|learned, got {critic_reduction!r}")

        self.embedding = nn.Embedding(table_size, embedding_dim)
        self.first_order = nn.Embedding(table_size, 1)
        nn.init.xavier_normal_(self.embedding.weight, gain=1e-3)
        nn.init.xavier_normal_(self.first_order.weight, gain=1e-3)
        self.bias = nn.Parameter(torch.zeros(1))      # FM global bias u0

        self.gate = Gate(num_fields, embedding_dim, gating_hidden, num_experts)
        self.experts = nn.ModuleList(
            VitalFeatureExpert(num_fields, embedding_dim, mlp_hidden, mlp_layers)
            for _ in range(num_experts)
        )

        pooled_dim = num_fields * embedding_dim + num_fields
        self.belief_head = None
        self.belief_actor_head = None
        self.central_value_head = None
        if self.belief_enabled:
            self.belief_head = nn.Sequential(
                nn.Linear(pooled_dim, self.belief_hidden_dim),
                nn.LayerNorm(self.belief_hidden_dim),
                nn.Tanh(),
                nn.Linear(self.belief_hidden_dim, self.global_state_dim),
                nn.Tanh(),
            )
            # The actor receives only this predicted belief at execution.  The
            # head emits one bounded logit correction per camera-target pair.
            self.belief_actor_head = nn.Sequential(
                nn.Linear(pooled_dim + self.global_state_dim, self.belief_hidden_dim),
                nn.Tanh(),
                nn.Linear(self.belief_hidden_dim, self.num_cameras * self.num_targets),
            )
            if self.critic_use_global_state:
                self.central_value_head = nn.Sequential(
                    nn.Linear(self.global_state_dim, self.belief_hidden_dim),
                    nn.Tanh(),
                    nn.Linear(self.belief_hidden_dim, 1),
                )

        # Tier B: a separate value head on the shared trunk (decoupled from the
        # actor's per-pair scores). Input = mean-pooled reweighted 2nd-order
        # embeddings (num_fields * d) concatenated with pooled 1st-order weights.
        if self.critic_reduction == 'learned':
            value_input_dim = pooled_dim + (self.global_state_dim if self.belief_enabled else 0)
            self.value_head = nn.Sequential(
                nn.Linear(value_input_dim, value_head_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(value_head_hidden, 1),
            )
            nn.init.constant_(self.value_head[-1].bias, 0.0)
        else:
            self.value_head = None

    def _scores_and_pool(self, obs: torch.Tensor):
        """Return (z [pairs] FM interaction scores, pooled [F*d + F] trunk summary)."""

        idx = obs.reshape(-1, self.num_fields).long()      # [pairs, F]
        e = self.embedding(idx)                            # [pairs, F, d]
        u = self.first_order(idx).squeeze(-1)              # [pairs, F]

        # --- Vital Feature Module (MoE) ---
        gate_logits = self.gate(e)
        top_val, top_idx = torch.topk(gate_logits, self.top_k, dim=1)
        weights = F.softmax(top_val, dim=1)
        expert_out = torch.stack([expert(e) for expert in self.experts], dim=1)  # [pairs, E, F]
        gathered = torch.gather(
            expert_out, 1, top_idx.unsqueeze(-1).expand(-1, -1, self.num_fields)
        )
        vital = (weights.unsqueeze(-1) * gathered).sum(dim=1)   # [pairs, F]

        # --- Reweighting + FM interaction ---
        e_star = e * vital.unsqueeze(-1)                   # [pairs, F, d]
        u_star = u * vital                                 # [pairs, F]
        sum_sq = e_star.sum(dim=1).pow(2).sum(dim=1)       # ||sum_i e*_i||^2
        sq_sum = e_star.pow(2).sum(dim=(1, 2))             # sum_i ||e*_i||^2
        z = self.bias + u_star.sum(dim=1) + 0.5 * (sum_sq - sq_sum)   # [pairs]

        # mean-pooled trunk summary for the learned critic (decoupled from z)
        pooled = torch.cat(
            [e_star.mean(dim=0).reshape(-1), u_star.mean(dim=0)]
        ) if (self.critic_reduction == 'learned' or self.belief_enabled) else None
        return z, pooled

    def _belief_from_pool(self, pooled):
        if self.belief_head is None or pooled is None:
            self.last_belief_state = None
            return None
        belief = self.belief_head(pooled).reshape(-1)
        self.last_belief_state = belief
        return belief

    def belief_loss(self, target: torch.Tensor, prediction: torch.Tensor | None = None) -> torch.Tensor:
        """Supervised local-to-global loss used only during centralized training."""
        if self.belief_head is None:
            return target.new_zeros(())
        prediction = self.last_belief_state if prediction is None else prediction
        if prediction is None:
            return target.new_zeros(())
        target = target.to(device=prediction.device, dtype=prediction.dtype).reshape_as(prediction)
        return torch.nn.functional.smooth_l1_loss(prediction, target)

    def _actor_logits(self, z: torch.Tensor, pooled):
        """Add belief-only context to actor logits without privileged state."""
        if self.belief_actor_head is None:
            return z
        belief = self._belief_from_pool(pooled)
        if belief is None:
            return z
        actor_input = torch.cat([pooled, belief], dim=-1)
        return z + self.belief_actor_head(actor_input).reshape_as(z)

    def _value(self, z: torch.Tensor, pooled, critic_state: torch.Tensor | None = None) -> torch.Tensor:
        if critic_state is not None and self.central_value_head is not None:
            state = critic_state.to(device=z.device, dtype=z.dtype).reshape(-1)
            return self.central_value_head(state).reshape(1)
        if self.critic_reduction == 'max':
            return z.max().reshape(1)
        if self.critic_reduction == 'mean':
            return z.mean().reshape(1)
        belief = self._belief_from_pool(pooled)
        if critic_state is not None and self.belief_enabled:
            context = critic_state.to(device=z.device, dtype=z.dtype).reshape(-1)
        elif self.belief_enabled:
            context = belief
        else:
            context = None
        value_input = pooled if context is None else torch.cat([pooled, context], dim=-1)
        return self.value_head(value_input).reshape(1)     # 'learned'

    def belief_from_observation(self, obs: torch.Tensor) -> torch.Tensor | None:
        """Return the local belief prediction for auxiliary supervised training."""
        _, pooled = self._scores_and_pool(obs)
        return self._belief_from_pool(pooled)

    def forward(self, obs: torch.Tensor, critic_state: torch.Tensor | None = None):
        z, pooled = self._scores_and_pool(obs)
        z = self._actor_logits(z, pooled)
        prob = torch.sigmoid(z)
        return prob.reshape(self.num_cameras, self.num_targets), self._value(z, pooled, critic_state)

    def act(self, obs: torch.Tensor, deterministic: bool = False, critic_state: torch.Tensor | None = None):
        """Returns ``(action[N_cam,N_tgt], log_prob, entropy, value)``."""

        z, pooled = self._scores_and_pool(obs)
        z = self._actor_logits(z, pooled)
        prob = torch.sigmoid(z).clamp(1e-6, 1.0 - 1e-6)
        dist = torch.distributions.Bernoulli(probs=prob)
        action = (prob > 0.5).float() if deterministic else dist.sample()
        log_prob = dist.log_prob(action).sum()
        entropy = dist.entropy().sum()
        value = self._value(z, pooled, critic_state)
        return (
            action.reshape(self.num_cameras, self.num_targets).long(),
            log_prob,
            entropy,
            value,
        )

    def value_only(self, obs: torch.Tensor, critic_state: torch.Tensor | None = None) -> torch.Tensor:
        z, pooled = self._scores_and_pool(obs)
        z = self._actor_logits(z, pooled)
        return self._value(z, pooled, critic_state)
