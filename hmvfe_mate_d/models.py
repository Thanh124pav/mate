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
    ) -> None:
        super().__init__()
        self.num_cameras = int(num_cameras)
        self.num_targets = int(num_targets)
        self.num_fields = int(num_fields)
        self.num_experts = int(num_experts)
        self.top_k = max(1, min(int(top_k), int(num_experts)))
        self.critic_reduction = str(critic_reduction).lower()
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

        # Tier B: a separate value head on the shared trunk (decoupled from the
        # actor's per-pair scores). Input = mean-pooled reweighted 2nd-order
        # embeddings (num_fields * d) concatenated with pooled 1st-order weights.
        if self.critic_reduction == 'learned':
            pooled_dim = num_fields * embedding_dim + num_fields
            self.value_head = nn.Sequential(
                nn.Linear(pooled_dim, value_head_hidden),
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
        ) if self.critic_reduction == 'learned' else None
        return z, pooled

    def _value(self, z: torch.Tensor, pooled) -> torch.Tensor:
        if self.critic_reduction == 'max':
            return z.max().reshape(1)
        if self.critic_reduction == 'mean':
            return z.mean().reshape(1)
        return self.value_head(pooled).reshape(1)          # 'learned'

    def forward(self, obs: torch.Tensor):
        z, pooled = self._scores_and_pool(obs)
        prob = torch.sigmoid(z)
        return prob.reshape(self.num_cameras, self.num_targets), self._value(z, pooled)

    def act(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        return_per_camera_log_prob: bool = False,
    ):
        """Returns ``(action[N_cam,N_tgt], log_prob, entropy, value)``."""

        z, pooled = self._scores_and_pool(obs)
        prob = torch.sigmoid(z).clamp(1e-6, 1.0 - 1e-6)
        dist = torch.distributions.Bernoulli(probs=prob)
        action = (prob > 0.5).float() if deterministic else dist.sample()
        pair_log_prob = dist.log_prob(action).reshape(self.num_cameras, self.num_targets)
        camera_log_prob = pair_log_prob.sum(dim=-1)
        log_prob = camera_log_prob.sum(dim=-1)
        entropy = dist.entropy().sum()
        value = self._value(z, pooled)
        result = (
            action.reshape(self.num_cameras, self.num_targets).long(),
            log_prob,
            entropy,
            value,
        )
        if return_per_camera_log_prob:
            return (*result, camera_log_prob)
        return result

    def value_only(self, obs: torch.Tensor) -> torch.Tensor:
        z, pooled = self._scores_and_pool(obs)
        return self._value(z, pooled)
