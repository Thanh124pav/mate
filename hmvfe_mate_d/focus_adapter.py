"""FOCUS responsibility adapter for HMVFE actor updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch


@dataclass
class HMVFEFocusOutput:
    rho: torch.Tensor
    confidence: Optional[torch.Tensor] = None
    valid: Optional[torch.Tensor] = None
    total_gain: Optional[torch.Tensor] = None


class HMVFEFocusAdapter:
    """Map FOCUS camera responsibility to baseline-preserving actor weights."""

    def __init__(
        self,
        eta: float,
        use_confidence: bool = True,
        eps: float = 1e-8,
        weight_min: Optional[float] = None,
        weight_max: Optional[float] = None,
    ) -> None:
        self.eta = float(eta)
        self.use_confidence = bool(use_confidence)
        self.eps = float(eps)
        self.weight_min = None if weight_min is None else float(weight_min)
        self.weight_max = None if weight_max is None else float(weight_max)
        if self.weight_min is not None and self.weight_min < 0.0:
            raise ValueError('focus weight_min must be non-negative')
        if self.weight_max is not None and self.weight_max <= 0.0:
            raise ValueError('focus weight_max must be positive')
        if (
            self.weight_min is not None
            and self.weight_max is not None
            and self.weight_min > self.weight_max
        ):
            raise ValueError('focus weight_min must be <= weight_max')

    @torch.no_grad()
    def _clip_preserve_mean(
        self, weights: torch.Tensor, active_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.weight_min is None and self.weight_max is None:
            return weights

        low = -torch.inf if self.weight_min is None else weights.new_tensor(self.weight_min)
        high = torch.inf if self.weight_max is None else weights.new_tensor(self.weight_max)
        if active_mask is None:
            mask = torch.ones_like(weights, dtype=torch.bool)
        else:
            mask = active_mask.to(device=weights.device, dtype=torch.bool)

        flat_weights = weights.reshape(-1, weights.shape[-1]).clone()
        flat_mask = mask.reshape(-1, mask.shape[-1])
        for row_idx in range(flat_weights.shape[0]):
            row = flat_weights[row_idx]
            row_mask = flat_mask[row_idx]
            active_count = int(row_mask.sum().item())
            if active_count == 0:
                continue
            target_sum = row.new_tensor(float(active_count))
            active = row[row_mask].clamp(low, high)
            # Project onto the bounded simplex with sum = active_count. This keeps
            # FOCUS non-uniform but prevents zero/oversized actor gradients.
            for _ in range(active.numel() + 2):
                diff = target_sum - active.sum()
                if diff.abs() <= self.eps:
                    break
                free = (active > low + self.eps) & (active < high - self.eps)
                if not bool(free.any().item()):
                    free = torch.ones_like(active, dtype=torch.bool)
                active = active.clone()
                active[free] = active[free] + diff / free.float().sum()
                active = active.clamp(low, high)
            row = row.clone()
            row[row_mask] = active
            row[~row_mask] = 0.0
            flat_weights[row_idx] = row
        return flat_weights.reshape_as(weights)

    @torch.no_grad()
    def compute_weights(
        self,
        rho: torch.Tensor,
        confidence: Optional[torch.Tensor] = None,
        active_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        rho = rho.detach().clamp_min(0.0)
        eta_t = rho.new_tensor(self.eta)
        if self.use_confidence and confidence is not None:
            confidence = confidence.detach().to(device=rho.device, dtype=rho.dtype).clamp(0.0, 1.0)
            while confidence.dim() < rho.dim() - 1:
                confidence = confidence.unsqueeze(-1)
            eta_t = eta_t * confidence.unsqueeze(-1)

        if active_mask is None:
            rho = rho / rho.sum(dim=-1, keepdim=True).clamp_min(self.eps)
            n_cam = rho.new_tensor(float(rho.shape[-1]))
            weights = 1.0 + eta_t * (n_cam * rho - 1.0)
            return self._clip_preserve_mean(weights).detach()

        mask = active_mask.detach().to(device=rho.device, dtype=rho.dtype)
        masked_rho = rho * mask
        rho = masked_rho / masked_rho.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        n_active = mask.sum(dim=-1, keepdim=True)
        weights = 1.0 + eta_t * (n_active.clamp_min(1.0) * rho - 1.0)
        weights = self._clip_preserve_mean(weights, active_mask=mask)
        return (weights * mask).detach()


def weighted_actor_log_prob(
    joint_log_prob: torch.Tensor,
    camera_log_prob: torch.Tensor,
    adapter: HMVFEFocusAdapter,
    rho: Optional[torch.Tensor] = None,
    confidence: Optional[torch.Tensor] = None,
    active_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Return the actor log-probability used by the policy loss."""

    if rho is None:
        return joint_log_prob, None
    weights = adapter.compute_weights(rho, confidence=confidence, active_mask=active_mask)
    return (camera_log_prob * weights).sum(dim=-1), weights


def policy_loss_from_log_prob(
    actor_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """HMVFE's A2C actor reduction, optionally over valid FOCUS decisions only."""

    advantages = advantages.detach()
    if mask is None:
        return -(actor_log_prob * advantages).mean()
    mask = mask.to(device=actor_log_prob.device, dtype=torch.bool)
    if not bool(mask.any().item()):
        raise RuntimeError('FOCUS actor loss has no valid decisions in this update batch.')
    return -(actor_log_prob[mask] * advantages[mask]).mean()


def focus_diagnostics(
    rho: torch.Tensor,
    weights: torch.Tensor,
    weighted_log_prob: torch.Tensor,
    vanilla_log_prob: torch.Tensor,
    confidence: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> Dict[str, float]:
    with torch.no_grad():
        rho = rho.detach().clamp_min(0.0)
        rho = rho / rho.sum(dim=-1, keepdim=True).clamp_min(eps)
        uniform = rho.new_full(rho.shape, 1.0 / max(rho.shape[-1], 1))
        entropy = -(rho * torch.log(rho + eps)).sum(dim=-1)
        metrics = {
            'focus/rho_entropy': float(entropy.mean().cpu()),
            'focus/rho_l1_uniform': float((rho - uniform).abs().sum(dim=-1).mean().cpu()),
            'focus/rho_min': float(rho.min().cpu()),
            'focus/rho_max': float(rho.max().cpu()),
            'focus/effective_num_cameras': float(torch.exp(entropy).mean().cpu()),
            'focus/weight_min': float(weights.detach().min().cpu()),
            'focus/weight_max': float(weights.detach().max().cpu()),
            'focus/weight_mean': float(weights.detach().mean().cpu()),
            'focus/weighted_log_prob_mean': float(weighted_log_prob.detach().mean().cpu()),
            'focus/vanilla_log_prob_mean': float(vanilla_log_prob.detach().mean().cpu()),
            'focus/actor_weight_delta': float(
                (weighted_log_prob.detach() - vanilla_log_prob.detach()).abs().mean().cpu()
            ),
        }
        if confidence is not None:
            metrics['focus/confidence_mean'] = float(confidence.detach().float().mean().cpu())
        return metrics


class HMVFEFocusResponsibilityEngine:
    """Thin HMVFE wrapper around the existing QPLEX FOCUS responsibility code."""

    def __init__(
        self,
        num_cameras: int,
        num_targets: int,
        state_dim: int,
        config: Dict[str, Any],
    ) -> None:
        from ray.rllib.agents.qplex_focus.qplex_policy import LearnedOccupancyModel, QPLEXFocusLoss

        self.num_cameras = int(num_cameras)
        self.num_targets = int(num_targets)
        focus_config = dict(config)
        focus_config.setdefault('n_agents', self.num_cameras)
        focus_config.setdefault('n_targets', self.num_targets)
        focus_config.setdefault('use_action_selection', False)
        focus_config.setdefault('belief_mode', 'oracle_next_ablation')

        occupancy_model = None
        if focus_config.get('belief_mode') == 'learned':
            occupancy_model = LearnedOccupancyModel(
                state_dim,
                self.num_cameras,
                self.num_targets,
                horizon=int(focus_config.get('horizon', 3)),
                hidden_dim=int(focus_config.get('belief_hidden_dim', 256)),
                max_delta=float(focus_config.get('belief_max_delta', 400.0)),
                min_std=float(focus_config.get('belief_min_std', 25.0)),
            )

        self.estimator = QPLEXFocusLoss(
            model=None,
            target_model=None,
            mixer=None,
            target_mixer=None,
            n_agents=self.num_cameras,
            n_actions=0,
            focus_config=focus_config,
            occupancy_model=occupancy_model,
        )

    def to(self, device: torch.device) -> 'HMVFEFocusResponsibilityEngine':
        self.estimator.to(device)
        return self

    @torch.no_grad()
    def compute(
        self,
        global_state: torch.Tensor,
        next_global_state: Optional[torch.Tensor],
        valid_mask: Optional[torch.Tensor] = None,
    ) -> HMVFEFocusOutput:
        if global_state.dim() == 2:
            global_state = global_state.unsqueeze(1)
        if next_global_state is not None and next_global_state.dim() == 2:
            next_global_state = next_global_state.unsqueeze(1)
        if valid_mask is None:
            valid_mask = torch.ones(
                (*global_state.shape[:2], self.num_cameras),
                dtype=global_state.dtype,
                device=global_state.device,
            )
        elif valid_mask.dim() == 2:
            valid_mask = valid_mask.unsqueeze(-1).expand(-1, -1, self.num_cameras)

        actions = torch.zeros(
            (*global_state.shape[:2], self.num_cameras),
            dtype=torch.long,
            device=global_state.device,
        )
        rho, valid, total_gain, _belief_loss, confidence, _mode = self.estimator._focus_credit_target(
            global_state,
            next_global_state,
            actions,
            valid_mask,
        )
        return HMVFEFocusOutput(
            rho=rho.detach(),
            confidence=confidence.detach(),
            valid=valid.detach(),
            total_gain=total_gain.detach(),
        )


def synthetic_responsibility(
    mode: str,
    shape: Tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
    eps: float,
    base_rho: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    mode = str(mode).lower()
    batch, num_cameras = shape
    if mode == 'uniform' or (base_rho is None and mode == 'shuffled'):
        return torch.full((batch, num_cameras), 1.0 / num_cameras, device=device, dtype=dtype)
    if mode == 'random':
        rho = torch.rand((batch, num_cameras), device=device, dtype=dtype)
        return rho / rho.sum(dim=-1, keepdim=True).clamp_min(eps)
    if mode == 'shuffled':
        rows = []
        for row in base_rho.detach():
            rows.append(row[torch.randperm(num_cameras, device=device)])
        return torch.stack(rows, dim=0)
    raise ValueError(f'Unknown synthetic FOCUS mode: {mode}')
