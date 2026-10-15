"""FOCUS responsibility adapter for HMVFE actor updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch


@dataclass
class HMVFEFocusOutput:
    rho: Optional[torch.Tensor]
    confidence: Optional[torch.Tensor] = None
    valid: Optional[torch.Tensor] = None
    total_gain: Optional[torch.Tensor] = None
    belief_loss: Optional[torch.Tensor] = None
    belief_stats: Optional[Dict[str, float]] = None
    confidence_mode: Optional[str] = None


class HMVFEFocusAdapter:
    """Map FOCUS camera responsibility to baseline-preserving actor weights."""

    def __init__(
        self,
        eta: float,
        use_confidence: bool = True,
        eps: float = 1e-8,
        weight_formula: str = 'rho',
        rho_temperature: float = 1.2,
    ) -> None:
        self.eta = float(eta)
        self.use_confidence = bool(use_confidence)
        self.eps = float(eps)
        self.weight_formula = str(weight_formula).lower()
        self.rho_temperature = float(rho_temperature)
        if self.weight_formula not in ('rho', 'affine'):
            raise ValueError('focus weight_formula must be one of: rho, affine')
        if self.rho_temperature <= 0.0:
            raise ValueError('focus rho_temperature must be positive')

    @torch.no_grad()
    def _apply_temperature(
        self, rho: torch.Tensor, active_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.rho_temperature == 1.0:
            return rho
        logits = torch.log(rho.clamp_min(self.eps)) / self.rho_temperature
        if active_mask is not None:
            mask = active_mask.to(device=rho.device, dtype=torch.bool)
            logits = logits.masked_fill(~mask, -torch.inf)
            inactive_rows = ~mask.any(dim=-1, keepdim=True)
            logits = torch.where(inactive_rows, torch.zeros_like(logits), logits)
        tempered = torch.softmax(logits, dim=-1)
        if active_mask is not None:
            mask = active_mask.to(device=rho.device, dtype=rho.dtype)
            tempered = tempered * mask
            tempered = tempered / tempered.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        return tempered

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
            rho = self._apply_temperature(rho)
            n_cam = rho.new_tensor(float(rho.shape[-1]))
            if self.weight_formula == 'rho':
                weights = n_cam * rho
            else:
                weights = 1.0 + eta_t * (n_cam * rho - 1.0)
            return weights.detach()

        mask = active_mask.detach().to(device=rho.device, dtype=rho.dtype)
        masked_rho = rho * mask
        rho = masked_rho / masked_rho.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        rho = self._apply_temperature(rho, active_mask=mask)
        n_active = mask.sum(dim=-1, keepdim=True)
        if self.weight_formula == 'rho':
            weights = n_active.clamp_min(1.0) * rho
        else:
            weights = 1.0 + eta_t * (n_active.clamp_min(1.0) * rho - 1.0)
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
        from ray.rllib.agents.qplex_focus.qplex_policy import (
            QPLEXFocusLoss,
            build_occupancy_model,
            resolve_focus_config,
        )

        self.num_cameras = int(num_cameras)
        self.num_targets = int(num_targets)
        raw_focus_config = dict(config)
        raw_focus_config.setdefault('use_env_params', False)
        raw_focus_config.setdefault('n_agents', self.num_cameras)
        raw_focus_config.setdefault('n_targets', self.num_targets)
        focus_config = resolve_focus_config(
            {'focus': raw_focus_config, 'env_config': {}}
        )
        focus_config.setdefault('use_action_selection', False)
        focus_config.setdefault('belief_mode', 'oracle_next_ablation')

        occupancy_model = None
        if focus_config.get('belief_mode') == 'learned':
            occupancy_model = build_occupancy_model(
                state_dim,
                self.num_cameras,
                self.num_targets,
                focus_config,
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

    @property
    def has_learned_belief(self) -> bool:
        return self.estimator.occupancy_model is not None

    def parameters(self):
        return self.estimator.parameters()

    def state_dict(self):
        return self.estimator.state_dict()

    def get_runtime_state(self):
        return self.estimator.get_focus_runtime_state()

    def set_runtime_state(self, state):
        self.estimator.set_focus_runtime_state(state)

    def set_learner_update(self, learner_update: int) -> None:
        self.estimator.set_focus_learner_update(learner_update)

    def to(self, device: torch.device) -> 'HMVFEFocusResponsibilityEngine':
        self.estimator.to(device)
        return self

    def compute(
        self,
        global_state: torch.Tensor,
        next_global_state: Optional[torch.Tensor],
        valid_mask: Optional[torch.Tensor] = None,
        update_belief_ema: bool = False,
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
        rho, valid, total_gain, belief_loss, confidence, mode = self.estimator._focus_credit_target(
            global_state,
            next_global_state,
            actions,
            valid_mask,
            update_belief_ema=update_belief_ema,
        )
        return HMVFEFocusOutput(
            rho=rho.detach(),
            confidence=confidence.detach(),
            valid=valid.detach(),
            total_gain=total_gain.detach(),
            belief_loss=belief_loss,
            belief_stats=dict(getattr(self.estimator, 'last_belief_stats', {}) or {}),
            confidence_mode=mode,
        )

    @torch.no_grad()
    def infer(
        self,
        global_state: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> HMVFEFocusOutput:
        """Predict camera responsibility from the current state only.

        ``compute`` also accepts a future state because it builds the auxiliary
        belief loss.  Action selection and evaluation must not have that future
        label, so this path directly turns the learned occupancy distribution
        into FOCUS geometric gains.  The safeguard blend/runtime state is read
        but never updated here.
        """

        from ray.rllib.agents.qplex_focus.qplex_policy import _extract_target_positions

        if global_state.dim() == 2:
            global_state = global_state.unsqueeze(1)
        batch, time = global_state.shape[:2]
        device, dtype = global_state.device, global_state.dtype
        if valid_mask is None:
            valid_mask = torch.ones(
                (batch, time, self.num_cameras), device=device, dtype=dtype
            )
        elif valid_mask.dim() == 2:
            valid_mask = valid_mask.unsqueeze(-1).expand(
                -1, -1, self.num_cameras
            )

        estimator = self.estimator
        occupancy = estimator.occupancy_model
        uniform = torch.full(
            (batch, time, self.num_cameras),
            1.0 / self.num_cameras,
            device=device,
            dtype=dtype,
        )
        if occupancy is None:
            return HMVFEFocusOutput(
                rho=uniform,
                confidence=torch.ones((batch, time), device=device, dtype=dtype),
                valid=torch.zeros((batch, time), device=device, dtype=torch.bool),
                total_gain=torch.zeros((batch, time), device=device, dtype=dtype),
                confidence_mode='inference',
            )

        actions = torch.zeros(
            (batch, time, self.num_cameras), device=device, dtype=torch.long
        )
        n_targets = self.num_targets
        blend = estimator._belief_blend_schedule() * estimator._belief_performance_gate()
        categorical = getattr(occupancy, 'distribution_type', None) == 'categorical_grid'
        if categorical:
            probabilities, _mean, _std, _logits = occupancy.distribution(global_state)
            persistence = occupancy.persistence_probabilities(global_state)
            effective = persistence + blend * (probabilities - persistence)
            gains = estimator._credit_from_categorical_grid(
                effective,
                occupancy.displacement_grid,
                global_state,
                actions,
                n_targets,
            )
        else:
            mean, std = occupancy(global_state)
            current = _extract_target_positions(
                global_state, self.num_cameras, n_targets
            ).unsqueeze(2)
            effective_mean = current + blend * (mean - current)
            integral_mode = str(estimator.focus_config.get('integral_mode', 'MC')).lower()
            if integral_mode == 'grid':
                gains = estimator._credit_from_grid(
                    effective_mean, std, global_state, actions, n_targets
                )
            elif integral_mode == 'sigma':
                samples, weights = estimator._sigma_points(effective_mean, std)
                gains = estimator._credit_from_sigma_points(
                    samples,
                    global_state,
                    actions,
                    n_targets,
                    sample_weights=weights,
                )
            elif integral_mode == 'mc':
                gains = estimator._credit_from_mc_points(
                    effective_mean, std, global_state, actions, n_targets
                )
            else:
                raise ValueError(f'Unknown FOCUS integral_mode: {integral_mode}')

        eps = float(estimator.focus_config.get('eps', 1e-8))
        total_gain = gains.sum(dim=-1)
        min_signal = float(estimator.focus_config.get('min_credit_signal', 1e-6))
        valid = (total_gain > min_signal) & (valid_mask[..., 0] > 0)
        rho = gains / total_gain.unsqueeze(-1).clamp_min(eps)
        rho = torch.where(valid.unsqueeze(-1), rho, uniform)
        return HMVFEFocusOutput(
            rho=rho,
            confidence=torch.ones((batch, time), device=device, dtype=dtype),
            valid=valid,
            total_gain=total_gain,
            confidence_mode='belief_inference',
        )

    def belief_loss_from_sequence(
        self,
        global_state: torch.Tensor,
        next_global_state: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> HMVFEFocusOutput:
        if not self.has_learned_belief:
            return HMVFEFocusOutput(
                rho=None,
                belief_loss=global_state.new_zeros(()),
                belief_stats={},
                confidence_mode='off',
            )
        return self.compute(
            global_state,
            next_global_state,
            valid_mask=valid_mask,
            update_belief_ema=True,
        )


def build_hmvfe_focus_engine(config, env, device=None):
    """Build the shared train/evaluation belief engine for an HMVFE env.

    ``env`` may be either a single ``MATECoordinatorEnv`` or the synchronous
    vector wrapper. Keeping this construction in one place prevents eval-only
    from silently using a different belief architecture than training.
    """

    focus_mode = str(config.focus_mode).lower()
    if not config.focus_enabled or focus_mode not in ('real', 'shuffled'):
        return None
    base = env.envs[0] if hasattr(env, 'envs') else env
    state_dim = 1
    for size in base.base_env.state_space.shape:
        state_dim *= int(size)
    engine = HMVFEFocusResponsibilityEngine(
        env.num_cameras,
        env.num_targets,
        state_dim,
        {
            'belief_mode': config.focus_belief_mode,
            'horizon': config.focus_horizon,
            'horizon_discount': config.focus_horizon_discount,
            'beta_belief': config.focus_beta_belief,
            'belief_hidden_dim': config.focus_belief_hidden_dim,
            'belief_arch': config.focus_belief_arch,
            'belief_num_layers': config.focus_belief_num_layers,
            'belief_dropout': config.focus_belief_dropout,
            'belief_distribution': config.focus_belief_distribution,
            'belief_max_delta': config.focus_belief_max_delta,
            'belief_min_std': config.focus_belief_min_std,
            'belief_grid_size': config.focus_belief_grid_size,
            'belief_grid_max_delta': config.focus_belief_grid_max_delta,
            'belief_grid_label_std': config.focus_belief_grid_label_std,
            'belief_persistence_prior_std': config.focus_belief_persistence_prior_std,
            'belief_position_loss_coeff': config.focus_belief_position_loss_coeff,
            'belief_grid_chunk_size': config.focus_belief_grid_chunk_size,
            'belief_safeguard_mode': config.focus_belief_safeguard_mode,
            'belief_error_ema_decay': config.focus_belief_error_ema_decay,
            'belief_gate_margin': config.focus_belief_gate_margin,
            'belief_gate_full_improvement': config.focus_belief_gate_full_improvement,
            'belief_blend_warmup_updates': config.focus_belief_blend_warmup_updates,
            'belief_blend_ramp_updates': config.focus_belief_blend_ramp_updates,
            'integral_mode': config.focus_integral_mode,
            'mc_num_points': config.focus_mc_num_points,
            'mc_chunk_size': config.focus_mc_chunk_size,
            'mc_seed': config.focus_mc_seed,
            'sample_chunk_size': config.focus_sample_chunk_size,
            'grid_size': config.focus_grid_size,
            'grid_chunk_size': config.focus_grid_chunk_size,
            'min_credit_signal': config.focus_min_credit_signal,
            'n_obstacles': base.num_obstacles,
            'obstacle_transmittance': config.focus_obstacle_transmittance,
            'eps': config.focus_eps,
            'use_action_selection': False,
        },
    )
    return engine.to(device) if device is not None else engine


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


def execution_camera_logit_bias(
    rho: torch.Tensor,
    eta: float,
    eps: float = 1e-8,
    max_abs: float = 2.0,
) -> torch.Tensor:
    """Baseline-preserving CTCE logit bias from camera responsibility.

    Uniform responsibility maps exactly to zero.  The clamp prevents an early,
    poorly calibrated belief from overwhelming the learned actor.
    """

    rho = rho.detach().clamp_min(0.0)
    rho = rho / rho.sum(dim=-1, keepdim=True).clamp_min(eps)
    relative = rho * float(rho.shape[-1])
    return (float(eta) * torch.log(relative.clamp_min(eps))).clamp(
        min=-float(max_abs), max=float(max_abs)
    )
