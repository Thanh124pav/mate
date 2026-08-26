import math

import numpy as np

from ray.rllib.utils.framework import try_import_torch


torch, _ = try_import_torch(error=True)


FOCUS_CONFIDENCE_DEFAULTS = {
    "confidence_gate_enabled": True,
    "confidence_gate_mode": "auto",
    "confidence_entropy_kappa": 2.0,
    "confidence_loss_threshold": None,
    "confidence_loss_temperature": 1.0,
}


def add_confidence_defaults(focus_config):
    for key, value in FOCUS_CONFIDENCE_DEFAULTS.items():
        focus_config.setdefault(key, value)
    return focus_config


def confidence_mode(focus_config, default_mode):
    if not focus_config.get("confidence_gate_enabled", True):
        return "off"
    mode = str(focus_config.get("confidence_gate_mode", "auto")).lower()
    if mode == "auto":
        return default_mode
    return mode


def entropy_confidence(probs, focus_config):
    eps = focus_config.get("eps", 1e-8)
    num_classes = max(int(probs.size(-1)), 2)
    entropy = -(probs * torch.log(probs + eps)).sum(dim=-1) / math.log(num_classes)
    while entropy.dim() > 2:
        entropy = entropy.mean(dim=-1)
    kappa = float(focus_config.get("confidence_entropy_kappa", 2.0))
    return torch.exp(-kappa * entropy).detach().clamp(0.0, 1.0)


def loss_confidence(per_step_loss, valid, focus_config):
    eps = focus_config.get("eps", 1e-8)
    loss = per_step_loss.detach()
    threshold = focus_config.get("confidence_loss_threshold")
    if threshold is None:
        if valid is not None and valid.any():
            threshold = loss[valid].mean().detach()
        else:
            threshold = loss.mean().detach()
    else:
        threshold = torch.as_tensor(threshold, dtype=loss.dtype, device=loss.device)
    temperature = max(float(focus_config.get("confidence_loss_temperature", 1.0)), eps)
    return torch.sigmoid((threshold - loss) / temperature).detach()


def belief_std_confidence(std, focus_config):
    """Return per-step confidence from belief predictive uncertainty.

    This is usable at action-selection time because it only needs the belief
    model's predicted std, not the future state. The output has the same
    leading [B, T] shape as the belief prediction.
    """
    eps = focus_config.get("eps", 1e-8)
    std_mean = std.detach().mean(dim=tuple(range(2, std.dim())))
    threshold = float(focus_config.get("action_bias_confidence_std_threshold", 80.0))
    temperature = max(float(focus_config.get("action_bias_confidence_std_temperature", 40.0)), eps)
    confidence = torch.sigmoid((threshold - std_mean) / temperature)
    min_conf = float(focus_config.get("action_bias_confidence_min", 0.0))
    max_conf = float(focus_config.get("action_bias_confidence_max", 1.0))
    return confidence.clamp(min_conf, max_conf), std_mean


def resolve_confidence(
    focus_config,
    valid,
    reference,
    default_mode,
    probs=None,
    per_step_loss=None,
):
    mode = confidence_mode(focus_config, default_mode)
    if mode == "off":
        confidence = torch.ones_like(reference, dtype=reference.dtype, device=reference.device)
    elif mode == "entropy" and probs is not None:
        confidence = entropy_confidence(probs, focus_config)
    elif mode == "loss" and per_step_loss is not None:
        confidence = loss_confidence(per_step_loss, valid, focus_config)
    else:
        confidence = torch.ones_like(reference, dtype=reference.dtype, device=reference.device)
        mode = "off"
    return confidence.to(dtype=reference.dtype, device=reference.device), mode


def gated_focus_loss(per_step_loss, valid, confidence, reference_loss, eps=1e-8, weights=None):
    if not valid.any():
        return torch.zeros_like(reference_loss)
    if weights is None:
        weights = torch.ones_like(per_step_loss)
    gated_weights = weights * confidence.detach()
    valid_weights = gated_weights[valid]
    denom = valid_weights.sum()
    if denom.detach().item() <= eps:
        return torch.zeros_like(reference_loss)
    return (per_step_loss[valid] * valid_weights).sum() / (denom + eps)


def confidence_stats(confidence, valid, mode):
    values = confidence[valid] if valid is not None and valid.any() else confidence.reshape(-1)
    return {
        "focus_confidence_mean": values.mean().detach().item(),
        "focus_confidence_min": values.min().detach().item(),
        "focus_confidence_max": values.max().detach().item(),
        "focus_confidence_mode": mode,
    }

PRESERVED_DIM = 13
CAMERA_STATE_DIM_PRIVATE = 9
TARGET_STATE_DIM_PRIVATE = 14
_CAM_LOW = torch.tensor([-2000., -2000., 0., -2000., -2000., 0., 0., 0., 0.])
_CAM_HIGH = torch.tensor([2000., 2000., 1000., 2000., 2000., 180., 2000., 180., 180.])


def _denorm_camera_state(value, idx):
    low = _CAM_LOW[idx].to(dtype=value.dtype, device=value.device)
    high = _CAM_HIGH[idx].to(dtype=value.dtype, device=value.device)
    return (value + 1.0) / 2.0 * (high - low) + low


def _extract_focus_camera_fov(state, n_agents):
    positions, orientations, sight_ranges, half_angles = [], [], [], []
    for i in range(int(n_agents)):
        start = PRESERVED_DIM + i * CAMERA_STATE_DIM_PRIVATE
        x = _denorm_camera_state(state[..., start], 0)
        y = _denorm_camera_state(state[..., start + 1], 1)
        vx = _denorm_camera_state(state[..., start + 3], 3)
        vy = _denorm_camera_state(state[..., start + 4], 4)
        va = _denorm_camera_state(state[..., start + 5], 5)
        positions.append(torch.stack([x, y], dim=-1))
        orientations.append(torch.atan2(vy, vx))
        sight_ranges.append(torch.sqrt(vx.square() + vy.square() + 1e-8))
        half_angles.append(va * (math.pi / 180.0) / 2.0)
    return (
        torch.stack(positions, dim=-2),
        torch.stack(orientations, dim=-1),
        torch.stack(sight_ranges, dim=-1),
        torch.stack(half_angles, dim=-1),
    )


def _extract_focus_target_positions(state, n_agents, n_targets):
    target_start = PRESERVED_DIM + int(n_agents) * CAMERA_STATE_DIM_PRIVATE
    positions = []
    for j in range(int(n_targets)):
        start = target_start + j * TARGET_STATE_DIM_PRIVATE
        x = (state[..., start] + 1.0) / 2.0 * 4000.0 - 2000.0
        y = (state[..., start + 1] + 1.0) / 2.0 * 4000.0 - 2000.0
        positions.append(torch.stack([x, y], dim=-1))
    return torch.stack(positions, dim=-2)


def focus_action_q_bias(
    q_values,
    state,
    focus_config,
    n_agents,
    n_actions,
    require_eta=True,
):
    """Return a geometry prior over camera discrete actions for CTDE training.

    The helper uses centralized normalized MATE state, so callers should only add
    this bias while training/collecting exploratory experience. Decentralized
    evaluation should call the policy with ``explore=False`` and skip the bias.
    """
    if state is None or q_values is None or not focus_config:
        return None
    eta = float(focus_config.get("action_bias_eta", 0.0))
    if require_eta and eta <= 0.0:
        return None
    n_agents = int(n_agents)
    n_actions = int(n_actions)
    levels = int(round(n_actions ** 0.5))
    if levels * levels != n_actions:
        return None

    n_targets = int(focus_config.get("n_targets", 8))
    required_dim = PRESERVED_DIM + n_agents * CAMERA_STATE_DIM_PRIVATE
    required_dim += n_targets * TARGET_STATE_DIM_PRIVATE
    if state.size(-1) < required_dim:
        return None

    leading = q_values.shape[:-2]
    if state.shape[:-1] != leading:
        try:
            state = state.reshape(*leading, state.size(-1))
        except RuntimeError:
            return None

    grid = torch.as_tensor(
        np.stack(
            np.meshgrid(
                np.linspace(-1.0, 1.0, num=levels),
                np.linspace(-1.0, 1.0, num=levels),
            ),
            axis=-1,
        ).reshape(-1, 2),
        dtype=q_values.dtype,
        device=q_values.device,
    )
    state = state.to(dtype=q_values.dtype, device=q_values.device)

    cam_pos, cam_orient, cam_range, cam_half_angle = _extract_focus_camera_fov(state, n_agents)
    target_pos = _extract_focus_target_positions(state, n_agents, n_targets)

    rotation_step = float(focus_config.get("action_bias_rotation_step", 5.0))
    zooming_step = float(focus_config.get("action_bias_zooming_step", 2.5))
    min_angle = float(focus_config.get("action_bias_min_viewing_angle", 30.0))
    max_angle = float(focus_config.get("action_bias_max_viewing_angle", 180.0))
    max_range = float(focus_config.get("action_bias_max_sight_range", 1500.0))

    delta_orient = grid[:, 0] * (rotation_step * math.pi / 180.0)
    orient = cam_orient.unsqueeze(-1) + delta_orient.view(*([1] * len(leading)), 1, n_actions)
    own_range = cam_range.clamp_min(1.0)
    own_half = cam_half_angle.clamp_min(1e-3)
    current_angle = (own_half * 2.0 * 180.0 / math.pi).clamp(min=min_angle, max=max_angle)
    next_angle = current_angle.unsqueeze(-1) + grid[:, 1].view(*([1] * len(leading)), 1, n_actions) * zooming_step
    next_angle = next_angle.clamp(min=min_angle, max=max_angle)
    area_product = own_range.square().unsqueeze(-1) * current_angle.unsqueeze(-1)
    next_range = torch.sqrt(area_product / next_angle.clamp_min(1e-3)).clamp(1.0, max_range)
    next_half = next_angle * (math.pi / 180.0) / 2.0

    rel = target_pos.unsqueeze(-3).unsqueeze(-3) - cam_pos.unsqueeze(-2).unsqueeze(-2)
    dist = torch.sqrt(rel.square().sum(dim=-1) + 1e-8)
    target_angle = torch.atan2(rel[..., 1], rel[..., 0])
    angle_delta = torch.atan2(
        torch.sin(target_angle - orient.unsqueeze(-1)),
        torch.cos(target_angle - orient.unsqueeze(-1)),
    ).abs()

    angular_temp = float(focus_config.get("action_bias_angular_temp", 0.18))
    range_temp = float(focus_config.get("action_bias_range_temp", 150.0))
    angular_score = torch.sigmoid((next_half.unsqueeze(-1) - angle_delta) / angular_temp)
    range_score = torch.sigmoid((next_range.unsqueeze(-1) - dist) / range_temp)

    target_weight = torch.ones_like(dist)
    if focus_config.get("action_bias_real_only", True):
        target_start = PRESERVED_DIM + n_agents * CAMERA_STATE_DIM_PRIVATE
        weights = []
        for j in range(n_targets):
            weights.append(state[..., target_start + j * TARGET_STATE_DIM_PRIVATE + 3])
        target_weight = torch.stack(weights, dim=-1).clamp(0.0, 1.0).unsqueeze(-2).unsqueeze(-2)

    score = (angular_score * range_score * target_weight).sum(dim=-1)
    centered = score - score.mean(dim=-1, keepdim=True)
    scale = centered.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-4)
    clip = float(focus_config.get("action_bias_clip", 3.0))
    return (centered / scale).clamp(-clip, clip)
