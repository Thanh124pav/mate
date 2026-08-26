from gym.spaces import Tuple, Discrete, Dict
import copy
import logging
import numpy as np
import os
import tree  # pip install dm_tree
import torch.nn.functional as F
from argparse import Namespace


import ray
from .mixers import FocusDuelMixer
from .model import RNNModel, _get_size
from ray.rllib.env.multi_agent_env import ENV_STATE
from ray.rllib.env.wrappers.group_agents_wrapper import GROUP_REWARDS
from ray.rllib.models.torch.torch_action_dist import TorchCategorical
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.rnn_sequencing import chop_into_sequences
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.modelv2 import _unpack_obs
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.annotations import override
from ray.rllib.agents.focus_utils import (
    add_confidence_defaults,
    confidence_stats,
    gated_focus_loss,
    resolve_confidence,
    focus_action_q_bias,
    belief_std_confidence,
)

torch, nn = try_import_torch(error=True)

logger = logging.getLogger(__name__)

PRESERVED_DIM = 13
CAMERA_STATE_DIM_PRIVATE = 9
TARGET_STATE_DIM_PUBLIC = 4
TARGET_STATE_DIM_PRIVATE = 14
OBSTACLE_STATE_DIM = 3
_CAM_LOW = torch.tensor([-2000., -2000., 0., -2000., -2000., 0., 0., 0., 0.])
_CAM_HIGH = torch.tensor([2000., 2000., 1000., 2000., 2000., 180., 2000., 180., 180.])


def _recursive_update(base, updates):
    for key, value in (updates or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _recursive_update(base[key], value)
        else:
            base[key] = value
    return base


def _load_mate_env_config(env_config):
    env_config = env_config or {}
    raw_config = env_config.get("config")
    if isinstance(raw_config, dict):
        mate_config = copy.deepcopy(raw_config)
    elif isinstance(raw_config, str):
        try:
            import yaml
        except ImportError:
            return {}

        candidates = [raw_config]
        if not os.path.isabs(raw_config):
            repo_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "../../../..")
            )
            candidates.append(os.path.join(repo_root, "mate", "assets", raw_config))

        mate_config = {}
        for path in candidates:
            if os.path.exists(path):
                with open(path, "r") as f:
                    mate_config = yaml.safe_load(f) or {}
                break
    else:
        mate_config = {}

    overrides = env_config.get("config_overrides", {})
    if isinstance(overrides, dict):
        mate_config = _recursive_update(mate_config, copy.deepcopy(overrides))
    return mate_config


def _infer_focus_env_params(config):
    mate_config = _load_mate_env_config((config or {}).get("env_config", {}))
    camera_ranges = mate_config.get("camera", {}).get("location_random_range", [])
    target_ranges = mate_config.get("target", {}).get("location_random_range", [])
    obstacle_cfg = mate_config.get("obstacle", {})
    obstacle_ranges = obstacle_cfg.get("location_random_range", [])

    inferred = {}
    if camera_ranges:
        inferred["n_agents"] = len(camera_ranges)
    if target_ranges:
        inferred["n_targets"] = len(target_ranges)
    if obstacle_cfg:
        inferred["n_obstacles"] = len(obstacle_ranges)
        inferred["obstacle_transmittance"] = float(
            obstacle_cfg.get("transmittance", 0.0)
        )
    return inferred


def resolve_focus_config(config):
    focus_config = copy.deepcopy((config or {}).get("focus", {}))
    focus_config.setdefault("use_env_params", True)

    if focus_config.get("use_env_params", True):
        focus_config.update(_infer_focus_env_params(config))

    focus_config.setdefault("n_targets", 8)
    focus_config.setdefault("n_agents", 4)
    focus_config.setdefault("n_obstacles", 0)
    focus_config.setdefault("obstacle_transmittance", 0.0)
    add_confidence_defaults(focus_config)
    return focus_config


def _denorm_camera(value, idx):
    low = _CAM_LOW[idx].to(value.device)
    high = _CAM_HIGH[idx].to(value.device)
    return (value + 1.0) / 2.0 * (high - low) + low


def _extract_camera_fov(state, n_agents):
    positions, orientations, sight_ranges, half_angles = [], [], [], []
    for i in range(n_agents):
        start = PRESERVED_DIM + i * CAMERA_STATE_DIM_PRIVATE
        x = _denorm_camera(state[:, :, start], 0)
        y = _denorm_camera(state[:, :, start + 1], 1)
        vx = _denorm_camera(state[:, :, start + 3], 3)
        vy = _denorm_camera(state[:, :, start + 4], 4)
        va = _denorm_camera(state[:, :, start + 5], 5)
        positions.append(torch.stack([x, y], dim=-1))
        orientations.append(torch.atan2(vy, vx))
        sight_ranges.append(torch.sqrt(vx ** 2 + vy ** 2 + 1e-8))
        half_angles.append(va * (np.pi / 180.0) / 2.0)
    return (
        torch.stack(positions, dim=2),
        torch.stack(orientations, dim=2),
        torch.stack(sight_ranges, dim=2),
        torch.stack(half_angles, dim=2),
    )


def _extract_target_positions(state, n_agents, n_targets):
    target_start = PRESERVED_DIM + n_agents * CAMERA_STATE_DIM_PRIVATE
    positions = []
    for j in range(n_targets):
        start = target_start + j * TARGET_STATE_DIM_PRIVATE
        x = (state[:, :, start] + 1.0) / 2.0 * 4000.0 - 2000.0
        y = (state[:, :, start + 1] + 1.0) / 2.0 * 4000.0 - 2000.0
        positions.append(torch.stack([x, y], dim=-1))
    return torch.stack(positions, dim=2)




def _extract_target_positions_normalized(state, n_agents, n_targets):
    target_start = PRESERVED_DIM + n_agents * CAMERA_STATE_DIM_PRIVATE
    positions = []
    for j in range(n_targets):
        start = target_start + j * TARGET_STATE_DIM_PRIVATE
        positions.append(state[..., start : start + 2])
    return torch.stack(positions, dim=-2)


def _normalized_target_pos_to_world(target_pos):
    return (target_pos + 1.0) / 2.0 * 4000.0 - 2000.0


def _local_camera_fov_from_obs(obs):
    self_state = obs[..., PRESERVED_DIM : PRESERVED_DIM + CAMERA_STATE_DIM_PRIVATE]
    x = _denorm_camera(self_state[..., 0], 0)
    y = _denorm_camera(self_state[..., 1], 1)
    vx = _denorm_camera(self_state[..., 3], 3)
    vy = _denorm_camera(self_state[..., 4], 4)
    va = _denorm_camera(self_state[..., 5], 5)
    positions = torch.stack([x, y], dim=-1)
    orientations = torch.atan2(vy, vx)
    sight_ranges = torch.sqrt(vx ** 2 + vy ** 2 + 1e-8)
    half_angles = va * (np.pi / 180.0) / 2.0
    return positions, orientations, sight_ranges, half_angles


def _local_visible_target_positions(obs, n_targets):
    target_start = PRESERVED_DIM + CAMERA_STATE_DIM_PRIVATE
    target_block = TARGET_STATE_DIM_PUBLIC + 1
    required_dim = target_start + int(n_targets) * target_block
    if obs.size(-1) < required_dim:
        return None, None
    cam_pos, _, _, _ = _local_camera_fov_from_obs(obs)
    positions, masks = [], []
    for j in range(int(n_targets)):
        start = target_start + j * target_block
        dx = (obs[..., start] + 1.0) / 2.0 * 4000.0 - 2000.0
        dy = (obs[..., start + 1] + 1.0) / 2.0 * 4000.0 - 2000.0
        positions.append(cam_pos + torch.stack([dx, dy], dim=-1))
        masks.append(obs[..., start + TARGET_STATE_DIM_PUBLIC] > 0.0)
    return torch.stack(positions, dim=-2), torch.stack(masks, dim=-1)


def focus_action_bias_from_local_targets(
    q_values, obs, target_pos, focus_config, n_actions, target_mask=None
):
    """Geometry action bias using only local obs and predicted targets."""
    if q_values is None or obs is None or target_pos is None:
        return None
    levels = int(round(int(n_actions) ** 0.5))
    if levels * levels != int(n_actions):
        return None
    leading = q_values.shape[:-1]
    if obs.shape[:-1] != leading or target_pos.shape[:-2] != leading:
        return None
    if target_mask is not None and target_mask.shape != target_pos.shape[:-1]:
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
    obs = obs.to(dtype=q_values.dtype, device=q_values.device)
    target_pos = target_pos.to(dtype=q_values.dtype, device=q_values.device)
    if target_mask is not None:
        target_mask = target_mask.to(dtype=q_values.dtype, device=q_values.device)

    cam_pos, cam_orient, cam_range, cam_half_angle = _local_camera_fov_from_obs(obs)
    rotation_step = float(focus_config.get("action_bias_rotation_step", 5.0))
    zooming_step = float(focus_config.get("action_bias_zooming_step", 2.5))
    min_angle = float(focus_config.get("action_bias_min_viewing_angle", 30.0))
    max_angle = float(focus_config.get("action_bias_max_viewing_angle", 180.0))
    max_range = float(focus_config.get("action_bias_max_sight_range", 1500.0))

    delta_orient = grid[:, 0] * (rotation_step * np.pi / 180.0)
    orient = cam_orient.unsqueeze(-1) + delta_orient.view(*([1] * len(leading)), n_actions)
    own_range = cam_range.clamp_min(1.0)
    own_half = cam_half_angle.clamp_min(1e-3)
    current_angle = (own_half * 2.0 * 180.0 / np.pi).clamp(min=min_angle, max=max_angle)
    next_angle = current_angle.unsqueeze(-1) + grid[:, 1].view(
        *([1] * len(leading)), n_actions
    ) * zooming_step
    next_angle = next_angle.clamp(min=min_angle, max=max_angle)
    area_product = own_range.square().unsqueeze(-1) * current_angle.unsqueeze(-1)
    next_range = torch.sqrt(area_product / next_angle.clamp_min(1e-3)).clamp(1.0, max_range)
    next_half = next_angle * (np.pi / 180.0) / 2.0

    rel = target_pos.unsqueeze(-3) - cam_pos.unsqueeze(-2).unsqueeze(-2)
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
    per_target_score = angular_score * range_score
    if target_mask is not None:
        per_target_score = per_target_score * target_mask.unsqueeze(-2)
        target_count = target_mask.sum(dim=-1).clamp_min(1.0).unsqueeze(-1)
        score = per_target_score.sum(dim=-1) / target_count
    else:
        score = per_target_score.sum(dim=-1)
    centered = score - score.mean(dim=-1, keepdim=True)
    scale = centered.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-4)
    clip = float(focus_config.get("action_bias_clip", 3.0))
    return (centered / scale).clamp(-clip, clip)

def _extract_obstacles(state, n_agents, n_targets, n_obstacles):
    obstacle_start = PRESERVED_DIM + n_agents * CAMERA_STATE_DIM_PRIVATE
    obstacle_start += n_targets * TARGET_STATE_DIM_PRIVATE
    positions, radii = [], []
    for o in range(n_obstacles):
        start = obstacle_start + o * OBSTACLE_STATE_DIM
        x = (state[:, :, start] + 1.0) / 2.0 * 4000.0 - 2000.0
        y = (state[:, :, start + 1] + 1.0) / 2.0 * 4000.0 - 2000.0
        radius = (state[:, :, start + 2] + 1.0) / 2.0 * 1000.0
        positions.append(torch.stack([x, y], dim=-1))
        radii.append(radius)
    return torch.stack(positions, dim=2), torch.stack(radii, dim=2)


class LearnedOccupancyModel(nn.Module):
    """Learned multi-horizon future occupancy belief over target positions."""

    def __init__(
        self,
        state_dim,
        n_agents,
        n_targets,
        horizon=3,
        hidden_dim=256,
        max_delta=400.0,
        min_std=25.0,
        architecture="mlp",
        num_layers=1,
        dropout=0.0,
    ):
        super(LearnedOccupancyModel, self).__init__()
        self.state_dim = int(np.prod(state_dim))
        self.n_agents = n_agents
        self.n_targets = n_targets
        self.horizon = horizon
        self.max_delta = max_delta
        self.min_std = min_std
        self.architecture = str(architecture).lower()
        self.num_layers = int(num_layers)

        if self.architecture == "mlp":
            self.net = nn.Sequential(
                nn.Linear(self.state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, horizon * n_targets * 4),
            )
        elif self.architecture == "lstm":
            self.rnn = nn.LSTM(
                input_size=self.state_dim,
                hidden_size=hidden_dim,
                num_layers=self.num_layers,
                batch_first=True,
                dropout=float(dropout) if self.num_layers > 1 else 0.0,
            )
            self.head = nn.Linear(hidden_dim, horizon * n_targets * 4)
        else:
            raise ValueError(f"Unknown belief architecture: {architecture}")

    def forward(self, state):
        B, T = state.shape[:2]
        current_pos = _extract_target_positions(state, self.n_agents, self.n_targets)
        state_flat = state.reshape(B, T, self.state_dim)
        if self.architecture == "mlp":
            out = self.net(state_flat.reshape(-1, self.state_dim)).view(
                B, T, self.horizon, self.n_targets, 4
            )
        elif self.architecture == "lstm":
            # Avoid cuDNN LSTM kernels here: some cluster/conda CUDA stacks can
            # report CUDNN_STATUS_VERSION_MISMATCH even when regular Torch CUDA
            # ops work. The belief model is small enough for the native kernel.
            with torch.backends.cudnn.flags(enabled=False):
                features, _ = self.rnn(state_flat)
            out = self.head(features).view(B, T, self.horizon, self.n_targets, 4)
        else:
            raise ValueError(f"Unknown belief architecture: {self.architecture}")
        delta = torch.tanh(out[..., :2]) * self.max_delta
        std = F.softplus(out[..., 2:]) + self.min_std
        mean = current_pos.unsqueeze(2) + delta
        return mean, std

    def nll(self, state, next_state):
        mean, std = self.forward(state)
        current_pos = _extract_target_positions(state, self.n_agents, self.n_targets)
        B, T = state.shape[:2]
        per_horizon_losses = []
        per_horizon_valid = []
        per_horizon_position_errors = []
        per_horizon_baseline_position_errors = []
        per_horizon_pred_stds = []
        for h in range(self.horizon):
            valid_t = T - h
            if valid_t <= 0:
                break
            future_state = next_state[:, h:, :]
            target = _extract_target_positions(future_state, self.n_agents, self.n_targets)
            pred_mean = mean[:, :valid_t, h, :, :]
            pred_std = std[:, :valid_t, h, :, :]
            z = (target - pred_mean) / pred_std
            nll = 0.5 * (z ** 2) + torch.log(pred_std) + 0.5 * np.log(2.0 * np.pi)
            per_step = nll.sum(dim=-1).mean(dim=-1)
            position_error = torch.linalg.norm(target - pred_mean, dim=-1).mean(dim=-1)
            baseline_pos = current_pos[:, :valid_t, :, :]
            baseline_position_error = torch.linalg.norm(target - baseline_pos, dim=-1).mean(dim=-1)
            pred_std_mean = pred_std.mean(dim=(-1, -2))
            padded = torch.zeros((B, T), dtype=per_step.dtype, device=per_step.device)
            padded_position_error = torch.zeros((B, T), dtype=per_step.dtype, device=per_step.device)
            padded_baseline_position_error = torch.zeros((B, T), dtype=per_step.dtype, device=per_step.device)
            padded_pred_std = torch.zeros((B, T), dtype=per_step.dtype, device=per_step.device)
            valid = torch.zeros((B, T), dtype=torch.bool, device=per_step.device)
            padded[:, :valid_t] = per_step
            padded_position_error[:, :valid_t] = position_error
            padded_baseline_position_error[:, :valid_t] = baseline_position_error
            padded_pred_std[:, :valid_t] = pred_std_mean
            valid[:, :valid_t] = True
            per_horizon_losses.append(padded)
            per_horizon_valid.append(valid)
            per_horizon_position_errors.append(padded_position_error)
            per_horizon_baseline_position_errors.append(padded_baseline_position_error)
            per_horizon_pred_stds.append(padded_pred_std)
        diagnostics = {
            "position_error": per_horizon_position_errors,
            "baseline_position_error": per_horizon_baseline_position_errors,
            "pred_std": per_horizon_pred_stds,
        }
        return per_horizon_losses, per_horizon_valid, mean, std, diagnostics


class LocalTargetBeliefModel(nn.Module):
    """Predict global target positions from each agent's local recurrent state."""

    def __init__(
        self,
        hidden_dim,
        n_targets,
        hidden_size=512,
        num_layers=2,
        dropout=0.0,
        min_std=0.05,
    ):
        super(LocalTargetBeliefModel, self).__init__()
        self.n_targets = int(n_targets)
        self.min_std = float(min_std)
        layers = []
        in_dim = int(hidden_dim)
        for _ in range(max(int(num_layers), 1)):
            layers.extend([nn.Linear(in_dim, int(hidden_size)), nn.LayerNorm(int(hidden_size)), nn.ReLU()])
            if float(dropout) > 0.0:
                layers.append(nn.Dropout(float(dropout)))
            in_dim = int(hidden_size)
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, self.n_targets * 4)

    def forward(self, hidden_features):
        leading = hidden_features.shape[:-1]
        flat = hidden_features.reshape(-1, hidden_features.size(-1))
        out = self.head(self.trunk(flat)).view(*leading, self.n_targets, 4)
        mean = torch.tanh(out[..., :2])
        std = F.softplus(out[..., 2:]) + self.min_std
        return mean, std

    def loss(self, hidden_features, state, valid, obs=None):
        eps = 1e-8
        mean, std = self.forward(hidden_features)
        n_agents = hidden_features.size(-2)
        n_targets = self.n_targets
        target = _extract_target_positions_normalized(state, n_agents, n_targets)
        target = target.unsqueeze(-3).expand_as(mean)
        valid_agents = valid.unsqueeze(-1).expand(*valid.shape, n_agents).bool()

        z = (target - mean) / (std + eps)
        per_target_nll = (
            0.5 * z.square() + torch.log(std + eps) + 0.5 * np.log(2.0 * np.pi)
        ).sum(dim=-1)
        per_agent_nll = per_target_nll.mean(dim=-1)
        per_target_mse = (target - mean).square().sum(dim=-1)
        per_agent_mse = per_target_mse.mean(dim=-1)

        visible_target_mask = None
        if obs is not None:
            _, observed_visible = _local_visible_target_positions(obs, n_targets)
            if observed_visible is not None and observed_visible.shape == mean.shape[:-1]:
                visible_target_mask = observed_visible.to(device=mean.device, dtype=torch.bool)
                visible_target_mask = visible_target_mask & valid_agents.unsqueeze(-1)

        if not valid_agents.any():
            zero = hidden_features.new_zeros(())
            return zero, mean, std, {
                "focus_local_belief_loss": 0.0,
                "focus_local_belief_nll": 0.0,
                "focus_local_belief_mse": 0.0,
                "focus_local_belief_visible_nll": 0.0,
                "focus_local_belief_visible_mse": 0.0,
                "focus_local_belief_visible_target_ratio": 0.0,
                "focus_local_belief_valid_ratio": 0.0,
            }

        nll_loss = per_agent_nll[valid_agents].mean()
        mse_loss = per_agent_mse[valid_agents].mean()
        nll_coeff = float(getattr(self, "nll_coeff", 1.0))
        mse_coeff = float(getattr(self, "mse_coeff", 1.0))
        visible_nll_coeff = float(getattr(self, "visible_nll_coeff", 0.0))
        visible_mse_coeff = float(getattr(self, "visible_mse_coeff", 0.0))
        visible_nll_loss = hidden_features.new_zeros(())
        visible_mse_loss = hidden_features.new_zeros(())
        if visible_target_mask is not None and visible_target_mask.any():
            visible_nll_loss = per_target_nll[visible_target_mask].mean()
            visible_mse_loss = per_target_mse[visible_target_mask].mean()

        loss = (
            nll_coeff * nll_loss
            + mse_coeff * mse_loss
            + visible_nll_coeff * visible_nll_loss
            + visible_mse_coeff * visible_mse_loss
        )

        world_per_target_err = torch.linalg.norm(
            _normalized_target_pos_to_world(mean.detach())
            - _normalized_target_pos_to_world(target.detach()),
            dim=-1,
        )
        world_err = world_per_target_err.mean(dim=-1)
        baseline = torch.zeros_like(mean.detach())
        baseline_err = torch.linalg.norm(
            _normalized_target_pos_to_world(baseline)
            - _normalized_target_pos_to_world(target.detach()),
            dim=-1,
        ).mean(dim=-1)
        std_world = (std.detach() * 2000.0).mean(dim=(-1, -2))

        valid_target_mask = valid_agents.unsqueeze(-1).expand_as(per_target_mse)
        visible_ratio = hidden_features.new_zeros(())
        visible_pos_error = hidden_features.new_zeros(())
        invisible_pos_error = hidden_features.new_zeros(())
        if visible_target_mask is not None:
            denom = valid_target_mask.float().sum().clamp_min(1.0)
            visible_ratio = visible_target_mask.float().sum() / denom
            if visible_target_mask.any():
                visible_pos_error = world_per_target_err[visible_target_mask].mean()
            invisible_mask = valid_target_mask & (~visible_target_mask)
            if invisible_mask.any():
                invisible_pos_error = world_per_target_err[invisible_mask].mean()

        stats = {
            "focus_local_belief_loss": loss.detach().item(),
            "focus_local_belief_nll": nll_loss.detach().item(),
            "focus_local_belief_mse": mse_loss.detach().item(),
            "focus_local_belief_visible_nll": visible_nll_loss.detach().item(),
            "focus_local_belief_visible_mse": visible_mse_loss.detach().item(),
            "focus_local_belief_visible_nll_coeff": visible_nll_coeff,
            "focus_local_belief_visible_mse_coeff": visible_mse_coeff,
            "focus_local_belief_pos_error": world_err[valid_agents].mean().detach().item(),
            "focus_local_belief_visible_pos_error": visible_pos_error.detach().item(),
            "focus_local_belief_invisible_pos_error": invisible_pos_error.detach().item(),
            "focus_local_belief_visible_target_ratio": visible_ratio.detach().item(),
            "focus_local_belief_origin_baseline_pos_error": baseline_err[valid_agents].mean().detach().item(),
            "focus_local_belief_pred_std_world": std_world[valid_agents].mean().detach().item(),
            "focus_local_belief_valid_ratio": valid_agents.float().mean().detach().item(),
        }
        return loss, mean, std, stats


class QPLEXFocusLoss(nn.Module):
    def __init__(
        self,
        model,
        target_model,
        mixer,
        target_mixer,
        n_agents,
        n_actions,
        double_q=True,
        gamma=0.99,
        focus_config=None,
        occupancy_model=None,
        local_belief_model=None,
    ):
        nn.Module.__init__(self)
        self.model = model
        self.target_model = target_model
        self.mixer = mixer
        self.target_mixer = target_mixer
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.double_q = double_q
        self.gamma = gamma
        self.focus_config = focus_config or {}
        self.occupancy_model = occupancy_model
        self.local_belief_model = local_belief_model
        self.last_focus_stats = {}

    def _decode_target_selection(self, actions, n_targets):
        if not self.focus_config.get("use_action_selection", True):
            return None
        if self.n_actions != 2 ** n_targets:
            return None
        bits = []
        for j in range(n_targets):
            shift = n_targets - 1 - j
            bits.append(((actions.long() // (2 ** shift)) % 2).float())
        return torch.stack(bits, dim=-1)

    def _legacy_sigma_points(self, mean, std):
        x = torch.stack([std[..., 0], torch.zeros_like(std[..., 0])], dim=-1)
        y = torch.stack([torch.zeros_like(std[..., 1]), std[..., 1]], dim=-1)
        points = torch.stack([mean, mean + x, mean - x, mean + y, mean - y], dim=4)
        weights = torch.full((5,), 1.0 / 5.0, device=mean.device, dtype=mean.dtype)
        return points, weights

    def _gauss_hermite_sigma_points(self, mean, std, order):
        nodes_np, weights_np = np.polynomial.hermite.hermgauss(order)
        nodes = torch.as_tensor(nodes_np, device=mean.device, dtype=mean.dtype)
        weights_1d = torch.as_tensor(weights_np, device=mean.device, dtype=mean.dtype)
        nodes = np.sqrt(2.0) * nodes
        weights_1d = weights_1d / np.sqrt(np.pi)

        yy, xx = torch.meshgrid(nodes, nodes, indexing="ij")
        wy, wx = torch.meshgrid(weights_1d, weights_1d, indexing="ij")
        normals = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
        weights = (wx * wy).reshape(-1)
        weights = weights / (weights.sum() + self.focus_config.get("eps", 1e-8))
        points = mean.unsqueeze(4) + std.unsqueeze(4) * normals.view(1, 1, 1, 1, -1, 2)
        return points, weights

    def _sigma_points(self, mean, std):
        method = str(self.focus_config.get("sigma_method", "legacy5")).lower()
        if method in ("legacy5", "legacy_5", "legacy"):
            return self._legacy_sigma_points(mean, std)
        if method in ("gauss_hermite", "gh"):
            order = int(self.focus_config.get("sigma_order", 3))
            if order < 1:
                raise ValueError(f"sigma_order must be >= 1, got {order}")
            return self._gauss_hermite_sigma_points(mean, std, order)
        raise ValueError(f"Unknown sigma_method: {method}")

    def _mc_points(self, mean, std):
        eps = self.focus_config.get("eps", 1e-8)
        num_points = int(self.focus_config.get("mc_num_points", 128))
        seed = int(self.focus_config.get("mc_seed", 0))
        normal = self._mc_normals(num_points, mean.device, mean.dtype, seed, eps)
        return mean.unsqueeze(4) + std.unsqueeze(4) * normal.view(1, 1, 1, 1, num_points, 2)

    def _mc_normals(self, num_points, device, dtype, seed=None, eps=None):
        if eps is None:
            eps = self.focus_config.get("eps", 1e-8)
        if seed is None:
            seed = int(self.focus_config.get("mc_seed", 0))
        engine = torch.quasirandom.SobolEngine(dimension=2, scramble=True, seed=seed)
        uniforms = engine.draw(num_points).to(device=device, dtype=dtype)
        uniforms = uniforms.clamp(min=eps, max=1.0 - eps)
        return np.sqrt(2.0) * torch.erfinv(2.0 * uniforms - 1.0)

    def _horizon_weights(self, horizon, device, dtype):
        discount = float(self.focus_config.get("horizon_discount", 0.9))
        weights = torch.tensor([discount ** h for h in range(horizon)], device=device, dtype=dtype)
        return weights / (weights.sum() + self.focus_config.get("eps", 1e-8))

    def _grid_points(self, device, dtype):
        grid_size = int(self.focus_config.get("grid_size", 64))
        x_min, x_max = self.focus_config.get("grid_x_range", (-1000.0, 1000.0))
        y_min, y_max = self.focus_config.get("grid_y_range", (-1000.0, 1000.0))
        cell_w = (float(x_max) - float(x_min)) / grid_size
        cell_h = (float(y_max) - float(y_min)) / grid_size
        xs = torch.linspace(
            float(x_min) + 0.5 * cell_w,
            float(x_max) - 0.5 * cell_w,
            grid_size,
            device=device,
            dtype=dtype,
        )
        ys = torch.linspace(
            float(y_min) + 0.5 * cell_h,
            float(y_max) - 0.5 * cell_h,
            grid_size,
            device=device,
            dtype=dtype,
        )
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)

    def _grid_cell_visibility(
        self,
        grid_chunk,
        cam_pos,
        cam_orient,
        cam_range,
        cam_half_angle,
        obstacle_pos=None,
        obstacle_radius=None,
    ):
        eps = self.focus_config.get("eps", 1e-8)
        rel = grid_chunk.view(1, 1, 1, -1, 2) - cam_pos.unsqueeze(3)
        dist = torch.sqrt((rel ** 2).sum(dim=-1) + eps)
        bearing = torch.atan2(rel[..., 1], rel[..., 0])
        angle = torch.atan2(
            torch.sin(bearing - cam_orient.unsqueeze(3)),
            torch.cos(bearing - cam_orient.unsqueeze(3)),
        )
        visible = (
            (dist <= cam_range.unsqueeze(3))
            & (angle.abs() <= cam_half_angle.unsqueeze(3))
        ).float()
        if obstacle_pos is None or obstacle_radius is None or obstacle_pos.size(2) == 0:
            return visible

        ray = grid_chunk.view(1, 1, 1, 1, -1, 2) - cam_pos.unsqueeze(3).unsqueeze(4)
        obs_rel = obstacle_pos.unsqueeze(2).unsqueeze(4) - cam_pos.unsqueeze(3).unsqueeze(4)
        ray_len_sq = (ray ** 2).sum(dim=-1) + eps
        proj = (obs_rel * ray).sum(dim=-1) / ray_len_sq
        closest = proj.unsqueeze(-1) * ray
        dist_to_segment = torch.sqrt(((obs_rel - closest) ** 2).sum(dim=-1) + eps)
        blocked = (
            (proj > 0.0)
            & (proj < 1.0)
            & (dist_to_segment <= obstacle_radius.unsqueeze(2).unsqueeze(4))
        ).any(dim=3)
        transmittance = float(self.focus_config.get("obstacle_transmittance", 0.0))
        return torch.where(blocked, visible * transmittance, visible)

    def _point_visibility(
        self,
        points,
        cam_pos,
        cam_orient,
        cam_range,
        cam_half_angle,
        obstacle_pos=None,
        obstacle_radius=None,
    ):
        eps = self.focus_config.get("eps", 1e-8)
        rel = points.unsqueeze(2) - cam_pos.unsqueeze(3).unsqueeze(4).unsqueeze(5)
        dist = torch.sqrt((rel ** 2).sum(dim=-1) + eps)
        bearing = torch.atan2(rel[..., 1], rel[..., 0])
        angle = torch.atan2(
            torch.sin(bearing - cam_orient.unsqueeze(3).unsqueeze(4).unsqueeze(5)),
            torch.cos(bearing - cam_orient.unsqueeze(3).unsqueeze(4).unsqueeze(5)),
        )
        visible = (
            (dist <= cam_range.unsqueeze(3).unsqueeze(4).unsqueeze(5))
            & (angle.abs() <= cam_half_angle.unsqueeze(3).unsqueeze(4).unsqueeze(5))
        ).float()
        if obstacle_pos is None or obstacle_radius is None or obstacle_pos.size(2) == 0:
            return visible

        ray = points.unsqueeze(2).unsqueeze(5)
        ray = ray - cam_pos.unsqueeze(3).unsqueeze(4).unsqueeze(5).unsqueeze(6)
        obs_rel = obstacle_pos.unsqueeze(2).unsqueeze(3).unsqueeze(4).unsqueeze(6)
        obs_rel = obs_rel - cam_pos.unsqueeze(3).unsqueeze(4).unsqueeze(5).unsqueeze(6)
        ray_len_sq = (ray ** 2).sum(dim=-1) + eps
        proj = (obs_rel * ray).sum(dim=-1) / ray_len_sq
        closest = proj.unsqueeze(-1) * ray
        dist_to_segment = torch.sqrt(((obs_rel - closest) ** 2).sum(dim=-1) + eps)
        blocked = (
            (proj > 0.0)
            & (proj < 1.0)
            & (dist_to_segment <= obstacle_radius.unsqueeze(2).unsqueeze(3).unsqueeze(4).unsqueeze(6))
        ).any(dim=5)
        transmittance = float(self.focus_config.get("obstacle_transmittance", 0.0))
        return torch.where(blocked, visible * transmittance, visible)

    def _gaussian_grid_mass(self, mean, std):
        eps = self.focus_config.get("eps", 1e-8)
        grid = self._grid_points(mean.device, mean.dtype)
        logits = self._gaussian_grid_logits(mean, std, grid)
        return torch.softmax(logits, dim=-1), grid

    def _gaussian_grid_logits(self, mean, std, grid):
        eps = self.focus_config.get("eps", 1e-8)
        diff = grid.view(1, 1, 1, 1, -1, 2) - mean.unsqueeze(-2)
        z = diff / (std.unsqueeze(-2) + eps)
        return -0.5 * (z ** 2).sum(dim=-1) - torch.log(std.unsqueeze(-2) + eps).sum(dim=-1)

    def _credit_from_grid(self, mean, std, state, actions, n_targets):
        eps = self.focus_config.get("eps", 1e-8)
        grid = self._grid_points(mean.device, mean.dtype)
        chunk_size = int(self.focus_config.get("grid_chunk_size", 512))
        chunk_size = max(1, min(chunk_size, grid.size(0)))
        cam_pos, cam_orient, cam_range, cam_half_angle = _extract_camera_fov(state, self.n_agents)
        selection = self._decode_target_selection(actions, n_targets)
        n_obstacles = int(self.focus_config.get("n_obstacles", 0))
        obstacle_pos = obstacle_radius = None
        required_obstacle_dim = (
            PRESERVED_DIM
            + self.n_agents * CAMERA_STATE_DIM_PRIVATE
            + n_targets * TARGET_STATE_DIM_PRIVATE
            + n_obstacles * OBSTACLE_STATE_DIM
        )
        if n_obstacles > 0 and state.size(-1) >= required_obstacle_dim:
            obstacle_pos, obstacle_radius = _extract_obstacles(
                state, self.n_agents, n_targets, n_obstacles
            )

        log_norm = torch.full_like(mean[..., 0], -float("inf"))
        for start in range(0, grid.size(0), chunk_size):
            grid_chunk = grid[start : start + chunk_size]
            logits = self._gaussian_grid_logits(mean, std, grid_chunk)
            log_norm = torch.logaddexp(log_norm, torch.logsumexp(logits, dim=-1))

        horizon_weights = self._horizon_weights(mean.size(2), actions.device, mean.dtype)
        g = mean.new_zeros(mean.size(0), mean.size(1), self.n_agents)
        for start in range(0, grid.size(0), chunk_size):
            grid_chunk = grid[start : start + chunk_size]
            occ = mean.new_zeros(mean.size(0), mean.size(1), n_targets, grid_chunk.size(0))
            for h in range(mean.size(2)):
                logits_h = self._gaussian_grid_logits(
                    mean[:, :, h : h + 1, :, :],
                    std[:, :, h : h + 1, :, :],
                    grid_chunk,
                ).squeeze(2)
                occ_h = torch.exp(logits_h - log_norm[:, :, h, :].unsqueeze(-1))
                occ = occ + horizon_weights[h] * occ_h

            visibility = self._grid_cell_visibility(
                grid_chunk,
                cam_pos,
                cam_orient,
                cam_range,
                cam_half_angle,
                obstacle_pos=obstacle_pos,
                obstacle_radius=obstacle_radius,
            )

            if selection is not None:
                visibility = visibility.unsqueeze(3) * selection.unsqueeze(-1)
            else:
                visibility = visibility.unsqueeze(3).expand(-1, -1, -1, n_targets, -1)

            one_minus = 1.0 - visibility
            all_uncovered_by_others = []
            for i in range(self.n_agents):
                if self.n_agents == 1:
                    all_uncovered_by_others.append(torch.ones_like(visibility[:, :, i, :, :]))
                else:
                    others = torch.cat([one_minus[:, :, :i, :, :], one_minus[:, :, i + 1 :, :, :]], dim=2)
                    all_uncovered_by_others.append(torch.prod(others, dim=2))
            unique_vis = visibility * torch.stack(all_uncovered_by_others, dim=2)
            # unique_vis: [B, T, C, J, g], occ: [B, T, J, g].
            # Horizon is aggregated before credit, keeping peak memory independent of H.
            g = g + torch.einsum("btcjg,btjg->btc", unique_vis, occ)
        return g

    def _credit_chunk_sum(
        self,
        target_samples,
        cam_pos,
        cam_orient,
        cam_range,
        cam_half_angle,
        selection,
        sample_weights=None,
        obstacle_pos=None,
        obstacle_radius=None,
    ):
        visible = self._point_visibility(
            target_samples,
            cam_pos,
            cam_orient,
            cam_range,
            cam_half_angle,
            obstacle_pos=obstacle_pos,
            obstacle_radius=obstacle_radius,
        )

        if selection is not None:
            visible = visible * selection.unsqueeze(3).unsqueeze(-1)

        one_minus = 1.0 - visible
        all_uncovered_by_others = []
        for i in range(self.n_agents):
            if self.n_agents == 1:
                all_uncovered_by_others.append(torch.ones_like(visible[:, :, i, :]))
            else:
                others = torch.cat([one_minus[:, :, :i, :], one_minus[:, :, i + 1 :, :]], dim=2)
                all_uncovered_by_others.append(torch.prod(others, dim=2))
        unique_gain = visible * torch.stack(all_uncovered_by_others, dim=2)
        if sample_weights is not None:
            unique_gain = unique_gain * sample_weights.view(1, 1, 1, 1, 1, -1)
        return unique_gain.sum(dim=-1).sum(dim=-1)

    def _credit_geometry(self, state, n_targets):
        cam_pos, cam_orient, cam_range, cam_half_angle = _extract_camera_fov(state, self.n_agents)
        n_obstacles = int(self.focus_config.get("n_obstacles", 0))
        obstacle_pos = obstacle_radius = None
        required_obstacle_dim = (
            PRESERVED_DIM
            + self.n_agents * CAMERA_STATE_DIM_PRIVATE
            + n_targets * TARGET_STATE_DIM_PRIVATE
            + n_obstacles * OBSTACLE_STATE_DIM
        )
        if n_obstacles > 0 and state.size(-1) >= required_obstacle_dim:
            obstacle_pos, obstacle_radius = _extract_obstacles(
                state, self.n_agents, n_targets, n_obstacles
            )
        return cam_pos, cam_orient, cam_range, cam_half_angle, obstacle_pos, obstacle_radius

    def _credit_from_sigma_points(
        self,
        target_samples,
        state,
        actions,
        n_targets,
        sample_weights=None,
    ):
        cam_pos, cam_orient, cam_range, cam_half_angle, obstacle_pos, obstacle_radius = (
            self._credit_geometry(state, n_targets)
        )
        selection = self._decode_target_selection(actions, n_targets)
        horizon_weights = self._horizon_weights(target_samples.size(2), actions.device, target_samples.dtype)
        chunk_size = int(self.focus_config.get("sample_chunk_size", self.focus_config.get("mc_chunk_size", 32)))
        chunk_size = max(1, min(chunk_size, target_samples.size(4)))
        total_samples = target_samples.size(4)
        if sample_weights is not None and sample_weights.numel() != total_samples:
            raise ValueError(
                f"Expected {total_samples} sigma weights, got {sample_weights.numel()}"
            )
        g = target_samples.new_zeros(target_samples.size(0), target_samples.size(1), self.n_agents)

        for h in range(target_samples.size(2)):
            for start in range(0, total_samples, chunk_size):
                end = start + chunk_size
                samples = target_samples[:, :, h : h + 1, :, start:end, :]
                chunk_weights = None
                if sample_weights is not None:
                    chunk_weights = sample_weights[start:end]
                chunk_sum = self._credit_chunk_sum(
                    samples,
                    cam_pos,
                    cam_orient,
                    cam_range,
                    cam_half_angle,
                    selection,
                    sample_weights=chunk_weights,
                    obstacle_pos=obstacle_pos,
                    obstacle_radius=obstacle_radius,
                ).squeeze(3)
                if sample_weights is None:
                    chunk_sum = chunk_sum / float(total_samples)
                g = g + horizon_weights[h] * chunk_sum
        return g

    def _credit_from_mc_points(self, mean, std, state, actions, n_targets):
        cam_pos, cam_orient, cam_range, cam_half_angle, obstacle_pos, obstacle_radius = (
            self._credit_geometry(state, n_targets)
        )
        selection = self._decode_target_selection(actions, n_targets)
        num_points = int(self.focus_config.get("mc_num_points", 128))
        chunk_size = int(self.focus_config.get("mc_chunk_size", self.focus_config.get("sample_chunk_size", 32)))
        chunk_size = max(1, min(chunk_size, num_points))
        normals = self._mc_normals(
            num_points,
            mean.device,
            mean.dtype,
            seed=int(self.focus_config.get("mc_seed", 0)),
            eps=self.focus_config.get("eps", 1e-8),
        )
        horizon_weights = self._horizon_weights(mean.size(2), actions.device, mean.dtype)
        g = mean.new_zeros(mean.size(0), mean.size(1), self.n_agents)

        for h in range(mean.size(2)):
            mean_h = mean[:, :, h : h + 1, :, :]
            std_h = std[:, :, h : h + 1, :, :]
            for start in range(0, num_points, chunk_size):
                normal_chunk = normals[start : start + chunk_size]
                samples = mean_h.unsqueeze(4) + std_h.unsqueeze(4) * normal_chunk.view(
                    1, 1, 1, 1, -1, 2
                )
                chunk_sum = self._credit_chunk_sum(
                    samples,
                    cam_pos,
                    cam_orient,
                    cam_range,
                    cam_half_angle,
                    selection,
                    obstacle_pos=obstacle_pos,
                    obstacle_radius=obstacle_radius,
                ).squeeze(3)
                g = g + horizon_weights[h] * chunk_sum / float(num_points)
        return g

    def _focus_credit_target(self, state, next_state, actions, mask):
        eps = self.focus_config.get("eps", 1e-8)
        n_targets = int(self.focus_config.get("n_targets", 8))
        min_signal = float(self.focus_config.get("min_credit_signal", 1e-6))
        belief_loss = torch.zeros((), dtype=torch.float, device=actions.device)
        confidence = torch.ones(actions.shape[:2], dtype=torch.float, device=actions.device)
        confidence_mode = "off"
        self.last_belief_stats = {}

        required_dim = PRESERVED_DIM + self.n_agents * CAMERA_STATE_DIM_PRIVATE + n_targets * TARGET_STATE_DIM_PRIVATE
        if state is None or state.size(-1) < required_dim:
            B, T = actions.shape[:2]
            rho = torch.full(
                (B, T, self.n_agents),
                1.0 / self.n_agents,
                dtype=torch.float,
                device=actions.device,
            )
            valid = torch.zeros((B, T), dtype=torch.bool, device=actions.device)
            total_g = torch.zeros((B, T), dtype=torch.float, device=actions.device)
            return rho, valid, total_g, belief_loss, confidence, confidence_mode

        belief_mode = self.focus_config.get("belief_mode", "learned")
        if belief_mode == "learned":
            if self.occupancy_model is None or next_state is None:
                B, T = actions.shape[:2]
                rho = torch.full(
                    (B, T, self.n_agents),
                    1.0 / self.n_agents,
                    dtype=torch.float,
                    device=actions.device,
                )
                valid = torch.zeros((B, T), dtype=torch.bool, device=actions.device)
                total_g = torch.zeros((B, T), dtype=torch.float, device=actions.device)
                return rho, valid, total_g, belief_loss, confidence, confidence_mode
            per_horizon_losses, per_horizon_valid, mean, std, diagnostics = self.occupancy_model.nll(state, next_state)
            valid_belief = mask[:, :, 0] > 0.0
            horizon_weights = self._horizon_weights(mean.size(2), actions.device, mean.dtype)
            weighted_losses = []
            per_step_belief_loss = torch.zeros_like(valid_belief, dtype=mean.dtype)
            weighted_position_error = torch.zeros_like(valid_belief, dtype=mean.dtype)
            weighted_baseline_position_error = torch.zeros_like(valid_belief, dtype=mean.dtype)
            weighted_pred_std = torch.zeros_like(valid_belief, dtype=mean.dtype)
            belief_stats = {}
            for h, (nll_per_step, valid_h) in enumerate(zip(per_horizon_losses, per_horizon_valid)):
                per_step_belief_loss = per_step_belief_loss + horizon_weights[h] * nll_per_step
                position_error = diagnostics["position_error"][h]
                baseline_position_error = diagnostics["baseline_position_error"][h]
                pred_std = diagnostics["pred_std"][h]
                weighted_position_error = weighted_position_error + horizon_weights[h] * position_error
                weighted_baseline_position_error = (
                    weighted_baseline_position_error + horizon_weights[h] * baseline_position_error
                )
                weighted_pred_std = weighted_pred_std + horizon_weights[h] * pred_std
                valid_h = valid_h & valid_belief
                if valid_h.any():
                    weighted_losses.append(horizon_weights[h] * nll_per_step[valid_h].mean())
                    horizon_id = h + 1
                    model_err = position_error[valid_h].mean()
                    baseline_err = baseline_position_error[valid_h].mean()
                    std_mean = pred_std[valid_h].mean()
                    belief_stats[f"focus_belief_loss_h{horizon_id}"] = nll_per_step[valid_h].mean().detach().item()
                    belief_stats[f"focus_belief_pos_error_h{horizon_id}"] = model_err.detach().item()
                    belief_stats[f"focus_belief_baseline_pos_error_h{horizon_id}"] = baseline_err.detach().item()
                    belief_stats[f"focus_belief_pos_error_improvement_h{horizon_id}"] = (
                        baseline_err - model_err
                    ).detach().item()
                    belief_stats[f"focus_belief_pred_std_h{horizon_id}"] = std_mean.detach().item()
                    belief_stats[f"focus_belief_error_to_std_ratio_h{horizon_id}"] = (
                        model_err / (std_mean + eps)
                    ).detach().item()
            if weighted_losses:
                belief_loss = torch.stack(weighted_losses).sum()
            confidence, confidence_mode = resolve_confidence(
                self.focus_config,
                valid_belief,
                per_step_belief_loss,
                default_mode="loss",
                per_step_loss=per_step_belief_loss,
            )
            if valid_belief.any():
                valid_conf = confidence[valid_belief]
                valid_pos_error = weighted_position_error[valid_belief]
                valid_baseline_error = weighted_baseline_position_error[valid_belief]
                valid_pred_std = weighted_pred_std[valid_belief]
                belief_stats["focus_belief_pos_error_weighted"] = valid_pos_error.mean().detach().item()
                belief_stats["focus_belief_baseline_pos_error_weighted"] = (
                    valid_baseline_error.mean().detach().item()
                )
                belief_stats["focus_belief_pos_error_improvement_weighted"] = (
                    valid_baseline_error.mean() - valid_pos_error.mean()
                ).detach().item()
                belief_stats["focus_belief_pred_std_weighted"] = valid_pred_std.mean().detach().item()
                belief_stats["focus_belief_error_to_std_ratio_weighted"] = (
                    valid_pos_error.mean() / (valid_pred_std.mean() + eps)
                ).detach().item()
                if valid_conf.numel() > 1:
                    centered_conf = valid_conf - valid_conf.mean()
                    centered_err = valid_pos_error.detach() - valid_pos_error.detach().mean()
                    denom = centered_conf.std(unbiased=False) * centered_err.std(unbiased=False) + eps
                    belief_stats["focus_belief_confidence_pos_error_corr"] = (
                        (centered_conf * centered_err).mean() / denom
                    ).detach().item()
                    low = torch.quantile(valid_conf, 0.25)
                    high = torch.quantile(valid_conf, 0.75)
                    low_mask = valid_conf <= low
                    high_mask = valid_conf >= high
                    if low_mask.any():
                        belief_stats["focus_belief_low_conf_pos_error"] = (
                            valid_pos_error[low_mask].mean().detach().item()
                        )
                    if high_mask.any():
                        belief_stats["focus_belief_high_conf_pos_error"] = (
                            valid_pos_error[high_mask].mean().detach().item()
                        )
            self.last_belief_stats = belief_stats
            integral_mode = self.focus_config.get("integral_mode", "MC")
            if integral_mode == "grid":
                g = self._credit_from_grid(mean.detach(), std.detach(), state, actions, n_targets)
            elif integral_mode == "sigma":
                target_samples, sample_weights = self._sigma_points(mean, std)
                g = self._credit_from_sigma_points(
                    target_samples.detach(),
                    state,
                    actions,
                    n_targets,
                    sample_weights=sample_weights.detach(),
                )
            elif integral_mode == "MC":
                g = self._credit_from_mc_points(mean.detach(), std.detach(), state, actions, n_targets)
            else:
                raise ValueError(f"Unknown FOCUS integral_mode: {integral_mode}")
        elif belief_mode == "oracle_next_ablation":
            target_samples = _extract_target_positions(next_state, self.n_agents, n_targets)
            target_samples = target_samples.unsqueeze(2).unsqueeze(4)
            g = self._credit_from_sigma_points(target_samples, state, actions, n_targets)
        else:
            raise ValueError(f"Unknown FOCUS belief_mode: {belief_mode}")
        total_g = g.sum(dim=-1)
        valid = (total_g > min_signal) & (mask[:, :, 0] > 0.0)
        rho = g / (total_g.unsqueeze(-1) + eps)
        uniform = torch.full_like(rho, 1.0 / self.n_agents)
        rho = torch.where(valid.unsqueeze(-1), rho, uniform)
        return rho.detach(), valid.detach(), total_g.detach(), belief_loss, confidence.detach(), confidence_mode

    def _signal_confidence_weights(self, total_g, valid):
        eps = self.focus_config.get("eps", 1e-8)
        weights = total_g.detach()
        if not self.focus_config.get("use_signal_confidence", True):
            return torch.ones_like(weights)
        if valid.any():
            weights = weights / (weights[valid].mean() + eps)
        else:
            weights = torch.ones_like(weights)
        min_weight = float(self.focus_config.get("signal_weight_min", 0.1))
        max_weight = float(self.focus_config.get("signal_weight_max", 3.0))
        return weights.clamp(min=min_weight, max=max_weight)

    def _weighted_focus_loss(self, per_step_loss, valid, total_g, reference_loss, confidence=None):
        weights = self._signal_confidence_weights(total_g, valid)
        if confidence is None:
            confidence = torch.ones_like(weights)
        focus_loss = gated_focus_loss(
            per_step_loss,
            valid,
            confidence,
            reference_loss,
            eps=self.focus_config.get("eps", 1e-8),
            weights=weights,
        )
        return focus_loss, weights

    def _student_action_logits(self, hidden_features):
        if hidden_features is None or not hasattr(self.model, "focus_bias"):
            return None
        return self.model.focus_bias(hidden_features)

    def _mask_action_logits(self, logits, action_mask):
        if action_mask is None:
            return logits
        return logits.masked_fill(action_mask <= 0.0, -1e9)

    def _belief_action_confidence(self, state):
        if (
            not self.focus_config.get("action_bias_confidence_enabled", True)
            or self.occupancy_model is None
            or state is None
        ):
            return None, {}
        try:
            with torch.no_grad():
                _, std = self.occupancy_model.forward(state)
                confidence, std_mean = belief_std_confidence(std, self.focus_config)
        except (RuntimeError, ValueError):
            return None, {}
        values = confidence.reshape(-1)
        std_values = std_mean.reshape(-1)
        return confidence.detach(), {
            "focus_action_bias_confidence_mean": values.mean().detach().item(),
            "focus_action_bias_confidence_min": values.min().detach().item(),
            "focus_action_bias_confidence_max": values.max().detach().item(),
            "focus_action_bias_belief_std_mean": std_values.mean().detach().item(),
            "focus_action_bias_confidence_source": "belief_std",
        }

    def _apply_confidence_to_action_bias(self, action_bias, state):
        confidence, stats = self._belief_action_confidence(state)
        if confidence is None:
            return action_bias, stats
        while confidence.dim() < action_bias.dim() - 1:
            confidence = confidence.unsqueeze(-1)
        return action_bias * confidence.unsqueeze(-1), stats

    def _local_belief_loss(self, hidden_features, state, valid, obs=None):
        zero = hidden_features.new_zeros(())
        stats = {
            "focus_local_belief_loss": 0.0,
            "focus_local_belief_nll": 0.0,
            "focus_local_belief_mse": 0.0,
            "focus_local_belief_valid_ratio": 0.0,
        }
        if (
            self.local_belief_model is None
            or state is None
            or not self.focus_config.get("local_belief_enabled", False)
        ):
            return zero, None, None, stats
        self.local_belief_model.nll_coeff = float(
            self.focus_config.get("local_belief_nll_coeff", 0.1)
        )
        self.local_belief_model.mse_coeff = float(
            self.focus_config.get("local_belief_mse_coeff", 1.0)
        )
        self.local_belief_model.visible_nll_coeff = float(
            self.focus_config.get("local_belief_visible_nll_coeff", 0.0)
        )
        self.local_belief_model.visible_mse_coeff = float(
            self.focus_config.get("local_belief_visible_mse_coeff", 0.0)
        )
        loss, mean, std, stats = self.local_belief_model.loss(
            hidden_features, state, valid, obs=obs
        )
        return loss, mean, std, stats

    def _local_belief_action_bias(self, q_values, obs, hidden_features, apply_confidence=True):
        if self.local_belief_model is None or hidden_features is None:
            return None, {}
        mean, std = self.local_belief_model(hidden_features)
        target_pos = _normalized_target_pos_to_world(mean)
        visible_ratio = target_pos.new_tensor(0.0)
        visible_step = None
        target_mask = None
        if self.focus_config.get("local_belief_use_visible_obs", True):
            observed_pos, visible_mask = _local_visible_target_positions(
                obs, int(self.focus_config.get("n_targets", 8))
            )
            if observed_pos is not None and observed_pos.shape == target_pos.shape:
                visible_mask = visible_mask.to(device=target_pos.device)
                visible = visible_mask.unsqueeze(-1).to(dtype=torch.bool)
                target_pos = torch.where(
                    visible, observed_pos.to(dtype=target_pos.dtype, device=target_pos.device), target_pos
                )
                visible_ratio = visible_mask.to(dtype=target_pos.dtype).mean()
                visible_step = visible_mask.any(dim=-1).to(dtype=target_pos.dtype)
                fallback_mask = torch.ones_like(visible_mask, dtype=torch.bool)
                target_mask = torch.where(visible_step.unsqueeze(-1).to(dtype=torch.bool), visible_mask, fallback_mask)
        bias = focus_action_bias_from_local_targets(
            q_values, obs, target_pos, self.focus_config, self.n_actions, target_mask=target_mask
        )
        if bias is None:
            return None, {}
        raw_bias = bias
        confidence, std_mean = belief_std_confidence(std.detach() * 2000.0, self.focus_config)
        if visible_step is not None:
            visible_step = visible_step.to(confidence.device, confidence.dtype)
            if confidence.dim() == visible_step.dim() - 1:
                confidence = confidence.unsqueeze(-1).expand_as(visible_step)
            confidence = torch.maximum(confidence, visible_step)
        values = confidence.reshape(-1)
        if apply_confidence:
            while confidence.dim() < bias.dim() - 1:
                confidence = confidence.unsqueeze(-1)
            bias = bias * confidence.unsqueeze(-1)
        return bias, {
            "focus_local_belief_action_bias_mean": bias.mean().detach().item(),
            "focus_local_belief_action_bias_std": bias.std(unbiased=False).detach().item(),
            "focus_local_belief_action_raw_bias_mean": raw_bias.mean().detach().item(),
            "focus_local_belief_action_raw_bias_std": raw_bias.std(unbiased=False).detach().item(),
            "focus_local_belief_action_confidence_mean": values.mean().detach().item(),
            "focus_local_belief_action_confidence_min": values.min().detach().item(),
            "focus_local_belief_action_confidence_max": values.max().detach().item(),
            "focus_local_belief_action_confidence_applied": bool(apply_confidence),
            "focus_local_belief_action_visible_target_ratio": visible_ratio.detach().item(),
            "focus_local_belief_action_visible_step_ratio": (
                visible_step.mean().detach().item() if visible_step is not None else 0.0
            ),
            "focus_local_belief_action_std_world_mean": std_mean.mean().detach().item(),
        }

    def _action_distill_loss(
        self,
        q_values,
        hidden_features,
        obs,
        state,
        action_mask,
        valid_mask,
    ):
        eps = self.focus_config.get("eps", 1e-8)
        zero = q_values.new_zeros(())
        stats = {
            "focus_action_distill_loss": 0.0,
            "focus_action_distill_policy_q_loss": 0.0,
            "focus_action_distill_bias_loss": 0.0,
            "focus_local_belief_action_distill_loss": 0.0,
            "focus_local_action_student_distill_loss": 0.0,
            "focus_action_distill_valid_ratio": 0.0,
        }
        if not self.focus_config.get("action_distill_enabled", False):
            return zero, stats

        action_bias_logits = focus_action_q_bias(
            q_values.detach(),
            state,
            self.focus_config,
            self.n_agents,
            self.n_actions,
            require_eta=False,
        )
        teacher_confidence_stats = {}
        if action_bias_logits is not None:
            action_bias_logits, teacher_confidence_stats = self._apply_confidence_to_action_bias(
                action_bias_logits, state
            )
        if action_bias_logits is None:
            return zero, stats

        eta = float(self.focus_config.get("action_bias_eta", 0.0))
        policy_teacher_logits = q_values.detach() + eta * action_bias_logits.detach()
        bias_teacher_logits = action_bias_logits.detach()

        teacher_temperature = max(float(self.focus_config.get("teacher_temperature", 1.0)), eps)
        student_temperature = max(float(self.focus_config.get("student_temperature", 1.0)), eps)

        valid = valid_mask > 0.0
        if action_mask is not None:
            valid = valid & (action_mask.sum(dim=-1) > 0.0)
        if not valid.any():
            return zero, stats

        masked_policy_teacher_logits = self._mask_action_logits(
            policy_teacher_logits, action_mask
        )
        policy_teacher_probs = torch.softmax(
            masked_policy_teacher_logits / teacher_temperature, dim=-1
        ).detach()
        teacher_entropy = -(
            policy_teacher_probs * torch.log(policy_teacher_probs + eps)
        ).sum(dim=-1)
        available_actions = (
            action_mask.sum(dim=-1).clamp_min(2.0)
            if action_mask is not None
            else torch.full_like(teacher_entropy, float(self.n_actions))
        )
        max_entropy = torch.log(available_actions)
        entropy_conf = (1.0 - teacher_entropy / (max_entropy + eps)).clamp(0.0, 1.0).detach()
        confidence_mode = str(self.focus_config.get("action_distill_confidence", "entropy")).lower()
        if confidence_mode in ("off", "none"):
            confidence = torch.ones_like(entropy_conf)
        elif confidence_mode == "entropy":
            confidence = entropy_conf
        else:
            raise ValueError(f"Unknown action_distill_confidence: {confidence_mode}")

        weights = confidence[valid]
        denom = weights.sum()
        if denom.detach().item() <= eps:
            return zero, stats

        mode = str(self.focus_config.get("action_distill_mode", "soft_kl")).lower()

        def component_loss(student_logits, teacher_logits):
            masked_teacher_logits = self._mask_action_logits(teacher_logits, action_mask)
            teacher_probs = torch.softmax(
                masked_teacher_logits / teacher_temperature, dim=-1
            ).detach()
            masked_student_logits = self._mask_action_logits(student_logits, action_mask)
            student_log_probs = F.log_softmax(masked_student_logits / student_temperature, dim=-1)
            if mode in ("hard", "hard_ce", "ce"):
                teacher_action = masked_teacher_logits.argmax(dim=-1)
                per_step = F.cross_entropy(
                    masked_student_logits.reshape(-1, self.n_actions),
                    teacher_action.reshape(-1),
                    reduction="none",
                ).reshape_as(teacher_action)
                metric_kl = F.kl_div(
                    student_log_probs,
                    teacher_probs,
                    reduction="none",
                ).sum(dim=-1)
            elif mode in ("soft_ce", "soft_cross_entropy"):
                per_step = -(teacher_probs * student_log_probs).sum(dim=-1)
                metric_kl = F.kl_div(
                    student_log_probs,
                    teacher_probs,
                    reduction="none",
                ).sum(dim=-1)
            elif mode in ("soft_kl", "kl"):
                per_step = F.kl_div(
                    student_log_probs,
                    teacher_probs,
                    reduction="none",
                ).sum(dim=-1)
                metric_kl = per_step
            else:
                raise ValueError(f"Unknown action_distill_mode: {mode}")
            loss = (per_step[valid] * weights).sum() / (denom + eps)
            top1 = masked_student_logits.argmax(dim=-1)
            teacher_top1 = masked_teacher_logits.argmax(dim=-1)
            return loss, metric_kl, top1, teacher_top1, masked_student_logits

        policy_teacher_top1 = masked_policy_teacher_logits.argmax(dim=-1)
        use_policy_q = self.focus_config.get("action_distill_policy_q_enabled", True)
        policy_q_coeff = float(self.focus_config.get("action_distill_policy_q_coeff", 1.0))
        bias_coeff = float(self.focus_config.get("action_distill_bias_coeff", 1.0))
        components = []
        policy_q_loss = zero
        policy_q_kl = torch.zeros_like(valid, dtype=q_values.dtype)
        policy_q_top1 = self._mask_action_logits(q_values.detach(), action_mask).argmax(dim=-1)
        if use_policy_q:
            policy_q_loss, policy_q_kl, policy_q_top1, policy_teacher_top1, _ = component_loss(
                q_values, policy_teacher_logits
            )
            components.append(policy_q_coeff * policy_q_loss)

        student_logits = self._student_action_logits(hidden_features)
        bias_loss = zero
        bias_kl = torch.zeros_like(valid, dtype=q_values.dtype)
        bias_top1 = None
        bias_teacher_top1 = self._mask_action_logits(bias_teacher_logits, action_mask).argmax(dim=-1)
        if student_logits is not None and bias_coeff != 0.0:
            bias_loss, bias_kl, bias_top1, bias_teacher_top1, _ = component_loss(
                student_logits, bias_teacher_logits
            )
            components.append(bias_coeff * bias_loss)

        local_belief_bias, local_belief_bias_stats = self._local_belief_action_bias(
            q_values.detach(), obs, hidden_features, apply_confidence=False
        )
        local_belief_action_coeff = float(
            self.focus_config.get("local_belief_action_distill_coeff", 0.0)
        )
        local_belief_action_loss = zero
        local_belief_action_kl = torch.zeros_like(valid, dtype=q_values.dtype)
        local_belief_action_top1 = None
        if local_belief_bias is not None and local_belief_action_coeff != 0.0:
            (
                local_belief_action_loss,
                local_belief_action_kl,
                local_belief_action_top1,
                _,
                _,
            ) = component_loss(local_belief_bias, bias_teacher_logits)
            components.append(local_belief_action_coeff * local_belief_action_loss)

        local_student_coeff = float(
            self.focus_config.get("local_action_student_distill_coeff", 0.0)
        )
        local_student_loss = zero
        local_student_kl = torch.zeros_like(valid, dtype=q_values.dtype)
        local_student_top1 = None
        local_student_teacher_top1 = None
        if (
            student_logits is not None
            and local_belief_bias is not None
            and local_student_coeff != 0.0
        ):
            (
                local_student_loss,
                local_student_kl,
                local_student_top1,
                local_student_teacher_top1,
                _,
            ) = component_loss(student_logits, local_belief_bias.detach())
            components.append(local_student_coeff * local_student_loss)

        if not components:
            return zero, stats
        loss = torch.stack(components).sum()

        valid_metric = valid
        stats = {
            "focus_action_distill_loss": loss.detach().item(),
            "focus_action_distill_policy_q_loss": policy_q_loss.detach().item(),
            "focus_action_distill_bias_loss": bias_loss.detach().item(),
            "focus_local_belief_action_distill_loss": local_belief_action_loss.detach().item(),
            "focus_local_belief_action_distill_coeff": local_belief_action_coeff,
            "focus_local_action_student_distill_loss": local_student_loss.detach().item(),
            "focus_local_action_student_distill_coeff": local_student_coeff,
            "focus_action_distill_policy_q_enabled": bool(use_policy_q),
            "focus_action_distill_policy_q_coeff": policy_q_coeff,
            "focus_action_distill_bias_coeff": bias_coeff,
            "focus_action_distill_teacher_policy_eta": eta,
            "focus_action_distill_mode": mode,
            "focus_action_distill_confidence_mode": confidence_mode,
            "focus_action_distill_valid_ratio": valid.float().mean().detach().item(),
            "focus_teacher_student_top1_agreement": (
                (policy_teacher_top1 == policy_q_top1).float()[valid_metric].mean().detach().item()
            ),
            "focus_teacher_policy_top1_agreement": (
                (policy_teacher_top1 == policy_q_top1).float()[valid_metric].mean().detach().item()
            ),
            "focus_teacher_q_top1_agreement": (
                (policy_teacher_top1 == policy_q_top1).float()[valid_metric].mean().detach().item()
            ),
            "focus_teacher_student_kl": policy_q_kl[valid_metric].mean().detach().item(),
            "focus_teacher_policy_kl": policy_q_kl[valid_metric].mean().detach().item(),
            "focus_teacher_entropy": teacher_entropy[valid_metric].mean().detach().item(),
            "focus_teacher_confidence": confidence[valid_metric].mean().detach().item(),
            "focus_policy_q_abs_mean": q_values.detach().abs()[valid_metric].mean().item(),
            "focus_policy_q_std": q_values.detach()[valid_metric].std(unbiased=False).item(),
        }
        if student_logits is not None:
            bias_agreement = (
                (bias_teacher_top1 == bias_top1).float()[valid_metric].mean().detach().item()
                if bias_top1 is not None
                else 0.0
            )
            local_student_agreement = (
                (local_student_teacher_top1 == local_student_top1)
                .float()[valid_metric]
                .mean()
                .detach()
                .item()
                if local_student_top1 is not None
                else 0.0
            )
            stats.update(
                {
                    "focus_teacher_bias_top1_agreement": bias_agreement,
                    "focus_teacher_bias_kl": bias_kl[valid_metric].mean().detach().item(),
                    "focus_local_action_student_top1_agreement": local_student_agreement,
                    "focus_local_action_student_kl": local_student_kl[valid_metric].mean().detach().item(),
                    "focus_student_bias_abs_mean": student_logits.detach().abs()[valid_metric].mean().item(),
                    "focus_student_bias_std": student_logits.detach()[valid_metric].std(unbiased=False).item(),
                }
            )
        if local_belief_bias is not None:
            local_agreement = (
                (bias_teacher_top1 == local_belief_action_top1)
                .float()[valid_metric]
                .mean()
                .detach()
                .item()
                if local_belief_action_top1 is not None
                else 0.0
            )
            stats.update(
                {
                    "focus_local_belief_action_top1_agreement": local_agreement,
                    "focus_local_belief_action_kl": local_belief_action_kl[valid_metric].mean().detach().item(),
                }
            )
            stats.update(local_belief_bias_stats)
        stats.update({f"focus_distill_teacher_{k}": v for k, v in teacher_confidence_stats.items()})
        return loss, stats

    def forward(
        self,
        rewards,
        actions,
        terminated,
        mask,
        obs,
        next_obs,
        action_mask,
        next_action_mask,
        state=None,
        next_state=None,
    ):
        """Forward pass of the loss.

        Args:
            rewards: Tensor of shape [B, T, n_agents]
            actions: Tensor of shape [B, T, n_agents]
            terminated: Tensor of shape [B, T, n_agents]
            mask: Tensor of shape [B, T, n_agents]
            obs: Tensor of shape [B, T, n_agents, obs_size]
            next_obs: Tensor of shape [B, T, n_agents, obs_size]
            action_mask: Tensor of shape [B, T, n_agents, n_actions]
            next_action_mask: Tensor of shape [B, T, n_agents, n_actions]
            state: Tensor of shape [B, T, state_dim] (optional)
            next_state: Tensor of shape [B, T, state_dim] (optional)

        According to https://github.com/wjh720/QPLEX/blob/master/pymarl-master/src/learners/dmaq_qatten_learner.py
        We have some notes:
            rewards = batch['reward'][:, :-1]
            actions = batch['actions'][:,:-1]
        """

        # Assert either none or both of state and next_state are given
        if state is None and next_state is None:
            state = obs  # default to state being all agents' observations
            next_state = next_obs
        elif (state is None) != (next_state is None):
            raise ValueError(
                "Expected either neither or both of `state` and "
                "`next_state` to be given. Got: "
                "\n`state` = {}\n`next_state` = {}".format(state, next_state)
            )

        # Calculate estimated Q-Values and local recurrent features.
        mac_out, mac_features = _unroll_mac(
            self.model, obs, return_features=True
        )  # [B, T, n_agents, n_actions], [B, T, n_agents, H]

        chosen_action_qvals = torch.gather(
            mac_out, dim=3, index=actions.unsqueeze(3)
        ).squeeze(3)
        ignore_action = (action_mask == 0) & (mask == 1).unsqueeze(-1)
        x_mac_out = mac_out.clone().detach()
        x_mac_out[ignore_action] = -np.inf
        max_action_vals, max_action_index = x_mac_out.max(dim=3)
        max_action_index = max_action_index.detach().unsqueeze(3)

        # Calculate the Q-Values necessary for the target
        target_mac_out = _unroll_mac(self.target_model, next_obs)

        # Mask out unavailable actions for the t+1 step
        ignore_action_tp1 = (next_action_mask == 0) & (mask == 1).unsqueeze(-1)
        target_mac_out[ignore_action_tp1] = -np.inf


        # Max over target Q-Values
        if self.double_q:
            # Double Q learning computes the target Q values by selecting the
            # t+1 timestep action according to the "policy" neural network and
            # then estimating the Q-value of that action with the "target"
            # neural network

            # Compute the t+1 Q-values to be used in action selection
            # using next_obs
            mac_out_tp1, mac_features_tp1 = _unroll_mac(
                self.model, next_obs, return_features=True
            )

            if self.focus_config.get("student_action_bias_bootstrap", False):
                student_bias_tp1 = self._student_action_logits(mac_features_tp1)
                if student_bias_tp1 is not None:
                    eta = float(self.focus_config.get("student_action_bias_eta", 0.0))
                    if eta != 0.0:
                        mac_out_tp1 = mac_out_tp1 + eta * student_bias_tp1

            # mask out unallowed actions
            if self.focus_config.get("action_bias_bootstrap", True):
                action_bias = focus_action_q_bias(
                    mac_out_tp1,
                    next_state,
                    self.focus_config,
                    self.n_agents,
                    self.n_actions,
                )
                if action_bias is not None:
                    action_bias, bias_confidence_stats = self._apply_confidence_to_action_bias(
                        action_bias, next_state
                    )
                    eta = float(self.focus_config.get("action_bias_eta", 0.0))
                    mac_out_tp1 = mac_out_tp1 + eta * action_bias
                    self.last_focus_stats.update(
                        {f"focus_bootstrap_{k}": v for k, v in bias_confidence_stats.items()}
                    )

            mac_out_tp1[ignore_action_tp1] = -np.inf

            # obtain best actions at t+1 according to policy NN
            cur_max_actions = mac_out_tp1.argmax(dim=3, keepdim=True)

            # use the target network to estimate the Q-values of policy
            # network's selected actions
            target_max_qvals = torch.gather(target_mac_out, 3, cur_max_actions).squeeze(
                3
            )
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        assert (
            target_max_qvals.min().item() != -np.inf
        ), "target_max_qvals contains a masked action; \
            there may be a state with no valid actions."

        # FOCUS: compute the environment-derived responsibility rho BEFORE mixing
        # so it directly replaces the mixer's softmax responsibility factor
        # (instead of being matched against it via a KL/cross-entropy loss).
        focus_enabled = self.focus_config.get("enabled", True) and self.mixer is not None
        rho = None
        valid = total_g = confidence = None
        belief_loss = torch.zeros((), dtype=chosen_action_qvals.dtype, device=chosen_action_qvals.device)
        confidence_mode = "off"
        if focus_enabled:
            rho, valid, total_g, belief_loss, confidence, confidence_mode = self._focus_credit_target(
                state, next_state, actions, mask
            )

        # Mix
        if self.mixer is not None:
            ans_chosen = self.mixer(chosen_action_qvals, state, is_v = True)
            actions_onehot = F.one_hot(actions, num_classes = self.n_actions)
            ans_adv, lambda_weights = self.mixer(
                chosen_action_qvals,
                state,
                actions_onehot,
                max_action_vals=max_action_vals,
                is_v=False,
                return_lambda=True,
                rho=rho,
            )
            chosen_action_qvals = ans_chosen + ans_adv

            target_chosen= self.target_mixer(target_max_qvals, next_state, is_v = True)
            cur_max_actions_onehot = F.one_hot(cur_max_actions, num_classes = self.n_actions)
            target_adv = self.target_mixer(target_max_qvals, next_state, cur_max_actions_onehot, target_max_qvals, is_v = False)
            target_max_qvals = target_chosen + target_adv

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = chosen_action_qvals - targets.detach()

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        td_loss = (masked_td_error ** 2).sum() / mask.sum()
        bootstrap_focus_stats = dict(self.last_focus_stats)
        self.last_focus_stats = bootstrap_focus_stats
        action_distill_loss, action_distill_stats = self._action_distill_loss(
            mac_out,
            mac_features,
            obs,
            state,
            action_mask,
            mask,
        )
        action_distill_coeff = float(self.focus_config.get("action_distill_coeff", 0.05))
        local_belief_valid = mask[:, :, 0] > 0.0
        local_belief_loss, _, _, local_belief_stats = self._local_belief_loss(
            mac_features, state, local_belief_valid, obs=obs
        )
        local_belief_coeff = float(self.focus_config.get("local_belief_coeff", 0.0))
        self.last_focus_stats.update(action_distill_stats)
        self.last_focus_stats.update(local_belief_stats)
        self.last_focus_stats["focus_action_distill_coeff"] = action_distill_coeff
        self.last_focus_stats["focus_local_belief_coeff"] = local_belief_coeff
        if focus_enabled:
            eps = self.focus_config.get("eps", 1e-8)
            beta = float(self.focus_config.get("beta_belief", 0.01))
            lambda_dist = lambda_weights / (lambda_weights.sum(dim=-1, keepdim=True) + eps)
            rho_entropy = -(rho * torch.log(rho + eps)).sum(dim=-1)
            lambda_entropy = -(lambda_dist * torch.log(lambda_dist + eps)).sum(dim=-1)
            signal_weights = self._signal_confidence_weights(total_g, valid)
            valid_signal_weights = signal_weights[valid] if valid.any() else signal_weights.reshape(-1)
            focus_stats = {
                "focus_belief_loss": belief_loss.detach().item(),
                "focus_valid_ratio": valid.float().mean().detach().item(),
                "focus_mean_signal": total_g.mean().detach().item(),
                "focus_signal_weight_mean": valid_signal_weights.mean().detach().item(),
                "focus_rho_entropy": rho_entropy.mean().detach().item(),
                "focus_lambda_entropy": lambda_entropy.mean().detach().item(),
                "focus_beta_belief": beta,
                "focus_action_distill_coeff": action_distill_coeff,
            }
            self.last_focus_stats.update(focus_stats)
            self.last_focus_stats.update(confidence_stats(confidence, valid, confidence_mode))
            self.last_focus_stats.update(self.last_belief_stats)
            # rho is injected directly into the mixer above; the belief model
            # and local action student still need supervised objectives.
            loss = (
                td_loss
                + beta * belief_loss
                + action_distill_coeff * action_distill_loss
                + local_belief_coeff * local_belief_loss
            )
        else:
            loss = td_loss + action_distill_coeff * action_distill_loss + local_belief_coeff * local_belief_loss
        self.last_focus_stats["td_loss"] = td_loss.detach().item()
        return loss, mask, masked_td_error, chosen_action_qvals, targets

def adjust_args(args):
    additional_arguments = {
        'target_update_interval': 200,
        'agent_output_type': "q",
        'double_q': True,

        'hypernet_embed': 64,
        'adv_hypernet_layers': 2,
        'adv_hypernet_embed': 64,
        'ffn_hidden_dim': 64,

        'num_kernel': 5,
        'is_minus_one': True,
        'weighted_head': True,
        'is_adv_attention': True,
        'is_stop_gradient': True,
    }
    for k,v in additional_arguments.items():
        if not hasattr(args, k):
            setattr(args, k, v)
    return args


class QPLEXFocusTorchPolicy(Policy):
    """QPLEX impl. Assumes homogeneous agents for now.

    You must use MultiAgentEnv.with_agent_groups() to group agents
    together for QPLEX. This creates the proper Tuple obs/action spaces and
    populates the '_group_rewards' info field.

    Action masking: to specify an action mask for individual agents, use a
    dict space with an action_mask key, e.g. {"obs": ob, "action_mask": mask}.
    The mask space must be `Box(0, 1, (n_actions,))`.
    Addition arguments for QPLEX:
    'target_update_interval': 200,
    'agent_output_type': "q",
    'double_q': True,

    'hypernet_embed': 64,
    'adv_hypernet_layers': 2,
    'adv_hypernet_embed': 64,

    'num_kernel': 5,
    'is_minus_one': True,
    'weighted_head': True,
    'is_adv_attention': True,
    'is_stop_gradient': True,
    """

    def __init__(self, obs_space, action_space, config):
        _validate(obs_space, action_space)
        config = dict(ray.rllib.agents.qplex_focus.qplex.DEFAULT_CONFIG, **config)

        self.args = Namespace(**config)
        self.args = adjust_args(self.args)
        # print(self.args)
        self.framework = "torch"
        super().__init__(obs_space, action_space, config)
        self.n_agents = len(obs_space.original_space.spaces)
        config["model"]["n_agents"] = self.n_agents
        self.n_actions = action_space.spaces[0].n
        self.h_size = config["model"]["lstm_cell_size"]
        self.has_env_global_state = False
        self.has_action_mask = False
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        agent_obs_space = obs_space.original_space.spaces[0]
        if isinstance(agent_obs_space, Dict):
            space_keys = set(agent_obs_space.spaces.keys())
            if "obs" not in space_keys:
                raise ValueError("Dict obs space must have subspace labeled `obs`")
            self.obs_size = _get_size(agent_obs_space.spaces["obs"])
            if "action_mask" in space_keys:
                mask_shape = tuple(agent_obs_space.spaces["action_mask"].shape)
                if mask_shape != (self.n_actions,):
                    raise ValueError(
                        "Action mask shape must be {}, got {}".format(
                            (self.n_actions,), mask_shape
                        )
                    )
                self.has_action_mask = True
            if ENV_STATE in space_keys:
                self.env_global_state_shape = _get_size(
                    agent_obs_space.spaces[ENV_STATE]
                )
                self.has_env_global_state = True
            else:
                self.env_global_state_shape = (self.obs_size, self.n_agents)
            # The real agent obs space is nested inside the dict
            config["model"]["full_obs_space"] = agent_obs_space
            agent_obs_space = agent_obs_space.spaces["obs"]
        else:
            self.obs_size = _get_size(agent_obs_space)
            self.env_global_state_shape = (self.obs_size, self.n_agents)

        self.model = ModelCatalog.get_model_v2(
            agent_obs_space,
            action_space.spaces[0],
            self.n_actions,
            config["model"],
            framework="torch",
            name="model",
            default_model=RNNModel,
        ).to(self.device)

        self.target_model = ModelCatalog.get_model_v2(
            agent_obs_space,
            action_space.spaces[0],
            self.n_actions,
            config["model"],
            framework="torch",
            name="target_model",
            default_model=RNNModel,
        ).to(self.device)

        self.exploration = self._create_exploration()

        # Setup the mixer network.
        self.mixer = FocusDuelMixer(self.args, self.n_agents, self.n_actions, self.env_global_state_shape, config['mixing_embed_dim'], self.args.ffn_hidden_dim, self.args.num_kernel).to(self.device)
        self.target_mixer = FocusDuelMixer(self.args, self.n_agents, self.n_actions, self.env_global_state_shape, config['mixing_embed_dim'], self.args.ffn_hidden_dim, self.args.num_kernel).to(self.device)
        assert config['mixer'] == 'qplex_focus', f"Expected qplex_focus, get {config['mixer']}"
        focus_config = resolve_focus_config(self.config)
        self.focus_config = focus_config
        self.last_action_bias_stats = {}
        self.occupancy_model = None
        if focus_config.get("enabled", True) and focus_config.get("belief_mode", "learned") == "learned":
            self.occupancy_model = LearnedOccupancyModel(
                self.env_global_state_shape,
                self.n_agents,
                int(focus_config.get("n_targets", 8)),
                horizon=int(focus_config.get("horizon", 3)),
                hidden_dim=int(focus_config.get("belief_hidden_dim", 256)),
                max_delta=float(focus_config.get("belief_max_delta", 400.0)),
                min_std=float(focus_config.get("belief_min_std", 25.0)),
                architecture=focus_config.get("belief_arch", focus_config.get("belief_architecture", "mlp")),
                num_layers=int(focus_config.get("belief_num_layers", 1)),
                dropout=float(focus_config.get("belief_dropout", 0.0)),
            ).to(self.device)

        self.local_belief_model = None
        if focus_config.get("local_belief_enabled", False):
            self.local_belief_model = LocalTargetBeliefModel(
                hidden_dim=self.h_size,
                n_targets=int(focus_config.get("n_targets", 8)),
                hidden_size=int(focus_config.get("local_belief_hidden_dim", 512)),
                num_layers=int(focus_config.get("local_belief_num_layers", 2)),
                dropout=float(focus_config.get("local_belief_dropout", 0.0)),
                min_std=float(focus_config.get("local_belief_min_std", 0.05)),
            ).to(self.device)

        self.cur_epsilon = 1.0
        self.update_target()  # initial sync

        # Setup optimizer
        self.params = list(self.model.parameters())
        if self.mixer:
            self.params += list(self.mixer.parameters())
        if self.occupancy_model:
            self.params += list(self.occupancy_model.parameters())
        if self.local_belief_model:
            self.params += list(self.local_belief_model.parameters())
        self.loss = QPLEXFocusLoss(
            self.model,
            self.target_model,
            self.mixer,
            self.target_mixer,
            self.n_agents,
            self.n_actions,
            self.config["double_q"],
            self.config["gamma"],
            focus_config,
            self.occupancy_model,
            self.local_belief_model,
        )
        from torch.optim import RMSprop

        self.optimiser = RMSprop(
            params=self.params,
            lr=config["lr"],
            alpha=config["optim_alpha"],
            eps=config["optim_eps"],
        )

    @override(Policy)
    def compute_actions(
        self,
        obs_batch,
        state_batches=None,
        prev_action_batch=None,
        prev_reward_batch=None,
        info_batch=None,
        episodes=None,
        explore=None,
        timestep=None,
        **kwargs
    ):
        explore = explore if explore is not None else self.config["explore"]
        obs_batch, action_mask, env_global_state = self._unpack_observation(obs_batch)
        self.last_action_bias_stats = {}
        # CTDE: centralized state may bias exploratory training actions, but
        # decentralized evaluation calls this with explore=False and skips it.

        # Compute actions
        with torch.no_grad():
            q_values, hiddens = _mac(
                self.model,
                torch.as_tensor(obs_batch, dtype=torch.float, device=self.device),
                [
                    torch.as_tensor(np.array(s), dtype=torch.float, device=self.device)
                    for s in state_batches
                ],
            )
            if self.focus_config.get("student_action_bias_enabled", False) and hasattr(
                self.model, "focus_bias"
            ):
                student_bias = self.model.focus_bias(hiddens[0])
                eta = float(self.focus_config.get("student_action_bias_eta", 0.0))
                if eta != 0.0:
                    q_values = q_values + eta * student_bias
                self.last_action_bias_stats.update(
                    {
                        "focus_student_action_bias_mean": student_bias.mean().detach().item(),
                        "focus_student_action_bias_std": student_bias.std(unbiased=False).detach().item(),
                        "focus_student_action_bias_eta": eta,
                    }
                )
            if (
                self.focus_config.get("local_belief_action_guide_enabled", False)
                and self.local_belief_model is not None
            ):
                local_belief_bias, local_belief_bias_stats = self.loss._local_belief_action_bias(
                    q_values.detach(),
                    torch.as_tensor(obs_batch, dtype=torch.float, device=self.device),
                    hiddens[0],
                )
                if local_belief_bias is not None:
                    eta = float(
                        self.focus_config.get(
                            "local_belief_action_eta",
                            self.focus_config.get("student_action_bias_eta", 0.0),
                        )
                    )
                    if eta != 0.0:
                        q_values = q_values + eta * local_belief_bias
                    self.last_action_bias_stats.update(
                        {
                            "focus_local_belief_action_eta": eta,
                            **local_belief_bias_stats,
                        }
                    )

            if explore and env_global_state is not None:
                action_bias = focus_action_q_bias(
                    q_values,
                    torch.as_tensor(env_global_state, dtype=torch.float, device=self.device),
                    self.focus_config,
                    self.n_agents,
                    self.n_actions,
                )
                if action_bias is not None:
                    eta = float(self.focus_config.get("action_bias_eta", 0.0))
                    eta_scale = None
                    bias_confidence_stats = {}
                    if (
                        self.focus_config.get("action_bias_confidence_enabled", True)
                        and self.occupancy_model is not None
                    ):
                        try:
                            state_tensor = torch.as_tensor(
                                env_global_state, dtype=torch.float, device=self.device
                            ).unsqueeze(1)
                            _, pred_std = self.occupancy_model.forward(state_tensor)
                            eta_confidence, pred_std_mean = belief_std_confidence(
                                pred_std, self.focus_config
                            )
                            eta_scale = eta_confidence.squeeze(1).view(-1, 1, 1)
                            bias_confidence_stats = {
                                "focus_action_bias_confidence_mean": eta_confidence.mean().detach().item(),
                                "focus_action_bias_confidence_min": eta_confidence.min().detach().item(),
                                "focus_action_bias_confidence_max": eta_confidence.max().detach().item(),
                                "focus_action_bias_belief_std_mean": pred_std_mean.mean().detach().item(),
                                "focus_action_bias_confidence_source": "belief_std",
                            }
                        except (RuntimeError, ValueError):
                            eta_scale = None
                    if eta_scale is None:
                        eta_scale = 1.0
                    q_values = q_values + eta * eta_scale * action_bias
                    eta_eff = eta * eta_scale
                    eta_eff_mean = (
                        eta_eff.mean().detach().item()
                        if hasattr(eta_eff, "mean")
                        else float(eta_eff)
                    )
                    self.last_action_bias_stats = {
                        "focus_action_bias_mean": action_bias.mean().detach().item(),
                        "focus_action_bias_std": action_bias.std(unbiased=False).detach().item(),
                        "focus_action_bias_eta": eta,
                        "focus_action_bias_eta_effective_mean": eta_eff_mean,
                    }
                    self.last_action_bias_stats.update(bias_confidence_stats)
            avail = torch.as_tensor(action_mask, dtype=torch.float, device=self.device)
            masked_q_values = q_values.clone()
            masked_q_values[avail == 0.0] = -float("inf")
            masked_q_values_folded = torch.reshape(
                masked_q_values, [-1] + list(masked_q_values.shape)[2:]
            )
            if timestep is None:
                timestep = int(1e9)
            actions, _ = self.exploration.get_exploration_action(
                action_distribution=TorchCategorical(masked_q_values_folded),
                timestep=timestep,
                explore=explore,
            )
            actions = (
                torch.reshape(actions, list(masked_q_values.shape)[:-1]).cpu().numpy()
            )
            hiddens = [s.cpu().numpy() for s in hiddens]

        return tuple(actions.transpose([1, 0])), hiddens, {}

    @override(Policy)
    def compute_log_likelihoods(
        self,
        actions,
        obs_batch,
        state_batches=None,
        prev_action_batch=None,
        prev_reward_batch=None,
    ):
        obs_batch, action_mask, _ = self._unpack_observation(obs_batch)
        return np.zeros(obs_batch.size()[0])

    @override(Policy)
    def learn_on_batch(self, samples):
        obs_batch, action_mask, env_global_state = self._unpack_observation(
            samples[SampleBatch.CUR_OBS]
        )
        (
            next_obs_batch,
            next_action_mask,
            next_env_global_state,
        ) = self._unpack_observation(samples[SampleBatch.NEXT_OBS])
        group_rewards = self._get_group_rewards(samples[SampleBatch.INFOS])

        input_list = [
            group_rewards,
            action_mask,
            next_action_mask,
            samples[SampleBatch.ACTIONS],
            samples[SampleBatch.DONES],
            obs_batch,
            next_obs_batch,
        ]
        if self.has_env_global_state:
            input_list.extend([env_global_state, next_env_global_state])

        output_list, _, seq_lens = chop_into_sequences(
            episode_ids=samples[SampleBatch.EPS_ID],
            unroll_ids=samples[SampleBatch.UNROLL_ID],
            agent_indices=samples[SampleBatch.AGENT_INDEX],
            feature_columns=input_list,
            state_columns=[],  # RNN states not used here
            max_seq_len=self.config["model"]["max_seq_len"],
            dynamic_max=True,
        )
        # These will be padded to shape [B * T, ...]
        if self.has_env_global_state:
            (
                rew,
                action_mask,
                next_action_mask,
                act,
                dones,
                obs,
                next_obs,
                env_global_state,
                next_env_global_state,
            ) = output_list
        else:
            (
                rew,
                action_mask,
                next_action_mask,
                act,
                dones,
                obs,
                next_obs,
            ) = output_list
        B, T = len(seq_lens), max(seq_lens)

        def to_batches(arr, dtype):
            new_shape = [B, T] + list(arr.shape[1:])
            return torch.as_tensor(
                np.reshape(arr, new_shape), dtype=dtype, device=self.device
            )

        rewards = to_batches(rew, torch.float)
        actions = to_batches(act, torch.long)
        obs = to_batches(obs, torch.float).reshape([B, T, self.n_agents, self.obs_size])
        action_mask = to_batches(action_mask, torch.float)
        next_obs = to_batches(next_obs, torch.float).reshape(
            [B, T, self.n_agents, self.obs_size]
        )
        next_action_mask = to_batches(next_action_mask, torch.float)
        if self.has_env_global_state:
            env_global_state = to_batches(env_global_state, torch.float)
            next_env_global_state = to_batches(next_env_global_state, torch.float)

        # TODO(ekl) this treats group termination as individual termination
        terminated = (
            to_batches(dones, torch.float).unsqueeze(2).expand(B, T, self.n_agents)
        )

        # Create mask for where index is < unpadded sequence length
        filled = np.reshape(
            np.tile(np.arange(T, dtype=np.float32), B), [B, T]
        ) < np.expand_dims(seq_lens, 1)
        mask = (
            torch.as_tensor(filled, dtype=torch.float, device=self.device)
            .unsqueeze(2)
            .expand(B, T, self.n_agents)
        )

        # Compute loss
        loss_out, mask, masked_td_error, chosen_action_qvals, targets = self.loss(
            rewards,
            actions,
            terminated,
            mask,
            obs,
            next_obs,
            action_mask,
            next_action_mask,
            env_global_state,
            next_env_global_state,
        )

        # Optimise
        self.optimiser.zero_grad()
        loss_out.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.params, self.config["grad_norm_clipping"]
        )
        self.optimiser.step()

        mask_elems = mask.sum().item()
        stats = {
            "loss": loss_out.item(),
            "grad_norm": grad_norm
            if isinstance(grad_norm, float)
            else grad_norm.item(),
            "td_error_abs": masked_td_error.abs().sum().item() / mask_elems,
            "q_taken_mean": (chosen_action_qvals * mask).sum().item() / mask_elems,
            "target_mean": (targets * mask).sum().item() / mask_elems,
        }
        stats.update(getattr(self.loss, "last_focus_stats", {}))
        stats.update(self.last_action_bias_stats)
        return {LEARNER_STATS_KEY: stats}

    @override(Policy)
    def get_initial_state(self):  # initial RNN state
        return [
            s.expand([self.n_agents, -1]).cpu().numpy()
            for s in self.model.get_initial_state()
        ]

    @override(Policy)
    def get_weights(self):
        return {
            "model": self._cpu_dict(self.model.state_dict()),
            "target_model": self._cpu_dict(self.target_model.state_dict()),
            "mixer": self._cpu_dict(self.mixer.state_dict()) if self.mixer else None,
            "target_mixer": self._cpu_dict(self.target_mixer.state_dict())
            if self.mixer
            else None,
            "occupancy_model": self._cpu_dict(self.occupancy_model.state_dict())
            if self.occupancy_model
            else None,
            "local_belief_model": self._cpu_dict(self.local_belief_model.state_dict())
            if self.local_belief_model
            else None,
        }

    @override(Policy)
    def set_weights(self, weights):
        self.model.load_state_dict(self._device_dict(weights["model"]), strict=False)
        self.target_model.load_state_dict(
            self._device_dict(weights["target_model"]), strict=False
        )
        if weights["mixer"] is not None:
            self.mixer.load_state_dict(self._device_dict(weights["mixer"]))
            self.target_mixer.load_state_dict(
                self._device_dict(weights["target_mixer"])
            )
        if self.occupancy_model is not None and weights.get("occupancy_model") is not None:
            self.occupancy_model.load_state_dict(self._device_dict(weights["occupancy_model"]))
        if self.local_belief_model is not None and weights.get("local_belief_model") is not None:
            self.local_belief_model.load_state_dict(self._device_dict(weights["local_belief_model"]))

    @override(Policy)
    def get_state(self):
        state = self.get_weights()
        state["cur_epsilon"] = self.cur_epsilon
        return state

    @override(Policy)
    def set_state(self, state):
        self.set_weights(state)
        self.set_epsilon(state["cur_epsilon"])

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        logger.debug("Updated target networks")

    def set_epsilon(self, epsilon):
        self.cur_epsilon = epsilon

    def _get_group_rewards(self, info_batch):
        group_rewards = np.array(
            [info.get(GROUP_REWARDS, [0.0] * self.n_agents) for info in info_batch]
        )
        return group_rewards

    def _device_dict(self, state_dict):
        return {
            k: torch.as_tensor(v, device=self.device) for k, v in state_dict.items()
        }

    @staticmethod
    def _cpu_dict(state_dict):
        return {k: v.cpu().detach().numpy() for k, v in state_dict.items()}

    def _unpack_observation(self, obs_batch):
        """Unpacks the observation, action mask, and state (if present)
        from agent grouping.

        Returns:
            obs (np.ndarray): obs tensor of shape [B, n_agents, obs_size]
            mask (np.ndarray): action mask, if any
            state (np.ndarray or None): state tensor of shape [B, state_size]
                or None if it is not in the batch
        """

        unpacked = _unpack_obs(
            np.array(obs_batch, dtype=np.float32),
            self.observation_space.original_space,
            tensorlib=np,
        )

        if isinstance(unpacked[0], dict):
            assert "obs" in unpacked[0]
            unpacked_obs = [np.concatenate(tree.flatten(u["obs"]), 1) for u in unpacked]
        else:
            unpacked_obs = unpacked

        obs = np.concatenate(unpacked_obs, axis=1).reshape(
            [len(obs_batch), self.n_agents, self.obs_size]
        )

        if self.has_action_mask:
            action_mask = np.concatenate(
                [o["action_mask"] for o in unpacked], axis=1
            ).reshape([len(obs_batch), self.n_agents, self.n_actions])
        else:
            action_mask = np.ones(
                [len(obs_batch), self.n_agents, self.n_actions], dtype=np.float32
            )

        if self.has_env_global_state:
            state = np.concatenate(tree.flatten(unpacked[0][ENV_STATE]), 1)
        else:
            state = None
        return obs, action_mask, state

def _validate(obs_space, action_space):
    if not hasattr(obs_space, "original_space") or not isinstance(
        obs_space.original_space, Tuple
    ):
        raise ValueError(
            "Obs space must be a Tuple, got {}. Use ".format(obs_space)
            + "MultiAgentEnv.with_agent_groups() to group related "
            "agents for QPLEX."
        )
    if not isinstance(action_space, Tuple):
        raise ValueError(
            "Action space must be a Tuple, got {}. ".format(action_space)
            + "Use MultiAgentEnv.with_agent_groups() to group related "
            "agents for QPLEX."
        )
    if not isinstance(action_space.spaces[0], Discrete):
        raise ValueError(
            "QPLEX requires a discrete action space, got {}".format(
                action_space.spaces[0]
            )
        )
    if len({str(x) for x in obs_space.original_space.spaces}) > 1:
        raise ValueError(
            "Implementation limitation: observations of grouped agents "
            "must be homogeneous, got {}".format(obs_space.original_space.spaces)
        )
    if len({str(x) for x in action_space.spaces}) > 1:
        raise ValueError(
            "Implementation limitation: action space of grouped agents "
            "must be homogeneous, got {}".format(action_space.spaces)
        )

def _mac(model, obs, h):
    """Forward pass of the multi-agent controller.

    Args:
        model: TorchModelV2 class
        obs: Tensor of shape [B, n_agents, obs_size]
        h: List of tensors of shape [B, n_agents, h_size]

    Returns:
        q_vals: Tensor of shape [B, n_agents, n_actions]
        h: Tensor of shape [B, n_agents, h_size]
    """
    B, n_agents = obs.size(0), obs.size(1)
    if not isinstance(obs, dict):
        obs = {"obs": obs}
    obs_agents_as_batches = {k: _drop_agent_dim(v) for k, v in obs.items()}
    h_flat = [s.reshape([B * n_agents, -1]) for s in h]
    q_flat, h_flat = model(obs_agents_as_batches, h_flat, None)
    return q_flat.reshape([B, n_agents, -1]), [
        s.reshape([B, n_agents, -1]) for s in h_flat
    ]

def _unroll_mac(model, obs_tensor, return_features=False):
    """Computes estimated Q values and, optionally, local RNN features."""
    B = obs_tensor.size(0)
    T = obs_tensor.size(1)
    n_agents = obs_tensor.size(2)

    mac_out = []
    features = []
    h = [s.expand([B, n_agents, -1]) for s in model.get_initial_state()]
    for t in range(T):
        q, h = _mac(model, obs_tensor[:, t], h)
        mac_out.append(q)
        if return_features:
            features.append(h[0])
    mac_out = torch.stack(mac_out, dim=1)  # Concat over time

    if return_features:
        return mac_out, torch.stack(features, dim=1)
    return mac_out

def _drop_agent_dim(T):
    '''T [B, n_agents, X] -> [B * n_agents, X]'''
    shape = list(T.shape)
    B, n_agents = shape[0], shape[1]
    return T.reshape([B * n_agents] + shape[2:])

def _add_agent_dim(T, n_agents):
    shape = list(T.shape)
    B = shape[0] // n_agents # T ban đầu có dạng (batch * n_agents, X)
    assert shape[0] % n_agents == 0
    return T.reshape([B, n_agents] + shape[1:])
