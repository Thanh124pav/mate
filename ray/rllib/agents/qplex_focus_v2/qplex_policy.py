import copy
import math

import numpy as np
import torch.nn.functional as F

import ray
from ray.rllib.agents.focus_utils import confidence_stats, gated_focus_loss, resolve_confidence
from ray.rllib.agents.qplex_focus.mixers import FocusDuelMixer
from ray.rllib.agents.qplex_focus.qplex_policy import (
    CAMERA_STATE_DIM_PRIVATE,
    OBSTACLE_STATE_DIM,
    PRESERVED_DIM,
    TARGET_STATE_DIM_PRIVATE,
    LearnedOccupancyModel,
    QPLEXFocusLoss,
    QPLEXFocusTorchPolicy,
    _extract_obstacles,
    _unroll_mac,
    adjust_args,
    resolve_focus_config,
)
from ray.rllib.utils.framework import try_import_torch


torch, nn = try_import_torch(error=True)


class QPLEXFocusV2Loss(QPLEXFocusLoss):
    """FOCUS-v2: dynamic reachable responsibility + reliability-aware KL."""

    def _reachable_params(self, cam_orient, cam_range, cam_half_angle, horizon_index):
        if not self.focus_config.get("reachable_visibility_enabled", True):
            return cam_orient, cam_range, cam_half_angle

        h = float(horizon_index + 1)
        rot = math.radians(float(self.focus_config.get("reachable_rotation_step_deg", 20.0)))
        range_step = float(self.focus_config.get("reachable_range_step", 0.0))
        angle_step = math.radians(float(self.focus_config.get("reachable_half_angle_step_deg", 0.0)))

        mode = str(self.focus_config.get("reachable_mode", "union")).lower()
        if mode == "conservative":
            return cam_orient, cam_range, cam_half_angle

        # Union footprint: a point is reachable if it lies within the future
        # orientation envelope. This is the Level-B geometric approximation in
        # the uploaded implementation plan.
        return (
            cam_orient,
            cam_range + h * range_step,
            (cam_half_angle + h * rot + h * angle_step).clamp(max=math.pi),
        )

    def _point_visibility_horizon(
        self,
        points,
        cam_pos,
        cam_orient,
        cam_range,
        cam_half_angle,
        horizon_index,
        obstacle_pos=None,
        obstacle_radius=None,
    ):
        eps = self.focus_config.get("eps", 1e-8)
        cam_orient, cam_range, cam_half_angle = self._reachable_params(
            cam_orient, cam_range, cam_half_angle, horizon_index
        )
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

    def _credit_chunk_sum_horizon(
        self,
        target_samples,
        cam_pos,
        cam_orient,
        cam_range,
        cam_half_angle,
        selection,
        horizon_index,
        obstacle_pos=None,
        obstacle_radius=None,
    ):
        visible = self._point_visibility_horizon(
            target_samples,
            cam_pos,
            cam_orient,
            cam_range,
            cam_half_angle,
            horizon_index,
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
        return unique_gain.sum(dim=-1).sum(dim=-1)

    def _credit_from_sigma_points(self, target_samples, state, actions, n_targets):
        cam_pos, cam_orient, cam_range, cam_half_angle, obstacle_pos, obstacle_radius = (
            self._credit_geometry(state, n_targets)
        )
        selection = self._decode_target_selection(actions, n_targets)
        horizon_weights = self._horizon_weights(target_samples.size(2), actions.device, target_samples.dtype)
        chunk_size = int(self.focus_config.get("sample_chunk_size", self.focus_config.get("mc_chunk_size", 32)))
        chunk_size = max(1, min(chunk_size, target_samples.size(4)))
        total_samples = target_samples.size(4)
        g = target_samples.new_zeros(target_samples.size(0), target_samples.size(1), self.n_agents)

        for h in range(target_samples.size(2)):
            for start in range(0, total_samples, chunk_size):
                samples = target_samples[:, :, h : h + 1, :, start : start + chunk_size, :]
                chunk_sum = self._credit_chunk_sum_horizon(
                    samples,
                    cam_pos,
                    cam_orient,
                    cam_range,
                    cam_half_angle,
                    selection,
                    h,
                    obstacle_pos=obstacle_pos,
                    obstacle_radius=obstacle_radius,
                ).squeeze(3)
                g = g + horizon_weights[h] * chunk_sum / float(total_samples)
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
                chunk_sum = self._credit_chunk_sum_horizon(
                    samples,
                    cam_pos,
                    cam_orient,
                    cam_range,
                    cam_half_angle,
                    selection,
                    h,
                    obstacle_pos=obstacle_pos,
                    obstacle_radius=obstacle_radius,
                ).squeeze(3)
                g = g + horizon_weights[h] * chunk_sum / float(num_points)
        return g

    def _credit_confidence(self, rho):
        if not self.focus_config.get("credit_ambiguity_gate_enabled", True):
            return torch.ones_like(rho[..., 0])
        eps = self.focus_config.get("eps", 1e-8)
        max_entropy = math.log(max(self.n_agents, 2))
        entropy = -(rho * torch.log(rho + eps)).sum(dim=-1)
        return (1.0 - entropy / max_entropy).clamp(0.0, 1.0).detach()

    def _teacher_rho(self, rho):
        mode = str(self.focus_config.get("teacher_mode", "uniform")).lower()
        if mode in ("off", "none", "predicted"):
            return rho
        return torch.full_like(rho, 1.0 / self.n_agents)

    def _focus_credit_target(self, state, next_state, actions, mask):
        rho, valid, total_g, belief_loss, confidence, confidence_mode = super()._focus_credit_target(
            state, next_state, actions, mask
        )
        credit_confidence = self._credit_confidence(rho)
        final_confidence = (confidence * credit_confidence * valid.float()).detach()
        teacher = self._teacher_rho(rho)
        rho_final = final_confidence.unsqueeze(-1) * rho + (1.0 - final_confidence.unsqueeze(-1)) * teacher
        rho_final = rho_final / (rho_final.sum(dim=-1, keepdim=True) + self.focus_config.get("eps", 1e-8))
        self.last_belief_stats.update(
            {
                "focus_v2_credit_confidence_mean": credit_confidence.mean().detach().item(),
                "focus_v2_final_confidence_mean": final_confidence.mean().detach().item(),
                "focus_v2_reachable_enabled": float(self.focus_config.get("reachable_visibility_enabled", True)),
            }
        )
        return rho_final.detach(), valid, total_g, belief_loss, final_confidence, f"v2:{confidence_mode}+credit"

    def _weighted_focus_loss(self, per_step_loss, valid, total_g, reference_loss, confidence=None):
        weights = self._signal_confidence_weights(total_g, valid)
        if confidence is None:
            confidence = torch.ones_like(weights)
        return gated_focus_loss(
            per_step_loss,
            valid,
            confidence,
            reference_loss,
            eps=self.focus_config.get("eps", 1e-8),
            weights=weights,
        )

    def forward(self, rewards, actions, terminated, mask, obs, next_obs,
                action_mask, next_action_mask, state=None, next_state=None):
        if state is None and next_state is None:
            state = obs
            next_state = next_obs
        elif (state is None) != (next_state is None):
            raise ValueError("Expected either neither or both state/next_state.")

        mac_out = _unroll_mac(self.model, obs)
        chosen_action_qvals = torch.gather(
            mac_out, dim=3, index=actions.unsqueeze(3)
        ).squeeze(3)
        ignore_action = (action_mask == 0) & (mask == 1).unsqueeze(-1)
        x_mac_out = mac_out.clone().detach()
        x_mac_out[ignore_action] = -np.inf
        max_action_vals, _ = x_mac_out.max(dim=3)

        target_mac_out = _unroll_mac(self.target_model, next_obs)
        ignore_action_tp1 = (next_action_mask == 0) & (mask == 1).unsqueeze(-1)
        target_mac_out[ignore_action_tp1] = -np.inf
        if self.double_q:
            mac_out_tp1 = _unroll_mac(self.model, next_obs)
            mac_out_tp1[ignore_action_tp1] = -np.inf
            cur_max_actions = mac_out_tp1.argmax(dim=3, keepdim=True)
            target_max_qvals = torch.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            cur_max_actions = target_mac_out.argmax(dim=3, keepdim=True)
            target_max_qvals = target_mac_out.max(dim=3)[0]

        focus_enabled = self.focus_config.get("enabled", True) and self.mixer is not None
        rho = valid = total_g = confidence = None
        belief_loss = torch.zeros((), dtype=chosen_action_qvals.dtype, device=chosen_action_qvals.device)
        confidence_mode = "off"
        if focus_enabled:
            rho, valid, total_g, belief_loss, confidence, confidence_mode = self._focus_credit_target(
                state, next_state, actions, mask
            )

        if self.mixer is not None:
            ans_chosen = self.mixer(chosen_action_qvals, state, is_v=True)
            actions_onehot = F.one_hot(actions, num_classes=self.n_actions)
            ans_adv, lambda_weights = self.mixer(
                chosen_action_qvals,
                state,
                actions_onehot,
                max_action_vals=max_action_vals,
                is_v=False,
                return_lambda=True,
                rho=None,
            )
            chosen_action_qvals = ans_chosen + ans_adv

            target_chosen = self.target_mixer(target_max_qvals, next_state, is_v=True)
            cur_max_actions_onehot = F.one_hot(cur_max_actions, num_classes=self.n_actions)
            target_adv = self.target_mixer(
                target_max_qvals,
                next_state,
                cur_max_actions_onehot,
                target_max_qvals,
                is_v=False,
            )
            target_max_qvals = target_chosen + target_adv

        targets = rewards + self.gamma * (1 - terminated) * target_max_qvals
        td_error = chosen_action_qvals - targets.detach()
        mask = mask.expand_as(td_error)
        masked_td_error = td_error * mask
        td_loss = (masked_td_error ** 2).sum() / mask.sum()

        self.last_focus_stats = {}
        if focus_enabled:
            eps = self.focus_config.get("eps", 1e-8)
            if hasattr(self.mixer, "credit_prior"):
                p_dist = self.mixer.credit_prior(state).view(state.size(0), state.size(1), self.n_agents)
            else:
                p_dist = lambda_weights / (lambda_weights.sum(dim=-1, keepdim=True) + eps)
            lambda_dist = lambda_weights / (lambda_weights.sum(dim=-1, keepdim=True) + eps)
            per_step_ce = -(rho * torch.log(p_dist + eps)).sum(dim=-1)
            focus_loss = self._weighted_focus_loss(
                per_step_ce, valid, total_g, td_loss, confidence=confidence
            )
            alpha = float(self.focus_config.get("alpha_credit", 0.05))
            beta = float(self.focus_config.get("beta_belief", 0.01))
            rho_entropy = -(rho * torch.log(rho + eps)).sum(dim=-1)
            p_entropy = -(p_dist * torch.log(p_dist + eps)).sum(dim=-1)
            lambda_entropy = -(lambda_dist * torch.log(lambda_dist + eps)).sum(dim=-1)
            signal_weights = self._signal_confidence_weights(total_g, valid)
            valid_signal_weights = signal_weights[valid] if valid.any() else signal_weights.reshape(-1)
            self.last_focus_stats = {
                "focus_credit_loss": focus_loss.detach().item(),
                "focus_belief_loss": belief_loss.detach().item(),
                "focus_valid_ratio": valid.float().mean().detach().item(),
                "focus_mean_signal": total_g.mean().detach().item(),
                "focus_signal_weight_mean": valid_signal_weights.mean().detach().item(),
                "focus_rho_entropy": rho_entropy.mean().detach().item(),
                "focus_p_entropy": p_entropy.mean().detach().item(),
                "focus_lambda_entropy": lambda_entropy.mean().detach().item(),
                "focus_alpha_credit": alpha,
                "focus_beta_belief": beta,
            }
            self.last_focus_stats.update(confidence_stats(confidence, valid, confidence_mode))
            self.last_focus_stats.update(self.last_belief_stats)
            loss = td_loss + alpha * focus_loss + beta * belief_loss
        else:
            loss = td_loss
        self.last_focus_stats["td_loss"] = td_loss.detach().item()
        return loss, mask, masked_td_error, chosen_action_qvals, targets


class QPLEXFocusV2TorchPolicy(QPLEXFocusTorchPolicy):
    """QPLEX + FOCUS-V2, built from the original QPLEX_FOCUS baseline."""

    def __init__(self, obs_space, action_space, config):
        from argparse import Namespace
        from torch.optim import RMSprop

        from ray.rllib.agents.qplex_focus_v2.qplex import DEFAULT_CONFIG

        config = copy.deepcopy(dict(DEFAULT_CONFIG, **config))
        bootstrap_config = copy.deepcopy(config)
        bootstrap_config["mixer"] = "qplex_focus"
        super().__init__(obs_space, action_space, bootstrap_config)

        self.config = config
        self.args = adjust_args(Namespace(**config))
        self.mixer = FocusDuelMixer(
            self.args,
            self.n_agents,
            self.n_actions,
            self.env_global_state_shape,
            config["mixing_embed_dim"],
            self.args.ffn_hidden_dim,
            self.args.num_kernel,
        ).to(self.device)
        self.target_mixer = FocusDuelMixer(
            self.args,
            self.n_agents,
            self.n_actions,
            self.env_global_state_shape,
            config["mixing_embed_dim"],
            self.args.ffn_hidden_dim,
            self.args.num_kernel,
        ).to(self.device)

        focus_config = resolve_focus_config(self.config)
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
            ).to(self.device)

        self.update_target()
        self.params = list(self.model.parameters())
        if self.mixer:
            self.params += list(self.mixer.parameters())
        if self.occupancy_model:
            self.params += list(self.occupancy_model.parameters())
        self.loss = QPLEXFocusV2Loss(
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
        )
        self.optimiser = RMSprop(
            params=self.params,
            lr=config["lr"],
            alpha=config["optim_alpha"],
            eps=config["optim_eps"],
        )

