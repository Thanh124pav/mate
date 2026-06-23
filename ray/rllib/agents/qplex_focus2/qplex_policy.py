import copy
import logging
import math
from argparse import Namespace

import numpy as np
import torch.nn.functional as F

import ray
from ray.rllib.agents.qplex_focus.qplex_policy import (
    CAMERA_STATE_DIM_PRIVATE,
    OBSTACLE_STATE_DIM,
    PRESERVED_DIM,
    TARGET_STATE_DIM_PRIVATE,
    QPLEXFocusTorchPolicy,
    _extract_camera_fov,
    _extract_obstacles,
    _extract_target_positions,
    _unroll_mac,
    adjust_args,
    resolve_focus_config,
)
from ray.rllib.agents.qplex_focus2.mixers import Focus2DuelMixer
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch(error=True)

logger = logging.getLogger(__name__)


def _make_qmc_points(num_points, x_range, y_range, seed, device=None, dtype=None):
    engine = torch.quasirandom.SobolEngine(dimension=2, scramble=True, seed=int(seed))
    pts = engine.draw(int(num_points))
    x_min, x_max = float(x_range[0]), float(x_range[1])
    y_min, y_max = float(y_range[0]), float(y_range[1])
    pts = torch.stack(
        [
            x_min + (x_max - x_min) * pts[:, 0],
            y_min + (y_max - y_min) * pts[:, 1],
        ],
        dim=-1,
    )
    if device is not None or dtype is not None:
        pts = pts.to(device=device, dtype=dtype)
    return pts


class QMCDiscreteBeliefModel(nn.Module):
    """Belief over fixed Sobol QMC support points, as in the methodology."""

    def __init__(self, state_dim, n_agents, n_targets, horizon=9, hidden_dim=512,
                 num_points=128, x_range=(-1000.0, 1000.0),
                 y_range=(-1000.0, 1000.0), soft_label_sigma=100.0, seed=0):
        super(QMCDiscreteBeliefModel, self).__init__()
        self.state_dim = int(np.prod(state_dim))
        self.n_agents = n_agents
        self.n_targets = n_targets
        self.horizon = horizon
        self.num_points = num_points
        self.soft_label_sigma = soft_label_sigma
        self.net = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, horizon * n_targets * num_points),
        )
        self.register_buffer(
            "qmc_points",
            _make_qmc_points(num_points, x_range, y_range, seed),
        )

    def forward(self, state):
        B, T = state.shape[:2]
        logits = self.net(state.reshape(-1, self.state_dim))
        return logits.view(B, T, self.horizon, self.n_targets, self.num_points)

    def belief_and_loss(self, state, next_state, mask, horizon_weights, eps=1e-8):
        logits = self.forward(state)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        B, T = state.shape[:2]
        belief_terms = []
        stats = {}
        valid_belief = mask[:, :, 0] > 0.0
        sigma2 = float(self.soft_label_sigma) ** 2
        qmc = self.qmc_points.to(device=state.device, dtype=state.dtype)

        for h in range(self.horizon):
            valid_t = T - h
            if valid_t <= 0:
                break
            future_state = next_state[:, h:, :]
            target = _extract_target_positions(
                future_state, self.n_agents, self.n_targets
            )
            pred_log_probs = log_probs[:, :valid_t, h, :, :]
            diff = qmc.view(1, 1, 1, self.num_points, 2) - target.unsqueeze(-2)
            soft_logits = -0.5 * (diff ** 2).sum(dim=-1) / max(sigma2, eps)
            soft_target = F.softmax(soft_logits, dim=-1)
            ce = -(soft_target * pred_log_probs).sum(dim=-1).mean(dim=-1)
            valid_h = valid_belief[:, :valid_t]
            if valid_h.any():
                term = horizon_weights[h] * ce[valid_h].mean()
                belief_terms.append(term)
                stats[f"focus_belief_loss_h{h + 1}"] = ce[valid_h].mean().detach().item()
        belief_loss = (
            torch.stack(belief_terms).sum()
            if belief_terms
            else torch.zeros((), dtype=state.dtype, device=state.device)
        )
        return probs, belief_loss, stats


class MixtureGaussianBeliefModel(nn.Module):
    """Mixture Gaussian option projected onto the same QMC support."""

    def __init__(self, state_dim, n_agents, n_targets, horizon=9, hidden_dim=512,
                 num_points=128, x_range=(-1000.0, 1000.0),
                 y_range=(-1000.0, 1000.0), seed=0, components=4,
                 min_std=25.0, max_delta=400.0):
        super(MixtureGaussianBeliefModel, self).__init__()
        self.state_dim = int(np.prod(state_dim))
        self.n_agents = n_agents
        self.n_targets = n_targets
        self.horizon = horizon
        self.num_points = num_points
        self.components = components
        self.min_std = min_std
        self.max_delta = max_delta
        self.net = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, horizon * n_targets * components * 5),
        )
        self.register_buffer(
            "qmc_points",
            _make_qmc_points(num_points, x_range, y_range, seed),
        )

    def _params(self, state):
        B, T = state.shape[:2]
        out = self.net(state.reshape(-1, self.state_dim))
        out = out.view(B, T, self.horizon, self.n_targets, self.components, 5)
        mix_logits = out[..., 0]
        delta = torch.tanh(out[..., 1:3]) * self.max_delta
        std = F.softplus(out[..., 3:5]) + self.min_std
        current_pos = _extract_target_positions(state, self.n_agents, self.n_targets)
        mean = current_pos.unsqueeze(2).unsqueeze(4) + delta
        return mix_logits, mean, std

    def belief_and_loss(self, state, next_state, mask, horizon_weights, eps=1e-8):
        mix_logits, mean, std = self._params(state)
        log_mix = F.log_softmax(mix_logits, dim=-1)
        qmc = self.qmc_points.to(device=state.device, dtype=state.dtype)
        diff = qmc.view(1, 1, 1, 1, 1, self.num_points, 2) - mean.unsqueeze(-2)
        z = diff / (std.unsqueeze(-2) + eps)
        log_comp = (
            -0.5 * (z ** 2).sum(dim=-1)
            - torch.log(std.unsqueeze(-2) + eps).sum(dim=-1)
            - math.log(2.0 * math.pi)
        )
        log_density = torch.logsumexp(log_mix.unsqueeze(-1) + log_comp, dim=-2)
        probs = F.softmax(log_density, dim=-1)

        B, T = state.shape[:2]
        valid_belief = mask[:, :, 0] > 0.0
        belief_terms = []
        stats = {}
        for h in range(self.horizon):
            valid_t = T - h
            if valid_t <= 0:
                break
            future_state = next_state[:, h:, :]
            target = _extract_target_positions(
                future_state, self.n_agents, self.n_targets
            )
            mean_h = mean[:, :valid_t, h, :, :, :]
            std_h = std[:, :valid_t, h, :, :, :]
            log_mix_h = log_mix[:, :valid_t, h, :, :]
            z = (target.unsqueeze(-2) - mean_h) / (std_h + eps)
            log_comp_target = (
                -0.5 * (z ** 2).sum(dim=-1)
                - torch.log(std_h + eps).sum(dim=-1)
                - math.log(2.0 * math.pi)
            )
            nll = -torch.logsumexp(log_mix_h + log_comp_target, dim=-1).mean(dim=-1)
            valid_h = valid_belief[:, :valid_t]
            if valid_h.any():
                belief_terms.append(horizon_weights[h] * nll[valid_h].mean())
                stats[f"focus_belief_loss_h{h + 1}"] = nll[valid_h].mean().detach().item()
        belief_loss = (
            torch.stack(belief_terms).sum()
            if belief_terms
            else torch.zeros((), dtype=state.dtype, device=state.device)
        )
        return probs, belief_loss, stats


class QPLEXFocus2Loss(nn.Module):
    def __init__(self, model, target_model, mixer, target_mixer, n_agents,
                 n_actions, double_q=True, gamma=0.99, focus_config=None,
                 belief_model=None):
        super(QPLEXFocus2Loss, self).__init__()
        self.model = model
        self.target_model = target_model
        self.mixer = mixer
        self.target_mixer = target_mixer
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.double_q = double_q
        self.gamma = gamma
        self.focus_config = focus_config or {}
        self.belief_model = belief_model
        self.last_focus_stats = {}

    def _horizon_weights(self, horizon, device, dtype):
        discount = float(self.focus_config.get("horizon_discount", 0.9))
        weights = torch.tensor(
            [discount ** h for h in range(horizon)], device=device, dtype=dtype
        )
        return weights / (weights.sum() + self.focus_config.get("eps", 1e-8))

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

    def _qmc_visibility(self, qmc_points, state, n_targets):
        eps = self.focus_config.get("eps", 1e-8)
        cam_pos, cam_orient, cam_range, cam_half_angle = _extract_camera_fov(
            state, self.n_agents
        )
        rel = qmc_points.view(1, 1, 1, -1, 2) - cam_pos.unsqueeze(3)
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

        n_obstacles = int(self.focus_config.get("n_obstacles", 0))
        required_dim = (
            PRESERVED_DIM
            + self.n_agents * CAMERA_STATE_DIM_PRIVATE
            + n_targets * TARGET_STATE_DIM_PRIVATE
            + n_obstacles * OBSTACLE_STATE_DIM
        )
        if n_obstacles <= 0 or state.size(-1) < required_dim:
            return visible

        obstacle_pos, obstacle_radius = _extract_obstacles(
            state, self.n_agents, n_targets, n_obstacles
        )
        ray = qmc_points.view(1, 1, 1, 1, -1, 2) - cam_pos.unsqueeze(3).unsqueeze(4)
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

    def _focus_credit_target(self, state, next_state, actions, mask):
        eps = self.focus_config.get("eps", 1e-8)
        n_targets = int(self.focus_config.get("n_targets", 8))
        if self.belief_model is None or state is None or next_state is None:
            B, T = actions.shape[:2]
            rho = torch.full(
                (B, T, self.n_agents), 1.0 / self.n_agents,
                dtype=torch.float, device=actions.device
            )
            valid = torch.zeros((B, T), dtype=torch.bool, device=actions.device)
            return rho, valid, torch.zeros((B, T), device=actions.device), torch.zeros((), device=actions.device), {}

        horizon_weights = self._horizon_weights(
            int(self.focus_config.get("horizon", 9)), actions.device, state.dtype
        )
        probs, belief_loss, belief_stats = self.belief_model.belief_and_loss(
            state, next_state, mask, horizon_weights, eps
        )
        qmc_points = self.belief_model.qmc_points.to(device=state.device, dtype=state.dtype)
        occ = torch.einsum("h,bthjm->btjm", horizon_weights[: probs.size(2)], probs)
        visible = self._qmc_visibility(qmc_points, state, n_targets)
        selection = self._decode_target_selection(actions, n_targets)
        if selection is not None:
            visible = visible.unsqueeze(3) * selection.unsqueeze(-1)
        else:
            visible = visible.unsqueeze(3).expand(-1, -1, -1, n_targets, -1)

        one_minus = 1.0 - visible
        unique_terms = []
        for i in range(self.n_agents):
            if self.n_agents == 1:
                unique_terms.append(torch.ones_like(visible[:, :, i, :, :]))
            else:
                others = torch.cat(
                    [one_minus[:, :, :i, :, :], one_minus[:, :, i + 1 :, :, :]],
                    dim=2,
                )
                unique_terms.append(torch.prod(others, dim=2))
        unique_vis = visible * torch.stack(unique_terms, dim=2)
        target_weights = self.focus_config.get("target_weights")
        if target_weights is None:
            tw = torch.ones(n_targets, dtype=state.dtype, device=state.device)
        else:
            tw = torch.as_tensor(target_weights, dtype=state.dtype, device=state.device)
            tw = tw[:n_targets]
        g = torch.einsum("btijm,btjm,j->bti", unique_vis, occ, tw)
        rho = (g + eps) / (g + eps).sum(dim=-1, keepdim=True)
        valid = (g.sum(dim=-1) > float(self.focus_config.get("min_credit_signal", 1e-6))) & (mask[:, :, 0] > 0.0)
        return rho.detach(), valid.detach(), g.sum(dim=-1).detach(), belief_loss, belief_stats

    def forward(self, rewards, actions, terminated, mask, obs, next_obs,
                action_mask, next_action_mask, state=None, next_state=None):
        if state is None and next_state is None:
            state = obs
            next_state = next_obs
        elif (state is None) != (next_state is None):
            raise ValueError("Expected either both or neither state/next_state.")

        mac_out = _unroll_mac(self.model, obs)
        chosen_action_qvals = torch.gather(
            mac_out, dim=3, index=actions.unsqueeze(3)
        ).squeeze(3)
        ignore_action = (action_mask == 0) & (mask == 1).unsqueeze(-1)
        x_mac_out = mac_out.clone().detach()
        x_mac_out[ignore_action] = -np.inf
        max_action_vals = x_mac_out.max(dim=3)[0]

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

        ans_chosen = self.mixer(chosen_action_qvals, state, is_v=True)
        actions_onehot = F.one_hot(actions, num_classes=self.n_actions)
        ans_adv, lambda_weights, p_dist = self.mixer(
            chosen_action_qvals, state, actions_onehot,
            max_action_vals=max_action_vals, is_v=False, return_credit=True
        )
        chosen_q_tot = ans_chosen + ans_adv

        target_chosen = self.target_mixer(target_max_qvals, next_state, is_v=True)
        cur_max_actions_onehot = F.one_hot(cur_max_actions.squeeze(3), num_classes=self.n_actions)
        target_adv = self.target_mixer(
            target_max_qvals, next_state, cur_max_actions_onehot,
            target_max_qvals, is_v=False
        )
        target_q_tot = target_chosen + target_adv

        targets = rewards + self.gamma * (1 - terminated) * target_q_tot
        td_error = chosen_q_tot - targets.detach()
        mask = mask.expand_as(td_error)
        masked_td_error = td_error * mask
        td_loss = (masked_td_error ** 2).sum() / mask.sum()

        loss = td_loss
        self.last_focus_stats = {"td_loss": td_loss.detach().item()}
        if self.focus_config.get("enabled", True):
            rho, valid, total_g, belief_loss, belief_stats = self._focus_credit_target(
                state, next_state, actions, mask
            )
            eps = self.focus_config.get("eps", 1e-8)
            per_step_kl = (rho * (torch.log(rho + eps) - torch.log(p_dist + eps))).sum(dim=-1)
            if valid.any():
                focus_loss = per_step_kl[valid].mean()
            else:
                focus_loss = torch.zeros_like(td_loss)
            alpha = float(self.focus_config.get("alpha_credit", 0.1))
            beta = float(self.focus_config.get("beta_belief", 0.05))
            loss = td_loss + alpha * focus_loss + beta * belief_loss
            self.last_focus_stats.update({
                "focus_credit_loss": focus_loss.detach().item(),
                "focus_belief_loss": belief_loss.detach().item(),
                "focus_valid_ratio": valid.float().mean().detach().item(),
                "focus_mean_signal": total_g.mean().detach().item(),
                "focus_rho_entropy": (-(rho * torch.log(rho + eps)).sum(dim=-1)).mean().detach().item(),
                "focus_p_entropy": (-(p_dist * torch.log(p_dist + eps)).sum(dim=-1)).mean().detach().item(),
                "focus_lambda_entropy": (-(lambda_weights / (lambda_weights.sum(dim=-1, keepdim=True) + eps) * torch.log(lambda_weights / (lambda_weights.sum(dim=-1, keepdim=True) + eps) + eps)).sum(dim=-1)).mean().detach().item(),
                "focus_alpha_credit": alpha,
                "focus_beta_belief": beta,
            })
            self.last_focus_stats.update(belief_stats)
        return loss, mask, masked_td_error, chosen_q_tot, targets


class QPLEXFocus2TorchPolicy(QPLEXFocusTorchPolicy):
    """QPLEX-FOCUS2 policy with QMC-discrete belief and explicit p_i m_i lambda."""

    def __init__(self, obs_space, action_space, config):
        from ray.rllib.agents.qplex_focus2.qplex import DEFAULT_CONFIG

        config = copy.deepcopy(dict(DEFAULT_CONFIG, **config))
        bootstrap_config = copy.deepcopy(config)
        bootstrap_config["mixer"] = "qplex_focus"
        bootstrap_focus = copy.deepcopy(bootstrap_config.get("focus", {}))
        bootstrap_focus["enabled"] = False
        bootstrap_config["focus"] = bootstrap_focus
        super().__init__(obs_space, action_space, bootstrap_config)

        self.config = config
        self.args = adjust_args(Namespace(**config))
        self.mixer = Focus2DuelMixer(
            self.args, self.n_agents, self.n_actions, self.env_global_state_shape,
            config["mixing_embed_dim"], self.args.ffn_hidden_dim, self.args.num_kernel
        ).to(self.device)
        self.target_mixer = Focus2DuelMixer(
            self.args, self.n_agents, self.n_actions, self.env_global_state_shape,
            config["mixing_embed_dim"], self.args.ffn_hidden_dim, self.args.num_kernel
        ).to(self.device)

        focus_config = resolve_focus_config(self.config)
        self.occupancy_model = None
        if focus_config.get("enabled", True):
            common = dict(
                state_dim=self.env_global_state_shape,
                n_agents=self.n_agents,
                n_targets=int(focus_config.get("n_targets", 8)),
                horizon=int(focus_config.get("horizon", 9)),
                hidden_dim=int(focus_config.get("belief_hidden_dim", 512)),
                num_points=int(focus_config.get("qmc_num_points", 128)),
                x_range=focus_config.get("qmc_x_range", (-1000.0, 1000.0)),
                y_range=focus_config.get("qmc_y_range", (-1000.0, 1000.0)),
                seed=int(focus_config.get("qmc_seed", 0)),
            )
            belief_type = focus_config.get("belief_type", "qmc_discrete")
            if belief_type == "qmc_discrete":
                self.occupancy_model = QMCDiscreteBeliefModel(
                    **common,
                    soft_label_sigma=float(focus_config.get("qmc_soft_label_sigma", 100.0)),
                ).to(self.device)
            elif belief_type == "mixture_gaussian":
                self.occupancy_model = MixtureGaussianBeliefModel(
                    **common,
                    components=int(focus_config.get("mixture_components", 4)),
                    min_std=float(focus_config.get("mixture_min_std", 25.0)),
                    max_delta=float(focus_config.get("mixture_max_delta", 400.0)),
                ).to(self.device)
            else:
                raise ValueError(f"Unknown QPLEX_FOCUS2 belief_type: {belief_type}")

        self.update_target()
        self.params = list(self.model.parameters()) + list(self.mixer.parameters())
        if self.occupancy_model:
            self.params += list(self.occupancy_model.parameters())

        self.loss = QPLEXFocus2Loss(
            self.model, self.target_model, self.mixer, self.target_mixer,
            self.n_agents, self.n_actions, self.config["double_q"],
            self.config["gamma"], focus_config, self.occupancy_model,
        )
        from torch.optim import RMSprop

        self.optimiser = RMSprop(
            params=self.params,
            lr=config["lr"],
            alpha=config["optim_alpha"],
            eps=config["optim_eps"],
        )
