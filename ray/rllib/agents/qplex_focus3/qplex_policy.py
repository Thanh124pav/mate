import copy
from argparse import Namespace

import numpy as np
import torch.nn.functional as F

from ray.rllib.agents.focus_utils import confidence_stats, resolve_confidence
from ray.rllib.agents.qplex_focus.qplex_policy import (
    CAMERA_STATE_DIM_PRIVATE,
    OBSTACLE_STATE_DIM,
    PRESERVED_DIM,
    TARGET_STATE_DIM_PRIVATE,
    QPLEXFocusTorchPolicy,
    _extract_obstacles,
    _extract_target_positions,
    adjust_args,
    resolve_focus_config,
)
from ray.rllib.agents.qplex_focus2.mixers import Focus2DuelMixer
from ray.rllib.agents.qplex_focus2.qplex_policy import QPLEXFocus2Loss
from ray.rllib.utils.framework import try_import_torch


torch, nn = try_import_torch(error=True)


def _make_cell_grid(
    x_range=(-1000.0, 1000.0),
    y_range=(-1000.0, 1000.0),
    grid_size=(10, 10),
    subpoints_per_cell=4,
    device=None,
    dtype=None,
):
    nx, ny = int(grid_size[0]), int(grid_size[1])
    if nx <= 0 or ny <= 0:
        raise ValueError(f"cell_grid_size must be positive, got {grid_size}")
    x_min, x_max = float(x_range[0]), float(x_range[1])
    y_min, y_max = float(y_range[0]), float(y_range[1])
    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny
    xs = torch.linspace(x_min + 0.5 * dx, x_max - 0.5 * dx, nx, device=device, dtype=dtype)
    ys = torch.linspace(y_min + 0.5 * dy, y_max - 0.5 * dy, ny, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    centers = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
    bounds = torch.stack(
        [
            centers[:, 0] - 0.5 * dx,
            centers[:, 0] + 0.5 * dx,
            centers[:, 1] - 0.5 * dy,
            centers[:, 1] + 0.5 * dy,
        ],
        dim=-1,
    )
    k = int(subpoints_per_cell)
    if k == 1:
        offsets = centers.new_tensor([[0.0, 0.0]])
    elif k == 4:
        offsets = centers.new_tensor(
            [[-0.25, -0.25], [-0.25, 0.25], [0.25, -0.25], [0.25, 0.25]]
        )
    else:
        side = int(np.ceil(np.sqrt(k)))
        vals = torch.linspace(-0.25, 0.25, side, device=device, dtype=dtype)
        oy, ox = torch.meshgrid(vals, vals, indexing="ij")
        offsets = torch.stack([ox.reshape(-1), oy.reshape(-1)], dim=-1)[:k]
    scaled_offsets = offsets * centers.new_tensor([dx, dy])
    subpoints = centers.unsqueeze(1) + scaled_offsets.unsqueeze(0)
    return centers, bounds, subpoints


class CellDiscreteBeliefModel(nn.Module):
    """Finite-volume belief over spatial cells, not point support samples."""

    def __init__(
        self,
        state_dim,
        n_agents,
        n_targets,
        horizon=3,
        hidden_dim=512,
        grid_size=(10, 10),
        x_range=(-1000.0, 1000.0),
        y_range=(-1000.0, 1000.0),
        subpoints_per_cell=4,
        soft_label_sigma=150.0,
    ):
        super(CellDiscreteBeliefModel, self).__init__()
        self.state_dim = int(np.prod(state_dim))
        self.n_agents = n_agents
        self.n_targets = n_targets
        self.horizon = horizon
        self.grid_size = tuple(int(v) for v in grid_size)
        self.num_cells = self.grid_size[0] * self.grid_size[1]
        self.subpoints_per_cell = int(subpoints_per_cell)
        self.soft_label_sigma = soft_label_sigma
        self.confidence_default_mode = "entropy"
        self.net = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, horizon * n_targets * self.num_cells),
        )
        centers, bounds, subpoints = _make_cell_grid(
            x_range=x_range,
            y_range=y_range,
            grid_size=self.grid_size,
            subpoints_per_cell=self.subpoints_per_cell,
        )
        self.register_buffer("cell_centers", centers)
        self.register_buffer("cell_bounds", bounds)
        self.register_buffer("cell_subpoints", subpoints)

    def forward(self, state):
        B, T = state.shape[:2]
        logits = self.net(state.reshape(-1, self.state_dim))
        return logits.view(B, T, self.horizon, self.n_targets, self.num_cells)

    def belief_and_loss(self, state, next_state, mask, horizon_weights, eps=1e-6):
        logits = self.forward(state)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        B, T = state.shape[:2]
        valid_belief = mask[:, :, 0] > 0.0
        centers = self.cell_centers.to(device=state.device, dtype=state.dtype)
        sigma2 = float(self.soft_label_sigma) ** 2
        belief_terms = []
        per_step_belief_loss = torch.zeros((B, T), dtype=state.dtype, device=state.device)
        stats = {}
        for h in range(self.horizon):
            valid_t = T - h
            if valid_t <= 0:
                break
            future_state = next_state[:, h:, :]
            target = _extract_target_positions(future_state, self.n_agents, self.n_targets)
            pred_log_probs = log_probs[:, :valid_t, h, :, :]
            diff = centers.view(1, 1, 1, self.num_cells, 2) - target.unsqueeze(-2)
            soft_logits = -0.5 * (diff ** 2).sum(dim=-1) / max(sigma2, eps)
            soft_target = F.softmax(soft_logits, dim=-1)
            ce = -(soft_target * pred_log_probs).sum(dim=-1).mean(dim=-1)
            per_step_belief_loss[:, :valid_t] = per_step_belief_loss[:, :valid_t] + horizon_weights[h] * ce
            valid_h = valid_belief[:, :valid_t]
            if valid_h.any():
                belief_terms.append(horizon_weights[h] * ce[valid_h].mean())
                stats[f"focus_belief_loss_h{h + 1}"] = ce[valid_h].mean().detach().item()
        belief_loss = (
            torch.stack(belief_terms).sum()
            if belief_terms
            else torch.zeros((), dtype=state.dtype, device=state.device)
        )
        return probs, belief_loss, stats, per_step_belief_loss


class QPLEXFocus3Loss(QPLEXFocus2Loss):
    def _horizon_weights(self, horizon, device, dtype):
        if self.focus_config.get("horizon_weights", "uniform") == "uniform":
            return torch.ones(horizon, device=device, dtype=dtype) / float(horizon)
        return super()._horizon_weights(horizon, device, dtype)

    def _cell_visibility(self, cell_subpoints, state, n_targets):
        B, T = state.shape[:2]
        M, K = cell_subpoints.shape[:2]
        flat_points = cell_subpoints.reshape(M * K, 2)
        visible = self._qmc_visibility(flat_points, state, n_targets)
        return visible.view(B, T, self.n_agents, M, K)

    def _focus_credit_target(self, state, next_state, actions, mask):
        eps = self.focus_config.get("eps", 1e-6)
        n_targets = int(self.focus_config.get("n_targets", 8))
        if self.belief_model is None or state is None or next_state is None:
            B, T = actions.shape[:2]
            rho = torch.full(
                (B, T, self.n_agents), 1.0 / self.n_agents,
                dtype=torch.float, device=actions.device,
            )
            valid = torch.zeros((B, T), dtype=torch.bool, device=actions.device)
            total_g = torch.zeros((B, T), dtype=torch.float, device=actions.device)
            confidence = torch.ones((B, T), dtype=torch.float, device=actions.device)
            return rho, valid, total_g, torch.zeros((), device=actions.device), {}, confidence, "off"

        horizon_weights = self._horizon_weights(
            int(self.focus_config.get("horizon", 3)), actions.device, state.dtype
        )
        probs, belief_loss, belief_stats, per_step_belief_loss = self.belief_model.belief_and_loss(
            state, next_state, mask, horizon_weights, eps
        )
        occ = torch.einsum("h,bthjm->btjm", horizon_weights[: probs.size(2)], probs)
        cell_subpoints = self.belief_model.cell_subpoints.to(device=state.device, dtype=state.dtype)
        visible = self._cell_visibility(cell_subpoints, state, n_targets)
        selection = self._decode_target_selection(actions, n_targets)
        if selection is not None:
            visible = visible.unsqueeze(3) * selection.unsqueeze(-1).unsqueeze(-1)
        else:
            visible = visible.unsqueeze(3).expand(-1, -1, -1, n_targets, -1, -1)

        one_minus = 1.0 - visible
        unique_terms = []
        for i in range(self.n_agents):
            if self.n_agents == 1:
                unique_terms.append(torch.ones_like(visible[:, :, i, :, :, :]))
            else:
                others = torch.cat(
                    [one_minus[:, :, :i, :, :, :], one_minus[:, :, i + 1 :, :, :, :]],
                    dim=2,
                )
                unique_terms.append(torch.prod(others, dim=2))
        unique_vis = visible * torch.stack(unique_terms, dim=2)
        unique_vis_cell = unique_vis.mean(dim=-1)
        target_weights = self.focus_config.get("target_weights")
        if target_weights is None:
            tw = torch.ones(n_targets, dtype=state.dtype, device=state.device)
        else:
            tw = torch.as_tensor(target_weights, dtype=state.dtype, device=state.device)[:n_targets]
        g = torch.einsum("btijm,btjm,j->bti", unique_vis_cell, occ, tw)
        total_g = g.sum(dim=-1)
        rho = (g + eps) / (g + eps).sum(dim=-1, keepdim=True)
        valid = (total_g > float(self.focus_config.get("min_credit_signal", 1e-6))) & (mask[:, :, 0] > 0.0)
        confidence, confidence_mode = resolve_confidence(
            self.focus_config,
            valid,
            total_g,
            default_mode="entropy",
            probs=probs,
            per_step_loss=per_step_belief_loss,
        )
        return (
            rho.detach(),
            valid.detach(),
            total_g.detach(),
            belief_loss,
            belief_stats,
            confidence.detach(),
            confidence_mode,
        )


class QPLEXFocus3TorchPolicy(QPLEXFocusTorchPolicy):
    """QPLEX-FOCUS3 with finite-volume cell belief over the terrain."""

    def __init__(self, obs_space, action_space, config):
        from ray.rllib.agents.qplex_focus3.qplex import DEFAULT_CONFIG

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
            config["mixing_embed_dim"], self.args.ffn_hidden_dim, self.args.num_kernel,
        ).to(self.device)
        self.target_mixer = Focus2DuelMixer(
            self.args, self.n_agents, self.n_actions, self.env_global_state_shape,
            config["mixing_embed_dim"], self.args.ffn_hidden_dim, self.args.num_kernel,
        ).to(self.device)

        focus_config = resolve_focus_config(self.config)
        self.occupancy_model = None
        if focus_config.get("enabled", True):
            self.occupancy_model = CellDiscreteBeliefModel(
                self.env_global_state_shape,
                self.n_agents,
                int(focus_config.get("n_targets", 8)),
                horizon=int(focus_config.get("horizon", 3)),
                hidden_dim=int(focus_config.get("belief_hidden_dim", 512)),
                grid_size=focus_config.get("cell_grid_size", [10, 10]),
                x_range=focus_config.get("cell_x_range", [-1000.0, 1000.0]),
                y_range=focus_config.get("cell_y_range", [-1000.0, 1000.0]),
                subpoints_per_cell=int(focus_config.get("cell_subpoints_per_cell", 4)),
                soft_label_sigma=float(focus_config.get("cell_soft_label_sigma", 150.0)),
            ).to(self.device)

        self.update_target()
        self.params = list(self.model.parameters()) + list(self.mixer.parameters())
        if self.occupancy_model:
            self.params += list(self.occupancy_model.parameters())

        self.loss = QPLEXFocus3Loss(
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
