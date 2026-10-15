"""QFree transformation, value/advantage mixers, and IGM regularization."""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from ray.rllib.utils.annotations import override

from ray.rllib.agents.value_factorization.components import (
    focus_agent_weights,
    forward_preserving_scale,
    masked_mean,
    unroll_current_and_next,
)
from ray.rllib.agents.value_factorization.policy_base import (
    GroupedCTDEPolicy,
    GroupedSequenceBatch,
    basic_stats,
    safe_action_values,
)


class QFreeAgent(nn.Module):
    """Local recurrent dueling network from QFree equations (7) and (11)."""

    def __init__(self, obs_dim, n_actions, n_agents, hidden_dim=64):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.n_agents = int(n_agents)
        self.hidden_dim = int(hidden_dim)
        input_dim = self.obs_dim + self.n_actions + self.n_agents
        self.fc1 = nn.Linear(input_dim, self.hidden_dim)
        self.rnn = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        self.value = nn.Linear(self.hidden_dim, 1)
        self.advantage = nn.Linear(self.hidden_dim, self.n_actions)

    def initial_state(self, batch_size, device=None, dtype=None):
        ref = self.fc1.weight
        return torch.zeros(
            int(batch_size), self.n_agents, self.hidden_dim,
            device=device or ref.device, dtype=dtype or ref.dtype,
        )

    def forward(self, obs, hidden, previous_action=None):
        if previous_action is None:
            previous_action = obs.new_zeros(*obs.shape[:-1], self.n_actions)
        ids = torch.eye(self.n_agents, dtype=obs.dtype, device=obs.device)
        ids = ids.view(*([1] * (obs.dim() - 2)), self.n_agents, self.n_agents)
        ids = ids.expand(*obs.shape[:-2], self.n_agents, self.n_agents)
        x = torch.cat([obs, previous_action.to(obs.dtype), ids], dim=-1)
        leading = x.shape[:-1]
        x = F.relu(self.fc1(x.reshape(-1, x.size(-1))))
        h = self.rnn(x, hidden.reshape(-1, self.hidden_dim))
        value = self.value(h)
        advantage = self.advantage(h)
        output = torch.cat([value, advantage], dim=-1).view(*leading, 1 + self.n_actions)
        return output, h.view(*leading, self.hidden_dim)


class QFreeTransformation(nn.Module):
    """State-conditioned positive affine transformation from equation (12)."""

    def __init__(self, state_dim, n_agents, hidden_dim=64):
        super().__init__()
        self.state_dim = int(state_dim)
        self.n_agents = int(n_agents)
        self.weight = nn.Sequential(
            nn.Linear(self.state_dim, int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), self.n_agents),
        )
        self.bias = nn.Sequential(
            nn.Linear(self.state_dim, int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), self.n_agents),
        )

    def forward(self, values, advantages, states):
        flat = states.reshape(-1, self.state_dim)
        weight = self.weight(flat).abs().view(*states.shape[:-1], self.n_agents)
        bias = self.bias(flat).view(*states.shape[:-1], self.n_agents)
        return weight * values + bias, weight * advantages


class QFreeFeedForwardMixer(nn.Module):
    """Unrestricted feed-forward mixer from equation (13)."""

    def __init__(self, n_agents, hidden_dim=64):
        super().__init__()
        self.n_agents = int(n_agents)
        self.net = nn.Sequential(
            nn.Linear(self.n_agents, int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, values):
        return self.net(values.reshape(-1, self.n_agents)).view(*values.shape[:-1], 1)


class QFreeJointMixer(nn.Module):
    def __init__(self, state_dim, n_agents, transform_hidden=64, mixing_hidden=64):
        super().__init__()
        self.transform = QFreeTransformation(state_dim, n_agents, transform_hidden)
        self.value_mixer = QFreeFeedForwardMixer(n_agents, mixing_hidden)
        self.advantage_mixer = QFreeFeedForwardMixer(n_agents, mixing_hidden)

    def forward(self, values, advantages, states, advantage_weights=None):
        transformed_v, transformed_a = self.transform(values, advantages, states)
        if advantage_weights is not None:
            transformed_a = forward_preserving_scale(
                transformed_a, advantage_weights
            )
        value_total = self.value_mixer(transformed_v)
        advantage_total = self.advantage_mixer(transformed_a)
        return value_total + advantage_total, value_total, advantage_total


class QFreeTorchPolicy(GroupedCTDEPolicy):
    algorithm_name = "QFree"

    def __init__(self, obs_space, action_space, config):
        from .qfree import DEFAULT_CONFIG

        merged = copy.deepcopy(DEFAULT_CONFIG)
        merged.update(config)
        merged["model"] = dict(DEFAULT_CONFIG["model"], **config.get("model", {}))
        merged["focus"] = dict(DEFAULT_CONFIG.get("focus", {}), **config.get("focus", {}))
        super().__init__(obs_space, action_space, merged)
        hidden = int(self.config["model"].get("lstm_cell_size", 64))
        self.model = QFreeAgent(
            self.obs_dim, self.n_actions, self.n_agents, hidden
        ).to(self.device)
        self.target_model = copy.deepcopy(self.model).to(self.device)
        self.mixer = QFreeJointMixer(
            self.state_dim,
            self.n_agents,
            int(self.config.get("transformation_hidden_dim", 64)),
            int(self.config.get("mixing_hidden_dim", 64)),
        ).to(self.device)
        self.target_mixer = copy.deepcopy(self.mixer).to(self.device)
        for name in ("model", "target_model", "mixer", "target_mixer"):
            self._register_module(name, getattr(self, name))
        self._initialize_focus()
        self.online_parameters = list(self.model.parameters()) + list(self.mixer.parameters())
        self.optimizer = torch.optim.RMSprop(
            self.online_parameters,
            lr=float(self.config["lr"]),
            alpha=float(self.config.get("optim_alpha", 0.99)),
            eps=float(self.config.get("optim_eps", 1e-5)),
        )
        self._register_optimizer("qfree", self.optimizer)
        self.belief_optimizer = None
        if self.focus_engine is not None:
            parameters = [p for p in self.focus_engine.parameters() if p.requires_grad]
            if parameters:
                self.belief_optimizer = torch.optim.Adam(
                    parameters, lr=float(self.focus_config.get("belief_lr", 1e-4))
                )
                self._register_optimizer("belief", self.belief_optimizer)
        self.update_target()

    def _dueling(self, output, action_mask=None, active=None):
        value = output[..., 0]
        raw_advantage = output[..., 1:]
        if action_mask is None:
            maximum = raw_advantage.max(dim=-1, keepdim=True).values
        else:
            safe_mask = action_mask.clone()
            if active is not None:
                live_agents = active.expand(*action_mask.shape[:-1]).bool()
                safe_mask[~live_agents] = 1.0
            maximum = raw_advantage.masked_fill(safe_mask <= 0.0, -torch.inf).max(
                dim=-1, keepdim=True
            ).values
            maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        centered = raw_advantage - maximum
        return value, centered, value.unsqueeze(-1) + centered

    @override(GroupedCTDEPolicy)
    def _reshape_model_output(self, output):
        return self._dueling(output)[2]

    @override(GroupedCTDEPolicy)
    def _learn_sequence_batch(self, batch: GroupedSequenceBatch):
        active = batch.mask.bool()
        focus = self._focus_output(batch)
        current_output, next_online_output, _ = unroll_current_and_next(
            self.model, batch.obs, batch.next_obs, batch.actions
        )
        with torch.no_grad():
            _, next_target_output, _ = unroll_current_and_next(
                self.target_model, batch.obs, batch.next_obs, batch.actions
            )
        current_v, current_a, current_q = self._dueling(
            current_output, batch.action_mask, active
        )
        next_v_online, next_a_online, next_q_online = self._dueling(
            next_online_output, batch.next_action_mask, active & (batch.terminated < 0.5)
        )
        next_v_target, next_a_target, _ = self._dueling(
            next_target_output, batch.next_action_mask, active & (batch.terminated < 0.5)
        )
        action_index = batch.actions.unsqueeze(-1)
        current_chosen_a = current_a.gather(-1, action_index).squeeze(-1)
        next_safe_q = safe_action_values(
            next_q_online,
            batch.next_action_mask,
            active & (batch.terminated < 0.5),
        )
        next_greedy = next_safe_q.argmax(dim=-1)
        next_target_a = next_a_target.gather(
            -1, next_greedy.unsqueeze(-1)
        ).squeeze(-1)
        weights = None
        if focus is not None:
            weights = focus_agent_weights(
                focus.rho,
                self.n_agents,
                float(self.focus_config.get("eta", 0.1)),
                focus.confidence,
            )
        current_total, current_value_total, current_advantage_total = self.mixer(
            current_v, current_chosen_a, batch.state, advantage_weights=weights
        )
        with torch.no_grad():
            next_target_total, _, _ = self.target_mixer(
                next_v_target, next_target_a, batch.next_state
            )
            targets = batch.rewards + float(self.config["gamma"]) * (
                1.0 - batch.terminated
            ) * next_target_total
        td_error = current_total - targets.detach()
        td_loss = masked_mean(td_error.square(), batch.mask)

        next_online_greedy_a = next_a_online.gather(
            -1, next_greedy.unsqueeze(-1)
        ).squeeze(-1)
        _, _, optimal_advantage_total = self.mixer(
            next_v_online, next_online_greedy_a, batch.next_state
        )
        replay_at_next_a = next_a_online.gather(-1, action_index).squeeze(-1)
        _, _, nonoptimal_advantage_total = self.mixer(
            next_v_online, replay_at_next_a, batch.next_state
        )
        optimal_loss = masked_mean(optimal_advantage_total.square(), batch.mask)
        nonoptimal = (batch.actions != next_greedy).any(dim=-1, keepdim=True)
        nonoptimal_mask = batch.mask * nonoptimal.to(batch.mask.dtype)
        # Eq. (10) requires A_tot <= 0.  The paper prints min(A, 0)^2 in
        # Eq. (19), but its adjacent prose says to penalize A_tot > 0.  ReLU is
        # the constraint-correct implementation of that stated objective.
        nonoptimal_loss = masked_mean(
            F.relu(nonoptimal_advantage_total).square(), nonoptimal_mask
        )
        loss = (
            td_loss
            + float(self.config.get("igm_optimal_coefficient", 1.0)) * optimal_loss
            + float(self.config.get("igm_nonoptimal_coefficient", 1.0)) * nonoptimal_loss
        )
        focuss_action_loss, focuss_stats = self._focuss_action_objective(
            current_q,
            focus,
            loss,
            self.online_parameters,
        )
        loss = loss + focuss_action_loss

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.online_parameters, float(self.config["grad_norm_clipping"])
        )
        self.optimizer.step()

        belief_loss = loss.new_zeros(())
        belief_norm = loss.new_zeros(())
        if focus is not None and focus.belief_loss is not None and self.belief_optimizer:
            belief_loss = float(self.focus_config.get("beta_belief", 0.1)) * focus.belief_loss
            self.belief_optimizer.zero_grad()
            belief_loss.backward()
            belief_norm = torch.nn.utils.clip_grad_norm_(
                self.focus_engine.parameters(), float(self.config["grad_norm_clipping"])
            )
            self.belief_optimizer.step()

        stats = basic_stats(loss, grad_norm, td_error, current_total, targets, batch.mask)
        stats.update(
            td_loss=td_loss.detach(),
            igm_optimal_loss=optimal_loss.detach(),
            igm_nonoptimal_loss=nonoptimal_loss.detach(),
            igm_optimal_advantage_abs=masked_mean(
                optimal_advantage_total.abs(), batch.mask
            ).detach(),
            igm_positive_nonoptimal_rate=masked_mean(
                (nonoptimal_advantage_total > 0.0).float(), nonoptimal_mask
            ).detach(),
            value_total_mean=masked_mean(current_value_total, batch.mask).detach(),
            advantage_total_mean=masked_mean(current_advantage_total, batch.mask).detach(),
            belief_loss=belief_loss.detach(),
            belief_grad_norm=belief_norm,
        )
        if focus is not None:
            stats.update(
                focus_valid_ratio=focus.valid.float().mean().detach(),
                focus_rho_entropy=(
                    -(focus.rho * (focus.rho + 1e-8).log()).sum(-1).mean().detach()
                ),
                focus_total_gain=focus.total_gain.mean().detach(),
                focus_confidence=focus.confidence.mean().detach(),
            )
            stats.update(self.focus_engine.last_belief_stats)
        stats.update(focuss_stats)
        return stats

    @override(GroupedCTDEPolicy)
    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())


class QFreeFocusTorchPolicy(QFreeTorchPolicy):
    algorithm_name = "QFree+FOCUS"

    def __init__(self, obs_space, action_space, config):
        from .qfree import FOCUS_CONFIG

        merged = copy.deepcopy(FOCUS_CONFIG)
        merged.update(config)
        merged["focus"] = dict(FOCUS_CONFIG["focus"], **config.get("focus", {}))
        merged["focus"]["enabled"] = True
        super().__init__(obs_space, action_space, merged)
