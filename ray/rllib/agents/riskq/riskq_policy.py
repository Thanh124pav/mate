"""RiskQ distributional recurrent policy and risk-aware attention mixer."""

import copy

import torch
import torch.nn.functional as F

from ray.rllib.utils.annotations import override

from ray.rllib.agents.value_factorization.components import (
    RecurrentAgent,
    RiskQMixer,
    distorted_expectation,
    focus_agent_weights,
    forward_preserving_scale,
    masked_mean,
    quantile_huber_loss,
    td_lambda_targets,
    unroll_current_and_next,
)
from ray.rllib.agents.value_factorization.policy_base import (
    GroupedCTDEPolicy,
    GroupedSequenceBatch,
    safe_action_values,
)


class RiskQTorchPolicy(GroupedCTDEPolicy):
    algorithm_name = "RiskQ"

    def __init__(self, obs_space, action_space, config):
        from .riskq import DEFAULT_CONFIG

        merged = copy.deepcopy(DEFAULT_CONFIG)
        merged.update(config)
        merged["model"] = dict(DEFAULT_CONFIG["model"], **config.get("model", {}))
        merged["focus"] = dict(DEFAULT_CONFIG.get("focus", {}), **config.get("focus", {}))
        super().__init__(obs_space, action_space, merged)
        self.n_quantiles = int(self.config.get("n_quantiles", 32))
        self.n_target_quantiles = int(
            self.config.get("n_target_quantiles", self.n_quantiles)
        )
        if self.n_target_quantiles != self.n_quantiles:
            raise ValueError(
                "the QR-DQN agent has one fixed atom count; n_target_quantiles "
                "must equal n_quantiles"
            )
        hidden = int(self.config["model"].get("lstm_cell_size", 64))
        self.model = RecurrentAgent(
            self.obs_dim,
            self.n_actions,
            self.n_agents,
            hidden_dim=hidden,
            output_dim=self.n_actions * self.n_quantiles,
        ).to(self.device)
        self.target_model = copy.deepcopy(self.model).to(self.device)
        key_dim = int(self.config.get("agent_own_state_size", 9))
        self.mixer = RiskQMixer(
            self.n_agents,
            self.state_dim,
            key_dim,
            n_heads=int(self.config.get("n_attention_head", 1)),
            query_hidden=int(self.config.get("n_query_embedding_layer1", 32)),
            attention_dim=int(self.config.get("n_query_embedding_layer2", 16)),
            key_embedding_dim=int(self.config.get("n_key_embedding_layer1", 16)),
            head_hidden=int(self.config.get("n_head_embedding_layer1", 32)),
            constant_hidden=int(self.config.get("n_constant_value", 16)),
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
        self._register_optimizer("riskq", self.optimizer)
        self.belief_optimizer = None
        if self.focus_engine is not None:
            parameters = [p for p in self.focus_engine.parameters() if p.requires_grad]
            if parameters:
                self.belief_optimizer = torch.optim.Adam(
                    parameters, lr=float(self.focus_config.get("belief_lr", 1e-4))
                )
                self._register_optimizer("belief", self.belief_optimizer)
        self.update_target()

    def _to_quantiles(self, flat):
        quantiles = flat.view(*flat.shape[:-1], self.n_actions, self.n_quantiles)
        if self.config.get("sort_quantiles", True):
            quantiles = quantiles.sort(dim=-1).values
        return quantiles

    @override(GroupedCTDEPolicy)
    def _reshape_model_output(self, output):
        return self._to_quantiles(output)

    @override(GroupedCTDEPolicy)
    def _scores_from_output(self, output):
        return distorted_expectation(
            output,
            self.config.get("risk_type", "wang"),
            float(self.config.get("risk_param", 0.75)),
        )

    def _agent_keys(self, states, observations):
        key_dim = int(self.config.get("agent_own_state_size", 9))
        start = int(self.config.get("agent_own_state_start", 13))
        required = start + self.n_agents * key_dim
        if states.size(-1) >= required:
            return torch.stack(
                [states[..., start + i * key_dim : start + (i + 1) * key_dim]
                 for i in range(self.n_agents)],
                dim=-2,
            ).detach()
        local = observations[..., : min(key_dim, observations.size(-1))]
        if local.size(-1) < key_dim:
            local = F.pad(local, (0, key_dim - local.size(-1)))
        return local.detach()

    def _gather_action_quantiles(self, values, actions):
        index = actions.unsqueeze(-1).unsqueeze(-1).expand(
            *actions.shape, 1, values.size(-1)
        )
        return values.gather(-2, index).squeeze(-2)

    @override(GroupedCTDEPolicy)
    def _learn_sequence_batch(self, batch: GroupedSequenceBatch):
        active = batch.mask.bool()
        focus = self._focus_output(batch)
        current_flat, online_next_flat, _ = unroll_current_and_next(
            self.model, batch.obs, batch.next_obs, batch.actions
        )
        with torch.no_grad():
            _, target_next_flat, _ = unroll_current_and_next(
                self.target_model, batch.obs, batch.next_obs, batch.actions
            )
        current = self._to_quantiles(current_flat)
        online_next = self._to_quantiles(online_next_flat)
        target_next = self._to_quantiles(target_next_flat)
        chosen = self._gather_action_quantiles(current, batch.actions)

        next_risk = self._scores_from_output(online_next)
        next_risk = safe_action_values(
            next_risk,
            batch.next_action_mask,
            active & (batch.terminated < 0.5),
        )
        next_actions = next_risk.argmax(dim=-1)
        target_chosen = self._gather_action_quantiles(target_next, next_actions)
        weights = self._focuss_route_weights(focus)
        if weights is not None:
            chosen = forward_preserving_scale(chosen, weights)
        keys = self._agent_keys(batch.state, batch.obs)
        next_keys = self._agent_keys(batch.next_state, batch.next_obs)
        joint_quantiles = self.mixer(chosen, batch.state, keys)
        with torch.no_grad():
            target_joint = self.target_mixer(
                target_chosen, batch.next_state, next_keys
            )
            quantile_targets = td_lambda_targets(
                batch.rewards.expand_as(target_joint),
                batch.terminated.expand_as(target_joint),
                batch.mask.expand_as(target_joint),
                target_joint,
                self.config["gamma"],
                self.config["td_lambda"],
            )
        quantile_loss = quantile_huber_loss(
            joint_quantiles,
            quantile_targets.detach(),
            batch.mask.squeeze(-1),
        )
        loss = float(self.config.get("qmix_loss", 1.0)) * quantile_loss
        focuss_credit_loss, focuss_credit_stats = self._focuss_agent_objective(
            joint_quantiles, chosen, focus
        )
        loss = loss + focuss_credit_loss
        focuss_action_loss, focuss_stats = self._focuss_action_objective(
            self._scores_from_output(current),
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

        mean_q = joint_quantiles.mean(dim=-1, keepdim=True)
        mean_target = quantile_targets.mean(dim=-1, keepdim=True)
        stats = {
            "loss": loss.detach(),
            "quantile_loss": quantile_loss.detach(),
            "grad_norm": grad_norm,
            "mean_q": masked_mean(mean_q, batch.mask).detach(),
            "mean_target": masked_mean(mean_target, batch.mask).detach(),
            "mean_risk": masked_mean(next_risk.max(dim=-1).values.mean(dim=-1, keepdim=True), batch.mask).detach(),
            "quantile_spread": (joint_quantiles[..., -1] - joint_quantiles[..., 0]).mean().detach(),
            "belief_loss": belief_loss.detach(),
            "belief_grad_norm": belief_norm,
        }
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
        stats.update(focuss_credit_stats)
        return stats

    @override(GroupedCTDEPolicy)
    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())


class RiskQFocusTorchPolicy(RiskQTorchPolicy):
    algorithm_name = "RiskQ+FOCUS"

    def __init__(self, obs_space, action_space, config):
        from .riskq import FOCUS_CONFIG

        merged = copy.deepcopy(FOCUS_CONFIG)
        merged.update(config)
        merged["focus"] = dict(FOCUS_CONFIG["focus"], **config.get("focus", {}))
        merged["focus"]["enabled"] = True
        super().__init__(obs_space, action_space, merged)
