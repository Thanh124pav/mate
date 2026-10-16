"""Paper-faithful discrete FACMAC policy with an optional FOCUS actor gate."""

import copy

import torch
import torch.nn.functional as F

from ray.rllib.utils.annotations import override

from ray.rllib.agents.value_factorization.components import (
    FACMACCritic,
    MonotonicMixer,
    RecurrentAgent,
    focus_agent_weights,
    forward_preserving_scale,
    masked_mean,
    td_lambda_targets,
    unroll_current_and_next,
)
from ray.rllib.agents.value_factorization.policy_base import (
    GroupedCTDEPolicy,
    GroupedSequenceBatch,
    basic_stats,
    safe_action_values,
)


def _optimizer(name, parameters, lr, config):
    if str(name).lower() == "adam":
        return torch.optim.Adam(
            parameters,
            lr=lr,
            eps=float(config.get("optimizer_epsilon", 1e-7)),
        )
    if str(name).lower() == "rmsprop":
        return torch.optim.RMSprop(
            parameters,
            lr=lr,
            alpha=config.get("optim_alpha", 0.99),
            eps=config.get("optim_eps", 1e-5),
        )
    raise ValueError(f"unknown FACMAC optimizer: {name}")


class FACMACTorchPolicy(GroupedCTDEPolicy):
    algorithm_name = "FACMAC"

    def __init__(self, obs_space, action_space, config):
        from .facmac import DEFAULT_CONFIG

        merged = copy.deepcopy(DEFAULT_CONFIG)
        merged.update(config)
        if "model" in config:
            merged["model"] = dict(DEFAULT_CONFIG["model"], **config["model"])
        if "focus" in config:
            merged["focus"] = dict(DEFAULT_CONFIG.get("focus", {}), **config["focus"])
        super().__init__(obs_space, action_space, merged)
        hidden = int(self.config["model"].get("lstm_cell_size", 64))
        self.model = RecurrentAgent(
            self.obs_dim, self.n_actions, self.n_agents, hidden_dim=hidden
        ).to(self.device)
        self.target_model = copy.deepcopy(self.model).to(self.device)
        self.critic = FACMACCritic(self.obs_dim, self.n_actions, hidden).to(self.device)
        self.target_critic = copy.deepcopy(self.critic).to(self.device)
        mixer_kwargs = dict(
            n_agents=self.n_agents,
            state_dim=self.state_dim,
            embed_dim=int(self.config.get("mixing_embed_dim", 32)),
            hypernet_layers=int(self.config.get("hypernet_layers", 2)),
            hypernet_embed=int(self.config.get("hypernet_embed", 64)),
            skip_connections=bool(self.config.get("skip_connections", False)),
            gated=bool(self.config.get("gated", False)),
        )
        self.mixer = MonotonicMixer(**mixer_kwargs).to(self.device)
        self.target_mixer = copy.deepcopy(self.mixer).to(self.device)
        for name in ("model", "target_model", "critic", "target_critic", "mixer", "target_mixer"):
            self._register_module(name, getattr(self, name))
        self._initialize_focus()

        opt_name = self.config.get("optimizer", "adam")
        self.actor_optimizer = _optimizer(
            opt_name, self.model.parameters(), float(self.config["actor_lr"]), self.config
        )
        critic_parameters = list(self.critic.parameters()) + list(self.mixer.parameters())
        self.critic_optimizer = _optimizer(
            opt_name, critic_parameters, float(self.config["critic_lr"]), self.config
        )
        self._register_optimizer("actor", self.actor_optimizer)
        self._register_optimizer("critic", self.critic_optimizer)
        self.belief_optimizer = None
        if self.focus_engine is not None:
            focus_parameters = [p for p in self.focus_engine.parameters() if p.requires_grad]
            if focus_parameters:
                self.belief_optimizer = torch.optim.Adam(
                    focus_parameters,
                    lr=float(self.focus_config.get("belief_lr", 1e-4)),
                )
                self._register_optimizer("belief", self.belief_optimizer)
        self.update_target()

    def _freeze_critic(self, frozen):
        for parameter in list(self.critic.parameters()) + list(self.mixer.parameters()):
            parameter.requires_grad_(not frozen)

    def _select_actions(self, scores, action_mask, explore, timestep):
        """Official discrete FACMAC categorical/epsilon behavior policy."""

        if not explore:
            return scores.argmax(dim=-1)
        epsilon = float(self.exploration.epsilon_schedule(timestep))
        self.cur_epsilon = epsilon
        probabilities = F.softmax(scores, dim=-1)
        count = action_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        probabilities = (1.0 - epsilon) * probabilities
        probabilities = probabilities + epsilon * action_mask / count
        probabilities = probabilities * action_mask
        probabilities = probabilities / probabilities.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-12)
        return torch.distributions.Categorical(probs=probabilities).sample()

    @override(GroupedCTDEPolicy)
    def _learn_sequence_batch(self, batch: GroupedSequenceBatch):
        active = batch.mask.bool()
        replay_onehot = F.one_hot(batch.actions, self.n_actions).float()
        focus = self._focus_output(batch)

        with torch.no_grad():
            _, target_next_logits, _ = unroll_current_and_next(
                self.target_model, batch.obs, batch.next_obs, batch.actions
            )
            target_next_logits = safe_action_values(
                target_next_logits,
                batch.next_action_mask,
                active & (batch.terminated < 0.5),
            )
            target_actions = target_next_logits.argmax(dim=-1)
            target_onehot = F.one_hot(target_actions, self.n_actions).float()
            target_agent_q = self.target_critic(batch.next_obs, target_onehot)
            target_total = self.target_mixer(target_agent_q, batch.next_state)
            targets = td_lambda_targets(
                batch.rewards,
                batch.terminated,
                batch.mask,
                target_total,
                self.config["gamma"],
                self.config["td_lambda"],
            )

        replay_agent_q = self.critic(batch.obs, replay_onehot)
        replay_total = self.mixer(replay_agent_q, batch.state)
        td_error = replay_total - targets.detach()
        critic_loss = masked_mean(td_error.square(), batch.mask)
        focuss_credit_loss, focuss_credit_stats = self._focuss_agent_objective(
            replay_total, replay_agent_q, focus
        )
        self.critic_optimizer.zero_grad()
        (critic_loss + focuss_credit_loss).backward()
        critic_norm = torch.nn.utils.clip_grad_norm_(
            list(self.critic.parameters()) + list(self.mixer.parameters()),
            float(self.config["grad_norm_clipping"]),
        )
        self.critic_optimizer.step()

        actor_logits, _, _ = unroll_current_and_next(
            self.model, batch.obs, batch.next_obs, batch.actions
        )
        safe_current_mask = batch.action_mask.clone()
        safe_current_mask[~active.expand(-1, -1, self.n_agents)] = 1.0
        actor_logits = actor_logits.masked_fill(safe_current_mask <= 0.0, -1e10)
        policy_action = F.gumbel_softmax(
            actor_logits,
            tau=float(self.config.get("gumbel_temperature", 1.0)),
            hard=True,
            dim=-1,
        )
        weights = self._focuss_route_weights(focus)
        if weights is not None:
            policy_action = forward_preserving_scale(policy_action, weights)
        self._freeze_critic(True)
        actor_agent_q = self.critic(batch.obs, policy_action)
        actor_total = self.mixer(actor_agent_q, batch.state)
        actor_loss = -masked_mean(actor_total, batch.mask)
        focuss_action_loss, focuss_stats = self._focuss_action_objective(
            actor_logits,
            focus,
            actor_loss,
            self.model.parameters(),
        )
        actor_loss = actor_loss + focuss_action_loss

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), float(self.config["grad_norm_clipping"])
        )
        self.actor_optimizer.step()
        self._freeze_critic(False)

        belief_loss = actor_loss.new_zeros(())
        belief_norm = actor_loss.new_zeros(())
        if focus is not None and focus.belief_loss is not None and self.belief_optimizer:
            belief_loss = float(self.focus_config.get("beta_belief", 0.1)) * focus.belief_loss
            self.belief_optimizer.zero_grad()
            belief_loss.backward()
            belief_norm = torch.nn.utils.clip_grad_norm_(
                self.focus_engine.parameters(), float(self.config["grad_norm_clipping"])
            )
            self.belief_optimizer.step()

        stats = basic_stats(
            critic_loss, critic_norm, td_error, replay_total, targets, batch.mask
        )
        stats.update(
            actor_loss=actor_loss.detach(),
            actor_grad_norm=actor_norm,
            belief_loss=belief_loss.detach(),
            belief_grad_norm=belief_norm,
            target_q_mean=masked_mean(target_total, batch.mask).detach(),
        )
        if focus is not None:
            stats.update(
                focus_valid_ratio=focus.valid.float().mean().detach(),
                focus_rho_entropy=(
                    -(focus.rho * (focus.rho + 1e-8).log()).sum(dim=-1).mean().detach()
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
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())


class FACMACFocusTorchPolicy(FACMACTorchPolicy):
    algorithm_name = "FACMAC+FOCUS"

    def __init__(self, obs_space, action_space, config):
        from .facmac import FOCUS_CONFIG

        merged = copy.deepcopy(FOCUS_CONFIG)
        merged.update(config)
        merged["focus"] = dict(FOCUS_CONFIG["focus"], **config.get("focus", {}))
        merged["focus"]["enabled"] = True
        super().__init__(obs_space, action_space, merged)
