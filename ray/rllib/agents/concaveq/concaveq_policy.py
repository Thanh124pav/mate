"""ConcaveQ critic, exact coordinate sweep, and decentralized soft policies."""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ray.rllib.utils.annotations import override

from ray.rllib.agents.value_factorization.components import (
    ConcaveMixer,
    RecurrentAgent,
    UnrestrictedMixer,
    coordinate_ascent_actions,
    focus_agent_weights,
    forward_preserving_scale,
    masked_mean,
    td_lambda_targets,
    unroll_current_and_next,
)
from ray.rllib.agents.value_factorization.policy_base import (
    GroupedCTDEPolicy,
    GroupedSequenceBatch,
)


class PolicyValueMixer(nn.Module):
    """State-conditioned one-layer q^pi operator from ConcaveQ Eq. (18)."""

    def __init__(self, n_agents, state_dim):
        super().__init__()
        self.n_agents = int(n_agents)
        self.state_dim = int(state_dim)
        self.weight = nn.Linear(self.state_dim, self.n_agents)
        self.bias = nn.Linear(self.state_dim, 1)

    def forward(self, agent_values, states):
        flat_s = states.reshape(-1, self.state_dim)
        weights = self.weight(flat_s).abs().view(*states.shape[:-1], self.n_agents)
        bias = self.bias(flat_s).view(*states.shape[:-1], 1)
        return (weights * agent_values).sum(dim=-1, keepdim=True) + bias


class ConcaveQTorchPolicy(GroupedCTDEPolicy):
    algorithm_name = "ConcaveQ"

    def __init__(self, obs_space, action_space, config):
        from .concaveq import DEFAULT_CONFIG

        merged = copy.deepcopy(DEFAULT_CONFIG)
        merged.update(config)
        merged["model"] = dict(DEFAULT_CONFIG["model"], **config.get("model", {}))
        merged["focus"] = dict(DEFAULT_CONFIG.get("focus", {}), **config.get("focus", {}))
        super().__init__(obs_space, action_space, merged)
        hidden = int(self.config["model"].get("lstm_cell_size", 64))
        agent_args = (self.obs_dim, self.n_actions, self.n_agents)
        # The local policy alone is active at execution time.
        self.model = RecurrentAgent(*agent_args, hidden_dim=hidden).to(self.device)
        self.q_model = RecurrentAgent(*agent_args, hidden_dim=hidden).to(self.device)
        self.target_q_model = copy.deepcopy(self.q_model).to(self.device)
        self.concave_mixer = ConcaveMixer(
            self.n_agents,
            self.state_dim,
            hidden_dims=tuple(self.config.get("concave_hidden_dims", (64, 64, 64))),
            hypernet_hidden=int(self.config.get("concave_hypernet_hidden", 64)),
        ).to(self.device)
        self.target_concave_mixer = copy.deepcopy(self.concave_mixer).to(self.device)
        unrestricted_hidden = tuple(
            self.config.get("unrestricted_hidden_dims", (64, 64))
        )
        self.unrestricted_mixer = UnrestrictedMixer(
            self.n_agents,
            self.state_dim,
            hidden_dims=unrestricted_hidden,
            include_state_value=True,
        ).to(self.device)
        self.target_unrestricted_mixer = copy.deepcopy(
            self.unrestricted_mixer
        ).to(self.device)
        self.policy_value_mixer = PolicyValueMixer(
            self.n_agents, self.state_dim
        ).to(self.device)
        self.log_alpha = nn.Parameter(
            torch.full(
                (self.n_agents,),
                float(self.config.get("initial_log_alpha", -0.07)),
                device=self.device,
            )
        )
        for name in (
            "model",
            "q_model",
            "target_q_model",
            "concave_mixer",
            "target_concave_mixer",
            "unrestricted_mixer",
            "target_unrestricted_mixer",
            "policy_value_mixer",
        ):
            self._register_module(name, getattr(self, name))
        self._initialize_focus()
        self.critic_parameters = (
            list(self.q_model.parameters())
            + list(self.concave_mixer.parameters())
            + list(self.unrestricted_mixer.parameters())
            + list(self.policy_value_mixer.parameters())
        )
        self.critic_optimizer = torch.optim.RMSprop(
            self.critic_parameters,
            lr=float(self.config["lr"]),
            alpha=float(self.config.get("optim_alpha", 0.99)),
            eps=float(self.config.get("optim_eps", 1e-5)),
        )
        self.actor_optimizer = torch.optim.RMSprop(
            self.model.parameters(),
            lr=float(self.config["lr"]),
            alpha=float(self.config.get("optim_alpha", 0.99)),
            eps=float(self.config.get("optim_eps", 1e-5)),
        )
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=float(self.config.get("alpha_lr", 3e-4))
        )
        self._register_optimizer("critic", self.critic_optimizer)
        self._register_optimizer("actor", self.actor_optimizer)
        self._register_optimizer("temperature", self.alpha_optimizer)
        self.belief_optimizer = None
        if self.focus_engine is not None:
            parameters = [p for p in self.focus_engine.parameters() if p.requires_grad]
            if parameters:
                self.belief_optimizer = torch.optim.Adam(
                    parameters, lr=float(self.focus_config.get("belief_lr", 1e-4))
                )
                self._register_optimizer("belief", self.belief_optimizer)
        self.update_target()

    def get_state(self):
        state = super().get_state()
        state["log_alpha"] = self.log_alpha.detach().cpu().numpy()
        return state

    def set_state(self, state):
        super().set_state(state)
        if "log_alpha" in state:
            self.log_alpha.data.copy_(
                torch.as_tensor(state["log_alpha"], device=self.device)
            )

    def _joint_utilities(self, q_values, actions):
        if actions.dim() == q_values.dim() - 1:
            return q_values.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        if actions.dim() != q_values.dim():
            raise ValueError("joint action rank is incompatible with q_values")
        count = actions.size(-2)
        expanded = q_values.unsqueeze(-3).expand(
            *q_values.shape[:-2], count, *q_values.shape[-2:]
        )
        return expanded.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

    def _joint_value(self, q_values, states, actions, mixer):
        utilities = self._joint_utilities(q_values, actions)
        if actions.dim() == q_values.dim():
            states = states.unsqueeze(-2).expand(
                *states.shape[:-1], actions.size(-2), states.size(-1)
            )
        return mixer(utilities, states)

    def _coordinate_search(self, q_values, states, action_mask, mixer):
        def evaluate(actions):
            return self._joint_value(q_values, states, actions, mixer)

        return coordinate_ascent_actions(
            q_values,
            action_mask,
            evaluate,
            initial_actions=None,
            candidate_chunk_size=int(
                self.config.get("coordinate_candidate_chunk_size", 32)
            ),
        )

    def _freeze(self, modules, frozen):
        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad_(not frozen)

    @override(GroupedCTDEPolicy)
    def _learn_sequence_batch(self, batch: GroupedSequenceBatch):
        active = batch.mask.bool()
        focus = self._focus_output(batch)
        current_q, next_online_q, _ = unroll_current_and_next(
            self.q_model, batch.obs, batch.next_obs, batch.actions
        )
        with torch.no_grad():
            _, next_target_q, _ = unroll_current_and_next(
                self.target_q_model, batch.obs, batch.next_obs, batch.actions
            )
            next_mask = batch.next_action_mask.clone()
            next_live = (active & (batch.terminated < 0.5)).expand(
                -1, -1, self.n_agents
            )
            next_mask[~next_live] = 1.0
            next_opt_actions, _ = self._coordinate_search(
                next_online_q,
                batch.next_state,
                next_mask,
                self.concave_mixer,
            )
            target_next_unrestricted = self._joint_value(
                next_target_q,
                batch.next_state,
                next_opt_actions,
                self.target_unrestricted_mixer,
            )
            targets = td_lambda_targets(
                batch.rewards,
                batch.terminated,
                batch.mask,
                target_next_unrestricted,
                self.config["gamma"],
                self.config["td_lambda"],
            )
            current_mask = batch.action_mask.clone()
            current_mask[~active.expand(-1, -1, self.n_agents)] = 1.0
            current_initial_actions = current_q.detach().masked_fill(
                current_mask <= 0.0, -torch.inf
            ).argmax(dim=-1)
            current_initial_value = self._joint_value(
                current_q.detach(),
                batch.state,
                current_initial_actions,
                self.concave_mixer,
            )
            current_opt_actions, current_opt_value = self._coordinate_search(
                current_q.detach(),
                batch.state,
                current_mask,
                self.concave_mixer,
            )
            coordinate_improvement = current_opt_value - current_initial_value

        unrestricted_optimal = self._joint_value(
            current_q,
            batch.state,
            current_opt_actions,
            self.unrestricted_mixer,
        )
        replay_concave = self._joint_value(
            current_q, batch.state, batch.actions, self.concave_mixer
        )
        unrestricted_error = unrestricted_optimal - targets.detach()
        concave_error = replay_concave - targets.detach()
        projection_weight = torch.where(
            concave_error.detach() < 0.0,
            torch.ones_like(concave_error),
            torch.full_like(concave_error, float(self.config.get("overestimate_weight", 0.5))),
        )
        unrestricted_loss = masked_mean(unrestricted_error.square(), batch.mask)
        concave_loss = masked_mean(
            projection_weight * concave_error.square(), batch.mask
        )

        actor_logits, _, _ = unroll_current_and_next(
            self.model, batch.obs, batch.next_obs, batch.actions
        )
        safe_actor_mask = batch.action_mask.clone()
        safe_actor_mask[~active.expand(-1, -1, self.n_agents)] = 1.0
        actor_logits = actor_logits.masked_fill(safe_actor_mask <= 0.0, -1e10)
        log_policy = F.log_softmax(actor_logits, dim=-1)
        policy = log_policy.exp()
        alpha = self.log_alpha.exp().view(1, 1, self.n_agents, 1)

        # Eq. (19) identifies q^pi(s, E_pi[q_i]) with E_pi[Q_tot].  The
        # one-layer policy mixer therefore needs a value-regression target;
        # directly minimizing -q^pi with respect to the mixer's own weights
        # makes its positive weights and unconstrained bias grow without
        # bound.  That failure mode collapses policy entropy even though all
        # tensors remain finite.
        policy_local_values = (policy.detach() * current_q).sum(dim=-1)
        critic_policy_value = self.policy_value_mixer(
            policy_local_values, batch.state
        )
        policy_critic_loss = -masked_mean(critic_policy_value, batch.mask)
        with torch.no_grad():
            sample_count = int(self.config.get("policy_value_num_samples", 4))
            if sample_count <= 0:
                raise ValueError("policy_value_num_samples must be positive")
            sampled_actions = torch.multinomial(
                policy.detach().reshape(-1, self.n_actions),
                num_samples=sample_count,
                replacement=True,
            ).view(*policy.shape[:-1], sample_count)
            # [B, T, agent, sample] -> [B, T, sample, agent]
            sampled_actions = sampled_actions.movedim(-1, -2)
            policy_value_target = self._joint_value(
                current_q.detach(),
                batch.state,
                sampled_actions,
                self.concave_mixer,
            ).mean(dim=-2)
        fitted_policy_value = self.policy_value_mixer(
            policy_local_values.detach(), batch.state
        )
        policy_value_loss = masked_mean(
            (fitted_policy_value - policy_value_target).square(), batch.mask
        )
        critic_loss = (
            unrestricted_loss
            + concave_loss
            + float(self.config.get("policy_critic_coefficient", 1.0))
            * policy_value_loss
        )
        focuss_credit_loss, focuss_credit_stats = self._focuss_agent_objective(
            critic_policy_value, policy_local_values, focus
        )
        self.critic_optimizer.zero_grad()
        (critic_loss + focuss_credit_loss).backward()
        critic_norm = torch.nn.utils.clip_grad_norm_(
            self.critic_parameters, float(self.config["grad_norm_clipping"])
        )
        self.critic_optimizer.step()

        # Decentralized soft-policy update.  FOCUS changes only per-agent
        # backward credit and leaves the exact ConcaveQ forward objective intact.
        actor_soft_local = (
            policy * (current_q.detach() - alpha.detach() * log_policy)
        ).sum(dim=-1)
        actor_weights = self._focuss_route_weights(focus)
        if actor_weights is not None:
            actor_soft_local = forward_preserving_scale(
                actor_soft_local, actor_weights
            )
        self._freeze([self.policy_value_mixer], True)
        actor_value = self.policy_value_mixer(actor_soft_local, batch.state)
        actor_loss = -masked_mean(actor_value, batch.mask)
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
        self._freeze([self.policy_value_mixer], False)

        target_entropy = float(self.config.get("target_entropy_ratio", 0.98)) * math.log(
            self.n_actions
        )
        entropy = -(policy.detach() * log_policy.detach()).sum(dim=-1)
        alpha_loss = masked_mean(
            self.log_alpha.view(1, 1, self.n_agents)
            * (entropy - target_entropy).detach(),
            batch.agent_mask,
        )
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        belief_loss = critic_loss.new_zeros(())
        belief_norm = critic_loss.new_zeros(())
        if focus is not None and focus.belief_loss is not None and self.belief_optimizer:
            belief_loss = float(self.focus_config.get("beta_belief", 0.1)) * focus.belief_loss
            self.belief_optimizer.zero_grad()
            belief_loss.backward()
            belief_norm = torch.nn.utils.clip_grad_norm_(
                self.focus_engine.parameters(), float(self.config["grad_norm_clipping"])
            )
            self.belief_optimizer.step()

        stats = {
            "loss": (critic_loss + actor_loss).detach(),
            "critic_loss": critic_loss.detach(),
            "unrestricted_loss": unrestricted_loss.detach(),
            "concave_loss": concave_loss.detach(),
            "policy_critic_loss": policy_critic_loss.detach(),
            "policy_value_loss": policy_value_loss.detach(),
            "policy_value_mean": fitted_policy_value.mean().detach(),
            "policy_value_target_mean": policy_value_target.mean().detach(),
            "actor_loss": actor_loss.detach(),
            "alpha_loss": alpha_loss.detach(),
            "alpha_mean": self.log_alpha.exp().mean().detach(),
            "policy_entropy": masked_mean(entropy, batch.agent_mask).detach(),
            "critic_grad_norm": critic_norm,
            "actor_grad_norm": actor_norm,
            "projection_weight_mean": projection_weight.mean().detach(),
            "coordinate_improvement": masked_mean(
                coordinate_improvement, batch.mask
            ).detach(),
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
        self.target_q_model.load_state_dict(self.q_model.state_dict())
        self.target_concave_mixer.load_state_dict(self.concave_mixer.state_dict())
        self.target_unrestricted_mixer.load_state_dict(
            self.unrestricted_mixer.state_dict()
        )


class ConcaveQFocusTorchPolicy(ConcaveQTorchPolicy):
    algorithm_name = "ConcaveQ+FOCUS"

    def __init__(self, obs_space, action_space, config):
        from .concaveq import FOCUS_CONFIG

        merged = copy.deepcopy(FOCUS_CONFIG)
        merged.update(config)
        merged["focus"] = dict(FOCUS_CONFIG["focus"], **config.get("focus", {}))
        merged["focus"]["enabled"] = True
        super().__init__(obs_space, action_space, merged)
