"""ResQ's main, residual, and unrestricted centralized value objectives."""

import copy

import torch
import torch.nn.functional as F

from ray.rllib.utils.annotations import override

from ray.rllib.agents.value_factorization.components import (
    MonotonicMixer,
    RecurrentAgent,
    UnrestrictedMixer,
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


class ResQTorchPolicy(GroupedCTDEPolicy):
    algorithm_name = "ResQ"

    def __init__(self, obs_space, action_space, config):
        from .resq import DEFAULT_CONFIG

        merged = copy.deepcopy(DEFAULT_CONFIG)
        merged.update(config)
        merged["model"] = dict(DEFAULT_CONFIG["model"], **config.get("model", {}))
        merged["focus"] = dict(DEFAULT_CONFIG.get("focus", {}), **config.get("focus", {}))
        super().__init__(obs_space, action_space, merged)
        hidden = int(self.config["model"].get("lstm_cell_size", 64))
        agent_args = (self.obs_dim, self.n_actions, self.n_agents)
        self.model = RecurrentAgent(*agent_args, hidden_dim=hidden).to(self.device)
        self.target_model = copy.deepcopy(self.model).to(self.device)
        self.residual_model = RecurrentAgent(*agent_args, hidden_dim=hidden).to(self.device)
        self.target_residual_model = copy.deepcopy(self.residual_model).to(self.device)
        self.central_model = RecurrentAgent(*agent_args, hidden_dim=hidden).to(self.device)
        self.target_central_model = copy.deepcopy(self.central_model).to(self.device)
        self.mixer = MonotonicMixer(
            self.n_agents,
            self.state_dim,
            embed_dim=int(self.config.get("mixing_embed_dim", 32)),
            hypernet_layers=int(self.config.get("hypernet_layers", 2)),
            hypernet_embed=int(self.config.get("hypernet_embed", 64)),
        ).to(self.device)
        self.target_mixer = copy.deepcopy(self.mixer).to(self.device)
        central_width = int(self.config.get("central_mixing_embed_dim", 128))
        self.residual_mixer = UnrestrictedMixer(
            self.n_agents,
            self.state_dim,
            hidden_dims=(central_width, central_width, central_width),
            include_state_value=True,
        ).to(self.device)
        self.target_residual_mixer = copy.deepcopy(self.residual_mixer).to(self.device)
        self.central_mixer = UnrestrictedMixer(
            self.n_agents,
            self.state_dim,
            hidden_dims=(central_width, central_width, central_width),
            include_state_value=True,
        ).to(self.device)
        self.target_central_mixer = copy.deepcopy(self.central_mixer).to(self.device)
        for name in (
            "model",
            "target_model",
            "residual_model",
            "target_residual_model",
            "central_model",
            "target_central_model",
            "mixer",
            "target_mixer",
            "residual_mixer",
            "target_residual_mixer",
            "central_mixer",
            "target_central_mixer",
        ):
            self._register_module(name, getattr(self, name))
        self._initialize_focus()
        self.online_parameters = []
        for module in (
            self.model,
            self.residual_model,
            self.central_model,
            self.mixer,
            self.residual_mixer,
            self.central_mixer,
        ):
            self.online_parameters.extend(module.parameters())
        self.optimizer = torch.optim.RMSprop(
            self.online_parameters,
            lr=float(self.config["lr"]),
            alpha=float(self.config.get("optim_alpha", 0.99)),
            eps=float(self.config.get("optim_eps", 1e-5)),
        )
        self._register_optimizer("resq", self.optimizer)
        self.belief_optimizer = None
        if self.focus_engine is not None:
            parameters = [p for p in self.focus_engine.parameters() if p.requires_grad]
            if parameters:
                self.belief_optimizer = torch.optim.Adam(
                    parameters, lr=float(self.focus_config.get("belief_lr", 1e-4))
                )
                self._register_optimizer("belief", self.belief_optimizer)
        self.update_target()

    @override(GroupedCTDEPolicy)
    def _learn_sequence_batch(self, batch: GroupedSequenceBatch):
        active = batch.mask.bool()
        focus = self._focus_output(batch)
        main_all, main_next, _ = unroll_current_and_next(
            self.model, batch.obs, batch.next_obs, batch.actions
        )
        residual_all, _, _ = unroll_current_and_next(
            self.residual_model, batch.obs, batch.next_obs, batch.actions
        )
        central_all, _, _ = unroll_current_and_next(
            self.central_model, batch.obs, batch.next_obs, batch.actions
        )
        action_index = batch.actions.unsqueeze(-1)
        main_chosen = main_all.gather(-1, action_index).squeeze(-1)
        residual_chosen = residual_all.gather(-1, action_index).squeeze(-1)
        central_chosen = central_all.gather(-1, action_index).squeeze(-1)

        masked_current = safe_action_values(
            main_all, batch.action_mask, active
        )
        current_greedy = masked_current.argmax(dim=-1)
        behavior_is_greedy = (batch.actions == current_greedy).all(dim=-1, keepdim=True)
        residual_weight = (~behavior_is_greedy).to(main_chosen.dtype)

        with torch.no_grad():
            main_next_safe = safe_action_values(
                main_next,
                batch.next_action_mask,
                active & (batch.terminated < 0.5),
            )
            next_actions = main_next_safe.argmax(dim=-1)
            _, target_central_next, _ = unroll_current_and_next(
                self.target_central_model, batch.obs, batch.next_obs, batch.actions
            )
            target_central_chosen = target_central_next.gather(
                -1, next_actions.unsqueeze(-1)
            ).squeeze(-1)
            target_total = self.target_central_mixer(
                target_central_chosen, batch.next_state
            )
            targets = td_lambda_targets(
                batch.rewards,
                batch.terminated,
                batch.mask,
                target_total,
                self.config["gamma"],
                self.config["td_lambda"],
            )

        weights = self._focuss_route_weights(focus)
        if weights is not None:
            main_chosen = forward_preserving_scale(main_chosen, weights)
            residual_chosen = forward_preserving_scale(residual_chosen, weights)
            central_chosen = forward_preserving_scale(central_chosen, weights)

        main_total = self.mixer(main_chosen, batch.state)
        residual_total_raw = self.residual_mixer(residual_chosen, batch.state)
        residual_total = (
            -residual_total_raw.abs()
            if self.config.get("residual_negative_abs", True)
            else residual_total_raw
        )
        central_total = self.central_mixer(central_chosen, batch.state)
        td_error = main_total + residual_weight.detach() * residual_total - targets.detach()
        central_error = central_total - targets.detach()
        qmix_loss = masked_mean(td_error.square(), batch.mask)
        central_loss = masked_mean(central_error.square(), batch.mask)
        noopt_loss = masked_mean(F.relu(residual_total).square(), batch.mask)
        loss = (
            float(self.config.get("qmix_loss", 1.0)) * qmix_loss
            + float(self.config.get("central_loss", 1.0)) * central_loss
            + float(self.config.get("noopt_loss", 1.0)) * noopt_loss
        )
        focuss_credit_loss, focuss_credit_stats = self._focuss_agent_objective(
            main_total, main_chosen, focus
        )
        loss = loss + focuss_credit_loss
        focuss_action_loss, focuss_stats = self._focuss_action_objective(
            main_all,
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

        stats = basic_stats(loss, grad_norm, td_error, main_total, targets, batch.mask)
        stats.update(
            qmix_loss=qmix_loss.detach(),
            central_loss=central_loss.detach(),
            noopt_loss=noopt_loss.detach(),
            residual_weight_mean=residual_weight.mean().detach(),
            residual_q_mean=masked_mean(residual_total, batch.mask).detach(),
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
        stats.update(focuss_credit_stats)
        return stats

    @override(GroupedCTDEPolicy)
    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_residual_model.load_state_dict(self.residual_model.state_dict())
        self.target_central_model.load_state_dict(self.central_model.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.target_residual_mixer.load_state_dict(self.residual_mixer.state_dict())
        self.target_central_mixer.load_state_dict(self.central_mixer.state_dict())


class ResQFocusTorchPolicy(ResQTorchPolicy):
    algorithm_name = "ResQ+FOCUS"

    def __init__(self, obs_space, action_space, config):
        from .resq import FOCUS_CONFIG

        merged = copy.deepcopy(FOCUS_CONFIG)
        merged.update(config)
        merged["focus"] = dict(FOCUS_CONFIG["focus"], **config.get("focus", {}))
        merged["focus"]["enabled"] = True
        super().__init__(obs_space, action_space, merged)
