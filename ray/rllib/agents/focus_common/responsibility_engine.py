from dataclasses import dataclass
from typing import Optional

from ray.rllib.agents.qplex_focus.qplex_policy import QPLEXFocusLoss
from ray.rllib.utils.framework import try_import_torch


torch, nn = try_import_torch(error=True)


@dataclass
class FocusOutput:
    rho: torch.Tensor
    gains: torch.Tensor
    pair_gains: Optional[torch.Tensor]
    total_gain: torch.Tensor
    confidence: torch.Tensor
    valid: torch.Tensor
    belief_loss: torch.Tensor = None


class FocusResponsibilityEngine(nn.Module):
    """Backbone-independent wrapper around the existing FOCUS estimator."""

    def __init__(self, occupancy_model, n_agents, n_actions, focus_config=None):
        super().__init__()
        self.estimator = QPLEXFocusLoss(
            model=None,
            target_model=None,
            mixer=None,
            target_mixer=None,
            n_agents=int(n_agents),
            n_actions=int(n_actions),
            focus_config=focus_config or {},
            occupancy_model=occupancy_model,
        )

    @property
    def n_agents(self):
        return self.estimator.n_agents

    @property
    def last_belief_stats(self):
        return getattr(self.estimator, "last_belief_stats", {})

    def get_runtime_state(self):
        return self.estimator.get_focus_runtime_state()

    def set_runtime_state(self, state):
        self.estimator.set_focus_runtime_state(state)

    def observe_belief_errors(self, model_error, baseline_error, learner_update=None):
        return self.estimator.observe_belief_errors(
            model_error,
            baseline_error,
            learner_update=learner_update,
        )

    def observe_belief_diagnostics(
        self,
        diagnostics,
        per_horizon_valid,
        horizon_weights,
        valid_mask=None,
    ):
        """Commit one learner batch of position errors to the causal EMA gate."""

        model_terms, baseline_terms = [], []
        for horizon, valid_h in enumerate(per_horizon_valid):
            valid = valid_h
            if valid_mask is not None:
                valid = valid & valid_mask
            if valid.any():
                weight = horizon_weights[horizon]
                model_terms.append(
                    weight * diagnostics["position_error"][horizon][valid].mean()
                )
                baseline_terms.append(
                    weight
                    * diagnostics["baseline_position_error"][horizon][valid].mean()
                )
        if not model_terms:
            return {}
        model_error = torch.stack(model_terms).sum().detach()
        baseline_error = torch.stack(baseline_terms).sum().detach()
        updated = self.observe_belief_errors(model_error.item(), baseline_error.item())
        runtime = self.get_runtime_state()
        stats = {
            "focus/belief_pos_error_weighted": model_error,
            "focus/belief_baseline_pos_error_weighted": baseline_error,
            "focus/belief_pos_error_improvement_weighted": baseline_error - model_error,
        }
        if not updated:
            return stats
        stats.update({
            "focus/belief_blend_schedule": torch.tensor(
                self.estimator._belief_blend_schedule(),
                device=model_error.device,
            ),
            "focus/belief_performance_gate": torch.tensor(
                self.estimator._belief_performance_gate(),
                device=model_error.device,
            ),
            "focus/belief_effective_blend": torch.tensor(
                self.estimator._belief_blend_schedule()
                * self.estimator._belief_performance_gate(),
                device=model_error.device,
            ),
            "focus/belief_gate_ema_model_error": torch.tensor(
                float(runtime.get("belief_model_error_ema") or 0.0),
                device=model_error.device,
            ),
            "focus/belief_gate_ema_baseline_error": torch.tensor(
                float(runtime.get("belief_baseline_error_ema") or 0.0),
                device=model_error.device,
            ),
            "focus/belief_gate_ema_updates": torch.tensor(
                float(runtime.get("belief_error_ema_updates", 0)),
                device=model_error.device,
            ),
        })
        return stats

    def forward(
        self,
        global_state,
        joint_actions,
        camera_state=None,
        future_target_positions=None,
        valid_mask=None,
        next_global_state=None,
        learner_update=None,
        update_belief_ema=True,
    ):
        del camera_state, future_target_positions

        if valid_mask is None:
            valid_mask = torch.ones(
                joint_actions.shape[:2],
                dtype=global_state.dtype,
                device=global_state.device,
            )
        if valid_mask.dim() == 2:
            valid_mask = valid_mask.unsqueeze(-1).expand(
                -1, -1, self.estimator.n_agents
            )

        # Rollout postprocessing does not know the learner update.  Preserve
        # the counter synchronized from the learner instead of resetting it.
        if learner_update is not None:
            self.estimator.set_focus_learner_update(learner_update)

        rho, valid, total_gain, belief_loss, confidence, _ = (
            self.estimator._focus_credit_target(
                global_state,
                next_global_state,
                joint_actions,
                valid_mask,
                update_belief_ema=update_belief_ema,
            )
        )

        gains = rho * total_gain.unsqueeze(-1)
        return FocusOutput(
            rho=rho,
            gains=gains,
            pair_gains=getattr(self.estimator, "last_pair_gains", None),
            total_gain=total_gain,
            confidence=confidence,
            valid=valid,
            belief_loss=belief_loss,
        )
