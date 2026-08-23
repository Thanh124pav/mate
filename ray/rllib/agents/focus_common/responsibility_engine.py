from dataclasses import dataclass

from ray.rllib.agents.qplex_focus.qplex_policy import QPLEXFocusLoss
from ray.rllib.utils.framework import try_import_torch


torch, nn = try_import_torch(error=True)


@dataclass
class FocusOutput:
    rho: torch.Tensor
    gains: torch.Tensor
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

    def compute_target(self, global_state, next_global_state, joint_actions, valid_mask):
        """Compute detached responsibility targets and belief supervision.

        This is the stable learner-facing API shared by QPLEX, QMIX, DuelMIX
        and policy-gradient adapters.  The wrapped legacy estimator remains the
        source of truth for geometry and belief semantics.
        """
        return self.estimator._focus_credit_target(
            global_state,
            next_global_state,
            joint_actions,
            valid_mask,
        )

    def signal_confidence_weights(self, total_gain, valid):
        """Expose signal weighting without exposing QPLEX internals."""
        return self.estimator._signal_confidence_weights(total_gain, valid)

    def forward(
        self,
        global_state,
        joint_actions,
        camera_state=None,
        future_target_positions=None,
        valid_mask=None,
        next_global_state=None,
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

        rho, valid, total_gain, belief_loss, confidence, _ = self.compute_target(
            global_state,
            next_global_state,
            joint_actions,
            valid_mask,
        )

        gains = rho * total_gain.unsqueeze(-1)
        return FocusOutput(
            rho=rho,
            gains=gains,
            total_gain=total_gain,
            confidence=confidence,
            valid=valid,
            belief_loss=belief_loss,
        )
