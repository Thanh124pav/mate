"""Shared, paper-faithful building blocks for grouped CTDE algorithms.

The modules in this package intentionally do not depend on a particular
RLlib policy.  This keeps the mathematical pieces independently testable and
lets FACMAC, ResQ, RiskQ, QFree, and ConcaveQ share only mechanics (sequence
unrolling, masks, and target construction), not objectives.
"""

from .components import (
    ConcaveMixer,
    FACMACCritic,
    MonotonicMixer,
    RecurrentAgent,
    RiskQMixer,
    UnrestrictedMixer,
    coordinate_ascent_actions,
    forward_preserving_scale,
    quantile_huber_loss,
    risk_distortion_weights,
    td_lambda_targets,
)
from .policy_base import GroupedCTDEPolicy, GroupedSequenceBatch
from .trainer import ValueFactorizationTrainer

__all__ = [
    "ConcaveMixer",
    "FACMACCritic",
    "MonotonicMixer",
    "RecurrentAgent",
    "RiskQMixer",
    "UnrestrictedMixer",
    "coordinate_ascent_actions",
    "forward_preserving_scale",
    "quantile_huber_loss",
    "risk_distortion_weights",
    "td_lambda_targets",
    "GroupedCTDEPolicy",
    "GroupedSequenceBatch",
    "ValueFactorizationTrainer",
]
