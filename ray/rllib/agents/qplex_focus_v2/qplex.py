import copy
from typing import Type

from ray.rllib.agents.qplex_focus.qplex import (
    DEFAULT_CONFIG as QPLEX_FOCUS_DEFAULT_CONFIG,
    QPlexFocusTrainer,
)
from ray.rllib.agents.qplex_focus_v2.qplex_policy import QPLEXFocusV2TorchPolicy
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import TrainerConfigDict


DEFAULT_CONFIG = copy.deepcopy(QPLEX_FOCUS_DEFAULT_CONFIG)
DEFAULT_CONFIG["mixer"] = "qplex_focus_v2"
DEFAULT_CONFIG["focus"].update(
    {
        "variant": "focus_v2",
        # V2 follows the uploaded method notes: do not inject rho into the
        # mixer; align the QPLEX allocation prior against a reliability-aware
        # dynamic responsibility target.
        "credit_mode": "kl",
        "alpha_credit": 0.05,
        "beta_belief": 0.01,
        "reachable_visibility_enabled": True,
        "reachable_mode": "union",
        "reachable_rotation_step_deg": 20.0,
        "reachable_range_step": 0.0,
        "reachable_half_angle_step_deg": 0.0,
        "credit_ambiguity_gate_enabled": True,
        "teacher_mode": "uniform",
        "focus_policy_coef": 0.0,
    }
)


class QPlexFocusV2Trainer(QPlexFocusTrainer):
    @classmethod
    @override(QPlexFocusTrainer)
    def get_default_config(cls) -> TrainerConfigDict:
        return DEFAULT_CONFIG

    @override(QPlexFocusTrainer)
    def validate_config(self, config: TrainerConfigDict) -> None:
        super().validate_config(config)
        if config["framework"] != "torch":
            raise ValueError("Only `framework=torch` supported for QPLEX_FOCUS-V2.")

    @override(QPlexFocusTrainer)
    def get_default_policy_class(self, config: TrainerConfigDict) -> Type[Policy]:
        return QPLEXFocusV2TorchPolicy

