import copy
from typing import Type

from ray.rllib.agents.qplex_focus2.qplex import (
    QPlexFocus2Trainer,
    DEFAULT_CONFIG as QPLEX_FOCUS2_DEFAULT_CONFIG,
)
from ray.rllib.agents.qplex_focus3.qplex_policy import QPLEXFocus3TorchPolicy
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import TrainerConfigDict


DEFAULT_CONFIG = copy.deepcopy(QPLEX_FOCUS2_DEFAULT_CONFIG)
DEFAULT_CONFIG["mixer"] = "qplex_focus3"
DEFAULT_CONFIG["focus"].update({
    "enabled": True,
    "mode": "cell",
    "use_env_params": True,
    "horizon": 3,
    "horizon_discount": 1.0,
    "horizon_weights": "uniform",
    "alpha_credit": 0.02,
    "beta_belief": 0.005,
    "cell_grid_size": [10, 10],
    "cell_subpoints_per_cell": 4,
    "cell_soft_label_sigma": 150.0,
    "cell_x_range": [-1000.0, 1000.0],
    "cell_y_range": [-1000.0, 1000.0],
    "confidence_gate_enabled": True,
    "confidence_gate_mode": "auto",
    "confidence_entropy_kappa": 2.0,
    "confidence_loss_threshold": None,
    "confidence_loss_temperature": 1.0,
    "use_action_selection": True,
    "target_weights": None,
    "min_credit_signal": 1e-6,
    "eps": 1e-6,
})


class QPlexFocus3Trainer(QPlexFocus2Trainer):
    @classmethod
    @override(QPlexFocus2Trainer)
    def get_default_config(cls) -> TrainerConfigDict:
        return DEFAULT_CONFIG

    @override(QPlexFocus2Trainer)
    def validate_config(self, config: TrainerConfigDict) -> None:
        super().validate_config(config)
        if config["framework"] != "torch":
            raise ValueError("Only `framework=torch` supported for QPLEX_FOCUS3.")

    @override(QPlexFocus2Trainer)
    def get_default_policy_class(self, config: TrainerConfigDict) -> Type[Policy]:
        return QPLEXFocus3TorchPolicy
