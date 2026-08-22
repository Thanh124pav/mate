import copy
from typing import Type

from ray.rllib.agents.qplex_focus.qplex import (
    DEFAULT_CONFIG as QPLEX_FOCUS_DEFAULT_CONFIG,
    QPlexFocusTrainer,
)
from ray.rllib.agents.qplex_focus2.qplex_policy import QPLEXFocus2TorchPolicy
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import TrainerConfigDict


DEFAULT_CONFIG = copy.deepcopy(QPLEX_FOCUS_DEFAULT_CONFIG)
DEFAULT_CONFIG["mixer"] = "qplex_focus2"
DEFAULT_CONFIG["focus"].update(
    {
        "variant": "focus2_lstm_gh",
        "belief_mode": "learned",
        "belief_arch": "lstm",
        "belief_num_layers": 1,
        "belief_dropout": 0.0,
        "belief_hidden_dim": 256,
        "integral_mode": "sigma",
        "sigma_method": "gauss_hermite",
        "sigma_order": 5,
        "sample_chunk_size": 32,
        "use_action_selection": False,
        "alpha_credit": 0.05,
        "beta_belief": 0.01,
    }
)


class QPlexFocus2Trainer(QPlexFocusTrainer):
    """FOCUS2: LSTM belief model + weighted Gaussian-Hermite sigma integration."""

    @classmethod
    @override(QPlexFocusTrainer)
    def get_default_config(cls) -> TrainerConfigDict:
        return DEFAULT_CONFIG

    @override(QPlexFocusTrainer)
    def validate_config(self, config: TrainerConfigDict) -> None:
        super().validate_config(config)
        if config["framework"] != "torch":
            raise ValueError("Only `framework=torch` supported for QPLEX_FOCUS2.")

    @override(QPlexFocusTrainer)
    def get_default_policy_class(self, config: TrainerConfigDict) -> Type[Policy]:
        return QPLEXFocus2TorchPolicy
