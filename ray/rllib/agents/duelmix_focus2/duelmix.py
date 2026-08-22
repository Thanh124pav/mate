import copy
from typing import Type

from ray.rllib.agents.duelmix_focus.duelmix import DuelMixTrainer as _DuelMixFocusTrainer
from ray.rllib.agents.duelmix_focus.duelmix import DEFAULT_CONFIG as DUELMIX_FOCUS_DEFAULT_CONFIG
from ray.rllib.agents.duelmix_focus2.duelmix_policy import DuelMixFocus2TorchPolicy
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import TrainerConfigDict


DEFAULT_CONFIG = copy.deepcopy(DUELMIX_FOCUS_DEFAULT_CONFIG)
DEFAULT_CONFIG["focus"].update({
    "enabled": True,
    "variant": "focus2_lstm_gh",
    "belief_mode": "learned",
    "belief_arch": "lstm",
    "belief_num_layers": 1,
    "belief_dropout": 0.0,
    "belief_hidden_dim": 256,
    "horizon": 3,
    "horizon_discount": 0.9,
    "belief_max_delta": 400.0,
    "belief_min_std": 25.0,
    "integral_mode": "sigma",
    "sigma_method": "gauss_hermite",
    "sigma_order": 5,
    "sample_chunk_size": 32,
    "use_action_selection": False,
    "alpha_credit": 0.05,
    "beta_belief": 0.01,
    "confidence_gate_enabled": True,
    "confidence_gate_mode": "auto",
    "confidence_entropy_kappa": 2.0,
    "confidence_loss_threshold": None,
    "confidence_loss_temperature": 1.0,
})


class DuelMixFocus2Trainer(_DuelMixFocusTrainer):
    @classmethod
    @override(_DuelMixFocusTrainer)
    def get_default_config(cls) -> TrainerConfigDict:
        return DEFAULT_CONFIG

    @override(_DuelMixFocusTrainer)
    def get_default_policy_class(self, config: TrainerConfigDict) -> Type[Policy]:
        return DuelMixFocus2TorchPolicy
