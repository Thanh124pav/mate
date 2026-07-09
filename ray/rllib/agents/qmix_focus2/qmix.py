import copy
from typing import Type

from ray.rllib.agents.qmix_focus.qmix import QMixTrainer as _QMixFocusTrainer
from ray.rllib.agents.qmix_focus.qmix import DEFAULT_CONFIG as QMIX_FOCUS_DEFAULT_CONFIG
from ray.rllib.agents.qmix_focus2.qmix_policy import QMixFocus2TorchPolicy
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import TrainerConfigDict


DEFAULT_CONFIG = copy.deepcopy(QMIX_FOCUS_DEFAULT_CONFIG)
DEFAULT_CONFIG["focus"].update({
    "enabled": True,
    "belief_type": "qmc_discrete",
    "alpha_credit": 0.05,
    "beta_belief": 0.01,
    "horizon": 9,
    "horizon_discount": 0.9,
    "belief_hidden_dim": 512,
    "qmc_num_points": 128,
    "qmc_chunk_size": 64,
    "qmc_seed": 0,
    "qmc_x_range": (-1000.0, 1000.0),
    "qmc_y_range": (-1000.0, 1000.0),
    "qmc_soft_label_sigma": 100.0,
    "mixture_components": 4,
    "mixture_min_std": 25.0,
    "mixture_max_delta": 400.0,
    "confidence_gate_enabled": True,
    "confidence_gate_mode": "auto",
    "confidence_entropy_kappa": 2.0,
    "confidence_loss_threshold": None,
    "confidence_loss_temperature": 1.0,
})


class QMixFocus2Trainer(_QMixFocusTrainer):
    @classmethod
    @override(_QMixFocusTrainer)
    def get_default_config(cls) -> TrainerConfigDict:
        return DEFAULT_CONFIG

    @override(_QMixFocusTrainer)
    def get_default_policy_class(self, config: TrainerConfigDict) -> Type[Policy]:
        return QMixFocus2TorchPolicy
