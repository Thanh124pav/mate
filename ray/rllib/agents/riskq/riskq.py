"""RLlib trainers and official defaults for RiskQ and RiskQ+FOCUS."""

import copy
from typing import Type

from ray.rllib.agents.trainer import with_common_config
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import TrainerConfigDict

from ray.rllib.agents.value_factorization.trainer import ValueFactorizationTrainer


DEFAULT_CONFIG = with_common_config(
    {
        "batch_mode": "complete_episodes",
        "framework": "torch",
        "double_q": True,
        "gamma": 0.99,
        "td_lambda": 0.6,
        "lr": 0.001,
        "optim_alpha": 0.99,
        "optim_eps": 1e-5,
        "grad_norm_clipping": 10.0,
        "n_quantiles": 32,
        "n_target_quantiles": 32,
        "risk_type": "wang",
        "risk_param": 0.75,
        "sort_quantiles": True,
        "n_query_embedding_layer1": 32,
        "n_query_embedding_layer2": 16,
        "n_key_embedding_layer1": 16,
        "n_head_embedding_layer1": 32,
        "n_head_embedding_layer2": 1,
        "n_attention_head": 1,
        "n_constant_value": 16,
        "agent_own_state_start": 13,
        "agent_own_state_size": 9,
        "qmix_loss": 1.0,
        "strict_shared_reward": True,
        "exploration_config": {
            "type": "EpsilonGreedy",
            "initial_epsilon": 1.0,
            "final_epsilon": 0.05,
            "epsilon_timesteps": 100000,
        },
        "evaluation_interval": None,
        "evaluation_duration": 10,
        "evaluation_config": {"explore": False},
        "timesteps_per_iteration": 1000,
        "target_network_update_freq": 200,
        "buffer_size": 5000,
        "learning_starts": 1000,
        "rollout_fragment_length": 0,
        "train_batch_size": 32,
        "num_workers": 0,
        "min_time_s_per_reporting": 1,
        "model": {"lstm_cell_size": 64, "max_seq_len": 999999},
        "focus": {"enabled": False},
    }
)

FOCUS_CONFIG = copy.deepcopy(DEFAULT_CONFIG)
FOCUS_CONFIG["focus"] = {
    "enabled": True,
    "official_mode": True,
    "credit_level": "both",
    "credit_mechanism": "align",
    "carrier": "grad",
    "route_strength": 1.0,
    "use_env_params": True,
    "eta": 1.0,
    "alpha_credit": 0.1,
    "beta_belief": 0.05,
    "belief_lr": 1e-4,
    "belief_mode": "learned",
    "horizon": 9,
    "horizon_discount": 0.9,
    "belief_hidden_dim": 512,
    "belief_arch": "mlp",
    "belief_num_layers": 1,
    "belief_dropout": 0.0,
    "belief_distribution": "gaussian",
    "belief_max_delta": 400.0,
    "belief_min_std": 25.0,
    "belief_input_normalization": True,
    "belief_input_clip": 5.0,
    "belief_grid_size": 21,
    "belief_grid_max_delta": 400.0,
    "belief_grid_label_std": None,
    "belief_persistence_prior_std": None,
    "belief_position_loss_coeff": 0.1,
    "belief_grid_chunk_size": 128,
    "belief_safeguard_mode": "zero",
    "belief_blend_warmup_updates": 20,
    "belief_blend_ramp_updates": 80,
    "belief_error_ema_decay": 0.95,
    "belief_gate_margin": 0.0,
    "belief_gate_full_improvement": 0.2,
    "integral_mode": "MC",
    "mc_num_points": 128,
    "mc_chunk_size": 32,
    "mc_seed": 0,
    "sigma_method": "gauss_hermite",
    "sigma_order": 3,
    "use_action_selection": True,
    "action_credit_enabled": True,
    "action_credit_coeff": 1.0,
    "action_alpha": 1.0,
    "action_credit_temperature": 1.0,
    "action_temperature": 1.0,
    "action_q_std_floor": 0.01,
    "action_gradient_floor": 0.10,
    "action_grad_floor": 0.10,
    "action_gradient_ema_decay": 0.9,
    "action_grad_ema": 0.9,
    "signal_weight_min": 0.1,
    "signal_weight_max": 3.0,
    "min_credit_signal": 1e-6,
    "use_signal_confidence": True,
    "eps": 1e-8,
}


class RiskQTrainer(ValueFactorizationTrainer):
    @classmethod
    @override(ValueFactorizationTrainer)
    def get_default_config(cls) -> TrainerConfigDict:
        return DEFAULT_CONFIG

    @override(ValueFactorizationTrainer)
    def get_default_policy_class(self, config: TrainerConfigDict) -> Type[Policy]:
        from .riskq_policy import RiskQTorchPolicy

        return RiskQTorchPolicy


class RiskQFocusTrainer(ValueFactorizationTrainer):
    @classmethod
    @override(ValueFactorizationTrainer)
    def get_default_config(cls) -> TrainerConfigDict:
        return FOCUS_CONFIG

    @override(ValueFactorizationTrainer)
    def get_default_policy_class(self, config: TrainerConfigDict) -> Type[Policy]:
        from .riskq_policy import RiskQFocusTorchPolicy

        return RiskQFocusTorchPolicy
