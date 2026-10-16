import copy

from examples.mappo.camera.config import config as _mappo_config
from examples.mappo.camera.config import make_env, target_agent_factory


config = copy.deepcopy(_mappo_config)
focus_config = config["model"]["custom_model_config"].setdefault("focus", {})
focus_config.update(
    {
        "enabled": True,
        "official_mode": True,
        "credit_level": "both",
        "credit_mechanism": "route",
        "belief_safeguard_mode": "zero",
        "belief_distribution": "gaussian",
        "belief_grid_size": 21,
        "belief_grid_max_delta": 400.0,
        "belief_grid_label_std": None,
        "belief_persistence_prior_std": None,
        "belief_position_loss_coeff": 0.1,
        "belief_grid_chunk_size": 128,
        "belief_blend_warmup_updates": 20,
        "belief_blend_ramp_updates": 80,
        "belief_error_ema_decay": 0.95,
        "belief_gate_margin": 0.0,
        "belief_gate_full_improvement": 0.2,
        "policy_eta": 1.0,
        "use_confidence": True,
        "alpha_credit": 0.1,
        # Keep the centralized occupancy predictor trainable; without this
        # loss EMA/blending modes would remain at persistence forever.
        "beta_belief": 0.05,
        "horizon": 9,
        "horizon_discount": 0.9,
        "belief_hidden_dim": 512,
        "belief_input_normalization": True,
        "belief_input_clip": 5.0,
        "integral_mode": "MC",
        "mc_num_points": 128,
        "mc_chunk_size": 32,
        "mc_seed": 0,
        "belief_state_enabled": False,
        "belief_state_coeff": 0.05,
        "belief_state_hidden_dim": 256,
        "belief_state_loss": "mse",
        "belief_state_detach_for_prior": True,
        "action_prior_enabled": True,
        "action_prior_state_source": "global",
        "action_prior_apply_to_logits": False,
        "action_prior_logit_eta": 1.0,
        "action_prior_coeff": 1.0,
        "action_credit_enabled": True,
        "action_alpha": 1.0,
        "action_temperature": 1.0,
        "action_q_std_floor": 0.01,
        "action_grad_floor": 0.10,
        "action_grad_ema": 0.9,
        "action_gradient_rule": "match",
        "action_gradient_ratio": 0.10,
        "action_gradient_ema_decay": 0.9,
        "action_prior_bias_eta": 3.0,
        "action_prior_temperature": 1.0,
        "action_prior_use_confidence": True,
        "action_prior_use_positive_advantage": False,
        "action_prior_center_advantage": True,
        "action_prior_advantage_clip": 5.0,
        "action_prior_weight_floor": 0.05,
        "action_prior_use_focus_valid": True,
        "action_bias_rotation_step": 5.0,
        "action_bias_zooming_step": 2.5,
        "action_bias_min_viewing_angle": 30.0,
        "action_bias_max_viewing_angle": 180.0,
        "action_bias_max_sight_range": 1500.0,
        "action_bias_real_only": True,
        "action_bias_clip": 3.0,
        "action_bias_angular_temp": 0.18,
        "action_bias_range_temp": 150.0,
    }
)
config["model"]["custom_model_config"]["env_config"] = config["env_config"]

__all__ = ["config", "make_env", "target_agent_factory"]
