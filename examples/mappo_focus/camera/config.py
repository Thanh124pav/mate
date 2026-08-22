import copy

from examples.mappo.camera.config import config as _mappo_config
from examples.mappo.camera.config import make_env, target_agent_factory


config = copy.deepcopy(_mappo_config)
focus_config = config["model"]["custom_model_config"].setdefault("focus", {})
focus_config.update(
    {
        "enabled": True,
        "policy_eta": 0.5,
        "use_confidence": True,
        "beta_belief": 0.0,
        "belief_state_enabled": True,
        "belief_state_coeff": 0.05,
        "belief_state_hidden_dim": 256,
        "belief_state_loss": "mse",
        "belief_state_detach_for_prior": True,
        "action_prior_enabled": True,
        "action_prior_state_source": "belief",
        "action_prior_apply_to_logits": True,
        "action_prior_logit_eta": 1.0,
        "action_prior_coeff": 0.1,
        "action_prior_bias_eta": 3.0,
        "action_prior_temperature": 0.75,
        "action_prior_use_confidence": True,
        "action_prior_use_positive_advantage": True,
        "action_prior_center_advantage": True,
        "action_prior_advantage_clip": 5.0,
        "action_prior_weight_floor": 0.05,
        "action_prior_use_focus_valid": False,
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
