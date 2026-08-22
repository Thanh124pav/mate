import copy

from examples.smpe2.camera.config import config as _smpe2_config
from examples.smpe2.camera.config import make_env, target_agent_factory
from examples.smpe2_focus.models import SMPE2FocusModel


config = copy.deepcopy(_smpe2_config)
config["model"]["custom_model"] = SMPE2FocusModel
config["env_config"]["reward_coefficients"] = {"real_coverage_rate": 1.0}
config["env_config"]["reward_reduction"] = "mean"
config["env_config"]["intrinsic_coeff"] = 0.2
focus_config = config["model"]["custom_model_config"].setdefault("focus", {})
focus_config.update(
    {
        "enabled": True,
        "policy_eta": 1.0,
        "use_confidence": True,
        "beta_belief": 0.02,
        "horizon": 5,
        "horizon_discount": 0.9,
        "belief_hidden_dim": 512,
        "belief_max_delta": 400.0,
        "belief_min_std": 25.0,
        "integral_mode": "MC",
        "mc_num_points": 128,
        "mc_chunk_size": 64,
        "min_credit_signal": 1e-6,
        "action_bias_eta": 3.0,
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
