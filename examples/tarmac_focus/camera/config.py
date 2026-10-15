import copy

from examples.tarmac.camera.config import config as _tarmac_config
from examples.tarmac.camera.config import make_env, target_agent_factory


config = copy.deepcopy(_tarmac_config)
focus_config = config["model"]["custom_model_config"].setdefault("focus", {})
focus_config.update(
    {
        "enabled": True,
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
        "policy_eta": 0.5,
        "use_confidence": True,
        "beta_belief": 0.01,
        "horizon": 3,
        "horizon_discount": 0.9,
        "belief_hidden_dim": 256,
        "belief_max_delta": 400.0,
        "belief_min_std": 25.0,
        "integral_mode": "MC",
        "mc_num_points": 128,
        "mc_chunk_size": 32,
        "mc_seed": 0,
        "grid_size": 64,
        "grid_chunk_size": 128,
        "grid_x_range": (-1000.0, 1000.0),
        "grid_y_range": (-1000.0, 1000.0),
        "use_signal_confidence": True,
        "signal_weight_min": 0.1,
        "signal_weight_max": 3.0,
        "eps": 1e-8,
    }
)
config["model"]["custom_model_config"]["env_config"] = config["env_config"]

__all__ = ["config", "make_env", "target_agent_factory"]
