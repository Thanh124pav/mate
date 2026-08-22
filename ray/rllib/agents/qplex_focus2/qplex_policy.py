import copy

from ray.rllib.agents.qplex_focus.qplex_policy import QPLEXFocusTorchPolicy


class QPLEXFocus2TorchPolicy(QPLEXFocusTorchPolicy):
    """QPLEX FOCUS2 policy using the new LSTM + Gauss-Hermite FOCUS stack."""

    def __init__(self, obs_space, action_space, config):
        from ray.rllib.agents.qplex_focus2.qplex import DEFAULT_CONFIG

        merged_config = copy.deepcopy(DEFAULT_CONFIG)
        for key, value in (config or {}).items():
            if isinstance(value, dict) and isinstance(merged_config.get(key), dict):
                merged_config[key].update(value)
            else:
                merged_config[key] = value

        bootstrap_config = copy.deepcopy(merged_config)
        bootstrap_config["mixer"] = "qplex_focus"
        super().__init__(obs_space, action_space, bootstrap_config)
        self.config = merged_config
