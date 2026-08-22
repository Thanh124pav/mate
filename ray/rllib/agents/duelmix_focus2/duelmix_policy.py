import copy

from ray.rllib.agents.duelmix_focus.duelmix_policy import DuelMixTorchPolicy


class DuelMixFocus2TorchPolicy(DuelMixTorchPolicy):
    """DuelMIX FOCUS2 policy using the new LSTM + Gauss-Hermite FOCUS stack."""

    def __init__(self, obs_space, action_space, config):
        from ray.rllib.agents.duelmix_focus2.duelmix import DEFAULT_CONFIG

        merged_config = copy.deepcopy(DEFAULT_CONFIG)
        for key, value in (config or {}).items():
            if isinstance(value, dict) and isinstance(merged_config.get(key), dict):
                merged_config[key].update(value)
            else:
                merged_config[key] = value
        super().__init__(obs_space, action_space, merged_config)
        self.config = merged_config
