import copy
import numpy as np

from ray.rllib.agents.qplex_wm.qplex_policy import QPLEXWMTorchPolicy

import mate
from examples.qplex_wm.camera.config import config as _config
from examples.qplex_wm.camera.config import make_env as _make_env
from examples.utils import RLlibGroupedPolicyMixIn


class QPLEXWM_CameraAgent(RLlibGroupedPolicyMixIn, mate.CameraAgentBase):
    """QPLEX + World Model Camera Agent (pure, DiscreteCamera).

    Each camera directly controls orientation & viewing angle.
    Discrete actions are converted back to continuous camera actions at inference.
    """

    POLICY_CLASS = QPLEXWMTorchPolicy
    DEFAULT_CONFIG = copy.deepcopy(_config)

    def __init__(self, config=None, checkpoint_path=None, make_env=_make_env, seed=None):
        super().__init__(
            config=config, checkpoint_path=checkpoint_path, make_env=make_env, seed=seed
        )

        self.frame_skip = self.config.get('env_config', {}).get('frame_skip', 1)
        self.discrete_levels = self.config.get('env_config', {}).get('discrete_levels', 5)
        self.normalized_action_grid = mate.DiscreteCamera.discrete_action_grid(
            levels=self.discrete_levels
        )
        self.last_action = None

    def reset(self, observation):
        super().reset(observation)
        self.last_action = None

    def act(self, observation, info=None, deterministic=None):
        self.state, observation, info, messages = self.check_inputs(observation, info)

        if self.episode_step % self.frame_skip == 0:
            action_idx, self.hidden_state = self.compute_single_action(
                observation, state=self.hidden_state, info=info, deterministic=deterministic
            )
            # Convert discrete action to continuous camera action
            self.last_action = self.action_space.high * self.normalized_action_grid[action_idx]

        return self.last_action
