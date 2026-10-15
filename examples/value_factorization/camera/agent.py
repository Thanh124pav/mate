"""Checkpoint-compatible camera-agent adapter shared by the new algorithms."""

import mate

from examples.utils import RLlibGroupedPolicyMixIn
from .config import make_env


class ValueFactorizationCameraAgent(RLlibGroupedPolicyMixIn, mate.CameraAgentBase):
    DEFAULT_CONFIG = None
    POLICY_CLASS = None

    def __init__(self, config=None, checkpoint_path=None, make_env_fn=make_env, seed=None):
        super().__init__(
            config=config,
            checkpoint_path=checkpoint_path,
            make_env=make_env_fn,
            seed=seed,
        )
        self.frame_skip = int(self.config.get("env_config", {}).get("frame_skip", 1))
        levels = self.config.get("env_config", {}).get("discrete_levels")
        if levels is None:
            raise ValueError("discrete_levels is required")
        self.normalized_action_grid = mate.DiscreteCamera.discrete_action_grid(levels=levels)
        self.last_action = None

    def reset(self, observation):
        super().reset(observation)
        self.last_action = None

    def act(self, observation, info=None, deterministic=None):
        self.state, observation, info, messages = self.check_inputs(observation, info)
        if self.episode_step % self.frame_skip == 0:
            self.last_action, self.hidden_state = self.compute_single_action(
                observation,
                state=self.hidden_state,
                info=info,
                deterministic=deterministic,
            )
            self.last_action = (
                self.action_space.high * self.normalized_action_grid[self.last_action]
            )
        return self.last_action
