"""SMPE2CameraAgent — inference agent for SMPE2."""

import copy

from ray.rllib.agents.ppo import PPOTorchPolicy

import mate
from examples.smpe2.camera.config import config as _config
from examples.smpe2.camera.config import make_env as _make_env
from examples.utils import RLlibPolicyMixIn


class SMPE2CameraAgent(RLlibPolicyMixIn, mate.CameraAgentBase):
    """SMPE2 Camera Agent using PPO with variational belief.

    Args:
        config: Config dict (default: SMPE2 config).
        checkpoint_path: Path to trained checkpoint.
        make_env: Factory function to create env (for obs/action spaces).
        seed: Random seed.
    """

    POLICY_CLASS = PPOTorchPolicy
    DEFAULT_CONFIG = copy.deepcopy(_config)

    def __init__(self, config=None, checkpoint_path=None, make_env=_make_env, seed=None):
        super().__init__(
            config=config,
            checkpoint_path=checkpoint_path,
            make_env=make_env,
            seed=seed,
        )
