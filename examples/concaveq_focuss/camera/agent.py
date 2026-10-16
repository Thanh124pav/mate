from ray.rllib.agents.concaveq import ConcaveQFocussTorchPolicy
from examples.value_factorization.camera.agent import ValueFactorizationCameraAgent
from .config import config


class ConcaveQFocussCameraAgent(ValueFactorizationCameraAgent):
    POLICY_CLASS = ConcaveQFocussTorchPolicy
    DEFAULT_CONFIG = config
