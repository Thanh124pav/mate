from ray.rllib.agents.concaveq import ConcaveQFocusTorchPolicy
from examples.value_factorization.camera.agent import ValueFactorizationCameraAgent
from .config import config


class ConcaveQFocussCameraAgent(ValueFactorizationCameraAgent):
    POLICY_CLASS = ConcaveQFocusTorchPolicy
    DEFAULT_CONFIG = config
