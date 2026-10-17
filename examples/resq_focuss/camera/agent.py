from ray.rllib.agents.resq import ResQFocusTorchPolicy
from examples.value_factorization.camera.agent import ValueFactorizationCameraAgent
from .config import config


class ResQFocussCameraAgent(ValueFactorizationCameraAgent):
    POLICY_CLASS = ResQFocusTorchPolicy
    DEFAULT_CONFIG = config
