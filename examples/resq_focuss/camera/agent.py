from ray.rllib.agents.resq import ResQFocussTorchPolicy
from examples.value_factorization.camera.agent import ValueFactorizationCameraAgent
from .config import config


class ResQFocussCameraAgent(ValueFactorizationCameraAgent):
    POLICY_CLASS = ResQFocussTorchPolicy
    DEFAULT_CONFIG = config
