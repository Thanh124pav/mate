from ray.rllib.agents.resq import ResQTorchPolicy
from examples.value_factorization.camera.agent import ValueFactorizationCameraAgent
from .config import config


class ResQCameraAgent(ValueFactorizationCameraAgent):
    POLICY_CLASS = ResQTorchPolicy
    DEFAULT_CONFIG = config
