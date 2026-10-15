from ray.rllib.agents.concaveq import ConcaveQTorchPolicy
from examples.value_factorization.camera.agent import ValueFactorizationCameraAgent
from .config import config


class ConcaveQCameraAgent(ValueFactorizationCameraAgent):
    POLICY_CLASS = ConcaveQTorchPolicy
    DEFAULT_CONFIG = config
