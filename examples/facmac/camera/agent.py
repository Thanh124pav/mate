from ray.rllib.agents.facmac import FACMACTorchPolicy
from examples.value_factorization.camera.agent import ValueFactorizationCameraAgent
from .config import config


class FACMACCameraAgent(ValueFactorizationCameraAgent):
    POLICY_CLASS = FACMACTorchPolicy
    DEFAULT_CONFIG = config
